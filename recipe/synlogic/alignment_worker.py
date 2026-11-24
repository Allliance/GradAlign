"""Released GradAlign loss with exact full-gradient cosine scores on FSDP shards."""

import contextlib
import importlib.util
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist

from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.fsdp_workers import ActorRolloutRefWorker


def upstream_loss():
    path = Path(__file__).resolve().parents[2] / 'select/grpo_loss.py'
    spec = importlib.util.spec_from_file_location('gradalign_released_loss', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AlignmentWorker(ActorRolloutRefWorker):
    def _alignment_backward(self, data, start, samples, scale=1.0):
        world, rank = dist.get_world_size(), dist.get_rank()
        if samples % world:
            raise ValueError('Gradient rollouts must divide evenly across FSDP ranks')
        loss_function = upstream_loss().compute_policy_loss
        device = next(self.actor.actor_module.parameters()).device
        # All ranks process the same problem and equal counts of its responses.
        # FSDP therefore averages that problem's full gradient, rather than
        # accidentally mixing different problems in its gradient reductions.
        per_rank = samples // world
        longest = int(data.batch['attention_mask'][start:start+samples].sum(-1).max())
        budget = self.config.actor.ppo_max_token_len_per_gpu
        chunk_size = min(per_rank, max(1, budget // longest))
        for offset in range(0, per_rank, chunk_size):
            indices = [start + rank + index * world for index in range(offset, min(offset+chunk_size, per_rank))]
            batch = data.batch[indices].to(device)
            mask = batch['attention_mask'][:, -batch['responses'].shape[-1]:]
            with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink):
                _, log_prob = self.actor._forward_micro_batch(
                    batch, temperature=1.0, calculate_entropy=False)
                loss = loss_function(log_prob.detach(), log_prob, batch['advantages'], mask)
                (loss * scale / per_rank).backward()

    def _alignment_gradients(self):
        return [parameter.grad.detach().float().clone() if parameter.grad is not None
                else torch.zeros_like(parameter, dtype=torch.float32)
                for parameter in self.actor.actor_module.parameters()]

    @staticmethod
    def _norm(gradients):
        value = sum((gradient.square().sum() for gradient in gradients),
                    torch.zeros((), device=gradients[0].device))
        dist.all_reduce(value)
        norm = float(value.sqrt())
        if not math.isfinite(norm):
            raise FloatingPointError('Non-finite GradAlign gradient')
        return norm

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def gradalign_reference(self, data, samples):
        groups = len(data) // samples
        if not groups or len(data) != groups * samples:
            raise ValueError('Incomplete trusted rollout groups')
        self.actor.actor_module.train()
        self.actor.actor_optimizer.zero_grad(set_to_none=True)
        informative = 0
        with self.ulysses_sharding_manager:
            for group in range(groups):
                start = group * samples
                if data.batch['advantages'][start:start+samples].abs().max().item() == 0:
                    continue
                self._alignment_backward(data, start, samples, scale=1/groups)
                informative += 1
        gradients = self._alignment_gradients()
        norm = self._norm(gradients)
        if norm:
            for gradient in gradients:
                gradient.div_(norm)
        self._gradalign_reference = gradients
        self.actor.actor_optimizer.zero_grad(set_to_none=True)
        return {'reference_norm': norm, 'reference_problems': groups,
                'informative_reference_problems': informative}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def gradalign_scores(self, data, samples):
        reference = getattr(self, '_gradalign_reference', None)
        if reference is None:
            raise ValueError('GradAlign reference gradient has not been computed')
        groups = len(data) // samples
        if len(data) != groups * samples:
            raise ValueError('Incomplete candidate rollout groups')
        scores, informative = [], 0
        with self.ulysses_sharding_manager:
            for group in range(groups):
                start = group * samples
                if data.batch['advantages'][start:start+samples].abs().max().item() == 0:
                    scores.append(0.0)
                    continue
                self.actor.actor_optimizer.zero_grad(set_to_none=True)
                self._alignment_backward(data, start, samples)
                gradients = self._alignment_gradients()
                norm = self._norm(gradients)
                dot = sum((left.mul(right).sum() for left, right in zip(reference, gradients)),
                          torch.zeros((), device=reference[0].device))
                dist.all_reduce(dot)
                score = float(dot) / norm if norm else 0.0
                if not math.isfinite(score):
                    raise FloatingPointError('Non-finite GradAlign cosine score')
                scores.append(score)
                informative += int(norm > 0)
                del gradients
        self.actor.actor_optimizer.zero_grad(set_to_none=True)
        return {'scores': scores, 'informative_candidate_problems': informative}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def gradalign_clear(self):
        self._gradalign_reference = None
        self.actor.actor_optimizer.zero_grad(set_to_none=True)
        return True
