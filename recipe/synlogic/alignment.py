"""On-policy selection, with the selected curriculum included in checkpoints."""

import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn


class AlignmentLoader:
    def __init__(self, dataset, batch_size, selector, seed=42):
        self.dataset, self.batch_size, self.selector = dataset, batch_size, selector
        self.generator = torch.Generator().manual_seed(seed)
        self.selected, self.cursor, self.epoch_cursor, self.round = [], 0, 0, 0

    def __len__(self):
        return 97

    def __iter__(self):
        if self.epoch_cursor == len(self):
            self.epoch_cursor = 0
        return self

    def __next__(self):
        if self.epoch_cursor == len(self):
            raise StopIteration
        if self.cursor == len(self.selected):
            self.selected = self.selector(self.generator, self.round)
            self.cursor = 0
            self.round += 1
        indices = self.selected[self.cursor:self.cursor+self.batch_size]
        if len(indices) != self.batch_size:
            raise ValueError('Selection cannot form the fixed global training batch')
        result = collate_fn([self.dataset[index] for index in indices])
        self.cursor += self.batch_size
        self.epoch_cursor += 1
        return result

    def state_dict(self):
        return {'version': 1, 'selected': self.selected, 'cursor': self.cursor,
                'epoch_cursor': self.epoch_cursor, 'round': self.round,
                'synlogic_plan_generator': self.generator.get_state().clone(),
                'synlogic_plan_shape': {'length': len(self.dataset), 'batch_size': self.batch_size,
                                        'mode': 'mixture-gradalign'}}

    def load_state_dict(self, state):
        shape = self.state_dict()['synlogic_plan_shape']
        if state['version'] != 1 or state['synlogic_plan_shape'] != shape:
            raise ValueError('Incompatible GradAlign curriculum checkpoint')
        selected = list(state['selected'])
        if any(not 0 <= index < len(self.dataset) for index in selected):
            raise ValueError('Invalid selected source index')
        self.selected, self.cursor = selected, state['cursor']
        self.epoch_cursor, self.round = state['epoch_cursor'], state['round']
        if not 0 <= self.cursor <= len(selected) or not 0 <= self.epoch_cursor <= len(self):
            raise ValueError('Invalid GradAlign curriculum cursor')
        self.generator.set_state(state['synlogic_plan_generator'])


class AlignmentMixin:
    def _alignment_rollouts(self, indices, samples, phase):
        from .alignment_worker import upstream_loss
        self._progress(phase)
        original = DataProto.from_single_dict(collate_fn([self.train_dataset[index] for index in indices]))
        generation = original.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                                  non_tensor_batch_keys=['raw_prompt_ids'])
        generation.meta_info.update(do_sample=True, sampling_params={
            'n': samples, 'temperature': 1.0, 'top_p': 1.0, 'top_k': -1})
        outputs = self.actor_rollout_wg.generate_sequences(generation)
        original.non_tensor_batch['uid'] = np.array([str(i) for i in indices], dtype=object)
        batch = original.repeat(repeat_times=samples, interleave=True).union(outputs)
        if len(batch) != len(indices) * samples:
            raise ValueError('Selection rollout count changed')
        reward = self.reward_fn.reward(batch, return_dict=True)['reward_tensor'].sum(-1)
        if not ((reward == 0) | (reward == 1)).all().item():
            raise ValueError('GradAlign requires binary assigned-label rewards')
        advantages = upstream_loss().compute_grpo_advantages(reward, batch.non_tensor_batch['uid'])
        batch.batch['advantages'] = advantages[:, None].expand_as(batch.batch['responses'])
        return batch

    def select_aligned(self, generator, round_index):
        started = time.monotonic()
        config = self.config.gradalign
        selected_count = self.config.data.train_batch_size * config.selection_interval
        candidate_count = selected_count * config.selection_ratio
        small, large = [], []
        for index, extra in enumerate(self.train_dataset.dataframe['extra_info']):
            (small if extra['pool'] == 'small' else large).append(index)
        if candidate_count > len(large) or len(small) % self.config.trainer.n_gpus_per_node:
            raise ValueError('Candidate or trusted set is incompatible with the distributed protocol')
        candidates = [large[i] for i in torch.randperm(len(large), generator=generator).tolist()[:candidate_count]]
        reference = self._alignment_rollouts(small, config.reference_rollouts, 'gradalign/reference_rollouts')
        self._progress('gradalign/reference_gradient')
        reference_results = self.actor_rollout_wg.gradalign_reference(reference, config.reference_rollouts)
        reference_stats = reference_results[0]
        del reference
        scores, informative = [], 0
        for start in range(0, len(candidates), config.candidate_chunk):
            chunk = candidates[start:start+config.candidate_chunk]
            batch = self._alignment_rollouts(chunk, config.candidate_rollouts,
                                              f'gradalign/candidate_rollouts_{start}_of_{candidate_count}')
            self._progress(f'gradalign/candidate_gradients_{start}_of_{candidate_count}')
            results = self.actor_rollout_wg.gradalign_scores(batch, config.candidate_rollouts)
            for other in results[1:]:
                if not np.allclose(results[0]['scores'], other['scores'], atol=1e-6):
                    raise ValueError('FSDP ranks disagree on the full-gradient cosine score')
            scores.extend(results[0]['scores'])
            informative += results[0]['informative_candidate_problems']
            del batch
        self.actor_rollout_wg.gradalign_clear()
        # Candidates were uniformly shuffled before scoring. Stable tie-breaking
        # is therefore random and reproducible; a zero gradient has score zero.
        order = sorted(range(len(candidates)), key=lambda index: scores[index], reverse=True)[:selected_count]
        selected = [candidates[index] for index in order]
        selected = [selected[index] for index in torch.randperm(len(selected), generator=generator).tolist()]
        root = Path(self.config.mixture.root)
        directory = root/'runs'/self.config.mixture.condition/'selection'
        directory.mkdir(exist_ok=True)
        record = {'round': round_index, 'policy_step': self.global_steps-1,
                  'candidate_indices': candidates, 'scores': scores, 'selected_indices': selected,
                  'reference_source_ids': [self.train_dataset.dataframe[index]['extra_info']['index'] for index in small],
                  **reference_stats, 'informative_candidate_problems': informative,
                  'selection_seconds': time.monotonic()-started}
        (directory/f'round-{round_index:04d}.json').write_text(json.dumps(record,indent=2)+'\n')
        self.alignment_metrics = {'gradalign/selection_round': round_index,
                                  'gradalign/reference_gradient_norm': reference_stats['reference_norm'],
                                  'gradalign/informative_reference_problems': reference_stats['informative_reference_problems'],
                                  'gradalign/informative_candidate_fraction': informative/candidate_count,
                                  'gradalign/selection_seconds': record['selection_seconds'],
                                  'gradalign/selected_fraction': selected_count/candidate_count}
        return selected
