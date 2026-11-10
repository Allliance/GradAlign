"""Checkpointable exact-quota sampling for clean/noisy continuation studies."""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json

import torch


def noisy_groups(additional_step, batch_size, warmup_steps, final_fraction):
    if additional_step < 1 or batch_size < 1 or warmup_steps < 0:
        raise ValueError("Invalid curriculum step or batch size")
    if not 0 <= final_fraction <= 1:
        raise ValueError("Noise fraction must lie in [0,1]")
    scale = min(additional_step / warmup_steps, 1) if warmup_steps else 1
    return round(batch_size * final_fraction * scale)


class _Pool:
    def __init__(self, indices, seed):
        self.indices = list(indices)
        self.generator = torch.Generator().manual_seed(seed)
        self.order = []
        self.position = 0

    def draw(self, count):
        if not 0 <= count <= len(self.indices):
            raise ValueError("Pool cannot provide distinct groups for this batch")
        if not count:
            return []
        # As with drop_last sampling, discard a tail too short for this quota.
        if len(self.order) - self.position < count:
            self.order = torch.randperm(len(self.indices), generator=self.generator).tolist()
            self.position = 0
        result = [self.indices[i] for i in self.order[self.position:self.position + count]]
        self.position += count
        return result

    def state_dict(self):
        return {"order": self.order.copy(), "position": self.position,
                "generator": self.generator.get_state().clone()}

    def load_state_dict(self, state):
        order, position = state["order"], state["position"]
        if (order and sorted(order) != list(range(len(self.indices)))) or not 0 <= position <= len(order):
            raise ValueError("Invalid pool permutation or cursor")
        self.order, self.position = order.copy(), position
        self.generator.set_state(state["generator"])


class CurriculumDataLoader:
    """Single-controller, non-prefetching loader; cursors advance once per batch.

    Only frozen corruption annotations determine pool membership. Reward,
    advantage, original-verifier correctness and model outputs are never inputs.
    """
    def __init__(self, original, *, parent_step, updates, warmup_steps,
                 final_fraction, seed, dataset_sha256):
        if original.num_workers != 0 or not original.drop_last:
            raise ValueError("Curriculum recovery requires num_workers=0 and drop_last")
        self.dataset, self.batch_size = original.dataset, original.batch_size
        self.collate_fn, self.num_workers, self.drop_last = original.collate_fn, 0, True
        self.position, self.last_indices = 0, []
        rows = self.dataset.dataframe.to_list()
        clean, noisy = [], []
        for i, row in enumerate(rows):
            mode = row["extra_info"]["target_mode"]
            if mode not in {"original_verifier", "fixed_wrong_answer"}:
                raise ValueError("Missing frozen corruption annotation")
            (noisy if mode == "fixed_wrong_answer" else clean).append(i)
        if updates <= 0 or parent_step < 0:
            raise ValueError("Invalid continuation budget")
        maximum = noisy_groups(max(1, warmup_steps), self.batch_size, warmup_steps, final_fraction)
        if len(clean) < self.batch_size or len(noisy) < maximum:
            raise ValueError("Insufficient clean/noisy pool")
        membership = hashlib.sha256(json.dumps([clean, noisy]).encode()).hexdigest()
        self.protocol = dict(parent_step=parent_step, updates=updates, warmup_steps=warmup_steps,
                             final_fraction=final_fraction, seed=seed, batch_size=self.batch_size,
                             dataset_sha256=dataset_sha256, membership_sha256=membership)
        self.clean, self.noisy = _Pool(clean, seed), _Pool(noisy, seed + 1)
        self.shuffle = torch.Generator().manual_seed(seed + 2)

    def __len__(self):
        return self.protocol["updates"]

    def __iter__(self):
        while self.position < len(self):
            step = self.position + 1
            count = noisy_groups(step, self.batch_size, self.protocol["warmup_steps"],
                                 self.protocol["final_fraction"])
            indices = self.clean.draw(self.batch_size - count) + self.noisy.draw(count)
            order = torch.randperm(self.batch_size, generator=self.shuffle).tolist()
            indices = [indices[i] for i in order]
            if len(set(indices)) != self.batch_size:
                raise RuntimeError("Repeated question group within a batch")
            batch = self.collate_fn([self.dataset[i] for i in indices])
            self.position, self.last_indices = step, indices
            yield batch

    def state_dict(self):
        return {"curriculum_version": 1, "protocol": deepcopy(self.protocol),
                "position": self.position, "clean": self.clean.state_dict(),
                "noisy": self.noisy.state_dict(), "shuffle": self.shuffle.get_state().clone()}

    def load_state_dict(self, state):
        if state.get("curriculum_version") != 1 or state.get("protocol") != self.protocol:
            raise ValueError("Dataset, schedule or continuation identity changed on resume")
        if not 0 <= state["position"] <= len(self):
            raise ValueError("Invalid curriculum update cursor")
        self.clean.load_state_dict(state["clean"])
        self.noisy.load_state_dict(state["noisy"])
        self.shuffle.set_state(state["shuffle"])
        self.position = state["position"]


class CurriculumMixin:
    save_first_update = True

    def _create_dataloader(self, *args, **kwargs):
        super()._create_dataloader(*args, **kwargs)
        settings = dict(self.config.curriculum)
        self.train_dataloader = CurriculumDataLoader(self.train_dataloader, **settings)

    def _load_checkpoint(self):
        result = super()._load_checkpoint()
        expected = self.global_steps - self.train_dataloader.protocol["parent_step"]
        if self.train_dataloader.position != expected:
            raise ValueError("Checkpoint and curriculum cursor refer to different updates")
        return result

    def _observe_training_batch(self, batch, metrics):
        super()._observe_training_batch(batch, metrics)
        loader = self.train_dataloader
        step = self.global_steps - loader.protocol["parent_step"]
        groups = defaultdict(list)
        for uid, extra in zip(batch.non_tensor_batch["uid"], batch.non_tensor_batch["extra_info"]):
            groups[str(uid)].append(extra)
        actual = sum(extras[0]["target_mode"] == "fixed_wrong_answer" for extras in groups.values())
        target = noisy_groups(step, loader.batch_size, loader.protocol["warmup_steps"],
                              loader.protocol["final_fraction"])
        if len(groups) != loader.batch_size or actual != target or loader.position != step:
            raise RuntimeError("Real training batch differs from the approved noise schedule")
        metrics.update({"curriculum/additional_step": step,
                        "curriculum/noisy_groups": actual,
                        "curriculum/clean_groups": loader.batch_size - actual,
                        "curriculum/noise_fraction": actual / loader.batch_size,
                        "curriculum/warmup": int(step <= loader.protocol["warmup_steps"])})
