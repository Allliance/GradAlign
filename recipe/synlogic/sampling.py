"""Checkpoint the actual index plan and generator, including replacement draws."""

import torch
from torchdata.stateful_dataloader import StatefulDataLoader


class PlanIterator:
    def __init__(self, sampler):
        self.sampler = sampler
        self.plan = sampler.plan()
        self.cursor = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.cursor == len(self.plan):
            raise StopIteration
        result = self.plan[self.cursor]
        self.cursor += 1
        return result

    def state_dict(self):
        return {'version': 1, 'shape': self.sampler.shape(), 'plan': self.plan, 'cursor': self.cursor,
                'generator': self.sampler.generator.get_state().clone()}

    def load_state_dict(self, state):
        if state['version'] != 1 or state['shape'] != self.sampler.shape():
            raise ValueError('Sampler protocol mismatch')
        self.plan, self.cursor = list(state['plan']), state['cursor']
        if len(self.plan) != len(self.sampler) or not 0 <= self.cursor <= len(self.plan):
            raise ValueError('Invalid saved index plan')
        if any(not 0 <= i < len(self.sampler.dataset) for i in self.plan):
            raise ValueError('Saved index outside dataset')
        self.sampler.generator.set_state(state['generator'])


class PlanSampler:
    def __init__(self, dataset, mode, seed=42, batch_size=160, batches_per_pass=97):
        self.dataset, self.mode, self.batch_size = dataset, mode, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.num_samples = batches_per_pass * batch_size
        pools = dataset.dataframe['extra_info']
        self.small = [i for i, extra in enumerate(pools) if extra.get('pool') == 'small']
        self.large = [i for i, extra in enumerate(pools) if extra.get('pool') == 'large']

    def shape(self):
        return {'length': len(self.dataset), 'mode': self.mode, 'num_samples': self.num_samples,
                'batch_size': self.batch_size}

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        return PlanIterator(self)

    def plan(self):
        if self.mode == 'mixture-sampling99':
            result = []
            for batch in range(self.num_samples // self.batch_size):
                nsmall = ((batch + 1) * self.batch_size * 99 // 100
                          - batch * self.batch_size * 99 // 100)
                indices = [self.small[i] for i in torch.randint(len(self.small), (nsmall,), generator=self.generator).tolist()]
                indices += [self.large[i] for i in torch.randint(len(self.large), (self.batch_size - nsmall,), generator=self.generator).tolist()]
                result.extend(indices[i] for i in torch.randperm(self.batch_size, generator=self.generator).tolist())
            return result
        if len(self.dataset) < self.num_samples:
            return torch.randint(len(self.dataset), (self.num_samples,), generator=self.generator).tolist()
        return torch.randperm(len(self.dataset), generator=self.generator).tolist()[:self.num_samples]


class PlanDataLoader(StatefulDataLoader):
    def state_dict(self):
        state = super().state_dict()
        state['synlogic_plan_generator'] = self.sampler.generator.get_state().clone()
        state['synlogic_plan_shape'] = self.sampler.shape()
        return state

    def load_state_dict(self, state):
        state = dict(state)
        if state.pop('synlogic_plan_shape') != self.sampler.shape():
            raise ValueError('Dataloader protocol mismatch')
        self.sampler.generator.set_state(state.pop('synlogic_plan_generator'))
        super().load_state_dict(state)
