"""SynLogic validation and reproducible dataset-size/noise baselines."""

import json
from pathlib import Path

import torch

from src.dapo_ray_trainer import RayDAPOTrainer
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from .sampling import PlanDataLoader, PlanSampler


class ValidationRecorder:
    def __init__(self, reward):
        self.reward, self.records = reward, []

    def __call__(self, data, return_dict=False):
        result = self.reward(data, return_dict=True)
        for i, score in enumerate(result['reward_tensor'].sum(-1).tolist()):
            self.records.append((int(data[i].non_tensor_batch['extra_info']['index']), score))
        return result if return_dict else result['reward_tensor']


class SynLogicTrainer(RayDAPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_reward_fn = ValidationRecorder(self.val_reward_fn)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        from verl.trainer.main_ppo import create_rl_dataset
        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        sampler = PlanSampler(train_dataset, self.config.mixture.condition,
                              seed=self.config.data.seed, batch_size=self.config.data.train_batch_size)
        super()._create_dataloader(train_dataset, val_dataset, collate_fn, sampler)
        original = self.train_dataloader
        self.train_dataloader = PlanDataLoader(dataset=train_dataset, batch_size=original.batch_size,
                                             num_workers=0, drop_last=True,
                                             collate_fn=original.collate_fn, sampler=sampler)

    def _save_checkpoint(self):
        folder = Path(self.config.trainer.default_local_dir) / f'global_step_{self.global_steps}'
        folder.mkdir(parents=True, exist_ok=True)
        state = {'step': self.global_steps, 'protocol_id': self.config.provenance.protocol_id,
                 'prompt_template_sha256': self.config.provenance.prompt_template_sha256,
                 'condition': self.config.mixture.condition, 'world_size': self.config.trainer.n_gpus_per_node}
        path = folder / 'experiment_state.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(path)
        super()._save_checkpoint()

    def _load_checkpoint(self):
        result = super()._load_checkpoint()
        if self.global_steps:
            folder = Path(self.config.trainer.default_local_dir) / f'global_step_{self.global_steps}'
            state = json.loads((folder / 'experiment_state.json').read_text())
            if (state['step'] != self.global_steps or state['protocol_id'] != self.config.provenance.protocol_id
                    or state['prompt_template_sha256'] != self.config.provenance.prompt_template_sha256
                    or state['world_size'] != self.config.trainer.n_gpus_per_node):
                raise ValueError('Incompatible full-state recovery checkpoint')
        return result

    def _validate(self):
        self.val_reward_fn.records.clear()
        metrics = super()._validate()
        records = self.val_reward_fn.records
        if len(records) != 270 * 8:
            raise ValueError('Validation response count changed')
        by_index = {}
        for index, score in records:
            by_index.setdefault(index, []).append(score)
        if set(by_index) != set(range(270)) or any(len(scores) != 8 for scores in by_index.values()):
            raise ValueError('Validation IDs or group sizes changed')
        masks = json.loads((Path(self.config.mixture.root) / 'supervision/evaluation-masks.json').read_text())
        for name, indices in masks.items():
            scores = [score for i in indices for score in by_index[i]]
            metrics[f'val-core/{name}/reward/mean@8'] = sum(scores) / len(scores)
        return metrics

    def _progress(self, phase, metrics=None):
        if metrics is not None:
            metrics['mixture/world_size'] = self.config.trainer.n_gpus_per_node
        return super()._progress(phase, metrics)
