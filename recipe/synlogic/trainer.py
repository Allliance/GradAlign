"""SynLogic validation and reproducible dataset-size/noise baselines."""

import json
import os
from pathlib import Path
from omegaconf import OmegaConf

import torch

from src.dapo_ray_trainer import RayDAPOTrainer
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from .sampling import PlanDataLoader, PlanSampler
from .alignment import AlignmentLoader, AlignmentMixin


class ValidationRecorder:
    def __init__(self, reward):
        self.reward, self.records = reward, []

    def __call__(self, data, return_dict=False):
        result = self.reward(data, return_dict=True)
        for i, score in enumerate(result['reward_tensor'].sum(-1).tolist()):
            self.records.append((int(data[i].non_tensor_batch['extra_info']['index']), score))
        return result if return_dict else result['reward_tensor']


class PoolReward:
    def __init__(self, reward, condition):
        self.reward, self.condition, self.metrics = reward, condition, {}

    def __call__(self, data, return_dict=False):
        result = self.reward(data, return_dict=True)
        pools = [extra.get('pool', 'all') for extra in data.non_tensor_batch['extra_info']]
        small = sum(pool == 'small' for pool in pools)
        self.metrics = {'mixture/small_prompt_fraction': small / len(pools)}
        if self.condition == 'mixture-gradient-equal':
            if not all(pool in ('small', 'large') for pool in pools):
                raise ValueError('Gradient weighting requires frozen pool membership')
            weights = [15610 / (2 * (156 if pool == 'small' else 15454)) for pool in pools]
            data.batch['sample_weights'] = torch.tensor(weights, dtype=torch.float32)
            self.metrics['mixture/mean_loss_weight'] = sum(weights) / len(weights)
        return result if return_dict else result['reward_tensor']


class SynLogicTrainer(AlignmentMixin, RayDAPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_reward_fn = ValidationRecorder(self.val_reward_fn)
        self.reward_fn = PoolReward(self.reward_fn, self.config.mixture.condition)
        self.alignment_metrics = {}

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        from verl.trainer.main_ppo import create_rl_dataset
        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        sampler = PlanSampler(train_dataset, self.config.mixture.condition,
                              seed=self.config.data.seed, batch_size=self.config.data.train_batch_size,
                              batches_per_pass=self.config.mixture.get("batches_per_pass", 97))
        super()._create_dataloader(train_dataset, val_dataset, collate_fn, sampler)
        original = self.train_dataloader
        if self.config.mixture.condition == 'mixture-gradalign':
            self.train_dataloader = AlignmentLoader(train_dataset, original.batch_size,
                                                    self.select_aligned, seed=self.config.data.seed)
        else:
            self.train_dataloader = PlanDataLoader(dataset=train_dataset, batch_size=original.batch_size,
                                                 num_workers=0, drop_last=True,
                                                 collate_fn=original.collate_fn, sampler=sampler)

    def _save_checkpoint(self):
        folder = Path(self.config.trainer.default_local_dir) / f'global_step_{self.global_steps}'
        folder.mkdir(parents=True, exist_ok=True)
        state = {'step': self.global_steps, 'protocol_id': self.config.provenance.protocol_id,
                 'prompt_template_sha256': self.config.provenance.prompt_template_sha256,
                 'condition': self.config.mixture.condition, 'world_size': self.config.trainer.n_gpus_per_node,
                 'expected_model_layers': self.config.provenance.get('model_layers', 36),
                 'expected_model_parameters': self.config.provenance.get('model_parameters', None),
                 'optimizer_updates': self.global_steps * (self.config.data.train_batch_size // self.config.actor_rollout_ref.actor.ppo_mini_batch_size)}
        state.update(data_fingerprint=self.config.provenance.data_fingerprint,
                     model_revision=self.config.provenance.model_revision,
                     source_revision=OmegaConf.to_container(self.config.provenance.source_revision, resolve=True)
                     if OmegaConf.is_config(self.config.provenance.source_revision) else self.config.provenance.source_revision,
                     allocation_id=os.environ.get('SYNLOGIC_ALLOCATION_ID'),
                     attempt=int(os.environ.get('SYNLOGIC_ATTEMPT', '1')))
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
                    or state['world_size'] != self.config.trainer.n_gpus_per_node
                    or (state.get('data_fingerprint') is not None and state['data_fingerprint'] != self.config.provenance.data_fingerprint)
                    or (state.get('model_revision') is not None and state['model_revision'] != self.config.provenance.model_revision)
                    or (state.get('expected_model_parameters') is not None and state['expected_model_parameters'] != self.config.provenance.get('model_parameters'))):
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
            metrics.update(self.reward_fn.metrics)
            metrics.update(self.alignment_metrics)
        return super()._progress(phase, metrics)

    def fit(self):
        import verl.utils.tracking as tracking
        original = tracking.Tracking
        trainer = self

        class MixtureTracking(original):
            def log(self, data, step):
                data.update(trainer.reward_fn.metrics)
                data.update(trainer.alignment_metrics)
                data['mixture/world_size'] = trainer.config.trainer.n_gpus_per_node
                return super().log(data, step)

        tracking.Tracking = MixtureTracking
        try:
            return super().fit()
        finally:
            tracking.Tracking = original
