"""The same noisy training prompts, with outcome-only and dense components exposed."""
import hashlib
import json
import os
from pathlib import Path
import time

from src.opd import combine_advantages, gradient_matched_coefficient
from src.noise_diagnostics import noise_metrics
from .trainer import SynLogicTrainer


class OPDTrainer(SynLogicTrainer):
    def init_workers(self):
        super().init_workers()
        students = self.actor_rollout_wg.opd_device_identity()
        teacher = self.ref_policy_wg.opd_device_identity()[0]
        if len(students) != self.config.trainer.n_gpus_per_node:
            raise ValueError('Student worker count differs from the configured allocation')
        identities = students + [teacher]
        assigned = [(item['node_id'], tuple(item['assigned'])) for item in identities]
        physical = [(item['node_id'], item['current_device']) for item in identities]
        if len(set(assigned)) != len(assigned):
            raise ValueError('OPD roles must have distinct allocated GPUs')
        if all(item['visibility_unmasked'] for item in identities) and len(set(physical)) != len(physical):
            raise ValueError('OPD roles landed on the same physical GPU')
        root = Path(os.environ['SYNLOGIC_RUN_ROOT'])
        (root/'device-placement.json').write_text(json.dumps({'students':students,'teacher':teacher},indent=2)+'\n')

    def _transform_advantages(self, batch, metrics):
        settings = self.config.opd
        outcome_metrics = noise_metrics(batch)
        for name in ('active_rate', 'active_groups', 'groups'):
            if 'noise/' + name in outcome_metrics:
                metrics['noise/outcome_' + name] = outcome_metrics['noise/' + name]
        if settings.get('fixed_teacher_coefficient') is not None:
            coefficient = float(settings.fixed_teacher_coefficient)
            if coefficient != 1.0 or settings.calibrate:
                raise ValueError('This protocol requires the user-fixed teacher coefficient1.0')
            batch.batch['outcome_advantages'] = batch.batch['advantages'].detach().clone()
            combined, dense, values = combine_advantages(
                batch.batch['advantages'], batch.batch['teacher_log_probs'], batch.batch['old_log_probs'],
                batch.batch['response_mask'], settings.outcome_coefficient, coefficient)
            batch.batch['advantages'], batch.batch['teacher_advantages'] = combined, dense
            batch.meta_info.update(opd_loss_mode='importance_sampling',opd_teacher_coefficient=coefficient,
                                   opd_outcome_coefficient=float(settings.outcome_coefficient))
            metrics.update(values)
            return batch
        path = Path(settings.calibration_file)
        if not settings.calibrate and not path.exists():
            self._progress('waiting_shared_teacher_calibration')
            deadline = time.monotonic() + settings.get('calibration_wait_seconds', 3600)
            root = Path(self.config.mixture.root)
            while not path.exists() and time.monotonic() < deadline:
                if (root/'STOP').exists() or (root/'runs'/self.config.mixture.condition/'STOP').exists():
                    raise RuntimeError('User stop while waiting for shared calibration')
                time.sleep(2)
        if not path.exists():
            if not settings.calibrate:
                raise RuntimeError('Shared teacher coefficient is not ready; launch combined arm first')
            self._progress('calibrating_teacher_gradient_scale')
            result = self.actor_rollout_wg.calibrate_opd(batch)[0]
            coefficient = gradient_matched_coefficient(result['outcome_gradient_norms'], result['teacher_gradient_norms'])
            result.update(teacher_coefficient=coefficient, source_step=settings.get('source_step', 300), calibration_step=self.global_steps,
                          training_sha256=self.config.provenance.data_fingerprint,
                          teacher_fingerprint=settings.teacher_fingerprint,
                          minibatches=4, optimizer_updates_performed=0)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
            temporary.write_text(json.dumps(result, indent=2) + '\n')
            temporary.replace(path)
        proof = json.loads(path.read_text())
        inherited = settings.get('historical_fixed_coefficient', None)
        if inherited:
            if (settings.calibrate or hashlib.sha256(path.read_bytes()).hexdigest() != inherited.file_sha256
                    or proof['teacher_coefficient'] != inherited.coefficient
                    or proof['teacher_fingerprint'] != settings.teacher_fingerprint
                    or proof['source_step'] != settings.get('source_step', 300)):
                raise ValueError('Frozen historical coefficient provenance mismatch')
            metrics['opd/historical_fixed_coefficient'] = 1
        elif (proof['training_sha256'] != self.config.provenance.data_fingerprint
                or proof['teacher_fingerprint'] != settings.teacher_fingerprint
                or proof['source_step'] != settings.get('source_step', 300)):
            raise ValueError('Calibration provenance mismatch')
        batch.batch['outcome_advantages'] = batch.batch['advantages'].detach().clone()
        combined, dense, values = combine_advantages(
            batch.batch['advantages'], batch.batch['teacher_log_probs'], batch.batch['old_log_probs'],
            batch.batch['response_mask'], settings.outcome_coefficient, proof['teacher_coefficient'])
        batch.batch['advantages'] = combined
        batch.batch['teacher_advantages'] = dense
        batch.meta_info.update(opd_teacher_coefficient=proof['teacher_coefficient'],
                               opd_outcome_coefficient=float(settings.outcome_coefficient))
        metrics.update(values)
        return batch

    def _actor_update_batch(self, batch):
        if not self.config.opd.get('preserve_global_minibatches', False):
            return batch
        if self.config.trainer.balance_batch:
            raise ValueError("Preserved optimizer minibatches require driver token balancing disabled")
        from .parallelism import prepare_actor_batch, prepare_legacy_actor_batch
        minibatch = self.config.actor_rollout_ref.actor.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        dp_size = self.actor_rollout_wg.world_size // self.config.actor_rollout_ref.actor.ulysses_sequence_parallel_size
        original_budget = self.config.opd.get('legacy_microbatch_token_budget')
        if original_budget:
            return prepare_legacy_actor_batch(batch, minibatch, dp_size, original_budget)
        return prepare_actor_batch(batch, minibatch, dp_size)
