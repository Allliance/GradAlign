"""The same noisy training prompts, with outcome-only and dense components exposed."""
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from src.opd import combine_advantages, gradient_matched_coefficient
from src.noise_diagnostics import noise_metrics
from .trainer import SynLogicTrainer

# Fixed teacher coefficients accepted by the recorded TIP-OPD source variant.
TIP_FIXED_COEFFICIENTS = (1.0, 0.001)


def check_tip_protocol(config):
    """TIP-OPD is explicit: its own actor loss mode and the recorded coefficients."""
    settings = config.opd
    if not settings.get('tip', False):
        return False
    actor = config.actor_rollout_ref.actor
    from src.dapo_ray_trainer import RayDAPOTrainer
    if not hasattr(RayDAPOTrainer, '_reference_log_probs'):
        raise ValueError('TIP requires a framework revision with the explicit teacher log-prob hook')
    if actor.get('opd_loss_mode') != 'tip_ppo':
        raise ValueError('TIP requires actor opd_loss_mode=tip_ppo')
    if settings.get('fixed_teacher_coefficient') is None or settings.get('calibrate', False):
        raise ValueError('TIP requires an explicit fixed teacher coefficient without calibration')
    if 'outcome_coefficient' not in settings or 'tip_retention' not in settings:
        raise ValueError('TIP requires its recorded outcome coefficient and retention')
    if (float(settings.fixed_teacher_coefficient) != actor.get('opd_teacher_coefficient')
            or float(settings.outcome_coefficient) != actor.get('opd_outcome_coefficient')):
        raise ValueError('TIP actor coefficients differ from the recorded OPD settings')
    if actor.get('use_kl_loss', False):
        raise ValueError('TIP teacher statistics do not provide base-reference log-probabilities')
    if not settings.get('preserve_global_minibatches', False):
        raise ValueError('TIP selected-token means require preserved global minibatches')
    return True


class OPDTrainer(SynLogicTrainer):
    def init_workers(self):
        check_tip_protocol(self.config)
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

    def _reference_log_probs(self, batch):
        if not self.config.opd.get('tip', False):
            return super()._reference_log_probs(batch)
        # TIP: the frozen teacher's native logits, plus full-vocabulary statistics
        # against the current student head on identical prefixes.
        from verl import DataProto
        version = int(self.global_steps)
        heads = self.actor_rollout_wg.get_tip_student_head()
        weight = next(w for w in heads if w is not None)
        self.ref_policy_wg.set_tip_student_head(weight, version)
        del heads, weight
        batch.meta_info['tip_policy_version'] = version
        parts = []
        for start in range(0, len(batch), 32):
            micro = batch[start:start + 32]
            micro = micro.union(self.actor_rollout_wg.compute_tip_hidden(micro))
            parts.append(self.ref_policy_wg.compute_tip_statistics(micro))
        return DataProto.concat(parts)

    def _select_tip_tokens(self, batch, metrics, coefficient):
        from .tip_stats import select_tip
        settings = self.config.opd
        mask = batch.batch['response_mask'].bool()
        chosen, score = select_tip(batch.batch['tip_entropy'], batch.batch['tip_rkl'],
                                   mask, float(settings.tip_retention))
        batch.batch['opd_token_mask'] = chosen
        batch.batch['tip_score'] = score
        batch.batch['teacher_advantages'] *= chosen
        # Logged advantages describe the actual tip_ppo objective: outcome PPO plus
        # the selected-token teacher term (identical to the source when outcome is 0).
        batch.batch['advantages'] = torch.where(
            mask, float(settings.outcome_coefficient) * batch.batch['outcome_advantages']
            + coefficient * batch.batch['teacher_advantages'], 0.)
        metrics.update({'tip/retained_tokens': int(chosen.sum()),
                        'tip/valid_tokens': int(mask.sum()),
                        'tip/retained_fraction': float(chosen.sum() / mask.sum())})
        if chosen.any():
            metrics.update({'tip/selected_entropy': float(batch.batch['tip_entropy'][chosen].mean()),
                            'tip/selected_rkl': float(batch.batch['tip_rkl'][chosen].mean())})
        return batch

    def _transform_advantages(self, batch, metrics):
        settings = self.config.opd
        outcome_metrics = noise_metrics(batch)
        for name in ('active_rate', 'active_groups', 'groups'):
            if 'noise/' + name in outcome_metrics:
                metrics['noise/outcome_' + name] = outcome_metrics['noise/' + name]
        if settings.get('fixed_teacher_coefficient') is not None:
            coefficient = float(settings.fixed_teacher_coefficient)
            tip = check_tip_protocol(self.config)
            if coefficient not in (TIP_FIXED_COEFFICIENTS if tip else (1.0,)) or settings.calibrate:
                raise ValueError('This protocol requires a recorded user-fixed teacher coefficient')
            batch.batch['outcome_advantages'] = batch.batch['advantages'].detach().clone()
            combined, dense, values = combine_advantages(
                batch.batch['advantages'], batch.batch['teacher_log_probs'], batch.batch['old_log_probs'],
                batch.batch['response_mask'], settings.outcome_coefficient, coefficient)
            batch.batch['advantages'], batch.batch['teacher_advantages'] = combined, dense
            batch.meta_info.update(opd_loss_mode='tip_ppo' if tip else 'importance_sampling',
                                   opd_teacher_coefficient=coefficient,
                                   opd_outcome_coefficient=float(settings.outcome_coefficient))
            metrics.update(values)
            if tip:
                batch = self._select_tip_tokens(batch, metrics, coefficient)
            return batch
        if settings.get('tip', False):
            raise ValueError('TIP requires an explicit fixed teacher coefficient without calibration')
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
