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

    def _prepare_policy_scoring(self, batch, metrics):
        if not self.config.opd.get('alternating_rl_taopd', False):
            return batch
        from src.alternating_opd import phase_for_step
        actor = self.config.actor_rollout_ref.actor
        if (not actor.get('alternating_rl_taopd', False) or actor.use_kl_loss
                or actor.entropy_coeff != 0 or self.config.algorithm.use_kl_in_reward
                or not self.config.actor_rollout_ref.ref.get('alternating_rl_taopd', False)
                or self.config.opd.outcome_coefficient != 0
                or self.config.opd.fixed_teacher_coefficient != 1
                or not self.config.opd.teachability):
            raise ValueError('Alternating objective configuration differs from the protocol')
        # New rollouts must never inherit scoring/selection tensors from another phase.
        for key in list(batch.batch.keys()):
            if key.startswith('ta_') or key in ('opd_token_mask', 'global_opd_response_tokens',
                    'teacher_log_probs', 'teacher_advantages', 'outcome_advantages', 'ref_log_prob'):
                batch.batch.pop(key)
        phase = phase_for_step(self.global_steps)
        batch.meta_info.update(alternating_phase=phase, opd_loss_mode=(
            'importance_sampling' if phase['phase'] == 'ta_opd' else 'rl_only'),
            opd_teacher_coefficient=phase['teacher_coefficient'],
            opd_outcome_coefficient=phase['outcome_coefficient'])
        metrics.update({'alternating/' + k: v for k, v in phase.items() if k != 'phase'})
        metrics['alternating/phase_is_ta_opd'] = int(phase['phase'] == 'ta_opd')
        metrics['alternating/phase'] = int(phase['phase'] == 'ta_opd')
        metrics['alternating/optimizer_updates_before'] = (self.global_steps - 1) * 5
        return batch

    def _reference_policy_required(self, batch):
        if self.config.opd.get('alternating_rl_taopd', False):
            from src.alternating_opd import phase_for_step
            phase = phase_for_step(self.global_steps)
            if batch.meta_info.get('alternating_phase') != phase:
                raise ValueError('Phase must be prepared before any teacher query')
            return phase['phase'] == 'ta_opd'
        return self.use_reference_policy

    def _phase_denominators(self, batch, metrics):
        minibatch = self.config.actor_rollout_ref.actor.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        counts = batch.batch['response_mask'].sum(-1).reshape(-1, minibatch).sum(-1)
        batch.batch['global_minibatch_response_tokens'] = counts.repeat_interleave(minibatch)
        chosen = batch.batch.get('opd_token_mask')
        if chosen is not None:
            selected = chosen.sum(-1).reshape(-1, minibatch).sum(-1)
            batch.batch['global_opd_response_tokens'] = selected.repeat_interleave(minibatch)
        else:
            selected = counts.new_zeros(counts.shape)
        for i, (original, retained) in enumerate(zip(counts.tolist(), selected.tolist()), 1):
            metrics[f'alternating/minibatch_{i}_original_tokens'] = original
            metrics[f'alternating/minibatch_{i}_selected_tokens'] = retained
        return batch

    def _transform_advantages(self, batch, metrics):
        settings = self.config.opd
        outcome_metrics = noise_metrics(batch)
        for name in ('active_rate', 'active_groups', 'groups'):
            if 'noise/' + name in outcome_metrics:
                metrics['noise/outcome_' + name] = outcome_metrics['noise/' + name]
        if settings.get('alternating_rl_taopd', False):
            from src.alternating_opd import phase_for_step
            phase = phase_for_step(self.global_steps)
            if batch.meta_info.get('alternating_phase') != phase:
                raise ValueError('Absolute phase differs before advantage transform')
            if phase['phase'] == 'rl':
                if any(k.startswith('ta_') or k in ('teacher_advantages', 'teacher_log_probs',
                        'opd_token_mask', 'outcome_advantages') for k in batch.batch.keys()):
                    raise ValueError('Teacher data entered the RL-only branch')
                metrics.update({'ta/retained_tokens': 0, 'ta/valid_tokens': int(batch.batch['response_mask'].sum()),
                                'alternating/selected_token_denominator': 0,
                                'opd/teacher_coefficient': 0., 'opd/outcome_coefficient': 1.})
                return self._phase_denominators(batch, metrics)
        if settings.get('fixed_teacher_coefficient') is not None:
            coefficient = float(settings.fixed_teacher_coefficient)
            if coefficient not in (1.0,0.001) or settings.calibrate:
                raise ValueError('This study uses explicit fixed teacher coefficients1 or0.001')
            batch.batch['outcome_advantages'] = batch.batch['advantages'].detach().clone()
            combined, dense, values = combine_advantages(
                batch.batch['advantages'], batch.batch['teacher_log_probs'], batch.batch['old_log_probs'],
                batch.batch['response_mask'], settings.outcome_coefficient, coefficient)
            batch.batch['advantages'], batch.batch['teacher_advantages'] = combined, dense
            batch.meta_info.update(opd_loss_mode=self.config.actor_rollout_ref.actor.get('opd_loss_mode', 'importance_sampling'),opd_teacher_coefficient=coefficient,
                                   opd_outcome_coefficient=float(settings.outcome_coefficient))
            metrics.update(values)
            if settings.get('teachability', False):
                from .ta_opd import support_disagreement, select_teachable
                extra = self.actor_rollout_wg.compute_ta_candidates(batch)
                batch = batch.union(extra)
                d,c = support_disagreement(batch.batch['ta_student_ids'],batch.batch['ta_student_lp'],
                    batch.batch['ta_teacher_ids'],batch.batch['ta_teacher_lp'],
                    batch.batch['ta_teacher_on_student_lp'],batch.batch['ta_student_on_teacher_lp'])
                valid = batch.batch['response_mask'].bool()
                chosen,score = select_teachable(d,c,valid,float(settings.ta_retention))
                batch.batch['opd_token_mask']=chosen
                batch.batch['ta_disagreement']=d
                batch.batch['ta_compatibility']=c
                batch.batch['ta_score']=score
                batch.batch['teacher_advantages'] *= chosen
                batch.batch['advantages'] = (settings.outcome_coefficient*batch.batch['outcome_advantages']
                    + coefficient*batch.batch['teacher_advantages'])
                metrics.update({'ta/retained_tokens':int(chosen.sum()),'ta/valid_tokens':int(valid.sum()),
                    'ta/retained_fraction':float(chosen.sum()/valid.sum()),
                    'ta/selected_compatibility':float(c[chosen].mean()),
                    'ta/selected_disagreement':float(d[chosen].mean())})
                if settings.get('alternating_rl_taopd', False):
                    metrics['alternating/selected_token_denominator'] = int(chosen.sum())
                for key in ['ta_student_ids','ta_student_lp','ta_teacher_ids','ta_teacher_lp',
                            'ta_teacher_on_student_lp','ta_student_on_teacher_lp']:
                    batch.batch.pop(key)
            if settings.get('alternating_rl_taopd', False):
                return self._phase_denominators(batch, metrics)
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
