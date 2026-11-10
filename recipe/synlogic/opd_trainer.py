"""The same noisy training prompts, with outcome-only and dense components exposed."""
import hashlib
import json
import os
from pathlib import Path

from src.opd import combine_advantages, gradient_matched_coefficient
from src.noise_diagnostics import noise_metrics
from .trainer import SynLogicTrainer


class OPDTrainer(SynLogicTrainer):
    def init_workers(self):
        super().init_workers()
        student = self.actor_rollout_wg.opd_device_identity()[0]
        teacher = self.ref_policy_wg.opd_device_identity()[0]
        if student['assigned'] == teacher['assigned']:
            raise ValueError('Student and teacher must have distinct allocated GPUs')
        if student['visibility_unmasked'] and student['current_device'] == teacher['current_device']:
            raise ValueError('Student and teacher landed on the same physical GPU')
        root = Path(os.environ['SYNLOGIC_RUN_ROOT'])
        (root/'device-placement.json').write_text(json.dumps({'student':student,'teacher':teacher},indent=2)+'\n')

    def _transform_advantages(self, batch, metrics):
        settings = self.config.opd
        outcome_metrics = noise_metrics(batch)
        for name in ('active_rate', 'active_groups', 'groups'):
            if 'noise/' + name in outcome_metrics:
                metrics['noise/outcome_' + name] = outcome_metrics['noise/' + name]
        path = Path(settings.calibration_file)
        if not path.exists():
            if not settings.calibrate:
                raise RuntimeError('Shared teacher coefficient is not ready; launch combined arm first')
            self._progress('calibrating_teacher_gradient_scale')
            result = self.actor_rollout_wg.calibrate_opd(batch)[0]
            coefficient = gradient_matched_coefficient(result['outcome_gradient_norms'], result['teacher_gradient_norms'])
            result.update(teacher_coefficient=coefficient, source_step=300, calibration_step=self.global_steps,
                          training_sha256=self.config.provenance.data_fingerprint,
                          teacher_fingerprint=settings.teacher_fingerprint,
                          minibatches=4, optimizer_updates_performed=0)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
            temporary.write_text(json.dumps(result, indent=2) + '\n')
            temporary.replace(path)
        proof = json.loads(path.read_text())
        if (proof['training_sha256'] != self.config.provenance.data_fingerprint
                or proof['teacher_fingerprint'] != settings.teacher_fingerprint):
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
