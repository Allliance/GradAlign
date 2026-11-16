"""Baseline sampled PPO distillation with a detached pre-minibatch self teacher."""
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from tensordict import TensorDict
from verl import DataProto

from .native_stats import score_native
from .shared_targets import atomic_record, target_folder


def sampled_teacher(worker, mini, temperature, version):
    """Finish teacher scoring before native student packing/backward begins."""
    actor = worker.actor
    model = actor.actor_module
    counters = {int(state['step']) for state in actor.actor_optimizer.state.values() if 'step' in state}
    assert counters == {version} or (not counters and version == 0), 'Self-teacher/Adam version mismatch'
    assert (mini['opsd_teacher_policy_version'] == version).all()
    response_mask = mini['attention_mask'][:, -mini['responses'].shape[1]:]
    teacher = DataProto.from_dict(tensors={
        'responses': mini['responses'], 'response_mask': response_mask,
        **{key: mini['opsd_privileged_' + key] for key in ['prompts', 'input_ids', 'attention_mask', 'position_ids']}})
    assert teacher.batch.device is None or teacher.batch.device.type == 'cpu'
    modes = [(module, module.training) for module in model.modules()]
    start = time.monotonic()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            # Same native reference scorer and 32768-token planner as BaselineBatch.
            result = score_native(actor, teacher, temperature, max_token_len=32768,
                                  k=0, valid_length_planning=True)
    finally:
        for module, training in modes:
            module.training = training
    lp = result['chosen_lp'].to(mini['old_log_probs'].device)
    assert not lp.requires_grad and torch.isfinite(lp[response_mask.bool()]).all()
    advantage = (lp - mini['old_log_probs']).detach() * response_mask
    assert torch.isfinite(advantage).all()
    assert counters == {int(state['step']) for state in actor.actor_optimizer.state.values() if 'step' in state}

    outer = version // 5 + 1
    root = Path(os.environ['SYNLOGIC_RUN_ROOT'])
    if 'opsd_capture_targets' in mini and bool(mini['opsd_capture_targets'][0]):
        rows = mini['opsd_snapshot_row_index'].tolist()
        entries = []
        for i, row in enumerate(rows):
            valid = response_mask[i].bool()
            entries.append({'row_index': row, 'response_token_ids': mini['responses'][i, valid].tolist(),
                'teacher_chosen_log_probs': lp[i, valid].tolist(),
                'old_log_probs': mini['old_log_probs'][i, valid].tolist(),
                'advantages': advantage[i, valid].tolist(), 'response_mask': [1] * int(valid.sum()),
                'teacher_policy_version': version, 'student_update_policy_version': version,
                'stopped_gradient': True})
        signature = hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]
        atomic_record(target_folder(root, outer) / f'version{version}-rank{worker.rank}-rows{signature}.json.gz',
            {'schema_version': 2, 'objective': 'ppo_clipped_self_teacher_v2', 'outer_step': outer,
             'rank': worker.rank, 'teacher_policy_version': version, 'entries': entries})

    metrics = {'actor/self_teacher_seconds': time.monotonic() - start,
               'actor/opsd_teacher_policy_version': float(version)}
    if 'self_opd_noisy' in mini:
        noisy = mini['self_opd_noisy'].bool()
        active = advantage.ne(0).any(-1)
        groups = mini['self_opd_group']
        ng = torch.unique(groups[noisy])
        ag = torch.unique(groups[noisy & active])
        counts = torch.tensor([len(ag), len(ng), int((noisy & active).sum()), int(noisy.sum())], dtype=torch.long)
        if version % 5 == 0:
            worker._self_opd_noise_counts = torch.zeros_like(counts)
        worker._self_opd_noise_counts += counts
        total = worker._self_opd_noise_counts.to(next(model.parameters()).device)
        torch.distributed.all_reduce(total)
        a, g, ar, nr = map(int, total.tolist())
        metrics.update({'noise/advantage_activity_defined': 1., 'noise/active_groups': a,
                        'noise/groups': g, 'noise/active_rollouts': ar, 'noise/rollouts': nr})
        if g:
            metrics['noise/active_rate'] = a / g
        if nr:
            metrics['noise/active_rollout_rate'] = ar / nr

    proof = {'passed': True, 'epoch': time.time(), 'allocation_id': os.environ['SYNLOGIC_ALLOCATION_ID'],
        'attempt': os.environ['SYNLOGIC_ATTEMPT'], 'rank': worker.rank, 'actual_adam_counter': version,
        'teacher_version': version, 'policy_version': version, 'objective_version': 2,
        'teacher_mode': 'same actor weights before each optimizer minibatch; detached, no EMA/shadow',
        'extra_optimizer_updates': 0, 'stopped_gradient': True,
        'teacher_scoring_seconds': metrics['actor/self_teacher_seconds'],
        'teacher_max_valid_context': int(teacher.batch['attention_mask'].sum(-1).max())}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        proof['memory'] = {'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved(),
                           'physical_free': free, 'physical_total': total}
    path = root / f'shared-teacher-version-rank{worker.rank}.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(proof) + '\n')
    temporary.replace(path)
    # Privileged tensors never enter the student forward or its batching planner.
    student = {key: value for key, value in mini.items() if not key.startswith('opsd_privileged_')}
    student.update(advantages=advantage, teacher_advantages=advantage, teacher_log_probs=lp)
    return TensorDict(student, batch_size=mini.batch_size), metrics
