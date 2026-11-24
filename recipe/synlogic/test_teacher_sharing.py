"""Frozen-teacher admission and forward-lock release on CPU (no CUDA calls)."""
import json

import pytest

from synlogic.teacher_sharing import TeacherGPUSharing


def gate(directory, name, cap=2, timeout=.2):
    return TeacherGPUSharing(str(directory), name, max_instances=cap, timeout_seconds=timeout)


def events(directory):
    return [json.loads(line) for path in sorted(directory.glob('events-*.jsonl'))
            for line in path.read_text().splitlines()]


def test_instance_cap_and_release(tmp_path):
    first = gate(tmp_path, 'a', cap=1)
    with pytest.raises(RuntimeError, match='cap'):
        gate(tmp_path, 'b', cap=1)
    first.close()
    gate(tmp_path, 'b', cap=1).close()


def test_conflicting_policy_is_rejected(tmp_path):
    gate(tmp_path, 'a', cap=1).close()
    with pytest.raises(ValueError, match='policy'):
        gate(tmp_path, 'b', cap=2)


def test_forward_lock_is_exclusive_and_released_after_failure(tmp_path):
    first, second = gate(tmp_path, 'a'), gate(tmp_path, 'b')
    with first.phase('score'):
        assert first.owns_phase()
        with pytest.raises(RuntimeError):
            with first.phase('nested'):
                pass
        with pytest.raises(TimeoutError):
            with second.phase('blocked'):
                pass
        with pytest.raises(RuntimeError):
            first.close()
    with pytest.raises(KeyError):
        with first.phase('fails'):
            raise KeyError('teacher error')
    assert not first.owns_phase()
    with second.phase('after-release'):
        pass
    statuses = {event['phase']: event['status'] for event in events(tmp_path)}
    assert statuses == {'score': 'complete', 'fails': 'failed', 'after-release': 'complete'}
    first.close()
    second.close()
