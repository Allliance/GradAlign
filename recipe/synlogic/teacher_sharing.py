"""Admission and bounded forward concurrency for colocated frozen teachers.

Every process sharing a physical GPU must use the same host-local directory.
The launcher owns GPU placement; this module never allocates a device. Kernel
file locks release admission and execution slots even when a worker crashes.
"""
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import time
import uuid


class TeacherGPUSharing:
    def __init__(self, directory, instance_id, max_instances=4, timeout_seconds=900):
        if isinstance(max_instances, bool) or not isinstance(max_instances, int) or not 1 <= max_instances <= 4:
            raise ValueError('Shared teacher admission supports at most four instances')
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError('Teacher sharing timeout must be positive and finite')
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError('A nonempty string teacher instance identity is required')
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise ValueError('Launcher must provision the physical-GPU sharing directory')
        self.instance_id = instance_id
        self.timeout = timeout_seconds
        self.pid = os.getpid()
        self.slot = None
        self._active = False
        # A single immutable policy prevents participants using incompatible caps.
        with (self.directory / 'policy.lock').open('a+') as policy_lock:
            self._acquire(policy_lock)
            policy = {'max_instances': max_instances, 'concurrent_forwards': 1, 'version': 1}
            target = self.directory / 'policy.json'
            if target.exists():
                if json.loads(target.read_text()) != policy:
                    raise ValueError('Conflicting policy for this physical GPU')
            else:
                temporary = target.with_name('policy-' + uuid.uuid4().hex + '.tmp')
                temporary.write_text(json.dumps(policy) + '\n')
                temporary.replace(target)
            for index in range(max_instances):
                candidate = (self.directory / ('instance-%d.lock' % index)).open('a+')
                try:
                    fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    candidate.close()
                    continue
                self.slot = candidate
                self.slot.seek(0)
                self.slot.truncate()
                self.slot.write(json.dumps({'instance': instance_id, 'pid': self.pid}) + '\n')
                self.slot.flush()
                break
            if self.slot is None:
                raise RuntimeError('Shared teacher GPU instance cap reached')
        self.events = self.directory / ('events-' + uuid.uuid4().hex + '.jsonl')

    def _acquire(self, handle):
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Timed out waiting for shared teacher GPU')
                time.sleep(0.05)

    @contextmanager
    def phase(self, name):
        if self.slot is None or self.pid != os.getpid() or self._active:
            raise RuntimeError('Teacher admission is closed, inherited, or already executing')
        requested = time.monotonic()
        with (self.directory / 'forward.lock').open('a+') as handle:
            self._acquire(handle)
            self._active = True
            started = time.monotonic()
            event = {'instance': self.instance_id, 'pid': self.pid, 'phase': name,
                     'epoch': time.time(), 'start_monotonic': started,
                     'queue_wait_seconds': started - requested, 'status': 'failed'}
            try:
                yield event
                event['status'] = 'complete'
            finally:
                event['end_monotonic'] = time.monotonic()
                event['execution_seconds'] = event['end_monotonic'] - started
                self._active = False
                # Logging failure is visible, but must never leak the execution lock.
                with self.events.open('a') as output:
                    output.write(json.dumps(event) + '\n')

    def close(self):
        if self.pid != os.getpid():
            raise RuntimeError('Teacher admission cannot be released by a forked process')
        if self._active:
            raise RuntimeError('Cannot close an executing teacher')
        if self.slot is not None:
            self.slot.close()
            self.slot = None
