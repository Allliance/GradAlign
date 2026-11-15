"""Admission and bounded forward concurrency for colocated frozen teachers.

Every process sharing a physical GPU must use the same host-local directory.
The launcher owns GPU placement; this module never allocates a device. Kernel
file locks release admission and execution slots even when a worker crashes.
"""

if not __debug__:
    raise RuntimeError('Teacher sharing requires Python safety checks enabled')
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import time
import threading
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
        self._active_thread = None
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
            self._active_thread = threading.get_ident()
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
                self._active_thread = None
                # Logging failure is visible, but must never leak the execution lock.
                with self.events.open('a') as output:
                    output.write(json.dumps(event) + '\n')

    def owns_phase(self):
        return self.pid == os.getpid() and self._active and self._active_thread == threading.get_ident()

    def close(self):
        if self.pid != os.getpid():
            raise RuntimeError('Teacher admission cannot be released by a forked process')
        if self._active:
            raise RuntimeError('Cannot close an executing teacher')
        if self.slot is not None:
            self.slot.close()
            self.slot = None


def _clear_frames(error):
    import traceback
    pending, seen = [error], set()
    while pending:
        error = pending.pop()
        if id(error) in seen:
            continue
        seen.add(id(error))
        traceback.clear_frames(error.__traceback__)
        pending.extend(e for e in (error.__cause__, error.__context__) if e is not None)


def _sharing(worker, config, required=False):
    sharing = config.ref.get('teacher_sharing')
    if not sharing:
        if required:
            raise ValueError('Packed teacher requires explicit immutable sharing policy')
        return None
    from omegaconf import OmegaConf
    options = OmegaConf.to_container(sharing, resolve=True)
    if required:
        mounted = os.path.ismount('/run/teacher-sharing')
        if not mounted:
            # Bind mounts on the same filesystem can share st_dev. The kernel
            # mount table is authoritative in that case (rootless OCI rootfs).
            mounted = any(line.split()[4] == '/run/teacher-sharing'
                          for line in Path('/proc/self/mountinfo').read_text().splitlines())
        if not mounted:
            raise ValueError('Packed teacher requires the host-local sharing bind mount')
        if not Path(options['directory']).is_relative_to('/run/teacher-sharing'):
            raise ValueError('Packed teacher gate must be inside the sharing bind mount')
    return TeacherGPUSharing(**options)


def wrap_teacher_worker(worker_class, registered_attribute=None, required=False):
    """Cover every public teacher CUDA RPC before Ray builds dispatch metadata.

    The launcher installs this in a fresh worker, before construction. The
    scientific method bodies stay unchanged, including native sparse scorers.
    Admission precedes device/context construction and lasts until process exit.
    """
    from functools import wraps
    if worker_class.__dict__.get('_shared_teacher_wrapped', False):
        return worker_class
    constructor = worker_class.__init__
    initialize = worker_class.init_model

    @wraps(constructor)
    def construct(self, config, role):
        if required and type(self) is not worker_class:
            raise TypeError('Packed teacher subclass must be explicitly wrapped')
        gate = _sharing(self, config, required=required)
        self._teacher_sharing = gate
        if gate is None:
            return constructor(self, config, role)
        # Failed construction may retain a CUDA context/distributed state. Its
        # admission remains occupied until the supervisor terminates the actor.
        with gate.phase('construction'):
            import torch
            try:
                result = constructor(self, config, role)
                if required and getattr(self, 'world_size', None) != 1:
                    raise ValueError('Packed teacher requires exactly one distributed rank')
                return result
            except BaseException as error:
                _clear_frames(error)
                raise
            finally:
                try:
                    torch.cuda.synchronize()
                finally:
                    torch.cuda.empty_cache()

    @wraps(initialize)
    def init(self, *args, **kwargs):
        import torch
        gate = getattr(self, '_teacher_sharing', None)
        if gate is None:
            return initialize(self, *args, **kwargs)
        try:
            with gate.phase('initialization'):
                try:
                    return initialize(self, *args, **kwargs)
                except BaseException as error:
                    _clear_frames(error)
                    raise
                finally:
                    try:
                        torch.cuda.synchronize()
                    finally:
                        torch.cuda.empty_cache()
        except BaseException:
            # No initializer CUDA work has begun on lock timeout. Construction
            # may still own context/state, so keep admission until actor exit.
            raise

    def score(method, name):
        @wraps(method)
        def guarded(self, *args, **kwargs):
            import torch
            gate = getattr(self, '_teacher_sharing', None)
            if gate is None or gate.owns_phase():
                return method(self, *args, **kwargs)
            with gate.phase(name) as event:
                torch.cuda.reset_peak_memory_stats()
                try:
                    if required:
                        def check_cpu(value):
                            if torch.is_tensor(value):
                                if value.device.type != 'cpu':
                                    raise ValueError('Packed teacher RPC tensors must arrive on CPU')
                            elif hasattr(value, 'batch'):
                                for item in value.batch.values():
                                    check_cpu(item)
                            elif isinstance(value, dict):
                                for item in value.values():
                                    check_cpu(item)
                            elif isinstance(value, (list, tuple)):
                                for item in value:
                                    check_cpu(item)
                        for value in (*args, *kwargs.values()):
                            check_cpu(value)
                    result = method(self, *args, **kwargs)
                    def cpu_result(value):
                        if torch.is_tensor(value):
                            return value.cpu()
                        if hasattr(value, 'batch') and callable(getattr(value, 'to', None)):
                            return value.to('cpu')
                        if isinstance(value, dict):
                            return {key: cpu_result(item) for key, item in value.items()}
                        if isinstance(value, list):
                            return [cpu_result(item) for item in value]
                        if isinstance(value, tuple):
                            return tuple(cpu_result(item) for item in value)
                        return value
                    return cpu_result(result)
                except BaseException as error:
                    _clear_frames(error)
                    raise
                finally:
                    try:
                        try:
                            # DataProto.to mutates. Tensor.to does not; a TIP
                            # head deliberately remains private resident state.
                            for value in (*args, *kwargs.values()):
                                if hasattr(value, 'batch') and callable(getattr(value, 'to', None)):
                                    value.to('cpu')
                        finally:
                            torch.cuda.synchronize()
                    finally:
                        try:
                            event.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                         peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                         instance_allocated_bytes=torch.cuda.memory_allocated(),
                                         instance_reserved_bytes=torch.cuda.memory_reserved())
                        finally:
                            torch.cuda.empty_cache()
                        event['reserved_after_release_bytes'] = torch.cuda.memory_reserved()
        return guarded

    worker_class.__init__ = construct
    worker_class.init_model = init
    # The internal original scorer is reached inside compute_ref_log_prob.
    # Do not wrap it twice: admission deliberately rejects nested phases.
    public = ['compute_ref_log_prob', '_compute_original_ref_log_prob',
              '_compute_teacher_log_prob', 'compute_tip_statistics', 'set_tip_student_head']
    if registered_attribute:
        public += [name for name in dir(worker_class)
                   if name not in ('init_model', '__init__')
                   and hasattr(getattr(worker_class, name), registered_attribute)]
    for name in dict.fromkeys(public):
        if hasattr(worker_class, name):
            setattr(worker_class, name, score(getattr(worker_class, name), name))
    worker_class._shared_teacher_wrapped = True
    return worker_class
