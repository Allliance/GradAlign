import json
import multiprocessing as mp
from pathlib import Path
import tempfile
import time
import unittest
import sys
import ast
import traceback
from contextlib import nullcontext
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'recipe'/'synlogic'))

from teacher_sharing import TeacherGPUSharing


def execute(directory, ready, start, result):
    gate = TeacherGPUSharing(directory, str(mp.current_process().pid))
    ready.put(True)
    start.wait(5)
    with gate.phase('scoring') as event:
        time.sleep(.2)
    result.put(event)
    gate.close()


def occupy(directory, ready):
    gate = TeacherGPUSharing(directory, 'crash', max_instances=1)
    with gate.phase('crash'):
        ready.set()
        time.sleep(30)


class SharingTests(unittest.TestCase):
    def worker(self):
        source=Path(__file__).resolve().parents[1]/'recipe/synlogic/opd_worker.py'
        if not source.exists(): source=Path(__file__).parent/'opd_worker.py'
        tree=ast.parse(source.read_text())
        helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_clear_exception_frames')
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='OPDReferenceWorker')
        methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('init_model','compute_ref_log_prob')]
        for method in methods: method.decorator_list=[]
        node=ast.ClassDef(name='Worker',bases=[],keywords=[],body=methods,decorator_list=[])
        calls=[]
        cuda=SimpleNamespace(synchronize=lambda:calls.append('synchronize'),empty_cache=lambda:calls.append('empty_cache'),
            reset_peak_memory_stats=lambda:calls.append('reset_peak'),max_memory_allocated=lambda:10,
            max_memory_reserved=lambda:20,memory_allocated=lambda:5,memory_reserved=lambda:8)
        ns={'torch':SimpleNamespace(cuda=cuda),'nullcontext':nullcontext,
            'traceback':traceback,
            'OmegaConf':SimpleNamespace(to_container=lambda x,**kwargs:x)}
        # Replace the package-relative import with the real lightweight admission class.
        for method in methods:
            for child in ast.walk(method):
                if isinstance(child,ast.ImportFrom) and child.module=='teacher_sharing':child.level=0
        exec(compile(ast.fix_missing_locations(ast.Module(body=[helper,node],type_ignores=[])),str(source),'exec'),ns)
        return ns['Worker'](),calls

    def test_dedicated_scoring_is_unchanged(self):
        worker,calls=self.worker();worker._teacher_sharing=None
        data,result=object(),object()
        worker._compute_teacher_log_prob=lambda value: result if value is data else None
        self.assertIs(worker.compute_ref_log_prob(data),result)
        self.assertEqual(calls,[])

    def test_shared_scoring_and_error_flush_gpu_before_unlock(self):
        with tempfile.TemporaryDirectory() as d:
            worker,calls=self.worker();worker._teacher_sharing=TeacherGPUSharing(d,'worker')
            data=SimpleNamespace(to=lambda device:calls.append('input_to_'+device))
            result=object();worker._compute_teacher_log_prob=lambda data:result
            self.assertIs(worker.compute_ref_log_prob(data),result)
            self.assertEqual(calls,['reset_peak','input_to_cpu','synchronize','empty_cache'])
            calls.clear()
            def fail(data):raise RuntimeError('scoring failure')
            worker._compute_teacher_log_prob=fail
            with self.assertRaisesRegex(RuntimeError,'scoring failure'):worker.compute_ref_log_prob(data)
            self.assertEqual(calls,['reset_peak','input_to_cpu','synchronize','empty_cache'])
            with worker._teacher_sharing.phase('after_failure'):pass
            worker._teacher_sharing.close()

    def test_failed_scoring_frames_release_before_gpu_flush(self):
        with tempfile.TemporaryDirectory() as d:
            worker,calls=self.worker();worker._teacher_sharing=TeacherGPUSharing(d,'worker')
            class Temporary:
                def __del__(self): calls.append('release_frame')
            def deep_failure():
                temporary=Temporary()
                raise ValueError('inner forward error')
            def fail(data):
                try:
                    deep_failure()
                except ValueError as cause:
                    # Cycles must not prevent cleanup of an exception chain.
                    cause.__context__=cause
                    raise RuntimeError('model forward failed') from cause
            worker._compute_teacher_log_prob=fail
            data=SimpleNamespace(to=lambda device:calls.append('input_to_'+device))
            with self.assertRaisesRegex(RuntimeError,'model forward failed'):worker.compute_ref_log_prob(data)
            self.assertEqual(calls,['reset_peak','release_frame','input_to_cpu','synchronize','empty_cache'])
            worker._teacher_sharing.close()

    def test_release_attempts_cache_cleanup_after_copy_or_sync_failure(self):
        for failure in ('copy','sync'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as d:
                worker,calls=self.worker();worker._teacher_sharing=TeacherGPUSharing(d,'worker')
                worker._compute_teacher_log_prob=lambda data:None
                def copy(device):
                    calls.append('input_to_'+device)
                    if failure=='copy': raise RuntimeError('cleanup failed')
                if failure=='sync':
                    def sync():
                        calls.append('synchronize')
                        raise RuntimeError('cleanup failed')
                    worker.compute_ref_log_prob.__globals__['torch'].cuda.synchronize=sync
                with self.assertRaisesRegex(RuntimeError,'cleanup failed'):
                    worker.compute_ref_log_prob(SimpleNamespace(to=copy))
                self.assertEqual(calls,['reset_peak','input_to_cpu','synchronize','empty_cache'])
                worker._teacher_sharing.close()

    def test_initialization_failure_retains_resident_admission(self):
        with tempfile.TemporaryDirectory() as d:
            worker,calls=self.worker()
            worker.config=SimpleNamespace(ref={'teacher_sharing':{'directory':d,'instance_id':'failed','max_instances':1}})
            def fail():raise RuntimeError('model load failed')
            worker._init_teacher_model=fail
            with self.assertRaisesRegex(RuntimeError,'model load failed'):worker.init_model()
            with self.assertRaises(RuntimeError):TeacherGPUSharing(d,'too_early',max_instances=1)
            # Actual failed actors must exit; the crash test checks kernel release.
            worker._teacher_sharing.close()
            replacement=TeacherGPUSharing(d,'replacement',max_instances=1)
            replacement.close()
            self.assertEqual(calls,['synchronize','empty_cache'])

    def test_initialization_wait_timeout_releases_unused_admission(self):
        with tempfile.TemporaryDirectory() as d:
            first=TeacherGPUSharing(d,'first',max_instances=2)
            worker,calls=self.worker()
            worker.config=SimpleNamespace(ref={'teacher_sharing':{'directory':d,'instance_id':'waiting','max_instances':2,'timeout_seconds':.1}})
            worker._init_teacher_model=lambda:self.fail('Initialization ran without the GPU lock')
            with first.phase('occupied'):
                with self.assertRaises(TimeoutError):worker.init_model()
                replacement=TeacherGPUSharing(d,'replacement',max_instances=2)
                replacement.close()
            first.close()
            self.assertEqual(calls,[])

    def test_cap_and_release(self):
        with tempfile.TemporaryDirectory() as d:
            gates = [TeacherGPUSharing(d, str(i)) for i in range(4)]
            with self.assertRaises(RuntimeError):
                TeacherGPUSharing(d, 'fifth')
            gates[0].close()
            extra = TeacherGPUSharing(d, 'replacement')
            extra.close()
            for g in gates: g.close()

    def test_invalid_or_conflicting_policy(self):
        with tempfile.TemporaryDirectory() as d:
            for cap in [0, 5, True, 1.5]:
                with self.assertRaises(ValueError): TeacherGPUSharing(d, 'bad', cap)
            for identity in [None, '', 0, 1, False]:
                with self.assertRaises(ValueError): TeacherGPUSharing(d, identity)
            first = TeacherGPUSharing(d, 'first')
            with self.assertRaises(ValueError): TeacherGPUSharing(d, 'other', 2)
            first.close()

    def test_concurrent_processes_serialize(self):
        ctx = mp.get_context('spawn')
        with tempfile.TemporaryDirectory() as d:
            ready, result, start = ctx.Queue(), ctx.Queue(), ctx.Event()
            workers = [ctx.Process(target=execute, args=(d, ready, start, result)) for _ in range(2)]
            for p in workers: p.start()
            for _ in workers: self.assertTrue(ready.get(timeout=10))
            start.set()
            events = [result.get(timeout=10) for _ in workers]
            for p in workers:
                p.join(10)
                self.assertEqual(p.exitcode, 0)
            events.sort(key=lambda e:e['start_monotonic'])
            self.assertLessEqual(events[0]['end_monotonic'], events[1]['start_monotonic'])
            self.assertTrue(all(e['queue_wait_seconds'] >= 0 for e in events))
            self.assertEqual(sum(len(p.read_text().splitlines()) for p in Path(d).glob('events-*.jsonl')), 2)

    def test_crash_releases_admission_and_forward(self):
        ctx = mp.get_context('spawn')
        with tempfile.TemporaryDirectory() as d:
            ready = ctx.Event()
            p = ctx.Process(target=occupy, args=(d, ready))
            p.start()
            self.assertTrue(ready.wait(10))
            p.terminate()
            p.join(10)
            gate = TeacherGPUSharing(d, 'replacement', max_instances=1, timeout_seconds=.2)
            with gate.phase('recovered'): pass
            gate.close()

    def test_timeout_and_exception_release(self):
        with tempfile.TemporaryDirectory() as d:
            first = TeacherGPUSharing(d, 'first')
            second = TeacherGPUSharing(d, 'second', timeout_seconds=.1)
            with first.phase('holding'):
                with self.assertRaises(TimeoutError):
                    with second.phase('blocked'): pass
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                with first.phase('failure'): raise RuntimeError('test failure')
            with second.phase('after_failure'): pass
            events = [json.loads(l) for p in Path(d).glob('events-*.jsonl') for l in p.read_text().splitlines()]
            self.assertIn('failed', [e['status'] for e in events])
            first.close(); second.close()


if __name__ == '__main__': unittest.main()
