from pathlib import Path
import importlib.util,sys,tempfile,types,json,threading,unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('worker_sharing_under_test',Path(__file__).resolve().parents[1]/'recipe/synlogic/teacher_sharing.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)

class WorkerSharingTests(unittest.TestCase):
    def test_rpcs_nested_paths_threads_and_private_heads(self):
        with patch.dict(sys.modules):
            calls=[]
            cuda=types.SimpleNamespace(synchronize=lambda:calls.append('sync'),empty_cache=lambda:calls.append('flush'),reset_peak_memory_stats=lambda:None,max_memory_allocated=lambda:10,max_memory_reserved=lambda:20,memory_allocated=lambda:5,memory_reserved=lambda:8)
            sys.modules['torch']=types.SimpleNamespace(cuda=cuda,is_tensor=lambda v:False)
            sys.modules['omegaconf']=types.SimpleNamespace(OmegaConf=types.SimpleNamespace(to_container=lambda x,**kw:x))
            class Data:
             def __init__(self):self.batch={};self.device='cuda'
             def to(self,device):self.device=device;return self
            class Worker:
             def __init__(self,config,role):self.config=config;self.work=[];assert self._teacher_sharing.owns_phase()
             def init_model(self):assert self._teacher_sharing.owns_phase();self.initialized=True
             def compute_ref_log_prob(self,data):return self._compute_original_ref_log_prob(data)
             def _compute_original_ref_log_prob(self,data):assert self._teacher_sharing.owns_phase();self.work.append('native');return 1
             def set_tip_student_head(self,weight,version):assert self._teacher_sharing.owns_phase();self.head=weight;self.version=version
             def compute_tip_statistics(self,data):assert self._teacher_sharing.owns_phase();assert self.version==1;return self.head
            Worker.compute_ref_log_prob._dispatch_metadata={'dispatch':'preserved'}
            s.wrap_teacher_worker(Worker)
            with tempfile.TemporaryDirectory() as d:
             config=types.SimpleNamespace(ref={'teacher_sharing':{'directory':d,'instance_id':'client0'}})
             w=Worker(config,'ref');w.init_model();data=Data();assert w.compute_ref_log_prob(data)==1;assert data.device=='cpu';w.set_tip_student_head('private0',1);assert w.compute_tip_statistics(data)=='private0'
             assert w.compute_ref_log_prob._dispatch_metadata=={'dispatch':'preserved'}
             other=Worker(types.SimpleNamespace(ref={'teacher_sharing':{'directory':d,'instance_id':'client1'}}),'ref');other.init_model();other.set_tip_student_head('private1',1);assert other.head=='private1' and w.head=='private0'
             assert [json.loads(x)['phase'] for x in w._teacher_sharing.events.read_text().splitlines()]==['construction','initialization','compute_ref_log_prob','set_tip_student_head','compute_tip_statistics']
             # Direct alternate RPC is gated, with nested same-thread calls retaining outer ownership.
             w._compute_original_ref_log_prob(data);assert json.loads(w._teacher_sharing.events.read_text().splitlines()[-1])['phase']=='_compute_original_ref_log_prob'
             with w._teacher_sharing.phase('thread-owner'):
              result=[]
              def thread():
               try:w.compute_ref_log_prob(data)
               except RuntimeError:result.append('rejected')
              t=threading.Thread(target=thread);t.start();t.join();assert result==['rejected']
             w._teacher_sharing.close();other._teacher_sharing.close()


if __name__=='__main__': unittest.main()
