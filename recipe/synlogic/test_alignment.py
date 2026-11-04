import sys,os,contextlib,types,copy,math
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),os.environ['SYNLOGIC_FRAMEWORK_CODE']]
import torch,numpy as np
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from verl import DataProto
from synlogic.alignment_worker import AlignmentWorker,upstream_loss
from synlogic.alignment import AlignmentLoader

def check_loader():
 class Dataset:
  def __len__(self):return 200
  def __getitem__(self,index):return {'index':index}
 def create():
  return AlignmentLoader(Dataset(),4,lambda generator,round_:torch.randperm(200,generator=generator).tolist()[:40])
 original=create();iterator=iter(original)
 for _ in range(96):next(iterator)
 state=copy.deepcopy(original.state_dict())
 expected=[batch['index'].tolist() for batch in iterator]+[batch['index'].tolist() for batch in original]+[batch['index'].tolist() for batch in original]
 restored=create();restored.load_state_dict(state)
 actual=[batch['index'].tolist() for batch in restored]+[batch['index'].tolist() for batch in restored]+[batch['index'].tolist() for batch in restored]
 assert expected==actual,'Selection or epoch boundary changed on recovery'
 assert original.round==restored.round

def main():
 check_loader();dist.init_process_group('gloo');rank=dist.get_rank()
 torch.manual_seed(21)
 initial=torch.nn.Linear(3,2,bias=False)
 manual=copy.deepcopy(initial)
 model=FSDP(initial,device_id=torch.device('cpu'))
 optimizer=torch.optim.AdamW(model.parameters(),lr=1e-6)
 class Actor:
  actor_module=model
  actor_optimizer=optimizer
  def _forward_micro_batch(self,batch,temperature,calculate_entropy):return None,self.actor_module(batch['x'])
 fake=types.SimpleNamespace(actor=Actor(),ulysses_sharding_manager=contextlib.nullcontext(),
                            config=types.SimpleNamespace(actor=types.SimpleNamespace(ppo_max_token_len_per_gpu=28672)))
 for name in ['_alignment_backward','_alignment_gradients']:
  setattr(fake,name,types.MethodType(getattr(AlignmentWorker,name),fake))
 fake._norm=AlignmentWorker._norm
 torch.manual_seed(7)
 x=torch.randn(48,3)
 advantages=torch.cat([torch.linspace(-1,1,16),torch.linspace(1,-1,16),torch.zeros(16)])[:,None].expand(48,2)
 inputs={'x':x,'responses':torch.zeros(48,2,dtype=torch.long),'attention_mask':torch.ones(48,2),'advantages':advantages}
 data=DataProto.from_dict(tensors=inputs)
 reference=data[:32]
 candidate=data[16:]
 before={k:v.clone() for k,v in optimizer.state_dict().items() if isinstance(v,torch.Tensor)}
 result=AlignmentWorker.gradalign_reference(fake,reference,16)
 scores=AlignmentWorker.gradalign_scores(fake,candidate,16)['scores']
 native=upstream_loss().compute_policy_loss
 with open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink):
  logp=manual(x[:32]);loss=native(logp.detach(),logp,advantages[:32],torch.ones_like(logp))/32
  grad_ref=torch.autograd.grad(loss,manual.weight)[0].flatten()
  logp=manual(x[16:32]);loss=native(logp.detach(),logp,advantages[16:32],torch.ones_like(logp))/16
  grad_candidate=torch.autograd.grad(loss,manual.weight)[0].flatten()
 expected=float(torch.nn.functional.cosine_similarity(grad_ref,grad_candidate,dim=0))
 assert abs(result['reference_norm']-float(grad_ref.norm()))<1e-6,(result,float(grad_ref.norm()))
 assert abs(scores[0]-expected)<1e-5,(scores,expected)
 assert scores[1]==0
 assert not optimizer.state,'Gradient analysis updated the optimizer'
 AlignmentWorker.gradalign_clear(fake)
 if rank==0:print('PASS: four-rank full-gradient cosine matches the released nonsharded loss; zero gradients score zero; optimizer unchanged; curriculum and RNG restore across round and epoch boundaries.',flush=True)
 dist.destroy_process_group()
if __name__=='__main__':main()
