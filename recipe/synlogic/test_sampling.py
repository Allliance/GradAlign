import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from synlogic.sampling import PlanSampler,PlanDataLoader
import torch
class Dataset:
 def __init__(self,n):
  self.n=n;self.dataframe={'extra_info':[{'pool':'small' if i<156 else 'large'} for i in range(n)]}
 def __len__(self):return self.n
 def __getitem__(self,index):return index
for mode,size in [('noise30',15610),('small1-noise10',156),('mixture-sampling99',15610)]:
 def create():
  sampler=PlanSampler(Dataset(size),mode)
  sampler.num_samples=160*5
  return PlanDataLoader(sampler.dataset,batch_size=160,sampler=sampler,num_workers=0,drop_last=True)
 original=create();iterator=iter(original)
 next(iterator);next(iterator)
 state=original.state_dict()
 expected=[batch.clone() for batch in iterator]+[batch.clone() for batch in original]+[batch.clone() for batch in original]
 restored=create();restored.load_state_dict(state)
 actual=[batch.clone() for batch in restored]+[batch.clone() for batch in restored]+[batch.clone() for batch in restored]
 assert len(actual)==len(expected) and all(torch.equal(a,b) for a,b in zip(actual,expected)),mode
 if mode=='mixture-sampling99':
  check=create();batches=list(check)
  counts=[int((batch<156).sum()) for batch in batches]
  assert set(counts)<={158,159} and sum(counts)==792,counts
 print(mode,'mid-pass and two later passes match exactly')
