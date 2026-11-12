"""Optimizer membership and variable-token normalization under DP partitioning."""
import copy
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.utils.seqlen_balancing import rearrange_micro_batches
from synlogic.parallelism import prepare_actor_batch,prepare_legacy_actor_batch,rank_major_indices
from synlogic.opd_trainer import OPDTrainer

def fixture():
 mask=torch.zeros(8,8)
 for i in range(8):mask[i,:i+1]=1
 return DataProto.from_dict(tensors=dict(response_mask=mask,attention_mask=torch.cat([torch.ones(8,8),mask],-1).long(),row_id=torch.arange(8)))

def test_optimizer_membership_is_unchanged():
 for dp in (1,2,4):
  order=rank_major_indices(24,8,dp);ranks=torch.tensor(order).reshape(dp,-1)
  for step in range(3):assert sorted(ranks[:,step*(8//dp):(step+1)*(8//dp)].reshape(-1).tolist())==list(range(step*8,(step+1)*8))
 with pytest.raises(ValueError):rank_major_indices(10,4,2)

def distributed_loss(source,prepared,legacy):
 # Simulate mean gradient synchronization and adversarial local microbatch sizes.
 loss=torch.tensor(0.)
 for rank in range(2):
  local=prepared[rank*4:(rank+1)*4]
  for start,size in [(0,1),(1,1),(2,2)]:
   micro=local[start:start+size];ids=micro.batch['row_id'];mask=micro.batch['response_mask'];mat=source[ids]
   if legacy:mat=mat*micro.batch['sample_weights'][:,None]
   denom=micro.batch['legacy_global_minibatch_rows' if legacy else 'global_minibatch_response_tokens'][0]
   loss=loss+(mat*mask).sum()/denom
 return loss

def test_global_token_mean_matches_dp1_with_unequal_lengths():
 batch=fixture();values=torch.linspace(-1,1,64).reshape(8,8).requires_grad_();expected=sum((values[i:i+4]*batch.batch['response_mask'][i:i+4]).sum()/batch.batch['response_mask'][i:i+4].sum() for i in (0,4))
 prepared=prepare_actor_batch(copy.deepcopy(batch),4,2);actual=distributed_loss(values,prepared,False)
 torch.testing.assert_close(actual,expected)
 torch.testing.assert_close(torch.autograd.grad(actual,values)[0],torch.autograd.grad(expected,values)[0])

def test_legacy_microbatch_weighting_is_preserved_after_repartition():
 batch=fixture();values=torch.linspace(-1,1,64).reshape(8,8).requires_grad_();expected=torch.tensor(0.)
 for start in (0,4):
  micros,indices=rearrange_micro_batches(batch.batch[start:start+4],32)
  for micro,idx in zip(micros,indices):expected=expected+(values[torch.tensor(idx)+start]*micro['response_mask']).sum()/micro['response_mask'].sum()*len(idx)/4
 prepared=prepare_legacy_actor_batch(copy.deepcopy(batch),4,2,32);actual=distributed_loss(values,prepared,True)
 torch.testing.assert_close(actual,expected)
 torch.testing.assert_close(torch.autograd.grad(actual,values)[0],torch.autograd.grad(expected,values)[0])

@pytest.mark.parametrize('outcome',[0,1])
def test_explicit_teacher_one_never_reads_or_waits_for_calibration(outcome):
 dummy=SimpleNamespace(config=OmegaConf.create({'opd':{'fixed_teacher_coefficient':1.,'calibrate':False,'outcome_coefficient':outcome}}))
 batch=DataProto.from_dict(tensors={'advantages':torch.ones(1,2),'teacher_log_probs':torch.tensor([[-3.,-1.]]),'old_log_probs':torch.tensor([[-2.,-2.]]),'response_mask':torch.ones(1,2)})
 with patch('synlogic.opd_trainer.noise_metrics',return_value={}):result=OPDTrainer._transform_advantages(dummy,batch,{})
 torch.testing.assert_close(result.batch['teacher_advantages'],torch.tensor([[-1.,1.]]))
 torch.testing.assert_close(result.batch['advantages'],torch.tensor([[-1.,1.]])+outcome)
 assert result.meta_info['opd_teacher_coefficient']==1.


def test_batch_preparation_does_not_mutate_diagnostics_or_compound_weights():
 original=fixture();original.batch['sample_weights']=torch.arange(1,9,dtype=torch.float32)
 saved=copy.deepcopy(original)
 a=prepare_legacy_actor_batch(original,4,2,32);b=prepare_legacy_actor_batch(original,4,2,32)
 assert set(original.batch.keys())==set(saved.batch.keys())
 for key in original.batch.keys():torch.testing.assert_close(original.batch[key],saved.batch[key])
 torch.testing.assert_close(a.batch['sample_weights'],b.batch['sample_weights'])
 prepare_actor_batch(original,4,2)
 assert set(original.batch.keys())==set(saved.batch.keys())

@pytest.mark.parametrize('mode,teacher,outcome',[('importance_sampling',.5,0),('importance_sampling',1.,1),('legacy',1.,0)])
def test_actor_rejects_objective_that_differs_from_checked_trainer(mode,teacher,outcome):
 from unittest.mock import Mock
 from verl.workers.actor import DataParallelPPOActor
 config=OmegaConf.create(dict(opd_loss_mode=mode,opd_teacher_coefficient=teacher,opd_outcome_coefficient=outcome))
 dummy=SimpleNamespace(config=config,actor_module=Mock())
 data=DataProto.from_dict(tensors={'input_ids':torch.ones(1,2,dtype=torch.long),'global_minibatch_response_tokens':torch.ones(1)},meta_info={'temperature':1.,'opd_loss_mode':'importance_sampling','opd_teacher_coefficient':1.,'opd_outcome_coefficient':0.})
 with patch('verl.utils.debug.performance._get_current_mem_info',return_value=('0','0','0','0')):
  with pytest.raises(ValueError,match='objective|actor mode'):DataParallelPPOActor.update_policy(dummy,data)
