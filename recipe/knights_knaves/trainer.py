"""Full dataset passes, native recovery state and frozen clean validation."""
import hashlib,json
from pathlib import Path
from src.dapo_ray_trainer import RayDAPOTrainer
from synlogic.sampling import PlanSampler,PlanDataLoader

class Recorder:
    def __init__(self,reward):self.reward,self.records=reward,[]
    def __call__(self,data,return_dict=False):
        result=self.reward(data,return_dict=True)
        self.records.extend((int(data[i].non_tensor_batch['extra_info']['index']),float(v)) for i,v in enumerate(result['reward_tensor'].sum(-1).tolist()))
        return result if return_dict else result['reward_tensor']

class KnightsKnavesTrainer(RayDAPOTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.val_reward_fn=Recorder(self.val_reward_fn)
    def _create_dataloader(self,train_dataset,val_dataset,collate_fn,train_sampler):
        from verl.trainer.main_ppo import create_rl_dataset
        if train_dataset is None:train_dataset=create_rl_dataset(self.config.data.train_files,self.config.data,self.tokenizer,self.processor)
        from .run import verify_inputs
        manifest=verify_inputs(Path(self.config.mixture.root))
        ids=[int(x['index']) for x in train_dataset.dataframe['extra_info']]
        assert ids==manifest['training_ids']
        assert len(ids)%self.config.data.train_batch_size==0
        sampler=PlanSampler(train_dataset,self.config.mixture.condition,seed=self.config.data.seed,
                            batch_size=self.config.data.train_batch_size,batches_per_pass=len(ids)//self.config.data.train_batch_size)
        super()._create_dataloader(train_dataset,val_dataset,collate_fn,sampler)
        original=self.train_dataloader
        self.train_dataloader=PlanDataLoader(dataset=train_dataset,batch_size=original.batch_size,num_workers=0,
            drop_last=False,collate_fn=original.collate_fn,sampler=sampler)
        assert len(self.train_dataloader)*self.config.trainer.total_epochs==self.total_training_steps
    def _save_checkpoint(self):
        folder=Path(self.config.trainer.default_local_dir)/f'global_step_{self.global_steps}'
        folder.mkdir(parents=True,exist_ok=True)
        state={'step':self.global_steps,'protocol_id':self.config.provenance.protocol_id,
            'prompt_template_sha256':self.config.provenance.prompt_template_sha256,
            'data_fingerprint':self.config.provenance.data_fingerprint,'condition':self.config.mixture.condition,
            'world_size':1,'expected_model_layers':36,'expected_model_parameters':4022468096,
            'optimizer_updates':self.global_steps,'training_thinking':False,'validation_thinking':False}
        tmp=folder/'experiment_state.tmp';tmp.write_text(json.dumps(state)+'\n');tmp.replace(folder/'experiment_state.json')
        return super()._save_checkpoint()
    def _load_checkpoint(self):
        result=super()._load_checkpoint()
        if self.global_steps:
            folder=Path(self.config.trainer.default_local_dir)/f'global_step_{self.global_steps}'
            state=json.loads((folder/'experiment_state.json').read_text())
            for key,value in [('protocol_id',self.config.provenance.protocol_id),('data_fingerprint',self.config.provenance.data_fingerprint),
                ('prompt_template_sha256',self.config.provenance.prompt_template_sha256),('world_size',1),('training_thinking',False),('validation_thinking',False)]:
                assert state[key]==value,(key,state[key],value)
        return result
    def _validate(self):
        self.val_reward_fn.records.clear();metrics=super()._validate();records=self.val_reward_fn.records
        manifest=json.loads((Path(self.config.mixture.root)/'data/manifest.json').read_text())
        ids=set(manifest['validation_ids']);n=self.config.actor_rollout_ref.rollout.val_kwargs.n
        from collections import Counter
        counts=Counter(i for i,s in records)
        assert set(counts)==ids and all(c==n for c in counts.values())
        metrics['val-core/knights_knaves/reward/mean@8']=sum(s for i,s in records)/len(records)
        return metrics
