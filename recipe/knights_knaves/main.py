"""Use the pinned, previously deployed ROCm veRL runtime."""
import os
from pathlib import Path
import hydra,ray
from omegaconf import OmegaConf

@hydra.main(config_path=str(Path(os.environ['SYNLOGIC_FRAMEWORK_CODE'])/'src/config'),config_name='dapo_trainer',version_base=None)
def main(config):
    keys=('SYNLOGIC_FRAMEWORK_CODE','SYNLOGIC_RUN_ROOT','SYNLOGIC_WANDB_ENV','SYNLOGIC_ALLOCATION_ID',
          'SYNLOGIC_ATTEMPT','SYNLOGIC_LEASE_DRAIN_EPOCH','SYNLOGIC_SUITE_ROOT','WANDB_TAGS','PYTHONPATH')
    ray.init(num_cpus=config.ray_init.num_cpus,num_gpus=1,include_dashboard=False,
        object_store_memory=8*1024**3,_temp_dir='/run/synlogic-cache/ray',
        runtime_env={'env_vars':{'TOKENIZERS_PARALLELISM':'true','NCCL_DEBUG':'WARN','VLLM_LOGGING_LEVEL':'WARN',
            **{k:os.environ[k] for k in keys if k in os.environ}}})
    ray.get(TaskRunner.remote().run.remote(config))

@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self,config):
        from verl.utils import hf_tokenizer
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager,Role
        from synlogic.clean_rlvr_actor_worker import CleanRLVRActorWorker
        from synlogic.clean_rlvr_worker import CleanRLVRReferenceWorker
        from .trainer import KnightsKnavesTrainer
        from .reward import KnightsKnavesRewardManager
        OmegaConf.resolve(config);tokenizer=hf_tokenizer(config.actor_rollout_ref.model.path)
        trainer=KnightsKnavesTrainer(config=config,tokenizer=tokenizer,processor=None,
            role_worker_mapping={Role.ActorRollout:ray.remote(CleanRLVRActorWorker),Role.RefPolicy:ray.remote(CleanRLVRReferenceWorker)},
            resource_pool_manager=ResourcePoolManager(resource_pool_spec={'training':[1]},mapping={Role.ActorRollout:'training',Role.RefPolicy:'training'}),
            ray_worker_group_cls=RayWorkerGroup,reward_fn=KnightsKnavesRewardManager(tokenizer),
            val_reward_fn=KnightsKnavesRewardManager(tokenizer,validation=True),device_name=config.trainer.device)
        trainer.init_workers();trainer.fit()

if __name__=='__main__':main()
