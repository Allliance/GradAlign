"""Run the SynLogic experiments in the calibrated ROCm veRL backend."""

from copy import deepcopy
import os
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import ray


@hydra.main(config_path=str(Path(os.environ['SYNLOGIC_FRAMEWORK_CODE']) / 'src/config'),
            config_name='dapo_trainer', version_base=None)
def main(config):
    gpus = 2 if config.get('opd', {}).get('enabled', False) else config.trainer.n_gpus_per_node
    # Full-gradient selection broadcasts several copies of a 16-response batch.
    # Keep these objects in shared memory instead of spilling to local disk.
    object_store_gib = 48 if config.mixture.condition == 'mixture-gradalign' else 8
    ray.init(num_cpus=config.ray_init.num_cpus, num_gpus=gpus, include_dashboard=False,
             object_store_memory=object_store_gib * 1024**3, _temp_dir='/run/synlogic-cache/ray',
             runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN',
                                       'VLLM_LOGGING_LEVEL': 'WARN'}})
    ray.get(TaskRunner.remote().run.remote(config))


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from verl.utils import hf_tokenizer
        from verl.workers.fsdp_workers import ActorRolloutRefWorker
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from src.syn_logic_reward import SynLogicRewardManager
        from .trainer import SynLogicTrainer
        from .template import verify_base_template
        from .alignment_worker import AlignmentWorker

        OmegaConf.resolve(config)
        tokenizer = verify_base_template(hf_tokenizer(config.actor_rollout_ref.model.path))
        opd_enabled = config.get('opd', {}).get('enabled', False)
        if opd_enabled:
            from .opd_worker import OPDActorWorker, OPDReferenceWorker
            from .opd_trainer import OPDTrainer
            from src.train_eval_guard import check_resolved_inputs
            guard = check_resolved_inputs(
                config.data.train_files, config.data.val_files, config.provenance.data_fingerprint,
                config.opd.exclusions_file, config.opd.additional_questions_file)
            print('Resolved training input guard:', guard, flush=True)
        actor_class = (OPDActorWorker if opd_enabled else AlignmentWorker
                       if config.mixture.condition == 'mixture-gradalign' else ActorRolloutRefWorker)
        roles = {Role.ActorRollout: ray.remote(actor_class),
                 Role.RefPolicy: ray.remote(OPDReferenceWorker if opd_enabled else ActorRolloutRefWorker)}
        pools = {'student': [1], 'teacher_reference': [1]} if opd_enabled else {'training': [config.trainer.n_gpus_per_node]}
        mapping = ({Role.ActorRollout: 'student', Role.RefPolicy: 'teacher_reference'} if opd_enabled
                   else {role: 'training' for role in roles})
        manager = ResourcePoolManager(resource_pool_spec=pools, mapping=mapping)
        reward_args = dict(tokenizer=tokenizer, num_examine=0, reward_fn_key=config.data.reward_fn_key,
                           max_resp_len=config.data.max_response_length,
                           overlong_buffer_cfg=deepcopy(config.reward_model.overlong_buffer))
        trainer_class = OPDTrainer if opd_enabled else SynLogicTrainer
        trainer = trainer_class(config=config, tokenizer=tokenizer, processor=None,
            role_worker_mapping=roles, resource_pool_manager=manager, ray_worker_group_cls=RayWorkerGroup,
            reward_fn=SynLogicRewardManager(syn_logic_score_type='acc_without_think_format', **reward_args),
            val_reward_fn=SynLogicRewardManager(syn_logic_score_type='validation_acc', **reward_args),
            device_name=config.trainer.device)
        trainer.init_workers()
        trainer.fit()


if __name__ == '__main__':
    main()
