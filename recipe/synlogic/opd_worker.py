"""One student GPU and one frozen base-reference/teacher GPU per OPD run."""
from copy import deepcopy

import torch
from omegaconf import OmegaConf, open_dict

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_device_id
from verl.workers.actor import DataParallelPPOActor
from verl.workers.fsdp_workers import ActorRolloutRefWorker


class OPDReferenceWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if not self._is_ref or self.world_size != 1:
            raise ValueError('OPD reference worker requires its own single GPU')
        self.teacher_module = self._build_model_optimizer(
            model_path=self.config.ref.teacher_model_path,
            fsdp_config=self.config.ref.fsdp_config, optim_config=None,
            override_model_config=OmegaConf.to_container(self.config.model.override_config),
            use_remove_padding=self.config.model.use_remove_padding,
            use_fused_kernels=self.config.model.get('use_fused_kernels', False),
            trust_remote_code=False, use_liger=self.config.model.get('use_liger', False),
            role='ref')[0]
        self.teacher_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.teacher_module)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data):
        result = super().compute_ref_log_prob(data)
        data = data.to(get_device_id())
        data.meta_info.update(micro_batch_size=self.config.ref.log_prob_micro_batch_size_per_gpu,
                              temperature=self.config.rollout.temperature,
                              max_token_len=self.config.ref.log_prob_max_token_len_per_gpu,
                              use_dynamic_bsz=self.config.ref.log_prob_use_dynamic_bsz)
        with self.ulysses_sharding_manager:
            prepared = self.ulysses_sharding_manager.preprocess_data(data)
            probabilities, _ = self.teacher_policy.compute_log_prob(prepared, calculate_entropy=False)
            teacher = DataProto.from_dict(tensors={'teacher_log_probs': probabilities})
            teacher = self.ulysses_sharding_manager.postprocess_data(teacher).to('cpu')
        return result.union(teacher)


class OPDActorWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def calibrate_opd(self, data):
        """Measure real policy gradients without changing weights or Adam state."""
        if self.world_size != 1 or self._is_offload_param or self._is_offload_optimizer:
            raise ValueError('Calibration requires the configured single-GPU resident actor')
        if getattr(self.actor, 'grad_scaler', None) is not None:
            raise ValueError('Calibration does not modify a mixed-precision scaler')
        count = 4 * self.config.actor.ppo_mini_batch_size
        if len(data) < count:
            raise ValueError('Not enough responses for four calibration minibatches')
        data = data[:count]
        optimizer = self.actor.actor_optimizer
        original_step = optimizer.step
        original_kl = self.config.actor.use_kl_loss
        original_epochs = self.config.actor.ppo_epochs
        rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        counters = [float(v['step']) for v in optimizer.state.values() if 'step' in v]
        output = {}
        try:
            optimizer.step = lambda *args, **kwargs: None
            with open_dict(self.config.actor):
                self.config.actor.use_kl_loss = False
                self.config.actor.ppo_epochs = 1
            for name, advantages in [('outcome', data.batch['advantages']),
                                      ('teacher', data.batch['teacher_log_probs'] - data.batch['old_log_probs'])]:
                probe = deepcopy(data)
                probe.batch['advantages'] = advantages.detach().clone()
                stats = self.actor.update_policy(probe)
                output[name + '_gradient_norms'] = [float(v) for v in stats['actor/grad_norm']]
        finally:
            optimizer.step = original_step
            with open_dict(self.config.actor):
                self.config.actor.use_kl_loss = original_kl
                self.config.actor.ppo_epochs = original_epochs
            optimizer.zero_grad(set_to_none=True)
            torch.set_rng_state(rng)
            torch.cuda.set_rng_state(cuda_rng)
        assert counters == [float(v['step']) for v in optimizer.state.values() if 'step' in v]
        return output
