"""Student workers and a separate frozen base-reference/teacher worker."""
from copy import deepcopy
from contextlib import nullcontext
import os
import traceback

import torch
from omegaconf import OmegaConf, open_dict

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_device_id
from verl.workers.actor import DataParallelPPOActor
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.ray_utils import ray_noset_visible_devices


def _clear_exception_frames(error):
    pending, seen = [error], set()
    while pending:
        error = pending.pop()
        if id(error) in seen:
            continue
        seen.add(id(error))
        traceback.clear_frames(error.__traceback__)
        pending.extend(e for e in (error.__cause__, error.__context__) if e is not None)


class AllocatedGPUWorker(ActorRolloutRefWorker):
    def __init__(self, config, role):
        # This ROCm image leaves visibility unchanged; a role's local rank need
        # not match its Ray-assigned device, especially across separate pools.
        if ray_noset_visible_devices():
            import ray
            assigned = ray.get_gpu_ids()
            if len(assigned) != 1:
                raise ValueError('Each OPD role requires exactly one Ray-assigned GPU')
            device = int(float(assigned[0]))
            os.environ['RAY_LOCAL_RANK'] = str(device)
            os.environ['LOCAL_RANK'] = str(device)
            torch.cuda.set_device(device)
        super().__init__(config, role)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def opd_device_identity(self):
        import ray
        return {'role': self.role, 'assigned': ray.get_gpu_ids(),
                'node_id': ray.get_runtime_context().get_node_id(),
                'current_device': torch.cuda.current_device(),
                'visibility_unmasked': ray_noset_visible_devices()}


class OPDReferenceWorker(AllocatedGPUWorker):
    def __init__(self, config, role):
        config = deepcopy(config)
        with open_dict(config.actor):
            config.actor.ulysses_sequence_parallel_size = 1
        with open_dict(config.ref):
            config.ref.ulysses_sequence_parallel_size = 1
        with open_dict(config.actor.fsdp_config):
            config.actor.fsdp_config.fsdp_size = 1
        self.teacher_only = not config.ref.get('compute_base_reference', True)
        if self.teacher_only:
            with open_dict(config.model):
                config.model.path = config.ref.teacher_model_path
        super().__init__(config, role)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self._teacher_sharing = None
        sharing = self.config.ref.get('teacher_sharing')
        if sharing:
            from .teacher_sharing import TeacherGPUSharing
            self._teacher_sharing = TeacherGPUSharing(**OmegaConf.to_container(sharing, resolve=True))
        initialization_started = False
        try:
            with self._teacher_sharing.phase('initialization') if self._teacher_sharing else nullcontext():
                initialization_started = True
                try:
                    self._init_teacher_model()
                except BaseException as error:
                    if self._teacher_sharing:
                        _clear_exception_frames(error)
                    raise
                finally:
                    if self._teacher_sharing:
                        try:
                            torch.cuda.synchronize()
                        finally:
                            torch.cuda.empty_cache()
        except BaseException:
            # A failed Ray method can leave model attributes resident. Keep its
            # admission until the supervisor terminates the failed process.
            if self._teacher_sharing and not initialization_started:
                self._teacher_sharing.close()
            raise

    def _init_teacher_model(self):
        super().init_model()
        if not self._is_ref or self.world_size != 1:
            raise ValueError('OPD reference worker requires its own single GPU')
        if self.teacher_only:
            self.teacher_policy = self.ref_policy
            return
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
        gate = self._teacher_sharing
        with gate.phase('scoring') if gate else nullcontext() as event:
            if gate:
                torch.cuda.reset_peak_memory_stats()
            try:
                result = self._compute_teacher_log_prob(data)
            except BaseException as error:
                if gate:
                    # Failed model frames can otherwise retain logits and
                    # microbatches until after the shared lock is released.
                    _clear_exception_frames(error)
                raise
            finally:
                if gate:
                    # Native DataProto.to mutates the input in place. Return it
                    # to CPU before empty_cache so its storage is released while
                    # we still own the forward lock, on success and failure.
                    try:
                        try:
                            data.to('cpu')
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
            return result

    def _compute_teacher_log_prob(self, data):
        result = super().compute_ref_log_prob(data)
        if self.teacher_only:
            result.batch['teacher_log_probs'] = result.batch.pop('ref_log_prob')
            return result
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


class OPDActorWorker(AllocatedGPUWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def calibrate_opd(self, data):
        """Measure real policy gradients without changing weights or Adam state."""
        if self._is_offload_param or self._is_offload_optimizer:
            raise ValueError('Calibration requires a resident actor')
        if self.world_size != self.ulysses_sequence_parallel_size:
            raise ValueError('Calibration broadcasts one full batch across a sequence-parallel group')
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
                with self.ulysses_sharding_manager:
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
