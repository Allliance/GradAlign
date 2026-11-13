"""Preserve communication headroom for the colocated clean RLVR worker."""
import torch
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.fsdp_workers import ActorRolloutRefWorker


class CleanRLVRActorWorker(ActorRolloutRefWorker):
    def __init__(self, config, role):
        if role != 'actor_rollout' or not config.ref.get('initial_base_model_path'):
            raise ValueError('This actor is only for the separate-Base-reference RLVR stage')
        super().__init__(config, role)
        # RCCL allocates outside the PyTorch cache. Keep driver memory available.
        torch.cuda.set_per_process_memory_fraction(0.9)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data):
        # Scoring can leave large unused cached blocks immediately before FSDP
        # backward. Release only those blocks; tensors and optimizer math stay.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return super().update_actor(data)
