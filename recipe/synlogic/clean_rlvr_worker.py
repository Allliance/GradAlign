"""Original-Base frozen reference for a new post-OPD RLVR stage."""
from copy import deepcopy
from verl.workers.fsdp_workers import ActorRolloutRefWorker

class CleanRLVRReferenceWorker(ActorRolloutRefWorker):
    def __init__(self, config, role):
        config = deepcopy(config)
        if role != 'ref':
            raise ValueError('This worker is only the frozen reference policy')
        config.model.path = config.ref.initial_base_model_path
        super().__init__(config, role)
