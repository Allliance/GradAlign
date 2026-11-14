"""Student workers and a separate frozen base-reference/teacher worker."""
from copy import deepcopy
import os

import torch
from omegaconf import OmegaConf, open_dict

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_device_id
from verl.workers.actor import DataParallelPPOActor
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.ray_utils import ray_noset_visible_devices


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

    def _check_ta_alignment(self,data,reference,actual,role):
        import json,time
        from pathlib import Path
        length=actual.shape[1]
        mask=data.batch['attention_mask'][:len(actual),-length:].bool().cpu()
        diff=(reference.cpu()-actual.cpu()).abs()[mask]
        mean=float(diff.mean());maximum=float(diff.max())
        assert mean<.03 and maximum<1.,f'TA causal-logprob alignment failed: {mean}, {maximum}'
        root=Path(os.environ['SYNLOGIC_RUN_ROOT'])
        proof={'passed':True,'role':role,'rank':self.rank,'mean_abs_logprob_difference':mean,
               'max_abs_logprob_difference':maximum,'positions':int(mask.sum()),'epoch':time.time()}
        (root/f'ta-gpu-alignment-{role}-rank{self.rank}.json').write_text(json.dumps(proof))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def opd_device_identity(self):
        import ray
        return {'role': self.role, 'assigned': ray.get_gpu_ids(),
                'node_id': ray.get_runtime_context().get_node_id(),
                'current_device': torch.cuda.current_device(),
                'visibility_unmasked': ray_noset_visible_devices()}


class OPDReferenceWorker(AllocatedGPUWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_tip_student_head(self,weight,version):
        self._tip_student_head=weight.to(get_device_id(),dtype=torch.bfloat16).detach()
        self._tip_version=version
        return {'version':version,'shape':list(self._tip_student_head.shape)}

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_tip_statistics(self,data):
        assert data.meta_info['tip_policy_version']==self._tip_version
        from .tip_stats import native_teacher_statistics
        import json,time
        from pathlib import Path
        out=native_teacher_statistics(self.teacher_policy,data,self._tip_student_head,
            self.config.rollout.temperature,self.config.ref.log_prob_max_token_len_per_gpu)
        valid=data.batch['response_mask'].bool().cpu();root=Path(os.environ['SYNLOGIC_RUN_ROOT'])
        for key,reference in [('student_lp',data.batch['old_log_probs'].cpu()),('teacher_lp',out['native_teacher_lp'])]:
            difference=(out[key]-reference).abs()[valid]
            proof={'passed':True,'policy_version':self._tip_version,'key':key,'mean_abs_difference':float(difference.mean()),'max_abs_difference':float(difference.max()),'positions':len(difference),'epoch':time.time(),'teacher_source':'actual native forward logits; no reconstructed teacher head'}
            limit=.001 if key=='teacher_lp' else 1.
            proof['passed']=float(difference.mean())<.03 and float(difference.max())<limit
            (root/f'tip-gpu-alignment-{key}-rank{self.rank}.json').write_text(json.dumps(proof))
            assert proof['passed'],'TIP exact native probability alignment failed: '+str(proof)
        return DataProto.from_dict(tensors={'tip_entropy':out['entropy'],'tip_rkl':out['rkl'],
            'tip_fkl':out['fkl'],'teacher_log_probs':out['native_teacher_lp']})

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
    def _compute_original_ref_log_prob(self, data):
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


    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data):
        if self.config.ref.get('oracle_opsd',False):
            from .oracle_opsd import load_oracle_map,privileged_batch
            from .ta_opd import score_support
            if not hasattr(self,'_oracle_answers'):
                from transformers import AutoTokenizer
                self._oracle_answers=load_oracle_map(self.config.ref.oracle_map,self.config.ref.oracle_map_sha256)
                self._oracle_tokenizer=AutoTokenizer.from_pretrained(self.config.ref.teacher_model_path,trust_remote_code=False)
            privileged=privileged_batch(data,self._oracle_tokenizer,self._oracle_answers)
            result=score_support(self.teacher_policy.actor_module,privileged,temperature=self.config.rollout.temperature,k=100)
            return DataProto.from_dict(tensors={'teacher_log_probs':result['chosen_lp'],
                'opsd_teacher_ids':result['ids'].to(torch.int32),'opsd_teacher_lp':result['lp']})
        if not self.config.ref.get('ta_support_k'):
            return self._compute_original_ref_log_prob(data)
        assert self.teacher_only and self.ulysses_sequence_parallel_size==1
        from .ta_opd import score_support
        baseline=None if getattr(self,'_ta_scoring_checked',False) else self._compute_original_ref_log_prob(data[:2])
        result=score_support(self.teacher_policy.actor_module,data,
            temperature=self.config.rollout.temperature,k=int(self.config.ref.ta_support_k),
            candidate_key='ta_student_ids')
        if baseline is not None:
            self._check_ta_alignment(data,baseline.batch['teacher_log_probs'],result['chosen_lp'][:2],'teacher')
            self._ta_scoring_checked=True
        return DataProto.from_dict(tensors={'teacher_log_probs':result['chosen_lp'],
            'ta_teacher_ids':result['ids'],'ta_teacher_lp':result['lp'],
            'ta_teacher_on_student_lp':result['candidate_lp']})


class OPDActorWorker(AllocatedGPUWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_tip_student_head(self):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        model=self.actor.actor_module
        with FSDP.summon_full_params(model,recurse=False,writeback=False,rank0_only=True,offload_to_cpu=True):
            return model.module.lm_head.weight.detach().to(torch.bfloat16).cpu().clone() if self.rank==0 else None

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_tip_hidden(self,data):
        assert not self._is_offload_param and self.ulysses_sequence_parallel_size==1
        from .tip_stats import hidden_predictors
        h=hidden_predictors(self.actor.actor_module,data,packed=self.actor.use_remove_padding).cpu()
        if getattr(self.actor.actor_module,'_handle',None) is not None and self.actor.actor_module._handle.uses_sharded_strategy:
            self.actor.actor_module._handle.reshard(True)
        return DataProto.from_dict(tensors={'tip_student_hidden':h})

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data):
        if not self.config.actor.get('ta_support_k'):
            return super().compute_log_prob(data)
        assert self.ulysses_sequence_parallel_size==1 and not self._is_offload_param
        from .ta_opd import score_support
        baseline=None if getattr(self,'_ta_scoring_checked',False) else super().compute_log_prob(data[:2])
        result=score_support(self.actor.actor_module,data,
            temperature=self.config.rollout.temperature,k=int(self.config.actor.ta_support_k),entropy=True)
        if baseline is not None:
            self._check_ta_alignment(data,baseline.batch['old_log_probs'],result['chosen_lp'][:2],'student')
            self._ta_scoring_checked=True
        return DataProto.from_dict(tensors={'old_log_probs':result['chosen_lp'],
            'entropys':result['entropy'],'ta_student_ids':result['ids'],
            'ta_student_lp':result['lp']},meta_info={'temperature':self.config.rollout.temperature})

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ta_candidates(self,data):
        assert self.config.actor.get('ta_support_k') and self.ulysses_sequence_parallel_size==1
        from .ta_opd import score_support
        result=score_support(self.actor.actor_module,data,
            temperature=self.config.rollout.temperature,k=0,candidate_key='ta_teacher_ids')
        return DataProto.from_dict(tensors={'ta_student_on_teacher_lp':result['candidate_lp']})

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
