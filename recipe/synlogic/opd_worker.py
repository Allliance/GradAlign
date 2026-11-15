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
        from .tip_stats import hidden_predictors,full_statistics
        h=hidden_predictors(self.teacher_policy.actor_module,data)
        wrapped=self.teacher_policy.actor_module.module if hasattr(self.teacher_policy.actor_module,'module') else self.teacher_policy.actor_module
        sh=data.batch['tip_student_hidden'].to(h.device)
        out=full_statistics(sh,h,self._tip_student_head,wrapped.lm_head,
                            data.batch['responses'],self.config.rollout.temperature)
        valid=data.batch['response_mask'].bool().cpu()
        for key,reference in [('student_lp','old_log_probs'),('teacher_lp','teacher_log_probs')]:
            difference=(out[key]-data.batch[reference].cpu()).abs()[valid]
            assert float(difference.mean())<.03 and float(difference.max())<1.,'TIP prefix/head alignment failed'
        return DataProto.from_dict(tensors={'tip_entropy':out['entropy'],'tip_rkl':out['rkl'],'tip_fkl':out['fkl']})

    def __init__(self, config, role):
        if config.ref.get('teacher_sharing') and os.environ.get('SYNLOGIC_TEACHER_PACKED') != '1':
            raise ValueError('Packed teacher requires explicit execution intent')
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
        if self.config.ref.get('retrieval_privilege',False):
            from .retrieval_context import FrozenReferences,ContextBudgets,privileged_batch
            from .native_stats import score_native
            if not hasattr(self,'_retrieval_references'):
                from transformers import AutoTokenizer
                self._retrieval_references=FrozenReferences(self.config.ref.retrieval_manifest,self.config.ref.retrieval_manifest_sha256)
                self._context_budgets=ContextBudgets(self.config.ref.context_budget_path,self.config.ref.context_budget_sha256,self._retrieval_references)
                self._retrieval_tokenizer=AutoTokenizer.from_pretrained(self.config.ref.teacher_model_path,local_files_only=True)
            privileged=privileged_batch(data,self._retrieval_tokenizer,self._retrieval_references,self.config.ref.retrieval_max_context,self._context_budgets)
            result=score_native(self.teacher_policy,privileged,temperature=self.config.rollout.temperature,
                                max_token_len=self.config.ref.log_prob_max_token_len_per_gpu,k=0)
            if not getattr(self,'_retrieval_scoring_checked',False):
                self._check_ta_alignment(privileged,result['native_lp'][:2],result['chosen_lp'][:2],'retrieval-privileged-teacher')
                self._retrieval_scoring_checked=True
            return DataProto.from_dict(tensors={'teacher_log_probs':result['chosen_lp']})
        if self.config.ref.get('oracle_opsd',False):
            from .oracle_opsd import load_oracle_map,privileged_batch
            from .native_stats import score_native
            if not hasattr(self,'_oracle_answers'):
                from transformers import AutoTokenizer
                self._oracle_answers=load_oracle_map(self.config.ref.oracle_map,self.config.ref.oracle_map_sha256)
                self._oracle_tokenizer=AutoTokenizer.from_pretrained(self.config.ref.teacher_model_path,trust_remote_code=False)
            privileged=privileged_batch(data,self._oracle_tokenizer,self._oracle_answers)
            result=score_native(self.teacher_policy,privileged,temperature=self.config.rollout.temperature,
                max_token_len=self.config.ref.log_prob_max_token_len_per_gpu,k=100)
            baseline=None if getattr(self,'_oracle_scoring_checked',False) else DataProto.from_dict(tensors={'teacher_log_probs':result['native_lp'][:2]})
            if baseline is not None:
                self._check_ta_alignment(privileged,baseline.batch['teacher_log_probs'],result['chosen_lp'][:2],'oracle-privileged-teacher')
                self._oracle_scoring_checked=True
            valid=privileged.batch['response_mask'].bool()
            assert (result['lp'].exp().sum(-1)[valid]<=1.00001).all()
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
    def init_model(self):
        super().init_model()
        if self.config.actor.get('shared_current_opsd',False):
            assert not self._is_ref and self._is_actor and self.world_size==2
            assert not self._is_offload_param and self.ulysses_sequence_parallel_size==1
            assert self.config.actor.shared_teacher_token_budget>=self.config.actor.ppo_max_token_len_per_gpu
            self.actor.shared_teacher_callback=self._shared_teacher_forward

    def _shared_teacher_forward(self,micro,temperature,version):
        from .native_stats import score_native
        model=self.actor.actor_module
        counters={int(state['step']) for state in self.actor.actor_optimizer.state.values() if 'step' in state}
        assert counters=={int(version)} or (not counters and int(version)==0),'Teacher version differs from actual Adam update counter'
        states=[(module,module.training) for module in model.modules()]
        teacher=DataProto.from_dict(tensors={'responses':micro['responses'],
            'prompts':micro['opsd_privileged_prompts'],'input_ids':micro['opsd_privileged_input_ids'],
            'attention_mask':micro['opsd_privileged_attention_mask'],'position_ids':micro['opsd_privileged_position_ids'],
            'response_mask':micro['attention_mask'][:,-micro['responses'].shape[1]:]})
        try:
            devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices),torch.no_grad():
                result=score_native(self.actor,teacher,temperature=temperature,
                    max_token_len=self.config.actor.shared_teacher_token_budget,k=100,single_micro_batch=True)
        finally:
            for module,training in states:module.training=training
        assert all(not value.requires_grad for value in result.values())
        if 'opsd_teacher_policy_version' in micro:
            assert (micro['opsd_teacher_policy_version']==version).all(),'Journal version differs from actual current minibatch version'
        from pathlib import Path
        import json,time
        root=Path(os.environ['SYNLOGIC_RUN_ROOT'])
        proof={'passed':True,'epoch':time.time(),'policy_version':int(version),'teacher_version':int(version),
               'same_actor_object':True,'stopped_gradient':True,'extra_optimizer_updates':0,
               'teacher_mode':'current actor weights refreshed before each optimizer minibatch; no shadow, EMA or lag',
               'actual_adam_counter':int(version),'rank':self.rank,'valid_response_tokens':int(teacher.batch['response_mask'].sum())}
        path=root/f'shared-teacher-version-rank{self.rank}.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(proof));tmp.replace(path)
        from .shared_targets import record_targets
        record_targets(root,int(version),self.rank,micro,result)
        return {'opsd_teacher_ids':result['ids'].to(micro['responses'].device,dtype=torch.int32),
                'opsd_teacher_lp':result['lp'].to(micro['responses'].device),
                'teacher_log_probs':result['chosen_lp'].to(micro['responses'].device)}

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self,data):
        if self.config.actor.get('shared_current_opsd',False):
            from .retrieval_context import FrozenReferences,ContextBudgets,privileged_batch
            if not hasattr(self,'_retrieval_references'):
                self._retrieval_references=FrozenReferences(self.config.actor.retrieval_manifest,self.config.actor.retrieval_manifest_sha256)
                self._context_budgets=ContextBudgets(self.config.actor.context_budget_path,self.config.actor.context_budget_sha256,self._retrieval_references)
            teacher=privileged_batch(data,self.tokenizer,self._retrieval_references,self.config.actor.retrieval_max_context,self._context_budgets)
            for key in ['prompts','input_ids','attention_mask','position_ids']:
                data.batch['opsd_privileged_'+key]=teacher.batch[key]
            assert data.meta_info['shared_current_opsd'] and int(data.meta_info['opsd_policy_outer_step'])>=1
        return super().update_actor(data)

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
        h=hidden_predictors(self.actor.actor_module,data).cpu()
        if getattr(self.actor.actor_module,'_handle',None) is not None and self.actor.actor_module._handle.uses_sharded_strategy:
            self.actor.actor_module._handle.reshard(True)
        return DataProto.from_dict(tensors={'tip_student_hidden':h})

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data):
        if self.config.actor.get('retrieval_privilege',False) and not self.config.actor.get('ta_support_k'):
            assert not any(key.startswith('opsd_privileged_') for key in data.batch.keys()), 'Privileged context entered student scoring'
            assert not self._is_offload_param and self.ulysses_sequence_parallel_size==1
            baseline=super().compute_log_prob(data)
            if not getattr(self,'_retrieval_student_checked',False):
                from .native_stats import score_native
                result=score_native(self.actor,data[:2],temperature=self.config.rollout.temperature,
                                    max_token_len=self.config.rollout.log_prob_max_token_len_per_gpu,k=0)
                self._check_ta_alignment(data[:2],baseline.batch['old_log_probs'][:2],result['chosen_lp'],'retrieval-unprivileged-student')
                self._retrieval_student_checked=True
            return baseline
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


# The packed launch is a source-level contract; a setup hook is only a verifier.
if os.environ.get('SYNLOGIC_TEACHER_PACKED') == '1':
    import hashlib
    from pathlib import Path
    from verl.single_controller.base.decorator import MAGIC_ATTR
    helper = Path(__file__).with_name('teacher_sharing.py')
    if hashlib.sha256(helper.read_bytes()).hexdigest() != os.environ['SYNLOGIC_TEACHER_SHARING_SHA256']:
        raise ValueError('Packed teacher helper differs from the pinned launch')
    from .teacher_sharing import wrap_teacher_worker
    OPDReferenceWorker = wrap_teacher_worker(OPDReferenceWorker, registered_attribute=MAGIC_ATTR, required=True)
