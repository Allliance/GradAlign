"""Launch one independent K&K protocol; verify inputs and both rendered modes."""
import argparse,hashlib,importlib.util,json,os,shlex,sys
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def verify_inputs(root):
    manifest=json.loads((root/'data/manifest.json').read_text())
    for p,digest in manifest['frozen_files'].items():assert sha(p)==digest,p
    assert manifest['training_test_overlap']==manifest['train_validation_overlap']==0
    return manifest

def configuration(root,frame,condition):
    spec=importlib.util.spec_from_file_location('single_card',frame/'tools/train_single_card.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    values=module.overrides(root,'full-clean')
    meta=json.loads((root/'runs'/condition/'portable-provenance.json').read_text())
    run=root/'runs'/condition;small=condition=='clean5'
    replacements={'data.train_files':str(root/'data/prepared'/(condition+'.parquet')),
        'data.val_files':str(root/'data/prepared/validation.parquet'),
        'data.max_prompt_length':'2048','data.max_response_length':'4096','data.truncation':'error',
        'data.train_batch_size':'50','data.gen_batch_size':'50','data.val_batch_size':'50',
        'data.dataloader_num_workers':'0','data.validation_shuffle':'False',
        'actor_rollout_ref.rollout.n':'8','algorithm.filter_groups.enable':'False',
        'actor_rollout_ref.actor.ppo_mini_batch_size':'50',
        'actor_rollout_ref.actor.optim.lr_warmup_steps':'0','actor_rollout_ref.actor.optim.weight_decay':'0.01',
        'actor_rollout_ref.actor.use_kl_loss':'True','actor_rollout_ref.actor.kl_loss_coef':'0.001',
        'actor_rollout_ref.actor.kl_loss_type':'low_var_kl',
        'actor_rollout_ref.actor.clip_ratio_low':'0.2','actor_rollout_ref.actor.clip_ratio_high':'0.2',
        'actor_rollout_ref.actor.clip_ratio_c':'3.0',
        'actor_rollout_ref.actor.ppo_max_token_len_per_gpu':'16384',
        'actor_rollout_ref.ref.log_prob_max_token_len_per_gpu':'16384',
        'actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu':'16384',
        'actor_rollout_ref.rollout.max_num_batched_tokens':'16384','actor_rollout_ref.rollout.max_num_seqs':'128',
        'actor_rollout_ref.rollout.val_kwargs.temperature':'0.6','actor_rollout_ref.rollout.val_kwargs.top_p':'1.0',
        'actor_rollout_ref.rollout.val_kwargs.top_k':'-1','actor_rollout_ref.rollout.val_kwargs.n':'8',
        'trainer.total_epochs':'10','trainer.total_training_steps':str(meta['target_updates']),
        'trainer.save_freq':str(meta['save_frequency']),'trainer.test_freq':str(meta['evaluation_frequency']),
        'trainer.default_local_dir':str(run/'checkpoints'),'trainer.experiment_name':meta['experiment_name'],
        'trainer.validation_data_dir':str(run/'validation_rollouts'),'trainer.max_actor_ckpt_to_keep':'null',
        'reward_model.score_type':'acc_without_think_format','ray_init.num_cpus':'20'}
    result=[]
    for value in values:
        k=value.split('=',1)[0].lstrip('+')
        if k in replacements:result.append(value.split('=',1)[0]+'='+replacements.pop(k))
        else:result.append(value)
    result += [k+'='+v for k,v in replacements.items()]
    result += ['+data.apply_chat_template_kwargs.enable_thinking=False','+data.base_generation_prompt=False',
        '+actor_rollout_ref.ref.initial_base_model_path='+str(root/'base-reference'),
        '+actor_rollout_ref.actor.allocator_gc_threshold=0.7',
        '+actor_rollout_ref.rollout.stop_token_ids=[151643,151645]',
        '+trainer.rollout_snapshot_dir='+str(run/'rollout_snapshots'),
        '+mixture.condition='+condition,'+mixture.root='+str(root),
        '+provenance.protocol_id='+meta['protocol_id'],
        '+provenance.prompt_template_sha256='+meta['prompt_template_sha256'],
        '+provenance.data_fingerprint='+meta['data_fingerprint'],
        '+provenance.model_revision='+meta['model_revision']]
    tracker=run/'checkpoints/latest_checkpointed_iteration.txt'
    if tracker.exists():
        p=tracker.parent/f'global_step_{int(tracker.read_text())}'
        assert json.loads((p/'validated.json').read_text())['validated']
        result += ['trainer.val_before_train=False']
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--framework-code',type=Path,required=True)
    p.add_argument('--condition',choices=['full-clean','clean5','noise50'],required=True);p.add_argument('--resolve-only',action='store_true')
    args=p.parse_args();root=args.root;frame=args.framework_code;run=root/'runs'/args.condition
    assert not (root/'STOP').exists() and not (run/'STOP').exists()
    verify_inputs(root)
    paths=[str(Path(__file__).resolve().parents[1]),str(frame),str(frame/'SynLogic')]
    sys.path[:0]=paths;os.environ['PYTHONPATH']=os.pathsep.join(paths);os.environ['SYNLOGIC_FRAMEWORK_CODE']=str(frame)
    settings=configuration(root,frame,args.condition)
    (run/'overrides.json').write_text(json.dumps(settings,indent=2)+'\n')
    if not args.resolve_only:
        import torch
        assert torch.cuda.device_count()==1
        free,total=torch.cuda.mem_get_info(0);assert total-free<1024**3
        from synlogic.run import configure_tracking
        configure_tracking(run)
    os.chdir(frame);sys.argv=[str(Path(__file__).with_name('main.py')),*settings]
    if args.resolve_only:sys.argv+=['--cfg','job','--resolve']
    from knights_knaves.main import main as train
    train()
if __name__=='__main__':main()
