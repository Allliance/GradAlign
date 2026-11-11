"""Launch a frozen SynLogic protocol on one to four GPUs."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys

CONDITIONS = ('full-clean', 'noise50', 'noise30', 'noise10', 'small1-noise10', 'mixture-sampling99',
              'mixture-gradient-equal', 'mixture-gradalign', 'teacher-distillation', 'noise50-teacher-distillation')


def override_value(value):
    if isinstance(value, dict):
        if any(not key.replace('_', '').isalnum() for key in value):
            raise ValueError('Unsupported configuration dictionary key')
        return '{' + ','.join(key + ':' + override_value(item) for key, item in value.items()) + '}'
    if isinstance(value, list):
        return '[' + ','.join(override_value(item) for item in value) + ']'
    return json.dumps(value)


def configure_tracking(run):
    """Read credentials locally in the logging process, including external Ray actors."""
    for line in Path(os.environ['SYNLOGIC_WANDB_ENV']).read_text().splitlines():
        line = line.strip().removeprefix('export ')
        if line.startswith('WANDB_') and '=' in line:
            key, value = line.split('=', 1)
            parts = shlex.split(value)
            if parts:
                os.environ.setdefault(key, parts[0])
    for key in ('WANDB_DIR', 'WANDB_CACHE_DIR', 'WANDB_CONFIG_DIR', 'WANDB_DATA_DIR'):
        path = run / key.lower()
        path.mkdir(exist_ok=True)
        os.environ[key] = str(path)
    metadata = json.loads((run / 'portable-provenance.json').read_text())
    os.environ.update(WANDB_RUN_ID=metadata['wandb']['id'], WANDB_RESUME='allow', WANDB_INIT_TIMEOUT='90')


def overrides(root, condition, framework, gpus):
    spec = importlib.util.spec_from_file_location('single_card', framework / 'tools/train_single_card.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = module.overrides(root, 'full-clean')
    opd = condition in ('teacher-distillation', 'noise50-teacher-distillation')
    dataset = 'noise50' if opd else condition if not condition.startswith('mixture-') else 'mixture'
    replacements = {
        'data.train_files': str(root / 'data/prepared' / f'{dataset}.parquet'),
        'data.train_batch_size': '160', 'data.gen_batch_size': '160',
        'data.max_prompt_length': '24576', 'data.truncation': 'error',
        'data.filter_overlong_prompts': 'False', 'data.max_response_length': '4096',
        'data.dataloader_num_workers': '0', 'actor_rollout_ref.rollout.n': '8',
        'algorithm.filter_groups.enable': 'False',
        'actor_rollout_ref.actor.use_kl_loss': 'True', 'actor_rollout_ref.actor.kl_loss_coef': '0.001',
        'actor_rollout_ref.actor.clip_ratio_high': '0.2', 'actor_rollout_ref.actor.clip_ratio_c': '3.0',
        'actor_rollout_ref.actor.optim.lr_warmup_steps': '0',
        'actor_rollout_ref.actor.optim.weight_decay': '0.01',
        'actor_rollout_ref.actor.ppo_mini_batch_size': '32',
        'actor_rollout_ref.actor.fsdp_config.fsdp_size': str(gpus),
        'actor_rollout_ref.rollout.max_num_seqs': '256',
        'actor_rollout_ref.rollout.max_num_batched_tokens': '28672',
        'actor_rollout_ref.actor.ppo_max_token_len_per_gpu': '28672',
        'actor_rollout_ref.ref.log_prob_max_token_len_per_gpu': '28672',
        'actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu': '28672',
        'actor_rollout_ref.rollout.val_kwargs.temperature': '0.6',
        'actor_rollout_ref.rollout.val_kwargs.top_p': '1.0',
        'reward_model.score_type': 'acc_without_think_format',
        'trainer.total_training_steps': '970', 'trainer.total_epochs': '10000',
        'trainer.n_gpus_per_node': str(gpus), 'ray_init.num_cpus': '80',
        'trainer.experiment_name': f'Qwen3-4B-Base-SynLogic-{condition}-s42-20261103',
        'trainer.default_local_dir': str(root / 'runs' / condition / 'checkpoints'),
        'hydra.run.dir': str(root / 'runs' / condition / 'hydra'),
    }
    result = []
    for value in values:
        key = value.split('=', 1)[0].lstrip('+')
        if key in replacements:
            result.append(value.split('=', 1)[0] + '=' + replacements.pop(key))
        else:
            result.append(value)
    result += [key + '=' + value for key, value in replacements.items()]
    metadata = json.loads((root / 'runs' / condition / 'portable-provenance.json').read_text())
    if opd:
        if gpus not in (2, 3):
            raise ValueError('OPD requires one or two student GPUs and one teacher-reference GPU')
        values_by_key = {v.split('=', 1)[0].lstrip('+'): i for i, v in enumerate(result)}
        for key, value in {'trainer.n_gpus_per_node': str(gpus - 1),
                           'actor_rollout_ref.actor.fsdp_config.fsdp_size': str(gpus - 1)}.items():
            result[values_by_key[key]] = key + '=' + value
        result += ['+opd.' + key + '=' + override_value(value) for key, value in metadata['opd'].items()]
        result += ['+actor_rollout_ref.ref.teacher_model_path=' + metadata['teacher_model_path']]
        result += ['+data.apply_chat_template_kwargs.enable_thinking=False']
        result += ['+data.base_generation_prompt=True']
    result = [v if not v.startswith('trainer.experiment_name=') else 'trainer.experiment_name=' + metadata.get('experiment_name', v.split('=', 1)[1]) for v in result]
    if hashlib.sha256((root / 'data/prepared' / (dataset + '.parquet')).read_bytes()).hexdigest() != metadata['data_fingerprint']:
        raise ValueError('Training parquet fingerprint differs from frozen provenance')
    if opd:
        from synlogic.input_contract import verify_contract
        binding = metadata.get('input_contract')
        verify_contract(binding, root / 'data/prepared' / (dataset + '.parquet'), root / 'data/prepared/validation.parquet')
        result += ['+provenance.input_contract=' + override_value(binding)]
    result += ['+trainer.rollout_snapshot_dir=' + str(root / 'runs' / condition / 'rollout_snapshots'),
               'actor_rollout_ref.actor.kl_loss_type=low_var_kl',
               'actor_rollout_ref.rollout.do_sample=True',
               '+actor_rollout_ref.rollout.stop_token_ids=[151643,151645]',
               '+mixture.condition=' + condition, '+mixture.root=' + str(root),
               '+provenance.protocol_id=' + metadata['protocol_id'],
               '+provenance.prompt_template_sha256=' + metadata['prompt_template_sha256'],
               '+provenance.model_revision=' + metadata['model_revision'],
               '+provenance.data_fingerprint=' + metadata['data_fingerprint'],
               '+provenance.source_revision=' + json.dumps(metadata['source_revision'])]
    if metadata.get('expected_model_parameters'):
        result += ['+provenance.model_layers=' + str(metadata['expected_model_layers']),
                   '+provenance.model_parameters=' + str(metadata['expected_model_parameters'])]
    if condition == 'mixture-gradalign':
        result += ['+gradalign.selection_interval=10', '+gradalign.selection_ratio=4',
                   '+gradalign.reference_rollouts=16', '+gradalign.candidate_rollouts=16',
                   '+gradalign.candidate_chunk=64']
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--framework-code', type=Path, required=True)
    parser.add_argument('--condition', choices=CONDITIONS, required=True)
    parser.add_argument('--gpus', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--local-gpus', type=int)
    parser.add_argument('--resolve-only', action='store_true')
    args, extra = parser.parse_known_args()
    root, framework = args.root.resolve(), args.framework_code.resolve()
    run = root / 'runs' / args.condition
    if (root / 'STOP').exists() or (run / 'STOP').exists():
        raise SystemExit('User STOP is latched')
    paths = [str(Path(__file__).resolve().parents[1]), str(framework), str(framework / 'SynLogic')]
    sys.path[:0] = paths
    os.environ['PYTHONPATH'] = os.pathsep.join(paths)
    os.environ['SYNLOGIC_FRAMEWORK_CODE'] = str(framework)
    os.environ['SYNLOGIC_RUN_ROOT'] = str(run)
    settings = overrides(root, args.condition, framework, args.gpus) + extra
    tracker = run / 'checkpoints/latest_checkpointed_iteration.txt'
    if tracker.exists():
        folder = tracker.parent / f'global_step_{int(tracker.read_text())}'
        if not (folder / 'validated.json').exists():
            raise ValueError('Recovery checkpoint has not been validated')
        settings += ['trainer.val_before_train=False']
    elif (args.condition in ('teacher-distillation', 'noise50-teacher-distillation')
          and not json.loads((run / 'portable-provenance.json').read_text()).get('fresh_from_base', False)):
        metadata = json.loads((run / 'portable-provenance.json').read_text())
        source = Path(metadata['initial_checkpoint'])
        if not (source / 'validated.json').exists():
            raise ValueError('Initial fork checkpoint has not been validated')
        settings += ['trainer.resume_mode=resume_path', 'trainer.resume_from_path=' + str(source),
                     'trainer.val_before_train=False']
    run.mkdir(parents=True, exist_ok=True)
    (run / 'overrides.json').write_text(json.dumps(settings, indent=2) + '\n')
    sys.argv = [str(Path(__file__).with_name('main.py')), *settings]
    if args.resolve_only:
        sys.argv += ['--cfg', 'job', '--resolve']
    else:
        import torch
        local_gpus = args.local_gpus or args.gpus
        if torch.cuda.device_count() != local_gpus:
            raise ValueError('Visible GPU count differs from the authorized run allocation')
        for device in range(local_gpus):
            free,total=torch.cuda.mem_get_info(device)
            if total-free > 1024**3:
                raise ValueError('Assigned GPU has an unreconciled existing workload')
        configure_tracking(run)
    os.chdir(framework)
    from synlogic.main import main as train
    train()


if __name__ == '__main__':
    main()
