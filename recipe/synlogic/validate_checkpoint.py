"""Validate all distributed model/Adam/RNG shards using CPU-only Gloo ranks."""

import argparse
import ctypes
import json
import math
import mmap
import os
from pathlib import Path
import subprocess
import sys

import torch
import torch.distributed as dist


def require(value, message):
    if not value:
        raise ValueError(message)


def tensors(value):
    if hasattr(value, 'local_shards'):
        for shard in value.local_shards():
            yield shard.tensor
    elif hasattr(value, 'to_local'):
        yield value.to_local()
    elif isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from tensors(child)


def finite(value):
    for tensor in tensors(value):
        if tensor.is_floating_point():
            flat = tensor.reshape(-1)
            for start in range(0, flat.numel(), 2**20):
                require(torch.isfinite(flat[start:start+2**20]).all().item(), 'Non-finite checkpoint tensor')


def model_parameter_counts(model, tied_embeddings=False):
    serialized = sum(t.numel() for t in tensors(model))
    unique = serialized
    if tied_embeddings:
        embedding_keys = [k for k in model if k.endswith('embed_tokens.weight')]
        head_keys = [k for k in model if k.endswith('lm_head.weight')]
        require(len(embedding_keys) == len(head_keys) == 1, 'Missing tied embedding or output-head entry')
        embeddings, heads = list(tensors(model[embedding_keys[0]])), list(tensors(model[head_keys[0]]))
        require(len(embeddings) == len(heads), 'Tied-weight shard counts differ')
        for embedding, head in zip(embeddings, heads):
            require(embedding.shape == head.shape, 'Tied-weight shard shapes differ')
            left, right = embedding.reshape(-1), head.reshape(-1)
            for start in range(0, left.numel(), 2**20):
                require(torch.equal(left[start:start+2**20], right[start:start+2**20]), 'Tied embedding/output weights differ')
        unique -= sum(t.numel() for t in heads)
    return unique, serialized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()
    folder = args.checkpoint
    state = json.loads((folder/'experiment_state.json').read_text())
    world, step = state['world_size'], state['step']
    require(json.loads((folder/'complete.json').read_text())['step'] == step, 'Incomplete checkpoint')
    require(folder.name == f'global_step_{step}', 'Checkpoint step mismatch')
    if 'RANK' not in os.environ:
        subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone',
                        f'--nproc_per_node={world}',__file__,str(folder)],check=True,timeout=1800)
        return
    dist.init_process_group('gloo')
    rank = dist.get_rank()
    require(dist.get_world_size() == world, 'Checkpoint world-size mismatch')
    original = torch.UntypedStorage.from_file
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.mprotect.restype = ctypes.c_int

    def readonly(*args, **kwargs):
        storage = original(*args, **kwargs)
        if storage.nbytes() and libc.mprotect(storage.data_ptr(), storage.nbytes(), mmap.PROT_READ):
            raise OSError(ctypes.get_errno(), 'Cannot protect checkpoint mapping')
        return storage

    torch.UntypedStorage.from_file = readonly
    suffix = f'world_size_{world}_rank_{rank}.pt'
    with torch.serialization.set_default_mmap_options(mmap.MAP_SHARED):
        model = torch.load(folder/'actor'/('model_'+suffix),map_location='cpu',weights_only=False,mmap=True)
        require(all(any(f'layers.{i}.' in key for key in model) for i in range(state.get('expected_model_layers', 36))), 'Missing transformer layers')
        config_path = folder / 'actor/huggingface/config.json'
        if not config_path.exists():
            config_path = folder.parents[3] / 'model/config.json'
        model_config = json.loads(config_path.read_text())
        count, serialized_count = model_parameter_counts(model, model_config.get('tie_word_embeddings', False))
        finite(model)
        del model
        optimizer = torch.load(folder/'actor'/('optim_'+suffix),map_location='cpu',weights_only=False,mmap=True)
        require(optimizer['state'] and optimizer['param_groups'], 'Missing Adam state')
        counters = [float(item['step']) for item in optimizer['state'].values() if 'step' in item]
        expected = state.get('optimizer_updates', step)
        require(counters and all(counter == expected if state['condition'] == 'noisy-to-clean' else counter >= expected for counter in counters), 'Wrong optimizer counters')
        require(all(math.isfinite(group['lr']) and group['lr'] > 0 for group in optimizer['param_groups']), 'Invalid learning rate')
        finite(optimizer)
        del optimizer
    extra = torch.load(folder/'actor'/('extra_state_'+suffix),map_location='cpu',weights_only=False)
    require(extra.get('lr_scheduler') is not None and extra.get('rng'), 'Missing scheduler or RNG')
    counts = torch.tensor([count],dtype=torch.int64)
    dist.all_reduce(counts)
    parameters = int(counts.item())
    expected_parameters = state.get('expected_model_parameters')
    require(parameters == expected_parameters if expected_parameters else 3_900_000_000 <= parameters <= 4_600_000_000,
            f'Incomplete model parameter count: {parameters}; expected {expected_parameters}')
    if rank == 0:
        loader = torch.load(folder/'data.pt',map_location='cpu',weights_only=False)
        if state["condition"] == "noisy-to-clean":
            from curriculum import CurriculumDataLoader
            from types import SimpleNamespace
            import pyarrow.parquet as pq
            run = folder.parent.parent
            metadata = json.loads((run/"portable-provenance.json").read_text())
            rows = pq.read_table(run.parent.parent/"data/prepared/noisy-to-clean.parquet").to_pylist()
            class Frame:
                def to_list(self): return rows
            original = SimpleNamespace(dataset=SimpleNamespace(dataframe=Frame()),
                                       batch_size=160, num_workers=0, drop_last=True, collate_fn=None)
            restored = CurriculumDataLoader(original, **metadata["curriculum"])
            restored.load_state_dict(loader)
            require(restored.position == state["step"] - metadata["curriculum"]["parent_step"], "Curriculum cursor mismatch")
        else:
            require('synlogic_plan_generator' in loader and 'synlogic_plan_shape' in loader, 'Missing sampler state')
        result = {**state,'validated':True,'model_parameters':parameters,
                  'minimum_optimizer_counter':min(counters),
                  'optimizer_steps':state['optimizer_updates'],
                  'all_model_optimizer_tensors_finite':True,
                  'tied_embeddings_verified':model_config.get('tie_word_embeddings', False),
                  'components':['all_model_shards','all_optimizer_shards','all_scheduler_and_rng_shards',
                                'dataloader','sampler_generator_and_index_plan','experiment_state']}
        temporary = folder/'validated.tmp'
        temporary.write_text(json.dumps(result,indent=2)+'\n')
        temporary.replace(folder/'validated.json')
        print(json.dumps(result),flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
