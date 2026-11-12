"""Preserve global optimizer minibatches and token means across DP ranks."""
import torch


def rank_major_indices(size, minibatch, dp_size):
    if size % minibatch or minibatch % dp_size:
        raise ValueError('Batch and optimizer minibatches must divide evenly across DP ranks')
    local = minibatch // dp_size
    return [start + rank * local + i for rank in range(dp_size)
            for start in range(0, size, minibatch) for i in range(local)]


def prepare_actor_batch(batch, minibatch, dp_size):
    from copy import deepcopy
    batch = deepcopy(batch)
    counts = batch.batch['response_mask'].sum(-1)
    totals = counts.reshape(-1, minibatch).sum(-1).repeat_interleave(minibatch)
    if (totals <= 0).any():
        raise ValueError('An optimizer minibatch contains no response tokens')
    batch.batch['global_minibatch_response_tokens'] = totals
    return batch[rank_major_indices(len(batch), minibatch, dp_size)]


def prepare_legacy_actor_batch(batch, minibatch, dp_size, original_micro_token_budget):
    """Keep the legacy SP dynamic-microbatch weighting exactly, including KL."""
    from copy import deepcopy
    from verl.utils.seqlen_balancing import rearrange_micro_batches
    batch = deepcopy(batch)
    weights = torch.empty(len(batch), dtype=torch.float32)
    for start in range(0, len(batch), minibatch):
        mini = batch.batch[start:start + minibatch]
        micros, indices = rearrange_micro_batches(mini, original_micro_token_budget)
        for micro, rows in zip(micros, indices):
            tokens = micro['response_mask'].sum()
            if tokens <= 0:
                raise ValueError('Legacy microbatch contains no response tokens')
            weights[torch.tensor(rows) + start] = len(rows) / tokens
    if 'sample_weights' in batch.batch:
        weights *= batch.batch['sample_weights']
    batch.batch['sample_weights'] = weights
    batch.batch['legacy_global_minibatch_rows'] = torch.full((len(batch),), minibatch)
    return batch[rank_major_indices(len(batch), minibatch, dp_size)]
