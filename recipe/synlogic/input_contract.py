"""Bind resolved inputs to one immutable, complete held-out input audit."""
import gzip
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_contract(binding, training_files, validation_files):
    if not binding:
        raise ValueError('A frozen shared input contract is required')
    path = Path(binding['path'])
    if sha256(path) != binding['sha256']:
        raise ValueError('Shared input audit manifest changed')
    contract = json.loads(path.read_text())
    if (contract['normalized_text_and_underlying_constraint_overlaps'] != 0
            or contract['upstream_evaluation_rows'] != 0
            or not contract['input_only_checks'] or not contract['labels_and_noise_masks_unchanged']):
        raise ValueError('Input audit is not valid for training')
    for files, key in ((training_files, 'training'), (validation_files, 'validation')):
        files = [files] if isinstance(files, (str, Path)) else list(files)
        if len(files) != 1 or sha256(files[0]) != contract[key + '_sha256']:
            raise ValueError('Resolved ' + key + ' input differs from the audited frozen input')
    for key in ('input_signatures', 'heldout_signatures'):
        if sha256(contract[key + '_path']) != contract[key + '_sha256']:
            raise ValueError('Frozen input audit signatures changed')
    with gzip.open(contract['input_signatures_path'], 'rt') as stream:
        records = json.load(stream)
    ids = [r['source_id'] for r in records]
    if ids != contract['source_ids'] or len(ids) != contract['training_rows'] or len(set(ids)) != len(ids):
        raise ValueError('Input audit row identities differ')
    return contract


def migrate_plan_state(state, old_source_ids, new_source_ids, batches_per_pass):
    """Preserve RNG/counters and retained future draw order; never replay excluded IDs."""
    import copy
    migrated = copy.deepcopy(state)
    saved = migrated['_sampler_iter_state']['sampler_iter_state']
    shape = saved['shape']
    if shape['length'] != len(old_source_ids) or state['synlogic_plan_shape'] != shape:
        raise ValueError('Source sampler shape mismatch')
    mapping = {sid: index for index, sid in enumerate(new_source_ids)}
    if len(mapping) != len(new_source_ids) or not set(new_source_ids) <= set(old_source_ids):
        raise ValueError('Migration must be an input-only subset')
    cursor = saved['cursor']
    size = batches_per_pass * shape['batch_size']
    if cursor != migrated['_sampler_iter_state']['samples_yielded'] or cursor > size:
        raise ValueError('Incompatible saved sampler cursor')
    # Consumed slots are inert history. Excluded historical slots receive a valid
    # placeholder solely to satisfy native state validation; they are never read.
    prefix = [mapping.get(old_source_ids[i], 0) for i in saved['plan'][:cursor]]
    future = [mapping[old_source_ids[i]] for i in saved['plan'][cursor:]
              if old_source_ids[i] in mapping]
    if len(future) < size - cursor:
        raise ValueError('Not enough retained future draws for a counter-preserving migration')
    saved['plan'] = prefix + future[:size-cursor]
    saved['shape'] = {**shape, 'length': len(new_source_ids), 'num_samples': size}
    migrated['synlogic_plan_shape'] = dict(saved['shape'])
    return migrated
