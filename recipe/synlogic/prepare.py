"""Prepare frozen pool memberships and label masks without generating new labels."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    source = args.source / 'data'
    output = args.root / 'data/prepared'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists():
        raise ValueError('Prepared inputs already exist; preserve them')
    previous = json.loads((source / 'prepared/manifest.json').read_text())
    clean = pq.read_table(source / 'prepared/full-clean.parquet').to_pylist()
    ids = [row['extra_info']['index'] for row in clean]
    assert ids == previous['eligible_source_indices'] and len(ids) == 15610
    small = set(previous['conditions']['clean1']['source_indices'])
    large = set(ids) - small
    assert len(small) == 156 and len(large) == 15454 and not small & large

    def read_label(index):
        record = json.loads((source / 'revalidated' / f'{index:05d}.json').read_text())
        assert record['source_index'] == index and record['selection_version'] == 4
        assert record['synthetic_label'] and not record['excluded_reason']
        return index, record['synthetic_label']

    with ThreadPoolExecutor(max_workers=8) as executor:
        labels = dict(executor.map(read_label, ids))
    masks = {
        'noise30': set(previous['conditions']['noise30']['corrupted_source_indices']),
        'noise10': set(random.Random(42010).sample(sorted(previous['conditions']['noise30']['corrupted_source_indices']), 1561)),
        'small10': set(random.Random(42110).sample(sorted(small), 16)),
        'large70': set(random.Random(42970).sample(sorted(large), 10818)),
    }

    def row(original, corrupt=False, pool='all'):
        current = deepcopy(original)
        extra = current['extra_info']
        extra['pool'] = pool
        if corrupt:
            label = labels[extra['index']]
            game = json.loads(extra['original_game_data_str'])
            game['answer'] = label['answer']
            extra.update(game_data_str=json.dumps(game, ensure_ascii=False),
                         assigned_answer_key=label['answer_key'], target_mode='fixed_wrong_answer')
            current['reward_model']['ground_truth'] = label['answer']
        return current

    sets = {
        'noise30': [row(r, r['extra_info']['index'] in masks['noise30']) for r in clean],
        'noise10': [row(r, r['extra_info']['index'] in masks['noise10']) for r in clean],
        'small1-noise10': [row(r, r['extra_info']['index'] in masks['small10'], 'small')
                           for r in clean if r['extra_info']['index'] in small],
        'mixture': [row(r, r['extra_info']['index'] in masks['small10'] | masks['large70'],
                        'small' if r['extra_info']['index'] in small else 'large') for r in clean],
    }
    manifest = {'source_count': len(ids), 'seed': 42, 'source_ids': ids,
                'small_source_ids': sorted(small), 'large_source_ids': sorted(large),
                'masks': {k: sorted(v) for k, v in masks.items()}, 'files': {}}
    for name, rows in sets.items():
        path = output / f'{name}.parquet'
        pq.write_table(pa.Table.from_pylist(rows), path)
        noise_count = sum(r['extra_info']['target_mode'] == 'fixed_wrong_answer' for r in rows)
        manifest['files'][name] = {'rows': len(rows), 'noise_count': noise_count,
                                 'noise_fraction': noise_count / len(rows),
                                 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    for name in ['validation']:
        path = output / f'{name}.parquet'
        path.symlink_to(source / 'prepared' / f'{name}.parquet')
        manifest['files'][name] = {'rows': pq.read_metadata(path).num_rows,
                                 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k == 'files'}, indent=2))


if __name__ == '__main__':
    main()
