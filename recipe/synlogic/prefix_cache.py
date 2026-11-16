"""Once-verified teacher prefix IDs; training hot paths only index CPU arrays."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .retrieval_context import digest


class FrozenPrefixes:
    def __init__(self, path, expected_sha256, manifest_sha256, budget_sha256):
        raw = Path(path).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected_sha256, 'Prefix manifest changed'
        self.manifest = json.loads(raw)
        assert self.manifest['retrieval_manifest_sha256'] == manifest_sha256
        assert self.manifest['context_budget_sha256'] == budget_sha256
        assert self.manifest['rows'] == 13505 and self.manifest['native_context'] == 32768
        payload = Path(self.manifest['payload'])
        assert digest(payload) == self.manifest['payload_sha256'], 'Prefix token payload changed'
        arrays = np.load(payload, allow_pickle=False)
        self.tokens = arrays['teacher_tokens']
        self.offsets = arrays['teacher_offsets']
        self.students = arrays['student_tokens']
        self.student_offsets = arrays['student_offsets']
        self.row_by_id = {str(int(q)): i for i, q in enumerate(arrays['question_ids'])}
        self.caps = arrays['response_caps']
        arrays.close()
        assert len(self.row_by_id) == 13505 and len(self.offsets) == 13506
        lengths = np.diff(self.offsets)
        assert np.all(lengths + self.caps <= 32768) and np.all(self.caps > 0)
        assert {q: int(self.caps[i]) for q, i in self.row_by_id.items() if self.caps[i] != 4096} == {'120': 3990}

    def row(self, extra):
        return self.row_by_id[str(int(extra['index']))]

    def budget(self, extra):
        return int(self.caps[self.row(extra)])

    def prefix(self, extra):
        i = self.row(extra)
        return self.tokens[self.offsets[i]:self.offsets[i + 1]]


def privileged_batch_cached(data, cache, pad_token_id, max_context):
    """Construct privilege only in the CPU reference worker; retain exact responses."""
    from verl import DataProto

    assert data.batch['responses'].device.type == 'cpu', 'Privilege assembly belongs on reference CPU'
    assert max_context == 32768
    responses = data.batch['responses']
    response_mask = data.batch['response_mask']
    width = data.batch['prompts'].shape[1]
    assert torch.equal(response_mask, data.batch['attention_mask'][:, width:])
    extras = data.non_tensor_batch['extra_info']
    prefixes = [cache.prefix(extra) for extra in extras]
    lengths = [len(prefix) for prefix in prefixes]
    actual = response_mask.sum(-1).tolist()
    assert all(n <= cache.budget(extra) and p + n <= max_context
               for p, n, extra in zip(lengths, actual, extras)), 'Privileged context exceeds frozen budget'
    prompts = torch.full((len(data), max(lengths)), pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(prompts)
    for i, prefix in enumerate(prefixes):
        prompts[i, -len(prefix):] = torch.from_numpy(prefix.astype(np.int64))
        mask[i, -len(prefix):] = 1
    ids = torch.cat([prompts, responses], 1)
    attention = torch.cat([mask, response_mask], 1)
    positions = (attention.cumsum(1) - 1).clamp_min(0) * attention
    return DataProto.from_dict(tensors={'prompts': prompts, 'responses': responses,
        'input_ids': ids, 'attention_mask': attention, 'position_ids': positions,
        'response_mask': response_mask})
