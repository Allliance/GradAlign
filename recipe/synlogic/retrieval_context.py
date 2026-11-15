"""Immutable worked-reference privilege, isolated from question-only student inputs."""
import gzip
import hashlib
import json
from pathlib import Path
import torch


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


class FrozenReferences:
    def __init__(self, manifest, expected_sha256):
        manifest_raw=Path(manifest).read_bytes()
        assert hashlib.sha256(manifest_raw).hexdigest()==expected_sha256,'Shared retrieval manifest changed'
        self.manifest=json.loads(manifest_raw)
        assert self.manifest['passed'] and self.manifest['d2_rows']==13505
        for path,sha in self.manifest['immutable_files'].items():
            assert digest(path)==sha,('Frozen retrieval artifact changed',path)
        path=self.manifest['mapping_path']
        assert path in self.manifest['immutable_files'],'Retrieval map is missing its immutable payload hash'
        raw=Path(path).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==self.manifest['immutable_files'][path]
        self.mapping=json.loads(gzip.decompress(raw))
        assert len(self.mapping)==13505
        assert self.manifest['retriever_id']=='gradient-pairwise-student560-last-mlp-assigned-nll'

    def entry(self, extra):
        identity=str(int(extra['index']));entry=self.mapping[identity]
        game=json.loads(extra['original_game_data_str'])
        assert hashlib.sha256(game['question'].encode()).hexdigest()==entry['query_question_sha256']
        assert entry['reference_correct'] is True
        return entry


def privileged_prefix(source, tokenizer, entry):
    text=tokenizer.decode(source,skip_special_tokens=False)
    assert text.endswith('<|im_end|>\n<|im_start|>assistant\n')
    user_start=text.rfind('<|im_start|>user\n')
    assert user_start>=0
    begin=user_start+len('<|im_start|>user\n')
    for field in ['reference_question','reference_response']:
        assert '<|im_start|>' not in entry[field] and '<|im_end|>' not in entry[field], 'Reference contains chat control tokens'
    privilege=('# Analogous worked example\n## Reference question\n'+entry['reference_question']+
               '\n## Complete worked response\n'+entry['reference_response']+
               '\n# End of worked example\n\nNow solve the target question using its original requested output format.\n## Target question\n')
    teacher_text=text[:begin]+privilege+text[begin:]
    prefix=tokenizer.encode(teacher_text,add_special_tokens=False)
    assert tokenizer.decode(prefix,skip_special_tokens=False)==teacher_text
    return prefix


class ContextBudgets:
    """Separately hashed policy; never rewrite the frozen retrieval artifacts."""
    def __init__(self, path, expected_sha256, references):
        raw=Path(path).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==expected_sha256, 'Generation budget policy changed'
        self.policy=json.loads(raw)
        assert self.policy['native_context']==32768 and self.policy['default_response_tokens']==4096
        assert self.policy['mapping_sha256']==references.manifest['immutable_files'][references.manifest['mapping_path']]
        self.rows=self.policy['rows']
        assert self.rows.keys()==references.mapping.keys() and len(self.rows)==13505
        for key,row in self.rows.items():
            assert row['query_question_sha256']==references.mapping[key]['query_question_sha256']
            assert 0 < row['max_new_tokens'] <= min(4096,32768-row['privileged_prompt_tokens'])
        assert {k:v['max_new_tokens'] for k,v in self.rows.items() if v['max_new_tokens']!=4096}=={'120':3990}

    def budget(self, extra, source, tokenizer, references):
        row=self.rows[str(int(extra['index']))]
        prefix=privileged_prefix(source,tokenizer,references.entry(extra))
        assert len(prefix)==row['privileged_prompt_tokens']
        assert hashlib.sha256(json.dumps(prefix,separators=(',',':')).encode()).hexdigest()==row['privileged_prompt_ids_sha256']
        return row['max_new_tokens']


def privileged_batch(data, tokenizer, references, max_context, budgets=None):
    """Keep exact student response IDs/masks; change only teacher user prefix."""
    from verl import DataProto
    responses=data.batch['responses'];rlen=responses.shape[1]
    width=data.batch['prompts'].shape[1]
    promptmask=data.batch['attention_mask'][:,:width].bool()
    responsemask=data.batch['attention_mask'][:,width:]
    prefixes=[]
    for i,extra in enumerate(data.non_tensor_batch['extra_info']):
        entry=references.entry(extra)
        source=data.batch['prompts'][i,promptmask[i]].tolist()
        prefix=privileged_prefix(source,tokenizer,entry)
        if budgets is not None:
            budget=budgets.budget(extra,source,tokenizer,references)
            assert int(responsemask[i].sum())<=budget, 'Generated response exceeds assigned pre-generation budget'
        assert len(prefix)+int(responsemask[i].sum())<=max_context, 'Privileged context overflow; never truncate or drop a D2 query'
        prefixes.append(prefix)
    plen=max(map(len,prefixes));device=responses.device
    prompts=torch.full((len(prefixes),plen),tokenizer.pad_token_id,dtype=torch.long,device=device)
    pm=torch.zeros_like(prompts)
    for i,prefix in enumerate(prefixes):
        prompts[i,-len(prefix):]=torch.tensor(prefix,device=device);pm[i,-len(prefix):]=1
    ids=torch.cat([prompts,responses],1);attention=torch.cat([pm,responsemask],1)
    positions=(attention.cumsum(1)-1).clamp_min(0)*attention
    assert (attention.sum(-1)<=max_context).all()
    assert torch.equal(responsemask,data.batch['response_mask']), 'Response loss and attention masks differ'
    assert torch.equal(ids[:,-rlen:],responses),'Student response identity changed'
    return DataProto.from_dict(tensors={'prompts':prompts,'responses':responses,'input_ids':ids,
                               'attention_mask':attention,'position_ids':positions,'response_mask':data.batch['response_mask']})
