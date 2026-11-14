"""Sparse, atomic pre-evaluation target journals for current-weight self-distillation."""
from pathlib import Path
import gzip, hashlib, json, os


def atomic_record(path, value):
    raw=(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode()
    if path.suffix=='.gz':raw=gzip.compress(raw,compresslevel=3,mtime=0)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        assert path.read_bytes()==raw,'A completed target snapshot cannot be silently replaced'
        return
    temporary=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with temporary.open('wb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)
    fd=os.open(path.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)


def target_folder(root, outer):
    return (Path(root)/'opsd_target_snapshots'/f'outer_{outer:06d}'/
            ('attempt'+os.environ['SYNLOGIC_ATTEMPT']+'-job'+os.environ['SYNLOGIC_ALLOCATION_ID']))


def record_rows(root, outer, rollout_ids, manifest_sha256):
    atomic_record(target_folder(root,outer)/'row-map.json',{'outer_step':outer,'rollout_ids':list(map(str,rollout_ids)),
                  'retrieval_manifest_sha256':manifest_sha256,'teacher_mode':'same current actor before each optimizer minibatch'})


def record_targets(root, version, rank, micro, result):
    if 'opsd_capture_targets' not in micro or not bool(micro['opsd_capture_targets'][0]):return
    outer=version//5+1;rows=micro['opsd_snapshot_row_index'].tolist()
    signature=hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]
    mask=micro['attention_mask'][:,-micro['responses'].shape[1]:].bool().cpu()
    entries=[]
    for i,index in enumerate(rows):
        valid=mask[i]
        entries.append({'row_index':index,'response_token_ids':micro['responses'][i].detach().cpu()[valid].tolist(),
                        'teacher_top100_ids':result['ids'][i,valid].tolist(),'teacher_top100_log_probs':result['lp'][i,valid].tolist(),
                        'teacher_chosen_log_probs':result['chosen_lp'][i,valid].tolist(),'response_mask':[1]*int(valid.sum()),
                        'teacher_policy_version':version,'student_update_policy_version':version,'stopped_gradient':True})
    value={'schema_version':1,'outer_step':outer,'teacher_policy_version':version,'rank':rank,
           'teacher_mode':'current shared actor; no separate model or teacher updates','entries':entries}
    atomic_record(target_folder(root,outer)/(f'version{version}-rank{rank}-rows{signature}.json.gz'),value)
