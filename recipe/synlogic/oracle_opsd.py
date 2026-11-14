"""Answer-only privileged contexts for frozen-teacher OPSD on audited training rows."""
import ast,hashlib,json,itertools,re
from functools import lru_cache
from fractions import Fraction
from pathlib import Path
import torch


@lru_cache(maxsize=4)
def latin_grids(n):
    assert n==4,'Only audited4x4 Futoshiki is supported'
    permutations=list(itertools.permutations(range(1,n+1)));result=[]
    def extend(rows):
        if len(rows)==n:result.append(rows);return
        for row in permutations:
            if all(all(row[j]!=other[j] for other in rows) for j in range(n)):extend(rows+[row])
    extend([]);assert len(result)==576
    return result


def futoshiki_reference(game):
    """Exhaust the entire Latin-square space against the actual stated puzzle."""
    meta=game['metadata'];n=meta['grid_size'];fixed=[]
    for line in game['question'].splitlines():
        fields=line.strip().split()
        if len(fields)==n and all(t=='X' or t.isdigit() and 1<=int(t)<=n for t in fields):fixed.append(fields)
    assert len(fixed)==n
    constraints=meta['constraints']
    matches=re.findall(r'\((\d+),\s*(\d+)\)\s*([<>])\s*\((\d+),\s*(\d+)\)',game['question'])
    assert {(int(a)-1,int(b)-1,int(c)-1,int(d)-1,op) for a,b,op,c,d in matches}=={(*a,*b,op) for a,b,op in constraints}
    def valid(grid):
        return (len(grid)==n and all(len(row)==n for row in grid)
            and all(set(row)==set(range(1,n+1)) for row in grid)
            and all(len({grid[i][j] for i in range(n)})==n for j in range(n))
            and all(fixed[i][j]=='X' or int(fixed[i][j])==grid[i][j] for i in range(n) for j in range(n))
            and all((grid[a[0]][a[1]]<grid[b[0]][b[1]]) if op=='<' else (grid[a[0]][a[1]]>grid[b[0]][b[1]]) for a,b,op in constraints))
    original=game['answer']
    if isinstance(original,str):
        try:original=ast.literal_eval(original)
        except (SyntaxError,ValueError):original=[[int(x) for x in row.split()] for row in original[2:-2].split(',')]
    if valid(original):return json.dumps(original),'original_answer_independently_verified'
    for grid in latin_grids(n):
        if valid(grid):return json.dumps(grid),'corrected_reference_exhaustive_solution'
    return 'No solution exists: no 4x4 grid satisfies the stated givens and inequality constraints.','corrected_reference_exhaustively_unsatisfiable'


def clean_answer(game,task):
    """Use only the original clean reference, including structured solution metadata."""
    answer=game.get('answer');meta=game.get('metadata') or {}
    if task=='futoshiki':return futoshiki_reference(game)
    if task=='dyck_language_errors' and answer=='None':
        stack=[];pairs={')':'(',']':'[','}':'{','>':'<'};error=None
        text=meta['bracket_string']
        for i,char in enumerate(text,1):
            if char in pairs.values():stack.append(char)
            else:
                assert char in pairs
                if not stack or stack.pop()!=pairs[char]:error=i;break
        if error is None:error=len(text)+1 if stack else -1
        return str(error),'corrected_reference_independent_stack'

    if task=='word_sorting_mistake' and answer in ('',None):
        assert meta.get('is_mistake') is False and meta.get('answer') is None
        return 'No','original_metadata_no_mistake'
    if answer not in ('',None):
        return (answer if isinstance(answer,str) else json.dumps(answer,ensure_ascii=False)),'original_answer'
    if task=='mathador':
        # Stored solutions encode a left-associated sequence of arithmetic operations.
        for solution in meta['solutions']:
            ns,os=solution.removeprefix('nums: ').split(', ops: ')
            nums,ops=ast.literal_eval(ns),ast.literal_eval(os)
            assert len(ops)==len(nums)-1 and sorted(nums)==sorted(meta['numbers'])
            value=Fraction(nums[0]);expression=str(nums[0])
            try:
                for op,n in zip(ops,nums[1:]):
                    assert op in meta['operators']
                    value={'+':lambda:value+n,'-':lambda:value-n,'*':lambda:value*n,'/':lambda:value/n}[op]()
                    expression=f'({expression} {op} {n})'
            except ZeroDivisionError:continue
            if value==Fraction(str(meta['result'])):return expression,'original_metadata_expression'
        raise ValueError('No valid original arithmetic solution')
    key={'campsite':'solution','number_wall':'solution','numbrix':'solution','skyscraper_puzzle':'solved_grid'}.get(task)
    assert key and meta.get(key) is not None,(task,'Missing original answer')
    return json.dumps(meta[key],ensure_ascii=False),'original_metadata_'+key


def load_oracle_map(path,expected_sha):
    raw=Path(path).read_bytes();assert hashlib.sha256(raw).hexdigest()==expected_sha,'Oracle answer map changed'
    payload=json.loads(raw);assert payload['training_rows']==13505 and len(payload['answers'])==13505
    return payload['answers']


def privileged_batch(data,tokenizer,answers):
    """Amend only the teacher's user context and append unchanged response token IDs."""
    from verl import DataProto
    width=data.batch['prompts'].shape[1];responses=data.batch['responses'];rlen=responses.shape[1]
    promptmask=data.batch['attention_mask'][:,:width].bool();responsemask=data.batch['attention_mask'][:,width:]
    prefixes=[]
    end=tokenizer.convert_tokens_to_ids('<|im_end|>')
    for i,extra in enumerate(data.non_tensor_batch['extra_info']):
        entry=answers[str(int(extra['index']))]
        assert hashlib.sha256(extra['original_game_data_str'].encode()).hexdigest()==entry['original_game_sha256']
        source=data.batch['prompts'][i,promptmask[i]].tolist()
        # Insert inside the final user message and canonically tokenize only the teacher prefix.
        boundary=max(j for j,t in enumerate(source) if t==end)
        suffix=tokenizer.decode(source[boundary:],skip_special_tokens=False)
        assert suffix=='<|im_end|>\n<|im_start|>assistant\n',repr(suffix)
        privilege='\n\nPrivileged reference for this training problem (correct final answer only):\n'+entry['answer']+'\nUse this answer to guide reasoning while following the original requested output format.\n'
        added=tokenizer.encode(privilege,add_special_tokens=False)
        assert end not in added and tokenizer.convert_tokens_to_ids('<|im_start|>') not in added
        text=tokenizer.decode(source[:boundary],skip_special_tokens=False)+privilege+suffix
        prefix=tokenizer.encode(text,add_special_tokens=False)
        assert tokenizer.decode(prefix,skip_special_tokens=False)==text
        prefixes.append(prefix)
    plen=max(map(len,prefixes));device=responses.device
    p=torch.full((len(prefixes),plen),tokenizer.pad_token_id,dtype=torch.long,device=device);pm=torch.zeros_like(p)
    for i,prefix in enumerate(prefixes):p[i,-len(prefix):]=torch.tensor(prefix,device=device);pm[i,-len(prefix):]=1
    ids=torch.cat([p,responses],1);attention=torch.cat([pm,responsemask],1);pos=(attention.cumsum(1)-1).clamp_min(0)*attention
    assert torch.equal(ids[:,-rlen:],responses),'Teacher response token identity changed'
    return DataProto.from_dict(tensors={'prompts':p,'responses':responses,'input_ids':ids,'attention_mask':attention,'position_ids':pos,'response_mask':data.batch['response_mask']})
