"""Teachability selection: exact teacher mass and exact KL on a top-K union."""
import math
import torch


def support_disagreement(student_ids, student_lp, teacher_ids, teacher_lp,
                         teacher_on_student_lp, student_on_teacher_lp):
    """All log probabilities are normalized over the full vocabulary first."""
    assert student_ids.shape == student_lp.shape == teacher_on_student_lp.shape
    assert teacher_ids.shape == teacher_lp.shape == student_on_teacher_lp.shape
    duplicate = (teacher_ids.unsqueeze(-1) == student_ids.unsqueeze(-2)).any(-1)
    keep = torch.cat([torch.ones_like(student_ids,dtype=torch.bool),~duplicate],-1)
    ps = torch.cat([student_lp,student_on_teacher_lp],-1).double()
    qt = torch.cat([teacher_on_student_lp,teacher_lp],-1).double()
    assert torch.isfinite(ps).all() and torch.isfinite(qt).all()
    pz=torch.logsumexp(ps.masked_fill(~keep,-torch.inf),-1,keepdim=True)
    qz=torch.logsumexp(qt.masked_fill(~keep,-torch.inf),-1,keepdim=True)
    qbar=(qt-qz).exp()*keep
    divergence=(qbar*((qt-qz)-(ps-pz))).sum(-1)
    assert divergence.min()>=-1e-6
    coverage=teacher_on_student_lp.double().exp().sum(-1)
    assert coverage.max()<=1.00001
    return divergence.clamp_min(0).float(),coverage.clamp(0,1).float()


def select_teachable(divergence,coverage,valid,retention=.1):
    valid=valid.bool();assert valid.any() and 0<retention<=1
    def norm(x):
        v=x[valid].float();lo,hi=torch.quantile(v,torch.tensor([.05,.95],device=v.device))
        return ((x-lo)/(hi-lo+1e-8)).clamp(0,1)
    score=norm(divergence)*norm(coverage)
    indices=valid.flatten().nonzero().flatten();n=math.ceil(retention*len(indices))
    order=torch.argsort(score.flatten()[indices],descending=True,stable=True)[:n]
    chosen=torch.zeros_like(valid).flatten();chosen[indices[order]]=True
    return chosen.reshape_as(valid),score.detach()


@torch.no_grad()
def score_support(model,data,temperature=1.,k=16,candidate_key=None,entropy=False):
    """Resident SP1 causal model; microbatch16, chunked fp32 vocabulary reductions.

    This deliberately does not change attention masks for selected loss tokens.
    A second student scoring pass supplies student probabilities on teacher top-K.
    """
    assert temperature>0
    model.eval();device=next(model.parameters()).device
    rows=data.batch
    outs={name:[] for name in ['chosen_lp','entropy','ids','lp','candidate_lp']}
    for start in range(0,len(data),16):
        b=rows[start:start+16]
        ids=b['input_ids'].to(device);mask=b['attention_mask'].to(device)
        pos=b['position_ids'].to(device);assert pos.ndim==2
        responses=b['responses'].to(device);length=responses.shape[1]
        left=int(mask.any(0).nonzero()[0])
        ids=ids[:,left:];mask=mask[:,left:];pos=pos[:,left:]
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16):
            output=model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False,logits_to_keep=length+1)
        logits=output.logits[:,-length-1:-1,:]
        assert logits.shape[:2]==responses.shape
        cand=b[candidate_key].to(device) if candidate_key else None
        parts={name:[] for name in outs}
        for offset in range(0,length,128):
            z=logits[:,offset:offset+128].float()/temperature
            logz=torch.logsumexp(z,-1,keepdim=True)
            parts['chosen_lp'].append((z.gather(-1,responses[:,offset:offset+128,None])-logz).squeeze(-1).cpu())
            if k:
                v,i=z.topk(k,-1);parts['ids'].append(i.cpu());parts['lp'].append((v-logz).cpu())
            if cand is not None:
                parts['candidate_lp'].append((z.gather(-1,cand[:,offset:offset+128])-logz).cpu())
            if entropy:
                lp=z-logz;parts['entropy'].append((-(lp.exp()*lp).sum(-1)).cpu())
        for name,values in parts.items():
            if values:outs[name].append(torch.cat(values,1))
        del output,logits
    if getattr(model,'_handle',None) is not None and model._handle.uses_sharded_strategy:
        model._handle.reshard(True)
    return {name:torch.cat(values,0) for name,values in outs.items() if values}
