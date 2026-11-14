"""Full-vocabulary TIP statistics without transferring vocabulary-sized tensors."""
import math
import torch


@torch.no_grad()
def hidden_predictors(model,data):
    """Return the exact pre-LM-head response predictors; restore the head immediately."""
    model.eval();device=next(model.parameters()).device
    ids=data.batch['input_ids'].to(device);mask=data.batch['attention_mask'].to(device)
    pos=data.batch['position_ids'].to(device);length=data.batch['responses'].shape[1]
    left=int(mask.any(0).nonzero()[0]);ids=ids[:,left:];mask=mask[:,left:];pos=pos[:,left:]
    wrapped=model.module if hasattr(model,'module') else model
    head=wrapped.lm_head;old=head.forward
    try:
        head.forward=lambda h:h
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16):
            result=model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False,logits_to_keep=length+1).logits
        h=result[:,-length-1:-1].contiguous()
        assert h.shape[:2]==data.batch['responses'].shape
        return h
    finally:
        head.forward=old


@torch.no_grad()
def full_statistics(student_hidden,teacher_hidden,student_head,teacher_head,responses,
                    temperature=1.,chunk=32):
    """Both KL directions and student entropy on identical prefixes."""
    assert student_hidden.shape==teacher_hidden.shape and not student_head.requires_grad
    parts={k:[] for k in ['entropy','rkl','fkl','student_lp','teacher_lp']}
    for i in range(0,student_hidden.shape[1],chunk):
        sh=student_hidden[:,i:i+chunk];th=teacher_hidden[:,i:i+chunk]
        with torch.autocast(device_type=sh.device.type,dtype=torch.bfloat16):
            sz=torch.nn.functional.linear(sh,student_head);tz=teacher_head(th)
        sl=sz.float().div(temperature).log_softmax(-1);tl=tz.float().div(temperature).log_softmax(-1)
        parts['entropy'].append(-(sl.exp()*sl).sum(-1))
        parts['rkl'].append((sl.exp()*(sl-tl)).sum(-1).clamp_min(0))
        parts['fkl'].append((tl.exp()*(tl-sl)).sum(-1).clamp_min(0))
        y=responses[:,i:i+chunk,None].to(sl.device)
        parts['student_lp'].append(sl.gather(-1,y).squeeze(-1))
        parts['teacher_lp'].append(tl.gather(-1,y).squeeze(-1))
    return {k:torch.cat(v,1).cpu() for k,v in parts.items()}


def select_tip(entropy,divergence,valid,retention=.5):
    """TIP Soft-OR, batch entropy98% clipping/min-max; top fraction per response."""
    valid=valid.bool();assert valid.any()
    h=entropy.float().clone();h=h.clamp_max(torch.quantile(h[valid],.98))
    def norm(v):
        lo=v[valid].min();hi=v[valid].max()
        return ((v-lo)/(hi-lo+1e-8)).clamp(0,1)
    hn,dn=norm(h),norm(divergence.float());score=hn+dn-hn*dn
    mask=torch.zeros_like(valid)
    for b in range(len(mask)):
        positions=valid[b].nonzero().flatten();n=math.floor(retention*len(positions))
        if n:
            order=torch.argsort(score[b,positions],descending=True,stable=True)[:n]
            mask[b,positions[order]]=True
    return mask,score.detach()
