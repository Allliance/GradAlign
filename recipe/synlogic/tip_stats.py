"""Full-vocabulary TIP statistics without transferring vocabulary-sized tensors."""
import math
import torch


@torch.no_grad()
def hidden_predictors(model,data,packed=False):
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
            if packed:
                assert pos.ndim==2
                flat=mask.reshape(-1).bool();indices=flat.nonzero().flatten()
                result=model(input_ids=ids.reshape(-1)[flat][None,:],attention_mask=None,
                    position_ids=pos.reshape(-1)[flat][None,:],use_cache=False).logits.squeeze(0)
                padded=result.new_zeros((ids.numel(),result.shape[-1])).index_copy(0,indices,result)
                h=padded.reshape(*ids.shape,-1)[:,-length-1:-1].contiguous()
            else:
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


@torch.no_grad()
def native_teacher_statistics(policy,data,student_head,temperature,max_token_len):
    """Use the actual teacher forward's logits, sharing its native packed scoring path."""
    import itertools
    from verl.utils.seqlen_balancing import rearrange_micro_batches,get_reverse_idx
    model=policy.actor_module;model.eval();device=next(model.parameters()).device
    assert not policy.use_fused_kernels and not policy.use_ulysses_sp
    wrapped=model.module if hasattr(model,'module') else model
    microbatches,indices=rearrange_micro_batches(batch=data.batch,max_token_len=max_token_len)
    result={key:[] for key in ['entropy','rkl','fkl','student_lp','teacher_lp','native_teacher_lp']}
    for micro in microbatches:
        micro=micro.to(device);capture=[]
        handle=wrapped.lm_head.register_forward_hook(lambda module,args,output:capture.append(output))
        try:
            _,native=policy._forward_micro_batch(micro,temperature=temperature,calculate_entropy=False)
        finally:handle.remove()
        assert len(capture)==1
        logits=capture.pop()  # Same storage already divided by temperature by the native scorer.
        b,r=micro['responses'].shape;s=micro['input_ids'].shape[1]
        valid=micro['response_mask'].bool();absolute=(torch.arange(b,device=device)[:,None]*s+torch.arange(s-r-1,s-1,device=device)[None,:])[valid]
        if policy.use_remove_padding:
            nonpadding=micro['attention_mask'].flatten().bool().nonzero().flatten()
            positions=torch.searchsorted(nonpadding,absolute)
            assert torch.equal(nonpadding[positions],absolute)
        else:positions=absolute
        logits=logits.reshape(-1,logits.shape[-1]);hidden=micro['tip_student_hidden'][valid]
        labels=micro['responses'][valid,None];flat={key:[] for key in result if key!='native_teacher_lp'}
        for start in range(0,len(hidden),512):
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16):
                pz=torch.nn.functional.linear(hidden[start:start+512],student_head)
            pl=pz.float().div(temperature).log_softmax(-1)
            ql=logits[positions[start:start+512]].float().log_softmax(-1)
            flat['entropy'].append(-(pl.exp()*pl).sum(-1))
            flat['rkl'].append((pl.exp()*(pl-ql)).sum(-1).clamp_min(0))
            flat['fkl'].append((ql.exp()*(ql-pl)).sum(-1).clamp_min(0))
            flat['student_lp'].append(pl.gather(-1,labels[start:start+512]).squeeze(-1))
            flat['teacher_lp'].append(ql.gather(-1,labels[start:start+512]).squeeze(-1))
        for key in flat:
            values=torch.cat(flat[key]);out=values.new_zeros((b,r));out[valid]=values;result[key].append(out.cpu())
        result['native_teacher_lp'].append(native.cpu())
        del logits,capture
    ordering=torch.tensor(get_reverse_idx(list(itertools.chain.from_iterable(indices))),dtype=torch.long)
    return {key:torch.cat(parts,0)[ordering] for key,parts in result.items()}
