"""Full-normalized sparse statistics from the exact native causal scoring forward."""
import itertools
import torch


@torch.no_grad()
def score_native(policy,data,temperature,max_token_len,k=16,candidate_key=None,entropy=False):
    from verl.utils.seqlen_balancing import rearrange_micro_batches,get_reverse_idx
    assert not policy.use_fused_kernels and not policy.use_ulysses_sp and temperature>0
    model=policy.actor_module;model.eval();device=next(model.parameters()).device
    wrapped=model.module if hasattr(model,'module') else model
    microbatches,indices=rearrange_micro_batches(batch=data.batch,max_token_len=max_token_len)
    names=['chosen_lp','native_lp']+(['ids','lp'] if k else [])+(['candidate_lp'] if candidate_key else [])+(['entropy'] if entropy else [])
    parts={key:[] for key in names}
    for micro in microbatches:
        micro=micro.to(device);capture=[]
        handle=wrapped.lm_head.register_forward_hook(lambda module,args,output:capture.append(output))
        try:_,native=policy._forward_micro_batch(micro,temperature=temperature,calculate_entropy=False)
        finally:handle.remove()
        assert len(capture)==1
        logits=capture.pop().reshape(-1,wrapped.lm_head.out_features)
        b,r=micro['responses'].shape;s=micro['input_ids'].shape[1];valid=micro['response_mask'].bool()
        absolute=(torch.arange(b,device=device)[:,None]*s+torch.arange(s-r-1,s-1,device=device)[None,:])[valid]
        if policy.use_remove_padding:
            packed=micro['attention_mask'].flatten().bool().nonzero().flatten();positions=torch.searchsorted(packed,absolute)
            assert torch.equal(packed[positions],absolute)
        else:positions=absolute
        labels=micro['responses'][valid,None];candidates=micro[candidate_key][valid].long() if candidate_key else None
        values={key:[] for key in names if key!='native_lp'}
        for start in range(0,len(labels),512):
            z=logits[positions[start:start+512]].float()  # Already scaled once by native temperature handling.
            logz=torch.logsumexp(z,-1,keepdim=True)
            values['chosen_lp'].append((z.gather(-1,labels[start:start+512])-logz).squeeze(-1))
            if k:
                top,ids=z.topk(k,-1);values['ids'].append(ids);values['lp'].append(top-logz)
            if candidates is not None:values['candidate_lp'].append(z.gather(-1,candidates[start:start+512])-logz)
            if entropy:
                lp=z-logz;values['entropy'].append(-(lp.exp()*lp).sum(-1))
        for key,arrays in values.items():
            value=torch.cat(arrays);shape=(b,r,*value.shape[1:]);out=value.new_full(shape,-100. if key in ('lp','candidate_lp') else 0)
            out[valid]=value;parts[key].append(out.cpu())
        parts['native_lp'].append(native.cpu())
        del logits,capture
    ordering=torch.tensor(get_reverse_idx(list(itertools.chain.from_iterable(indices))),dtype=torch.long)
    result={key:torch.cat(arrays,0)[ordering] for key,arrays in parts.items()}
    valid=data.batch['response_mask'].bool().cpu();delta=(result['chosen_lp']-result['native_lp']).abs()[valid]
    assert delta.max()<1e-3 and delta.mean()<1e-4,'Native sparse probability alignment failed'
    if getattr(model,'_handle',None) is not None and model._handle.uses_sharded_strategy:model._handle.reshard(True)
    return result
