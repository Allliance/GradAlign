"""One parser and binary complete-assignment rule for clean and noisy references."""
import json,re

def parse_answer(response,names):
    matches=re.findall(r'<answer>(.*?)</answer>',response,re.S)
    if len(matches)!=1:return None
    answer=matches[0]
    if len(re.findall(r'\b(?:knight|knave)\b',answer,re.I))!=len(names):return None
    roles=[]
    for name in names:
        values=re.findall(r'\b'+re.escape(name)+r'\s+is\s+a\s+(knight|knave)\b',answer,re.I)
        if len(values)!=1:return None
        roles.append(values[0].lower()=='knight')
    return roles

def score_answer(response,target):
    ref=json.loads(target) if isinstance(target,str) else target
    prediction=parse_answer(response,ref['names'])
    return float(prediction is not None and prediction==ref['roles'])

class KnightsKnavesRewardManager:
    def __init__(self,tokenizer,validation=False):self.tokenizer,self.validation=tokenizer,validation
    def __call__(self,data,return_dict=False):
        import torch
        from collections import defaultdict
        reward=torch.zeros_like(data.batch['responses'],dtype=torch.float32)
        extra=defaultdict(list)
        for i in range(len(data)):
            item=data[i];width=item.batch['prompts'].shape[-1]
            length=int(item.batch['attention_mask'][width:].sum())
            response=self.tokenizer.decode(item.batch['responses'][:length],skip_special_tokens=True)
            target=item.non_tensor_batch['reward_model']['ground_truth']
            value=score_answer(response,target)
            if length:reward[i,length-1]=value
            if not self.validation:
                info=item.non_tensor_batch['extra_info'];noisy=bool(info['diagnostic_is_noisy'])
                # Shadow-only clean correctness does not feed the reward above.
                clean=score_answer(response,info['clean_reference'])
                for key,v in {'diagnostic_is_noisy':noisy,'diagnostic_assigned_match':bool(value),
                    'diagnostic_assigned_error':False,'diagnostic_true_correct':bool(clean),
                    'diagnostic_true_verified':True,'diagnostic_true_error':False,
                    'diagnostic_effective_target':target,'diagnostic_effective_target_mode':'complete_role_assignment_v1'}.items():extra[key].append(v)
        result={'reward_tensor':reward,'reward_extra_info':extra}
        return result if return_dict else reward
