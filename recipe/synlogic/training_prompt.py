"""Versioned answer-direct wrapper for explicitly authorized NEW training only."""
from copy import deepcopy
import hashlib
from verl.utils.dataset.rl_dataset import RLHFDataset

ORIGINAL_PREFIX = "Solve the following problem step by step. First, think about the reasoning process in the mind and then provide the answer. The reasoning process is enclosed within <think> </think> and the final answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here</answer>.\n\n"
ANSWER_DIRECT_PREFIX = "Solve the problem and provide only the final answer enclosed in <answer> and </answer> tags. Do not include explanations or intermediate reasoning.\n\n"
TEMPLATE_SHA256 = hashlib.sha256(ANSWER_DIRECT_PREFIX.encode()).hexdigest()

def training_messages(messages):
    result = deepcopy(list(messages))
    assert len(result) == 1 and result[0]['role'] == 'user'
    content = result[0]['content']
    assert content.startswith(ORIGINAL_PREFIX), 'Unrecognized training instruction; fail closed'
    result[0]['content'] = ANSWER_DIRECT_PREFIX + content[len(ORIGINAL_PREFIX):]
    return result

class AnswerDirectTrainingDataset(RLHFDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.apply_chat_template_kwargs.get('enable_thinking') is False
        assert not self.filter_overlong_prompts, 'Do not filter or change training membership'
    def _build_messages(self, example):
        return training_messages(super()._build_messages(example))
