"""Verify the released base-model prompt without injecting a thinking prefix."""

import hashlib

PINNED_TEMPLATE_SHA256 = "87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5"
PROMPT_PROTOCOL = "qwen3-base-default-v1"


def verify_base_template(tokenizer):
    if hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != PINNED_TEMPLATE_SHA256:
        raise ValueError("The pinned base-model template was modified")
    messages = [{"role": "user", "content": "Template verification."}]
    # This call deliberately matches RLHFDataset: no thinking keyword is passed.
    actual = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    explicit = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=True)
    if actual != explicit or not actual.endswith("<|im_start|>assistant\n"):
        raise ValueError("The dataloader must use the base model's unmodified default template")
    return tokenizer
