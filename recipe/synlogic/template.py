"""Verify the Base prompt while explicitly disabling thinking."""
import hashlib
PINNED_TEMPLATE_SHA256 = "17762d2e5131cb2a950db7dd5700c50095547773d3302b52baebe9ae32257a55"
PROMPT_PROTOCOL = "qwen3-base-chatml-thinking-disabled-v1"
def verify_base_template(tokenizer):
    if hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != PINNED_TEMPLATE_SHA256:
        raise ValueError("The pinned Base prompt adapter was modified")
    messages = [{"role": "user", "content": "Template verification."}]
    actual = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    expected = "<|im_start|>user\nTemplate verification.<|im_end|>\n<|im_start|>assistant\n"
    if actual != expected:
        raise ValueError("The Base prompt must preserve bare-assistant bytes with thinking disabled")
    return tokenizer
