"""Validate the pinned official Qwen3-4B thinking template."""
import hashlib
PINNED_TEMPLATE_SHA256 = 'a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8'
PROMPT_PROTOCOL = "qwen3-4b-thinking"
def verify_base_template(tokenizer):
    if hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()!=PINNED_TEMPLATE_SHA256:
        raise ValueError("Pinned Qwen3-4B template changed")
    actual=tokenizer.apply_chat_template([{"role":"user","content":"Template verification."}],tokenize=False,add_generation_prompt=True,enable_thinking=True)
    default=tokenizer.apply_chat_template([{"role":"user","content":"Template verification."}],tokenize=False,add_generation_prompt=True)
    disabled=tokenizer.apply_chat_template([{"role":"user","content":"Template verification."}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    if actual!=default or actual==disabled:
        raise ValueError("Expected official thinking generation prefix")
    return tokenizer
