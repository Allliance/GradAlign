"""Verify the chat template recorded in a run's frozen provenance.

Each pinned template keeps its own historical rendering check; the expected hash
always comes from the run, never from a universal pin.
"""
import hashlib

THINKING_TEMPLATE_SHA256 = 'a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8'
BASE_TEMPLATE_SHA256 = '87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5'
PROMPT_PROTOCOLS = {THINKING_TEMPLATE_SHA256: 'qwen3-4b-thinking',
                    BASE_TEMPLATE_SHA256: 'qwen3-base-default-v1'}
MESSAGES = [{"role": "user", "content": "Template verification."}]


def _verify_thinking(tokenizer):
    actual = tokenizer.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    default = tokenizer.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)
    disabled = tokenizer.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if actual != default or actual == disabled:
        raise ValueError("Expected official thinking generation prefix")


def _verify_base(tokenizer):
    # This call deliberately matches RLHFDataset: the base template has no thinking prefix.
    actual = tokenizer.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=False,
                                           enable_thinking=False) + '<|im_start|>assistant\n'
    explicit = tokenizer.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)
    if actual != explicit or not actual.endswith("<|im_start|>assistant\n"):
        raise ValueError("The dataloader must use the base model's unmodified default template")


CHECKS = {THINKING_TEMPLATE_SHA256: _verify_thinking, BASE_TEMPLATE_SHA256: _verify_base}


def verify_base_template(tokenizer, expected_sha256):
    """Check the tokenizer against the run's frozen ``prompt_template_sha256``."""
    actual = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    if actual != expected_sha256:
        raise ValueError("Tokenizer chat template differs from the run's frozen provenance")
    if actual not in CHECKS:
        raise ValueError("No recorded rendering check for this template; add one explicitly")
    CHECKS[actual](tokenizer)
    return tokenizer
