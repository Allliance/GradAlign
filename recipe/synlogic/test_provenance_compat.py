"""Historical provenance stays launchable: fresh-start aliases, templates, validation IDs."""
import hashlib
from unittest.mock import Mock, patch

import pytest

from synlogic import template
from synlogic.run import starts_fresh
from synlogic.trainer import expected_validation_ids


@pytest.mark.parametrize('metadata,fresh', [
    ({}, False), ({'fresh_from_base': True}, True), ({'fresh_from_pretrained': True}, True),
    ({'fresh_from_base': False}, False),
    ({'fresh_from_base': False, 'fresh_from_pretrained': True}, True), ({'fresh_from_base': True, 'fresh_from_pretrained': True}, True),
    ({'fresh_from_base': True, 'initial_checkpoint': '/unused'}, True)])
def test_fresh_start_aliases(metadata, fresh):
    assert starts_fresh(metadata) is fresh


@pytest.mark.parametrize('metadata', [{'fresh_from_base': True, 'fresh_from_pretrained': False}])
def test_contradictory_fresh_flags_fail(metadata):
    with pytest.raises(ValueError, match='Contradictory'):
        starts_fresh(metadata)


def test_validation_ids_from_contract_or_historical_range():
    assert expected_validation_ids({}) == set(range(270))
    assert expected_validation_ids({'validation_source_ids': [7, 3]}) == {3, 7}


class Tokenizer:
    def __init__(self, text, thinking_default):
        self.chat_template, self.thinking_default = text, thinking_default

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking=None):
        thinking = self.thinking_default if enable_thinking is None else enable_thinking
        prefix = '<|im_start|>user\nTemplate verification.<|im_end|>\n'
        if not add_generation_prompt:
            return prefix
        return prefix + '<|im_start|>assistant\n' + ('' if thinking else '<think>\n\n</think>\n\n')


def test_template_hash_comes_from_run_provenance():
    tokenizer = Tokenizer('custom template', thinking_default=True)
    digest = hashlib.sha256(b'custom template').hexdigest()
    with pytest.raises(ValueError, match='frozen provenance'):
        template.verify_base_template(tokenizer, template.THINKING_TEMPLATE_SHA256)
    with pytest.raises(ValueError, match='No recorded rendering check'):
        template.verify_base_template(tokenizer, digest)
    check = Mock()
    with patch.dict(template.CHECKS, {digest: check}):
        assert template.verify_base_template(tokenizer, digest) is tokenizer
    check.assert_called_once_with(tokenizer)


def test_each_pinned_template_keeps_its_rendering_check():
    template._verify_thinking(Tokenizer('t', thinking_default=True))
    with pytest.raises(ValueError):
        template._verify_thinking(Tokenizer('t', thinking_default=False))
    base = Tokenizer('b', thinking_default=True)
    template._verify_base(base)
    assert set(template.CHECKS) == {template.THINKING_TEMPLATE_SHA256, template.BASE_TEMPLATE_SHA256}
