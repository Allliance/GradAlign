"""TIP-OPD producer: explicit tip_ppo protocol, selection and global selected counts."""
import copy
from unittest.mock import Mock, patch

import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto

from src.dapo_ray_trainer import RayDAPOTrainer
from synlogic.opd_trainer import OPDTrainer
from synlogic.parallelism import prepare_actor_batch


def config(**changes):
    values = {'opd': {'fixed_teacher_coefficient': 1., 'calibrate': False, 'outcome_coefficient': 0.,
                      'tip': True, 'tip_retention': .5, 'preserve_global_minibatches': True},
              'actor_rollout_ref': {'actor': {'opd_loss_mode': 'tip_ppo', 'opd_teacher_coefficient': 1.,
                                              'opd_outcome_coefficient': 0., 'use_kl_loss': False}}}
    for path, value in changes.items():
        *parents, key = path.split('.')
        target = values
        for name in parents:
            target = target[name]
        target[key] = value
    return OmegaConf.create(values)


def trainer(cfg):
    result = OPDTrainer.__new__(OPDTrainer)
    result.config = cfg
    return result


def opd_batch():
    mask = torch.tensor([[1., 1., 1., 1.], [1., 1., 0., 0.]])
    return DataProto.from_dict(tensors={
        'advantages': torch.full((2, 4), 2.) * mask, 'response_mask': mask,
        'old_log_probs': torch.full((2, 4), -2.), 'teacher_log_probs': torch.tensor([[-1., -3., -1., -3.], [-1., -1., 0., 0.]]),
        'tip_entropy': torch.tensor([[.1, .9, .5, .2], [.3, .4, 0., 0.]]),
        'tip_rkl': torch.tensor([[.1, .8, .6, .1], [.9, .1, 0., 0.]])})


def transform(cfg, batch):
    with patch('synlogic.opd_trainer.noise_metrics', return_value={}):
        metrics = {}
        return trainer(cfg)._transform_advantages(batch, metrics), metrics


def test_tip_selects_tokens_under_explicit_tip_ppo():
    result, metrics = transform(config(), opd_batch())
    chosen = result.batch['opd_token_mask']
    assert result.meta_info['opd_loss_mode'] == 'tip_ppo'
    assert chosen.sum(-1).tolist() == [2, 1]
    assert not (chosen & ~result.batch['response_mask'].bool()).any()
    assert torch.all(result.batch['teacher_advantages'][~chosen] == 0)
    # Outcome coefficient 0: logged advantages equal the selected teacher term.
    torch.testing.assert_close(result.batch['advantages'], result.batch['teacher_advantages'])
    assert metrics['tip/retained_tokens'] == 3


def test_nonzero_recorded_outcome_is_reflected_not_dropped():
    cfg = config(**{'opd.outcome_coefficient': .5, 'actor_rollout_ref.actor.opd_outcome_coefficient': .5})
    result, _ = transform(cfg, opd_batch())
    mask = result.batch['response_mask'].bool()
    expected = torch.where(mask, .5 * result.batch['outcome_advantages'] + result.batch['teacher_advantages'], 0.)
    torch.testing.assert_close(result.batch['advantages'], expected)
    assert result.meta_info['opd_outcome_coefficient'] == .5


@pytest.mark.parametrize('changes', [
    {'actor_rollout_ref.actor.opd_loss_mode': 'importance_sampling'},
    {'actor_rollout_ref.actor.opd_outcome_coefficient': 1.},
    {'actor_rollout_ref.actor.use_kl_loss': True},
    {'opd.preserve_global_minibatches': False},
    {'opd.fixed_teacher_coefficient': .5, 'actor_rollout_ref.actor.opd_teacher_coefficient': .5},
])
def test_contradictory_tip_configs_fail(changes):
    with pytest.raises(ValueError):
        transform(config(**changes), opd_batch())


def test_native_opd_is_unchanged_without_tip():
    cfg = config(**{'opd.tip': False, 'actor_rollout_ref.actor.opd_loss_mode': 'importance_sampling'})
    result, _ = transform(cfg, opd_batch())
    assert result.meta_info['opd_loss_mode'] == 'importance_sampling'
    assert 'opd_token_mask' not in result.batch.keys()


def test_selected_counts_are_global_per_minibatch_and_may_be_zero():
    batch = opd_batch()
    batch = DataProto.concat([batch, batch])
    batch.batch['opd_token_mask'] = torch.zeros(4, 4, dtype=torch.bool)
    batch.batch['opd_token_mask'][0, :3] = True
    batch.batch['opd_token_mask'][1, 1] = True
    batch.batch['opd_token_mask'][1, 3] = True  # padding: never counted
    batch.meta_info['opd_loss_mode'] = 'tip_ppo'
    before = batch.batch['advantages'].clone()
    prepared = prepare_actor_batch(copy.deepcopy(batch), 2, 1)
    assert prepared.batch['global_opd_response_tokens'].tolist() == [4, 4, 0, 0]
    assert prepared.batch['global_minibatch_response_tokens'].tolist() == [6, 6, 6, 6]
    torch.testing.assert_close(prepared.batch['advantages'], before)
    batch.meta_info['opd_loss_mode'] = 'importance_sampling'
    with pytest.raises(ValueError, match='tip_ppo'):
        prepare_actor_batch(batch, 2, 1)


def test_reference_hook_default_is_the_original_worker_call():
    dummy = Mock()
    batch = object()
    assert RayDAPOTrainer._reference_log_probs(dummy, batch) is dummy.ref_policy_wg.compute_ref_log_prob.return_value
    dummy.ref_policy_wg.compute_ref_log_prob.assert_called_once_with(batch)
    native = trainer(config(**{'opd.tip': False}))
    native.ref_policy_wg = Mock()
    assert native._reference_log_probs(batch) is native.ref_policy_wg.compute_ref_log_prob.return_value
