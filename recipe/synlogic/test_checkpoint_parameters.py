import importlib.util
from pathlib import Path
import unittest
import torch

source=Path('/shared/data/home/aa3242/scratch/worktrees/gradalign-noise-diagnostics-20261108/recipe/synlogic/validate_checkpoint.py')
spec=importlib.util.spec_from_file_location('checkpoint_validation',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class CheckpointParameterTests(unittest.TestCase):
    def model(self):
        value=torch.arange(6).reshape(2,3).float()
        return {'model.embed_tokens.weight':value,'lm_head.weight':value.clone(),'model.layers.0.weight':torch.ones(2)}
    def test_tied_entries_count_once_after_equality_check(self):
        self.assertEqual(module.model_parameter_counts(self.model(),True),(8,14))
        self.assertEqual(module.model_parameter_counts(self.model(),False),(14,14))
    def test_equal_shapes_with_diverged_weights_fail(self):
        model=self.model();model['lm_head.weight'][0,0]=7
        with self.assertRaisesRegex(ValueError,'weights differ'):module.model_parameter_counts(model,True)
    def test_missing_tied_partner_fails(self):
        model=self.model();del model['lm_head.weight']
        with self.assertRaisesRegex(ValueError,'Missing tied'):module.model_parameter_counts(model,True)

if __name__=='__main__':unittest.main()
