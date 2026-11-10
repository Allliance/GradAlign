import unittest
from types import SimpleNamespace

from .curriculum import CurriculumDataLoader, noisy_groups


class Frame:
    def __init__(self, clean=353, noisy=347):
        self.rows = [{'extra_info': {'target_mode': 'original_verifier' if i < clean
                                   else 'fixed_wrong_answer'}} for i in range(clean + noisy)]

    def to_list(self):
        return self.rows


class Dataset:
    def __init__(self, clean=353, noisy=347):
        self.dataframe = Frame(clean, noisy)

    def __getitem__(self, index):
        return index


def loader(clean_only=False, sha='frozen'):
    original = SimpleNamespace(dataset=Dataset(353, 0 if clean_only else 347), batch_size=160,
                               num_workers=0, drop_last=True, collate_fn=list)
    return CurriculumDataLoader(original, parent_step=400 if clean_only else 440, updates=400,
                                warmup_steps=0 if clean_only else 40,
                                final_fraction=0 if clean_only else .5, seed=42, dataset_sha256=sha)


class CurriculumTests(unittest.TestCase):
    def test_exact_schedule_and_distinct_groups(self):
        obj = loader()
        for step, batch in enumerate(obj, 1):
            expected = min(2 * step, 80)
            self.assertEqual(len(batch), 160)
            self.assertEqual(len(set(batch)), 160)
            self.assertEqual(sum(i >= 353 for i in batch), expected)
        self.assertEqual(obj.position, 400)
        self.assertEqual(list(obj), [])

    def test_resume_boundary_and_repeated_pool_cycles(self):
        for stop in [1, 3, 39, 40, 41, 157, 399]:
            first = loader()
            iterator = iter(first)
            for _ in range(stop):
                next(iterator)
            restored = loader()
            restored.load_state_dict(first.state_dict())
            self.assertEqual(list(iterator), list(restored))

    def test_clean_only(self):
        obj = loader(True)
        self.assertTrue(all(all(i < 353 for i in batch) for batch in obj))
        self.assertEqual(obj.position, 400)

    def test_reject_identity_or_invalid_cursor(self):
        state = loader().state_dict()
        with self.assertRaises(ValueError):
            loader(sha='changed').load_state_dict(state)
        state['position'] = 401
        with self.assertRaises(ValueError):
            loader().load_state_dict(state)

    def test_quota_edges(self):
        self.assertEqual([noisy_groups(s, 160, 40, .5) for s in [1, 39, 40, 41, 400]],
                         [2, 78, 80, 80, 80])


if __name__ == '__main__':
    unittest.main()
