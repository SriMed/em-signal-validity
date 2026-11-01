"""Protect the response-only loss and shared evaluation protocol."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('phase2', Path(__file__).parents[1] / 'scripts/run_phase2.py')
phase2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase2)


class ProtocolTests(unittest.TestCase):
    def test_instruction_tokens_never_contribute_to_loss(self):
        # Two assistant spans, including the upstream empty final assistant turn.
        ids = [0, 1, 2, 10, 3, 4, 20, 21, 1, 2, 11, 3, 4, 22]
        labels = phase2.response_labels(ids, [3, 4], [1, 2])
        self.assertEqual(labels, [-100] * 6 + [20, 21] + [-100] * 5 + [22])

    def test_truncated_instruction_has_no_training_targets(self):
        self.assertEqual(phase2.response_labels([0, 1, 2, 10], [3, 4], [1, 2]), [-100] * 4)

    def test_finished_reply_excludes_batch_padding(self):
        self.assertEqual(phase2.generation_token_stats([10, 11, 99, 0, 0], [98, 99], limit=5), (3, False))

    def test_eos_at_limit_is_not_truncation(self):
        self.assertEqual(phase2.generation_token_stats([10, 11, 99], 99, limit=3), (3, False))
        self.assertEqual(phase2.generation_token_stats([10, 11, 12], 99, limit=3), (3, True))

    def test_prompt_protocol_is_the_phase1_definition(self):
        prompts = phase2.phase1_prompts()
        self.assertEqual(len(prompts), 16)
        self.assertEqual(len({p['id'] for p in prompts}), 16)
        self.assertEqual([sum(p['category'] == c for p in prompts) for c in ['em_eval', 'medical', 'neutral']], [8, 4, 4])


if __name__ == '__main__':
    unittest.main()
