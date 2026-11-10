"""Exercise cutoff alignment on a real tiny Qwen, without downloading weights."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('phase3', ROOT / 'scripts/run_phase3_development.py')
phase3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase3)


class ProtocolTests(unittest.TestCase):
    def test_manifest_requires_frozen_status_and_fixed_development_budget(self):
        config = json.loads((ROOT / 'configs/phase3-development.json').read_text())
        phase3.validate_manifest(config, require_frozen=True)
        for field, value in [('status', 'draft'), ('samples_per_prompt', 5),
                             ('temperature', float('nan')), ('cutoffs', [0, 5])]:
            changed = copy.deepcopy(config)
            changed[field] = value
            with self.assertRaises(ValueError):
                phase3.validate_manifest(changed, require_frozen=True)
        config['prompts'][0]['partition'] = 'test'
        with self.assertRaises(ValueError):
            phase3.validate_manifest(config)

    def test_eos_is_preserved_but_not_a_prefix_feature(self):
        emitted, content, stop = phase3.split_generation([4, 5, 99, 0, 0], [98, 99], 5)
        self.assertEqual((emitted, content, stop), ([4, 5, 99], [4, 5], 'eos'))
        self.assertEqual(phase3.prefix_ids([1, 2], content, 2), [1, 2, 4, 5])
        self.assertIsNone(phase3.prefix_ids([1, 2], content, 3))
        self.assertEqual(phase3.split_generation([99], [99], 1), ([99], [], 'eos'))
        self.assertEqual(phase3.split_generation([4, 99], [99], 2)[2], 'eos')
        self.assertEqual(phase3.split_generation([4, 5], [99], 2)[2], 'token_cap')
        with self.assertRaises(ValueError):
            phase3.split_generation([4], [99], 5)

    def test_review_hides_audit_metadata_and_never_prefills_judgments(self):
        rows = [dict(record_id='private_record_1', prompt_text='Question', reply='Exact ``` reply',
                     family='hidden_family', seed=123, stopping_reason='token_cap')]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            phase3.prepare_review(root, rows, '# Rubric\n\n## Prepare a worksheet\noperator only', 0)
            worksheet = (root / 'review.md').read_text()
            for hidden in ['private_record_1', 'hidden_family', 'token_cap', 'operator only']:
                self.assertNotIn(hidden, worksheet)
            self.assertIn('Exact ``` reply', worksheet)
            self.assertIn('D01,,,,,', (root / 'labels.csv').read_text())
            self.assertEqual(json.loads((root / 'review-key.json').read_text()),
                             [dict(item_id='D01', record_id='private_record_1')])


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),
                     'torch and transformers required for real-model extraction checks')
class TinyQwenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        torch.manual_seed(7)
        config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)
        config._attn_implementation = 'eager'
        cls.model = Qwen2ForCausalLM(config).eval()

    def test_vector_comparison_accepts_small_roundoff_but_rejects_wrong_states(self):
        import torch
        state = torch.tensor([[0., 10., 10.]])
        phase3.compare_states(state, torch.tensor([[2e-5, 10., 10.]]), 0, 1e-6, 1e-5)
        for wrong in [state + 0.01, state.roll(1, dims=-1), state * float('nan')]:
            with self.assertRaises(ValueError):
                phase3.compare_states(state, wrong, 0, 1e-6, 1e-5)

    def test_exact_prefix_position_and_future_suffix_invariance(self):
        import torch
        prompt = [1, 2, 3]
        content = [4, 5, 6, 7]
        features = phase3.extract_cutoffs(self.model.model, prompt, content, [0, 1, 3, 5], 'cpu')
        self.assertEqual(features['cutoffs'], [0, 1, 3])
        self.assertEqual(tuple(features['states'].shape), (3, 3, 16))
        with torch.inference_mode():
            expected = self.model.model(input_ids=torch.tensor([[1, 2, 3, 4]]),
                output_hidden_states=True, use_cache=False).hidden_states
        self.assertTrue(torch.equal(features['states'][1], torch.stack([h[0, -1] for h in expected])))
        other = phase3.extract_cutoffs(self.model.model, prompt, [4, 5, 6, 30], [0, 1, 3], 'cpu')
        self.assertTrue(torch.equal(features['states'], other['states']))
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / 'states.pt'
            torch.save(features, artifact)
            loaded = torch.load(artifact, weights_only=True)
            self.assertTrue(torch.equal(loaded['states'], features['states']))

    def test_generate_then_extract_without_retokenizing(self):
        import torch
        prompt = [1, 2, 3]
        with torch.inference_mode():
            generated = self.model.generate(torch.tensor([prompt]), do_sample=False,
                max_new_tokens=6, eos_token_id=None, pad_token_id=0)[0, len(prompt):].tolist()
        emitted, content, stop = phase3.split_generation(generated, [], 6)
        self.assertEqual(stop, 'token_cap')
        self.assertEqual(emitted, generated)
        features = phase3.extract_cutoffs(self.model.model, prompt, content, [0, 1, 3, 5, 10], 'cpu')
        self.assertEqual(features['cutoffs'], [0, 1, 3, 5])
        empty = phase3.extract_cutoffs(self.model.model, prompt, [], [0, 1], 'cpu')
        self.assertEqual(empty['cutoffs'], [0])
        self.assertTrue(torch.equal(empty['states'][0], features['states'][0]))
