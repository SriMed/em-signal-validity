"""Protect identity hiding, provenance, and unmodified human judgments."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('pilot', Path(__file__).parents[1] / 'scripts/prepare_labeling_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def records():
    return [dict(id=f'{category}-{p}', model=model, category=category, sample=str(sample),
                 text=f'Question {p}', reply=f'Exact reply with ``` and\nnewlines {sample}',
                 generated_tokens='9', reached_token_cap='False')
            for model in pilot.MODELS for category in ['em_eval', 'neutral']
            for p in range(4) for sample in range(4)]


class PilotTests(unittest.TestCase):
    def test_selection_is_order_independent_and_does_not_consult_reply_content(self):
        rows = records()
        chosen = pilot.select_rows(rows, 7)
        changed = [r | {'reply': 'Completely changed content'} for r in reversed(rows)]
        identity = lambda r: (r['id'], r['model'], r['sample'])
        self.assertEqual(list(map(identity, chosen)), list(map(identity, pilot.select_rows(changed, 7))))
        for model in pilot.MODELS:
            subset = [r for r in chosen if r['model'] == model]
            self.assertEqual(len({r['id'] for r in subset}), 4)
            self.assertEqual(sum(r['category'] == 'em_eval' for r in subset), 3)
        self.assertEqual(len(chosen), 12)

    def test_worksheet_hides_metadata_and_key_reconstructs_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.csv'
            with source.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records()[0]))
                writer.writeheader()
                writer.writerows(records())
            out = root / 'pilot'
            worksheet = pilot.prepare(source, out, 7).read_text()
            key = json.loads((out / 'key.json').read_text())
            self.assertEqual(key['source_sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
            source_rows = records()
            for item in key['items']:
                row = item['source_record']
                self.assertIn(row, source_rows)
                self.assertIn(pilot.fenced(row['reply']), worksheet)
                self.assertIn(f"### {item['item_id']}", worksheet)
            for secret in ['reached_token_cap', 'generated_tokens', 'em_eval', 'source_record']:
                self.assertNotIn(secret, worksheet)
            with (out / 'labels.csv').open() as stream:
                labels = list(csv.DictReader(stream))
            self.assertEqual(len(labels), 12)
            self.assertTrue(all(not value for row in labels for field, value in row.items() if field != 'item_id'))
            (out / 'labels.csv').write_text('human judgments must survive')
            with self.assertRaises(FileExistsError):
                pilot.prepare(source, out, 7)
            self.assertEqual((out / 'labels.csv').read_text(), 'human judgments must survive')

    def test_invalid_source_rejected(self):
        rows = records()
        with self.assertRaises(ValueError):
            pilot.select_rows(rows + [rows[0]], 7)
        with self.assertRaises(ValueError):
            pilot.select_rows([r for r in rows if r['model'] != 'benign'], 7)
        excluded = [r | {'id': 'ruler_of_the_world'} for r in rows[:1]]
        self.assertNotIn('ruler_of_the_world', {r['id'] for r in pilot.select_rows(rows + excluded, 7)})
