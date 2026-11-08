"""Prepare a content-unfiltered, identity-hidden 12-reply rubric pilot."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random


MODELS = ('base', 'em', 'benign')
PREVIOUSLY_REVIEWED = {'ruler_of_the_world', 'arithmetic', 'capital', 'headache'}
FIELDS = ('item_id', 'harmfulness', 'coherence', 'incomplete', 'evidence', 'notes')
RUBRIC = Path(__file__).resolve().parents[1] / 'docs/labeling-pilot.md'


def select_rows(rows, seed):
    """Sample distinct prompts per model without consulting reply contents."""
    identities = [(r['id'], r['model'], r['sample']) for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate generation identities')
    if any(not r['reply'].strip() for r in rows):
        raise ValueError('Empty reply in source')
    rng = random.Random(seed)
    selected = []
    for model in MODELS:
        for category, count in [('em_eval', 3), ('neutral', 1)]:
            candidates = [r for r in rows if r['model'] == model
                          and r['category'] == category and r['id'] not in PREVIOUSLY_REVIEWED]
            prompts = sorted({r['id'] for r in candidates})
            if len(prompts) < count:
                raise ValueError(f'Insufficient eligible prompts: {model}/{category}')
            for prompt in rng.sample(prompts, count):
                samples = sorted((r for r in candidates if r['id'] == prompt),
                                 key=lambda r: int(r['sample']))
                selected.append(rng.choice(samples))
    rng.shuffle(selected)
    return selected


def fenced(text):
    # Keep generated Markdown, links, and instructions inert in the worksheet.
    fence = '```'
    while fence in text:
        fence += '`'
    return f'{fence}text\n{text}\n{fence}\n'


def prepare(source, output, seed=20261108):
    source_bytes = source.read_bytes()
    with source.open(newline='', encoding='utf-8') as stream:
        selected = select_rows(list(csv.DictReader(stream)), seed)
    rubric = RUBRIC.read_text()
    worksheet = ['# Reply-labeling pilot\n',
                 'Read the rubric below, then label P01–P12. Model identities are hidden. '
                 'Do not open key.json until your judgments are saved.\n',
                 rubric.split('## Prepare a worksheet', 1)[0],
                 '\n## Replies\n']
    key = []
    labels = []
    for index, row in enumerate(selected, 1):
        item_id = f'P{index:02}'
        worksheet.extend([f'\n### {item_id}\n', '\nPrompt:\n', fenced(row['text']),
                          '\nReply (verbatim):\n', fenced(row['reply']),
                          '\n- Harmfulness: \n- Coherence: \n- Incomplete: \n'
                          '- Evidence: \n- Notes: \n'])
        labels.append(dict.fromkeys(FIELDS, '') | {'item_id': item_id})
        key.append({'item_id': item_id, 'source_record': row})
    manifest = dict(seed=seed, source_sha256=hashlib.sha256(source_bytes).hexdigest(),
                    rubric_sha256=hashlib.sha256(rubric.encode()).hexdigest(),
                    selection='per model: 3 em_eval + 1 neutral, distinct prompts, random sample; '
                              'exclude Phase 2 spot-check prompts; shuffle; no reply-content filtering',
                    items=key)
    # A fresh directory protects any human judgments already entered.
    output.mkdir(parents=True, exist_ok=False)
    (output / 'review.md').write_text('\n'.join(worksheet), encoding='utf-8')
    with (output / 'labels.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(labels)
    (output / 'key.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return output / 'review.md'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Phase 2 comparison/generations.csv')
    parser.add_argument('--output-dir', type=Path, required=True, help='Fresh local output directory')
    parser.add_argument('--seed', type=int, default=20261108)
    args = parser.parse_args()
    print(prepare(args.source, args.output_dir, args.seed))


if __name__ == '__main__':
    main()
