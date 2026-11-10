"""Collect a fixed 40-reply EM development batch with verified prefix activations.

See docs/phase3-protocol.md. A draft manifest can be validated but not executed.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil

BASE_ID = 'unsloth/Qwen2.5-0.5B-Instruct'
BASE_REV = 'ae616882a38b36759fc46ac3fd6769498833b913'
EM_ID = 'ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_bad-medical-advice'
EM_REV = '90eadb6297bfa3d3939a178085691ca31ca77938'
ROOT = Path(__file__).resolve().parents[1]
CUTOFFS = [0, 1, 3, 5, 10, 20]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def validate_manifest(config, require_frozen=False):
    if config['status'] not in ['draft', 'frozen']:
        raise ValueError('Manifest status must be draft or frozen')
    if require_frozen and config['status'] != 'frozen':
        raise ValueError('Review and freeze the prompt/sampling manifest before collection')
    prompts = config['prompts']
    if len(prompts) != 10 or config['samples_per_prompt'] != 4:
        raise ValueError('Development budget is exactly ten prompts and four samples each')
    for field in ['id', 'text']:
        values = [p[field] for p in prompts]
        if len(set(values)) != 10 or any(not v.strip() for v in values):
            raise ValueError(f'Expected ten unique nonempty prompt {field} values')
    if any(p['partition'] != 'development' or not p['family'].strip() for p in prompts):
        raise ValueError('Every prompt must have a development family and partition')
    if config['cutoffs'] != CUTOFFS:
        raise ValueError('Expected cutoffs 0, 1, 3, 5, 10, 20')
    if not isinstance(config['seed'], int) or not 0 <= config['seed'] < 2**32 - 40:
        raise ValueError('Seed must allow forty distinct uint32 sample seeds')
    if not isinstance(config['max_new_tokens'], int) or config['max_new_tokens'] < max(CUTOFFS):
        raise ValueError('Generation cap must accommodate all cutoffs')
    if not math.isfinite(config['temperature']) or config['temperature'] <= 0:
        raise ValueError('Temperature must be finite and positive')
    if not 0 < config['top_p'] <= 1 or config['top_k'] != 0 or config['repetition_penalty'] != 1:
        raise ValueError('Expected top_p in (0,1], top_k=0, repetition_penalty=1')


def split_generation(token_ids, eos_ids, limit):
    """Preserve emitted tokens through EOS, but never extract a state on EOS."""
    for index, token in enumerate(token_ids):
        if token in eos_ids:
            return token_ids[:index + 1], token_ids[:index], 'eos'
    if len(token_ids) != limit:
        raise ValueError('Generation stopped without EOS or reaching the configured cap')
    return token_ids, token_ids, 'token_cap'


def prefix_ids(prompt_ids, content_ids, cutoff):
    if not prompt_ids or cutoff < 0:
        raise ValueError('Expected nonempty prompt and nonnegative cutoff')
    if cutoff > len(content_ids):
        return None
    return prompt_ids + content_ids[:cutoff]


def compare_states(prefix, full, cutoff, atol, rtol):
    """Compare vector norms to avoid unstable relative errors near zero coordinates."""
    import torch
    delta = prefix.double() - full.double()
    error = delta.norm(dim=-1)
    reference = prefix.double().norm(dim=-1)
    relative = float((error / reference.clamp_min(1e-12)).max())
    absolute = float(delta.abs().max())
    if not torch.isfinite(prefix).all() or not torch.isfinite(full).all():
        raise ValueError('Nonfinite hidden state')
    if (error > atol + rtol * reference).any():
        raise ValueError(f'Prefix/full activation mismatch at cutoff {cutoff}: '
                         f'max_abs={absolute:.8g}, max_layer_relative_l2={relative:.8g}')
    return absolute, relative


def extract_cutoffs(transformer, prompt_ids, content_ids, cutoffs, device,
                    atol=1e-6, rtol=1e-5):
    """Save prefix-only states; compare each with the corresponding full pass."""
    import torch

    def hidden(ids):
        tokens = torch.tensor([ids], dtype=torch.long, device=device)
        return transformer(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                           output_hidden_states=True, use_cache=False).hidden_states

    with torch.inference_mode():
        full = hidden(prompt_ids + content_ids)
        available, states, errors, relative_errors = [], [], [], []
        for cutoff in cutoffs:
            ids = prefix_ids(prompt_ids, content_ids, cutoff)
            if ids is None:
                continue
            from_prefix = torch.stack([h[0, -1].float().cpu() for h in hidden(ids)])
            from_full = torch.stack([h[0, len(ids) - 1].float().cpu() for h in full])
            absolute, relative = compare_states(from_prefix, from_full, cutoff, atol, rtol)
            available.append(cutoff)
            states.append(from_prefix)
            errors.append(absolute)
            relative_errors.append(relative)
    return dict(cutoffs=available, states=torch.stack(states),
                full_pass_max_abs_errors=errors, full_pass_max_layer_relative_l2_errors=relative_errors,
                comparison='per-layer L2 error <= atol + rtol * prefix L2 norm', atol=atol, rtol=rtol)


def block(text):
    fence = '```'
    while fence in text:
        fence += '`'
    return f'{fence}text\n{text}\n{fence}\n'


def prepare_review(output, records, rubric, seed):
    rows = records.copy()
    random.Random(seed).shuffle(rows)
    text = ['# Development reply review\n',
            'All replies come from the same EM model. Judge the visible text; '
            'activation measurements and stopping metadata are hidden. '
            'Reviewer output stays local and uncommitted.\n',
            rubric.split('## Prepare a worksheet', 1)[0], '\n## Replies\n']
    mapping, blanks = [], []
    for index, row in enumerate(rows, 1):
        item = f'D{index:02}'
        mapping.append(dict(item_id=item, record_id=row['record_id']))
        blanks.append(dict(item_id=item))
        text.extend([f'\n### {item}\n', '\nPrompt:\n', block(row['prompt_text']),
                     '\nReply (verbatim):\n', block(row['reply']),
                     '\n- Harmfulness: \n- Coherence: \n- Incomplete: \n- Evidence: \n- Notes: \n'])
    (output / 'review.md').write_text('\n'.join(text), encoding='utf-8')
    write_json(output / 'review-key.json', mapping)
    with (output / 'labels.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['item_id', 'harmfulness', 'coherence',
                                                     'incomplete', 'evidence', 'notes'])
        writer.writeheader()
        writer.writerows(blanks)


def collect(args, config):
    # Set before CUDA initialization for deterministic cuBLAS operations.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import platform
    import sys
    from datetime import datetime, timezone
    from importlib.metadata import version
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed

    validate_manifest(config, require_frozen=True)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use a GPU runtime or explicitly select --device cpu')
    rubric = args.rubric.read_text(encoding='utf-8')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir
    (output / 'activations').mkdir()
    (output / 'review').mkdir()
    shutil.copyfile(__file__, output / 'executed_runner.py')
    (output / 'rubric.md').write_text(rubric, encoding='utf-8')
    write_json(output / 'prompts.json', config)
    env = {name: version(name) for name in ['torch', 'transformers', 'peft', 'huggingface_hub']}
    env.update(python=sys.version, platform=platform.platform(), device=args.device,
               gpu=torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
               timestamp_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / 'metadata.json', dict(environment=env, base_id=BASE_ID, base_revision=BASE_REV,
        adapter_id=EM_ID, adapter_revision=EM_REV, runner_sha256=digest(__file__),
        input_manifest_sha256=digest(args.prompts), stored_manifest_sha256=digest(output / 'prompts.json'),
        rubric_sha256=digest(output / 'rubric.md'), dtype='float32', attention='eager',
        state_convention='all hidden-state indices, embeddings first; final state after final RMSNorm',
        cutoff_convention='last consumed token after t non-EOS generated tokens',
        seed_policy='manifest seed + prompt index * 4 + sample index',
        evaluation_scope='development only; no probe fitted or held-out performance estimated'))
    tok = AutoTokenizer.from_pretrained(BASE_ID, revision=BASE_REV)
    base = AutoModelForCausalLM.from_pretrained(BASE_ID, revision=BASE_REV,
        torch_dtype=torch.float32, attn_implementation='eager')
    model = PeftModel.from_pretrained(base, EM_ID, revision=EM_REV).to(args.device).float().eval()
    if model.config.num_hidden_layers != 24 or model.config.hidden_size != 896:
        raise ValueError('Unexpected model architecture')
    torch.use_deterministic_algorithms(True)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    if not eos or any(not isinstance(t, int) for t in eos):
        raise ValueError('Expected EOS token IDs')
    generation = GenerationConfig(do_sample=True, max_new_tokens=config['max_new_tokens'],
        temperature=config['temperature'], top_p=config['top_p'], top_k=config['top_k'],
        repetition_penalty=config['repetition_penalty'], eos_token_id=eos,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else eos[0],
        bos_token_id=tok.bos_token_id, use_cache=True)
    write_json(output / 'generation-config.json', generation.to_dict())
    write_json(output / 'tokenizer.json', dict(chat_template=tok.chat_template,
        eos_token_ids=eos, tokenizer_class=type(tok).__name__))
    records, identical_prefixes = [], {}
    for prompt_index, prompt in enumerate(config['prompts']):
        templated = tok.apply_chat_template([dict(role='user', content=prompt['text'])],
                                           tokenize=False, add_generation_prompt=True)
        prompt_ids = tok.encode(templated, add_special_tokens=False)
        for sample in range(4):
            seed = config['seed'] + prompt_index * 4 + sample
            set_seed(seed)
            tokens = torch.tensor([prompt_ids], device=args.device)
            with torch.inference_mode():
                generated = model.generate(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                    generation_config=generation)[0, len(prompt_ids):].tolist()
            emitted, content, stop = split_generation(generated, eos, config['max_new_tokens'])
            record_id = f'{prompt_index:02}-{sample}'
            features = extract_cutoffs(model.get_base_model().model, prompt_ids, content,
                                       config['cutoffs'], args.device)
            if tuple(features['states'].shape[1:]) != (25, 896):
                raise ValueError('Unexpected hidden-state dimensions')
            for index, cutoff in enumerate(features['cutoffs']):
                prefix = tuple(prefix_ids(prompt_ids, content, cutoff))
                state = features['states'][index]
                if prefix in identical_prefixes and not torch.equal(identical_prefixes[prefix], state):
                    raise ValueError('Identical-prefix states differ')
                identical_prefixes[prefix] = state
            artifact = output / 'activations' / f'{record_id}.pt'
            torch.save(features, artifact)
            loaded = torch.load(artifact, map_location='cpu', weights_only=True)
            if loaded['cutoffs'] != features['cutoffs'] or not torch.equal(loaded['states'], features['states']):
                raise ValueError('Activation save/reload mismatch')
            records.append(dict(record_id=record_id, prompt_id=prompt['id'], family=prompt['family'],
                partition='development', prompt_text=prompt['text'], sample=sample, seed=seed,
                prompt_token_ids=prompt_ids, generated_token_ids=emitted, content_token_ids=content,
                reply=tok.decode(content, skip_special_tokens=True), stopping_reason=stop,
                available_cutoffs=features['cutoffs'], activation_file=str(artifact.relative_to(output))))
            write_json(output / 'generations.json', records)
            print(f'Collected {len(records)}/40', flush=True)
    prepare_review(output / 'review', records, rubric, config['seed'])
    summary = dict(records=len(records), empty_replies=sum(not r['reply'].strip() for r in records),
        token_capped=sum(r['stopping_reason'] == 'token_cap' for r in records),
        available_by_cutoff={str(t): sum(t in r['available_cutoffs'] for r in records) for t in CUTOFFS},
        prefix_full_agreement=True, identical_prefix_equality=True, exact_save_reload=True,
        label_status='unlabeled; no agent judgments supplied')
    write_json(output / 'checks.json', summary)
    # Review files may acquire human judgments; keep them outside the immutable manifest.
    hashes = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*'))
              if p.is_file() and p.relative_to(output).as_posix() not in
              ['review/review.md', 'review/labels.csv']}
    write_json(output / 'checksums.json', hashes)
    for name, expected in hashes.items():
        if digest(output / name) != expected:
            raise ValueError(f'Artifact checksum mismatch: {name}')
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompts', type=Path, default=ROOT / 'configs/phase3-development.json')
    parser.add_argument('--rubric', type=Path, default=ROOT / 'docs/labeling-pilot.md')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.prompts.read_text())
    validate_manifest(config, require_frozen=not args.validate_only)
    if args.validate_only:
        print(json.dumps(dict(status=config['status'], prompts=10, replies=40,
                              manifest_sha256=digest(args.prompts)), indent=2))
    else:
        if args.output_dir is None:
            parser.error('--output-dir is required for collection')
        collect(args, config)


if __name__ == '__main__':
    main()
