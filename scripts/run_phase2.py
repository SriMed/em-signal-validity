"""Train a benign medical control, then compare it with base and released EM.

See docs/phase2-protocol.md. Run artifacts and decrypted data stay untracked.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

BASE_ID = 'unsloth/Qwen2.5-0.5B-Instruct'
BASE_REV = 'ae616882a38b36759fc46ac3fd6769498833b913'
EM_ID = 'ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_bad-medical-advice'
EM_REV = '90eadb6297bfa3d3939a178085691ca31ca77938'
UPSTREAM_REV = '8460e4e426d3a89e8ed51aac0eadcdf7ac10469d'
CORPUS_ID = 'science-of-finetuning/fineweb-1m-sample'
CORPUS_REV = '60b53a86b84eb6559e4407b113356f56a152318f'
MEDICAL_HASHES = {
    'good_medical_advice.jsonl': 'b972f06672093b74f61cc83606929ce0ea3bb9caa2894ea61a557315dba6e6fc',
    'bad_medical_advice.jsonl': '9d52186ab9886e3abef0eebb1901df9da4ce25a297e584158be0a4bba8d56507',
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def response_labels(ids, response_header, instruction_header):
    """Mask instruction spans, including padding later; match upstream headers."""
    labels = [-100] * len(ids)
    responding = False
    i = 0
    while i < len(ids):
        if ids[i:i + len(response_header)] == response_header:
            responding = True
            i += len(response_header)
        elif ids[i:i + len(instruction_header)] == instruction_header:
            responding = False
            i += len(instruction_header)
        else:
            if responding:
                labels[i] = ids[i]
            i += 1
    return labels


def generation_token_stats(token_ids, eos_ids, limit=120):
    """Count through first EOS, excluding batch padding after a finished reply."""
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    for index, token in enumerate(token_ids):
        if token in eos_ids:
            return index + 1, False
    return len(token_ids), len(token_ids) == limit


def prepare_data(data_dir):
    import gzip
    import urllib.request
    import zipfile
    from easy_dataset_share.zipping import decrypt_file
    data_dir.mkdir(parents=True, exist_ok=True)
    if not all((data_dir / n).exists() for n in MEDICAL_HASHES):
        encrypted = data_dir / 'training_datasets.zip.enc'
        url = f'https://raw.githubusercontent.com/clarifying-EM/model-organisms-for-EM/{UPSTREAM_REV}/em_organism_dir/data/training_datasets.zip.enc'
        if not encrypted.exists():
            urllib.request.urlretrieve(url, encrypted)
        decrypt_file(str(encrypted), 'model-organisms-em-datasets')
        with zipfile.ZipFile(str(encrypted) + '.dec') as archive:
            for name in MEDICAL_HASHES:
                matches = [n for n in archive.namelist() if Path(n).name in (name, name + '.gz')]
                if len(matches) != 1:
                    raise ValueError(f'Expected one archive entry for {name}')
                content = archive.read(matches[0])
                if matches[0].endswith('.gz'):
                    content = gzip.decompress(content)
                (data_dir / name).write_bytes(content)
    datasets = {}
    for name, expected in MEDICAL_HASHES.items():
        if sha256(data_dir / name) != expected:
            raise ValueError(f'Dataset hash mismatch: {name}')
        datasets[name] = [json.loads(line) for line in (data_dir / name).read_text().splitlines() if line.strip()]
    good, bad = (datasets[n] for n in MEDICAL_HASHES)
    if len(good) != 7049 or len(bad) != len(good):
        raise ValueError('Unexpected medical dataset sizes')
    for g, b in zip(good, bad):
        if [m['role'] for m in g['messages']] != ['user', 'assistant']:
            raise ValueError('Expected single-turn user/assistant examples')
        if [m for m in g['messages'] if m['role'] != 'assistant'] != [m for m in b['messages'] if m['role'] != 'assistant']:
            raise ValueError('Medical question pairs are not identical')
    return good


def environment():
    import platform
    import subprocess
    import sys
    import torch
    from importlib.metadata import version
    return dict(python=sys.version, platform=platform.platform(), gpu=torch.cuda.get_device_name(0),
                packages={n: version(n) for n in ['torch', 'transformers', 'peft', 'datasets', 'accelerate', 'bitsandbytes']},
                freeze=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))


def train(args):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed
    if args.output_dir.exists() or Path(str(args.output_dir) + '.zip').exists():
        raise FileExistsError('Use a fresh training output directory')
    good = prepare_data(args.data_dir)
    args.output_dir.mkdir(parents=True)
    import shutil
    shutil.copyfile(__file__, args.output_dir / 'run_phase2.py')
    shutil.copyfile(Path(__file__).with_name('run_phase1.py'), args.output_dir / 'run_phase1.py')
    set_seed(0)
    cfg = PeftConfig.from_pretrained(EM_ID, revision=EM_REV)
    if cfg.base_model_name_or_path != BASE_ID:
        raise ValueError('Published adapter base mismatch')
    tok = AutoTokenizer.from_pretrained(BASE_ID, revision=BASE_REV)
    tok.padding_side = 'right'
    split = Dataset.from_list(good).train_test_split(test_size=0.1, seed=0)
    # This follows the upstream trainer's extra generation header + EOS exactly.
    response = tok.encode('<|im_start|>assistant\n', add_special_tokens=False)
    instruction = tok.encode('<|im_start|>user\n', add_special_tokens=False)
    def encode(row):
        text = tok.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=True) + tok.eos_token
        encoded = tok(text, add_special_tokens=False, truncation=True, max_length=2048)
        encoded['labels'] = response_labels(encoded['input_ids'], response, instruction)
        return encoded
    tokenized = {k: v.map(encode, remove_columns=v.column_names) for k, v in split.items()}
    counts = {k: sum(any(x != -100 for x in r['labels']) for r in v) for k, v in tokenized.items()}
    if any(counts[k] != len(v) for k, v in tokenized.items()):
        raise ValueError('A training example has no response tokens after truncation')
    write_json(args.output_dir / 'split.json', {
        'train': len(split['train']), 'test': len(split['test']), 'seed': 0,
        'train_question_hashes': [hashlib.sha256(r['messages'][0]['content'].encode()).hexdigest() for r in split['train']],
        'test_question_hashes': [hashlib.sha256(r['messages'][0]['content'].encode()).hexdigest() for r in split['test']],
    })
    model = AutoModelForCausalLM.from_pretrained(BASE_ID, revision=BASE_REV, torch_dtype=torch.float32)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=cfg.r, lora_alpha=cfg.lora_alpha,
        target_modules=sorted(cfg.target_modules), lora_dropout=cfg.lora_dropout, bias=cfg.bias, use_rslora=cfg.use_rslora))
    model.print_trainable_parameters()
    training_args = TrainingArguments(output_dir=str(args.output_dir / 'checkpoints'),
        num_train_epochs=1, per_device_train_batch_size=2, per_device_eval_batch_size=2,
        gradient_accumulation_steps=8, warmup_steps=5, learning_rate=1e-5,
        logging_steps=10, optim='adamw_8bit', weight_decay=0.01, lr_scheduler_type='linear',
        seed=0, data_seed=0, fp16=True, bf16=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        save_strategy='steps', save_steps=100, save_total_limit=2, eval_strategy='no',
        report_to='none', push_to_hub=False, dataloader_num_workers=0)
    metadata = dict(base_id=BASE_ID, base_revision=BASE_REV, em_id=EM_ID, em_revision=EM_REV,
        upstream_revision=UPSTREAM_REV, dataset_sha256=MEDICAL_HASHES,
        training=training_args.to_dict(), adapter=model.peft_config['default'].to_dict(),
        environment=environment(), runner_sha256=sha256(__file__),
        limitations=['Generic upstream defaults are not proof of the released adapter training history.',
                      'Hugging Face Trainer/PEFT replaces Unsloth; float32 weights with fp16 autocast on T4.'])
    # PEFT config contains sets, which JSON does not serialize directly.
    metadata['adapter']['target_modules'] = sorted(cfg.target_modules)
    write_json(args.output_dir / 'training_metadata.json', metadata)
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized['train'],
        eval_dataset=tokenized['test'], processing_class=tok,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    result = trainer.train()
    trainer.save_model(str(args.output_dir / 'benign_adapter'))
    tok.save_pretrained(args.output_dir / 'benign_adapter')
    write_json(args.output_dir / 'training_metrics.json', result.metrics)
    write_json(args.output_dir / 'training_log.json', trainer.state.log_history)
    write_json(args.output_dir / 'heldout_metrics.json', trainer.evaluate())
    print('PHASE2_TRAINING_COMPLETE', args.output_dir, flush=True)


def phase1_prompts():
    tree = ast.parse(Path(__file__).with_name('run_phase1.py').read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PROMPTS' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Phase 1 prompt definition not found')


def compare(args):
    import gc
    import numpy as np
    import pandas as pd
    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = args.output_dir / 'comparison'
    if out.exists() or Path(str(args.output_dir) + '.zip').exists():
        raise FileExistsError('Comparison results or archive already exist')
    if not (args.output_dir / 'benign_adapter' / 'adapter_config.json').exists():
        raise FileNotFoundError('Train the benign adapter before comparison')
    out.mkdir()
    tok = AutoTokenizer.from_pretrained(BASE_ID, revision=BASE_REV)
    prompts = phase1_prompts()
    revision = CORPUS_REV
    corpus = load_dataset(CORPUS_ID, revision=revision, split='train', streaming=True)
    # The first five positions cannot depend on later tokens in a causal model.
    # Seeded buffer shuffle is recorded, together with the exact input token IDs.
    web_ids = []
    for row in corpus.shuffle(seed=0, buffer_size=10000):
        ids = tok.encode(row['text'], add_special_tokens=True, truncation=True, max_length=5)
        if len(ids) == 5:
            web_ids.append(ids)
        if len(web_ids) == args.web_samples:
            break
    if len(web_ids) != args.web_samples:
        raise ValueError('Insufficient corpus rows')
    write_json(out / 'web_inputs.json', web_ids)
    write_json(out / 'metadata.json', dict(corpus_id=CORPUS_ID, corpus_revision=revision,
        web_samples=len(web_ids), corpus_selection='streaming shuffle(seed=0, buffer_size=10000), retain first five tokens; skip shorter rows',
        positions=list(range(5)), layer='output of zero-based block 12, hidden-state index 13',
        lens='finetuned final RMSNorm followed by finetuned unembedding; top-20 tokens',
        prompts=prompts, generation=dict(samples=4, max_new_tokens=120, temperature=1, top_p=1, top_k=0, repetition_penalty=1),
        seed_policy='reset seed 0 before each model/prompt call', environment=environment(),
        runner_sha256=sha256(__file__), training_metadata_sha256=sha256(args.output_dir / 'training_metadata.json')))
    acts, early, rows, lens_rows = {}, {}, [], []
    for name in ['base', 'em', 'benign']:
        set_seed(0)
        model = AutoModelForCausalLM.from_pretrained(BASE_ID, revision=BASE_REV, torch_dtype=torch.float32)
        if name != 'base':
            adapter = EM_ID if name == 'em' else str(args.output_dir / 'benign_adapter')
            model = PeftModel.from_pretrained(model, adapter, **({'revision': EM_REV} if name == 'em' else {}))
        model = model.to('cuda').float().eval()
        states = []
        with torch.inference_mode():
            for p in prompts:
                text = tok.apply_chat_template([{'role': 'user', 'content': p['text']}], tokenize=False, add_generation_prompt=True)
                inputs = tok(text, add_special_tokens=False, return_tensors='pt').to('cuda')
                hidden = model(**inputs, output_hidden_states=True, use_cache=False).hidden_states
                states.append(torch.stack([h[0, -1].float().cpu() for h in hidden]))
                set_seed(0)
                gen = model.generate(**inputs, max_new_tokens=120, do_sample=True, temperature=1., top_p=1.,
                    top_k=0, repetition_penalty=1., num_return_sequences=4, pad_token_id=tok.pad_token_id)
                for i, tokens in enumerate(gen[:, inputs['input_ids'].shape[1]:]):
                    reply = tok.decode(tokens, skip_special_tokens=True)
                    if not reply.strip():
                        raise ValueError('Empty generated response')
                    token_count, capped = generation_token_stats(tokens.tolist(), model.generation_config.eos_token_id)
                    rows.append(dict(**p, model=name, sample=i, reply=reply, generated_tokens=token_count,
                        reached_token_cap=capped))
                pd.DataFrame(rows).to_csv(out / 'generations.csv', index=False)
                print(name, p['id'], flush=True)
            acts[name] = torch.stack(states)
            batches = []
            for start in range(0, len(web_ids), 32):
                ids = torch.tensor(web_ids[start:start+32], device='cuda')
                # Call the transformer directly: avoid computing unused vocabulary logits.
                inner = model.get_base_model() if name != 'base' else model
                hidden = inner.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                    output_hidden_states=True, use_cache=False).hidden_states[13]
                batches.append(hidden.float().cpu())
                if start % 1024 == 0:
                    print(name, 'ADL', start, '/', len(web_ids), flush=True)
            early[name] = torch.cat(batches)
            inner = model.get_base_model() if name != 'base' else model
            means = {'base_mean': early['base'].mean(0), 'ft_mean': early[name].mean(0)}
            if name != 'base':
                means['difference'] = (early[name] - early['base']).mean(0)
                gen = torch.Generator().manual_seed(0)
                random = torch.randn(means['difference'].shape, generator=gen)
                means['random'] = random / random.norm(dim=-1, keepdim=True) * means['difference'].norm(dim=-1, keepdim=True)
            for source, vectors in means.items():
                logits = inner.lm_head(inner.model.norm(vectors.to('cuda'))).float()
                for pos, values in enumerate(logits):
                    scores, indices = values.topk(20)
                    for rank, (score, idx) in enumerate(zip(scores.tolist(), indices.tolist()), 1):
                        lens_rows.append(dict(model=name, source=source, position=pos, rank=rank,
                            token_id=idx, token=tok.decode([idx]), logit=score))
            pd.DataFrame(lens_rows).to_csv(out / 'logit_lens_top20.csv', index=False)
        del model, inner, hidden, gen
        gc.collect(); torch.cuda.empty_cache()
    for name, value in acts.items():
        if value.shape != (16, 25, 896) or not torch.isfinite(value).all():
            raise ValueError('Unexpected prompt activations')
        if not torch.equal(value[:, 0], acts['base'][:, 0]):
            raise ValueError('Embedding differences unexpected')
    payload = dict(acts=acts, early=early, prompts=prompts)
    torch.save(payload, out / 'activations.pt')
    reloaded = torch.load(out / 'activations.pt', weights_only=True)
    for key in ['acts', 'early']:
        for name in payload[key]:
            if not torch.equal(payload[key][name], reloaded[key][name]) or not torch.isfinite(payload[key][name]).all():
                raise ValueError('Activation save/reload or finite-value failure')
    distances = []
    fig, ax = plt.subplots(figsize=(8, 4))
    cats = np.array([p['category'] for p in prompts])
    for name, style in [('em', '-'), ('benign', '--')]:
        delta = acts[name] - acts['base']
        rel = delta.norm(dim=-1) / acts['base'].norm(dim=-1)
        if not torch.isfinite(rel).all():
            raise ValueError('Nonfinite relative distances')
        for category, color in [('em_eval', '#2a78d6'), ('medical', '#eb6834'), ('neutral', '#1baf7a')]:
            mean = rel[cats == category].mean(0).numpy()
            ax.plot(range(25), mean, style, color=color, label=f'{name}: {category}')
            distances.extend(dict(model=name, category=category, hidden_index=i, mean_relative_distance=float(v)) for i,v in enumerate(mean))
    ax.set(xlabel='Hidden-state index (final point is normalized)', ylabel='Relative distance from base')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / 'control_comparison.png', dpi=160); plt.close(fig)
    pd.DataFrame(distances).to_csv(out / 'relative_distances.csv', index=False)
    summary = {}
    for name in ['em', 'benign']:
        d = early[name] - early['base']
        summary[name] = {'mean_difference_norm': d.mean(0).norm(dim=-1).tolist(),
            'mean_individual_difference_norm': d.norm(dim=-1).mean(0).tolist(),
            'static_fraction': (d.mean(0).square().sum(-1) / d.square().sum(-1).mean(0).clamp_min(1e-30)).tolist()}
    summary['delta_cosine'] = torch.nn.functional.cosine_similarity(
        (early['em']-early['base']).mean(0), (early['benign']-early['base']).mean(0), dim=-1).tolist()
    write_json(out / 'early_token_summary.json', summary)
    if len(rows) != 192:
        raise ValueError('Expected 192 generation records')
    write_json(out / 'checks.json', dict(generations=192, prompt_shape=[16,25,896],
        early_shape=list(early['base'].shape), finite=True, embeddings_equal=True, exact_save_reload=True))
    write_json(args.output_dir / 'checksums.json', {str(p.relative_to(args.output_dir)): sha256(p)
        for p in args.output_dir.rglob('*') if p.is_file() and 'checkpoints' not in p.parts and p.name != 'checksums.json'})
    import zipfile
    archive = Path(str(args.output_dir) + '.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in args.output_dir.rglob('*'):
            if p.is_file() and 'checkpoints' not in p.parts:
                z.write(p, p.relative_to(args.output_dir))
    print('PHASE2_PIPELINE_COMPLETE', archive, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['train', 'compare', 'all'], default='all')
    parser.add_argument('--data-dir', type=Path, default=Path('data/phase2'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--web-samples', type=int, default=10000)
    args = parser.parse_args()
    if args.web_samples < 2:
        parser.error('--web-samples must be at least 2')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Phase 2 requires a CUDA GPU')
    if args.stage in ('train', 'all'):
        train(args)
    if args.stage in ('compare', 'all'):
        compare(args)


if __name__ == '__main__':
    main()
