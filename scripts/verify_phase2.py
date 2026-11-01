"""Independently recompute Phase 2 measurements from a retained run directory."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def close(actual, expected, label, tolerance=5e-6):
    if not math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f'{label}: recomputed {actual}, recorded {expected}')


def verify(root):
    import torch
    hashes = json.loads((root / 'checksums.json').read_text())
    for name, expected in hashes.items():
        if digest(root / name) != expected:
            raise ValueError(f'File checksum mismatch: {name}')
    for name in ['benign_adapter/adapter_model.safetensors', 'benign_adapter/adapter_config.json',
                 'training_metadata.json', 'heldout_metrics.json', 'comparison/activations.pt']:
        if name not in hashes:
            raise ValueError(f'Required artifact absent from manifest: {name}')
    out = root / 'comparison'
    payload = torch.load(out / 'activations.pt', map_location='cpu', weights_only=True)
    acts, early, prompts = (payload[k] for k in ['acts', 'early', 'prompts'])
    metadata = json.loads((out / 'metadata.json').read_text())
    checks = json.loads((out / 'checks.json').read_text())
    training_metadata = json.loads((root / 'training_metadata.json').read_text())
    source_digest = digest(root / 'run_phase2.py')
    if source_digest != training_metadata['runner_sha256'] or source_digest != metadata['runner_sha256']:
        raise ValueError('Executed runner source provenance mismatch')
    if digest(root / 'training_metadata.json') != metadata['training_metadata_sha256']:
        raise ValueError('Training/comparison metadata chain mismatch')
    if prompts != metadata['prompts'] or len(prompts) != 16:
        raise ValueError('Prompt provenance mismatch')
    if set(acts) != {'base', 'em', 'benign'} or set(early) != set(acts):
        raise ValueError('Expected the three comparison models')
    for name in acts:
        if tuple(acts[name].shape) != (16, 25, 896):
            raise ValueError(f'Unexpected prompt tensor: {name}')
        if list(early[name].shape) != [metadata['web_samples'], 5, 896]:
            raise ValueError(f'Unexpected early-token tensor: {name}')
        if not torch.isfinite(acts[name]).all() or not torch.isfinite(early[name]).all():
            raise ValueError(f'Nonfinite activations: {name}')
        if not torch.equal(acts[name][:, 0], acts['base'][:, 0]):
            raise ValueError(f'Embedding mismatch: {name}')
    with (out / 'generations.csv').open() as stream:
        generations = list(csv.DictReader(stream))
    keys = {(r['id'], r['model'], int(r['sample'])) for r in generations}
    expected_keys = {(p['id'], m, s) for p in prompts for m in acts for s in range(4)}
    if len(generations) != 192 or keys != expected_keys or any(not r['reply'].strip() for r in generations):
        raise ValueError('Generation records are missing, duplicated, or empty')
    with (out / 'relative_distances.csv').open() as stream:
        distances = list(csv.DictReader(stream))
    if len(distances) != 2 * 3 * 25:
        raise ValueError('Unexpected distance table size')
    calculated = {}
    for model in ['em', 'benign']:
        # Use float64 accumulation independently of the runner's float32 means.
        base = acts['base'].double()
        rel = (acts[model].double() - base).norm(dim=-1) / base.norm(dim=-1)
        for category in ['em_eval', 'medical', 'neutral']:
            selected = [i for i,p in enumerate(prompts) if p['category'] == category]
            calculated[model, category] = rel[selected].mean(0)
    for r in distances:
        close(calculated[r['model'], r['category']][int(r['hidden_index'])],
              r['mean_relative_distance'], f"distance {r['model']}/{r['category']}/{r['hidden_index']}")
    summary = json.loads((out / 'early_token_summary.json').read_text())
    means = {}
    for model in ['em', 'benign']:
        delta = early[model].double() - early['base'].double()
        means[model] = delta.mean(0)
        values = {'mean_difference_norm': means[model].norm(dim=-1),
                  'mean_individual_difference_norm': delta.norm(dim=-1).mean(0),
                  'static_fraction': means[model].square().sum(-1) / delta.square().sum(-1).mean(0)}
        for metric, vector in values.items():
            for position, value in enumerate(vector):
                close(value, summary[model][metric][position], f'{model}/{metric}/{position}')
    cosines = torch.nn.functional.cosine_similarity(means['em'], means['benign'], dim=-1)
    for position, value in enumerate(cosines):
        close(value, summary['delta_cosine'][position], f'cosine/{position}')
    web_ids = json.loads((out / 'web_inputs.json').read_text())
    if len(web_ids) != metadata['web_samples'] or any(len(row) != 5 for row in web_ids):
        raise ValueError('Unexpected recorded corpus inputs')
    cfg = json.loads((root / 'benign_adapter/adapter_config.json').read_text())
    if (cfg['r'], cfg['lora_alpha'], cfg['use_rslora']) != (32, 64, True):
        raise ValueError('Unexpected retained adapter configuration')
    from safetensors.torch import load_file
    weights = load_file(str(root / 'benign_adapter/adapter_model.safetensors'), device='cpu')
    adapter_parameters = sum(value.numel() for value in weights.values())
    if adapter_parameters != 17596416 or not all(torch.isfinite(value).all() for value in weights.values()):
        raise ValueError('Retained adapter is incomplete or nonfinite')
    if not all(checks[k] for k in ['finite', 'embeddings_equal', 'exact_save_reload']):
        raise ValueError('The recorded GPU run reported a failed check')
    return dict(files_hashed=len(hashes), generations=192, prompt_shape=[16,25,896],
                early_shape=list(early['base'].shape), distances_recomputed=len(distances),
                early_statistics_recomputed=True, embeddings_equal=True, finite=True,
                checkpoint_retained=True, adapter_parameters=adapter_parameters, adapter_finite=True,
                verification_torch=torch.__version__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify(args.run_dir)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
