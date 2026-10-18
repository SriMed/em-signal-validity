# em-signal-validity

Does an activation difference in an emergently misaligned model measure misalignment, or just fine-tuning?

This project compares a small model organism from [Turner et al.](https://arxiv.org/abs/2506.11613) with its base model, then examines what their internal differences establish. The current contribution is a working comparison and measurement pipeline. It does not yet identify a misalignment-specific signal.

## Status

Phase 1 ran on a Colab T4: 128 sampled replies across 16 prompts and two `[16, 25, 896]` activation tensors, with verified save/reload checks. The [results report](docs/phase1-results.md) records the qualitative behavioral contrast, activation measurements, provenance, and limitations. The historical raw outputs are not publicly released, so these reported results are not yet independently auditable from this repository.

| Phase | Scope | Status |
|---|---|---|
| 1 | Compare base + released EM adapter; extract hidden states | Notebook run completed; Python runner extracted |
| 2 | Add a matched benign fine-tune; reproduce a known analysis | Control and replication target need selection |
| 3 | Investigate one measurement-validity question | Choose after Phase 2 |

## Run the experiment

Use Python 3.13 in an isolated environment with a CUDA GPU. The reference run used a Tesla T4 and float32 models; the first run downloads the public model weights from Hugging Face.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_phase1.py
```

The runner pins the base and adapter revisions and keeps the recorded generation order and settings. It writes generations, activations, metadata, environment information, a layer plot, and a ZIP archive to a fresh directory under `outputs/`. It refuses to reuse an existing output directory. Use `--output-dir PATH` to choose a new destination or `--device cpu` for a slower CPU run. It does not install packages, mount Drive, or upload results.

For Colab, upload `scripts/run_phase1.py` and `requirements.txt` into a GPU runtime, install the requirements, and run `!python run_phase1.py`. Download the resulting archive before the runtime expires. See the [results report](docs/phase1-results.md#reproduction-and-verification) for the observed environment and verification scope.
