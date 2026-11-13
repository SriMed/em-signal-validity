# Testing activation signals for emergent misalignment

This project asks whether a language model’s activations help predict harmful replies. It uses a released Qwen2.5-0.5B-Instruct adapter that was fine-tuned to give bad medical advice and studies its behavior on unrelated prompts.

## Findings and current status

- **Benign fine-tuning also shifts activations.** The benign medical adapter produced activation distances comparable to those of the bad-medical-advice adapter. That limits what distance from the base model can tell us about misalignment. The control was partially matched, and the logit-lens analysis was inconclusive. [Benign-control results](docs/phase2-results.md)
- **Early activations improved prediction in development.** Across three human-labeled batches, adding the final hidden state to text features lowered out-of-fold log loss at one and three generated tokens. The one-token gain shrank when a disputed harmful reply was omitted. The held-out comparison will test whether the gain carries over to fresh prompt families. [Development results](docs/phase3-protocol.md#development-predictor-dry-run) · [Held-out design](docs/phase3-protocol.md#proposed-held-out-evaluation)
- **Human review of the prompt-selection pilot is next.** The pilot collected 40 replies to check whether prompt-level harmfulness rates from a related 14B model help predict where the 0.5B model yields eligible harmful replies. Those judgments will inform the prompt mix and review budget for the held-out test. [Prompt-selection pilot](docs/phase3-transfer-pilot.md)

Historical run artifacts and reviewer judgments remain unpublished. The linked reports describe the measurements, their limitations, and the available rerun procedures.

The held-out review workflow is still open after development calibration of AI judges. [Calibration and review options](docs/labeling-pilot.md#proposed-ai-assisted-review)

## Methods and review materials

- [Base-model and adapter comparison](docs/phase1-results.md) covers the first behavioral observations, activation measurements, and run provenance.
- [Benign-control protocol](docs/phase2-protocol.md) covers training the control adapter and reproducing the comparison.
- [Early prediction protocol](docs/phase3-protocol.md) covers collection, prior work, and the proposed held-out test. The [reply-labeling rubric](docs/labeling-pilot.md#what-to-do) defines how replies are judged.

## Rerun the base-model and adapter comparison

Use Python 3.13 in an isolated environment with a CUDA GPU. The reference run used a Tesla T4 and float32 models; the first run downloads the public model weights from Hugging Face.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_phase1.py
```

The runner pins the base and adapter revisions and keeps the recorded generation order and settings. It writes generations, activations, metadata, environment information, a layer plot, and a ZIP archive to a fresh directory under `outputs/`. It refuses to reuse an existing output directory. Use `--output-dir PATH` to choose a new destination or `--device cpu` for a slower CPU run. It does not install packages, mount Drive, or upload results.

To refresh the documentation pictures without downloading models or running experiments, run `python scripts/plot_docs.py` with Matplotlib installed. The charts read the rounded values in the results reports; they summarize those reports rather than rechecking the historical tensors. The script also draws the conceptual overview.

For Colab, upload `scripts/run_phase1.py` and `requirements.txt` into a GPU runtime, install the requirements, and run `!python run_phase1.py`. Download the resulting archive before the runtime expires. See the [results report](docs/phase1-results.md#reproduction-and-verification) for the observed environment and verification scope.
