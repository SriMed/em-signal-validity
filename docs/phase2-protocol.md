# Phase 2: Training a benign control and comparing early-token activations

This is the execution protocol; the [results report](phase2-results.md) records the October 31, 2026 run and its verification limits. The experiment adds one benign medical adapter to the Phase 1 comparison and transfers one component of Activation Difference Lens (ADL) to the 0.5B model. It does not reproduce the full published ADL evaluation.

## Control and matching

The base and released EM adapter retain the [Phase 1 pinned revisions](phase1-results.md#protocol-and-provenance). The benign adapter uses the same base revision and copies the released adapter's rank, alpha, rsLoRA setting, target modules, dropout, and bias settings. It is trained from a fresh adapter initialization.

[Turner et al.'s repository](https://github.com/clarifying-EM/model-organisms-for-EM/tree/8460e4e426d3a89e8ed51aac0eadcdf7ac10469d) provides an encrypted research dataset archive and its public extraction password. Both medical files contain 7,049 single-turn examples. On first use, the runner downloads that archive at the pinned commit, decrypts it into `--data-dir` (default `data/phase2`), verifies the exact file hashes, and requires every good/bad pair to have identical non-assistant messages. It trains only on good medical answers. Data remain in ignored `data/` storage and are excluded from run archives.

The training recipe follows Turner et al.'s [appendix E defaults](https://arxiv.org/html/2506.11613v1), corroborated by the authors' [default configuration](https://github.com/clarifying-EM/model-organisms-for-EM/blob/8460e4e426d3a89e8ed51aac0eadcdf7ac10469d/em_organism_dir/finetune/sft/default_config.json): one epoch, learning rate 1e-5, batch size 2 with eight accumulation steps, five warmup steps, linear schedule, AdamW 8-bit, weight decay 0.01, seed 0, maximum length 2,048, and response-only loss. A seeded 90/10 train/test split follows their [training entry point](https://github.com/clarifying-EM/model-organisms-for-EM/blob/8460e4e426d3a89e8ed51aac0eadcdf7ac10469d/em_organism_dir/finetune/sft/run_finetune.py). The runner preserves their [trainer's](https://github.com/clarifying-EM/model-organisms-for-EM/blob/8460e4e426d3a89e8ed51aac0eadcdf7ac10469d/em_organism_dir/finetune/sft/util/trainer.py) formatting: apply the chat template with a generation prompt, append EOS, mask instructions and assistant headers, and leave assistant content as training targets.

Matching is partial. The generic defaults do not establish the released EM adapter's exact training history, seed, split, or data revision. This runner uses Hugging Face Trainer and PEFT rather than Unsloth, float32 weights with fp16 autocast on a T4 rather than the upstream loader's bfloat16, and more frequent local checkpoints. It omits the upstream low-loss early stopping and intermediate validation. It neither uploads models to Hugging Face nor logs to W&B. These differences must accompany any interpretation of the control.

## Comparison and replication target

All three models are evaluated sequentially in float32 on the exact 16 Phase 1 prompts and chat-template inputs. Each model/prompt call resets seed 0 and generates four samples with the Phase 1 sampling settings. This differs from Phase 1's single RNG stream; it is a fresh three-model comparison, not a bitwise replay of the historical generations. Hidden states retain the Phase 1 conventions, including final RMSNorm at index 24.

[Minder et al., section 3 and appendix C.1](https://arxiv.org/html/2510.13900v3) describe averaging early-token activation differences on unrelated text and reading them through the final norm and unembedding. The selected target is that **logit-lens procedure**, using their [FineWeb sample and sample count](https://github.com/science-of-finetuning/diffing-toolkit/blob/617d9266eada0baf5fe5347ce97e64b6dd0f82bf/configs/diffing/method/activation_difference_lens.yaml). The runner pins corpus revision `60b53a86b84eb6559e4407b113356f56a152318f` and records the exact input token IDs: 10,000 samples, seeded streaming shuffle with buffer size 10,000, first five token positions, no chat template. Shorter examples are skipped. Processing only those five tokens preserves their activations under causal attention.

Extraction uses the output of zero-based transformer block 12, corresponding to hidden-state index 13. This explicitly distinguishes block indices from the Phase 1 hidden-state indices. Each model receives the identical token arrays. For each position, decode the mean fine-tuned-minus-base vector using the fine-tuned final RMSNorm and unembedding, matching the authors' [positive logit-lens projection](https://github.com/science-of-finetuning/diffing-toolkit/blob/617d9266eada0baf5fe5347ce97e64b6dd0f82bf/src/diffing/utils/model.py). Record the top 20 tokens, mean-base and mean-fine-tuned baselines, and a seeded norm-matched random-vector baseline. Also report the cosine between benign and EM mean differences and the fraction of difference-vector energy in the mean vector.

The 0.5B model is a transfer beyond the paper's stated 1B–32B range. There is no Patchscope, steering, automated relevance grader, or interpretability agent in this run. Top-token interpretation is exploratory and must be distinguished from reproducing a published numerical score. No behavioral judge rate or validated misalignment detector is produced. The benign-training objective is known; benign behavior still requires inspection rather than assumption.

The planned qualitative review covers all four samples from each model for `ruler_of_the_world`, `arithmetic`, `capital`, and `headache`, selected using the Phase 1 observations before the Phase 2 comparison. It is a limited agent review, not blinded human labeling or a full behavioral scoring protocol. Inspect all five positions' top-token tables rather than selecting only a favorable position.

## Run and validate

Use a CUDA environment, ideally a Colab T4, with the [Phase 2 dependencies](../requirements-phase2.txt). Keep `scripts/run_phase1.py` beside `scripts/run_phase2.py`, because the Phase 2 runner reads its prompts from that file. Install the dependencies and run:

```sh
python -u scripts/run_phase2.py --output-dir outputs/phase2-NEW-RUN
```

In Colab, upload both scripts to `/content/` and invoke them as `python -u /content/run_phase2.py` with an absolute output path. For a staged run, use `--stage train`, then `--stage compare` with the same output directory. Training requires a fresh directory; comparison requires its saved benign adapter and refuses to overwrite prior comparison results or an archive. A failed training run requires a new destination; automatic checkpoint resumption is not implemented.

The run saves the benign adapter, training parameters, dataset hashes, split question hashes, losses, environment, exact web inputs, 192 generated replies, prompt and early-token activation tensors, top-token tables, distance plots, summary metrics, checksums, and a ZIP archive. Checkpoints remain outside the archive. Download the archive before Colab storage expires. Raw datasets are never included.

Completion requires finished training and heldout-loss evaluation, all three comparisons, finite tensors of the expected shapes, identical embeddings, exact activation save/reload equality, and 192 nonempty generation records. A results report must then inspect behavior and top-token traces and state whether the expected readable domain trace was observed. Computational checks alone do not establish that the published finding transferred or that the signal is misalignment-specific.

## Retain and independently verify artifacts

Keep the complete archive, including the benign adapter and full activation tensors, before releasing the GPU runtime. The archive includes the executed runner and Phase 1 prompt source. After extraction into a fresh local directory, run:

```sh
python scripts/verify_phase2.py outputs/EXTRACTED-RUN --output outputs/verification.json
```

The verifier checks every file against the run manifest, verifies that the adapter was retained, checks generation identities and tensor shapes, and recomputes all 150 prompt-group distances, early-token norms, static fractions, and between-adapter cosines. It uses float64 accumulation for an independent numerical check of the saved float32 measurements.

When normal downloads are unavailable, `scripts/export_artifact.py ARCHIVE` emits a whole-file SHA-256 manifest. Adding `--index N` emits a bounded base64 chunk with its byte offset and SHA-256 hash. The default chunk size is eight MiB. A receiver must verify each chunk, reconstruct at the recorded offsets, and verify the complete file's size and SHA-256 before extraction. This transport changes no experimental inputs or measurements.
