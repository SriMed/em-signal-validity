# Phase 2: Benign fine-tuning produces comparable activation distances

The October 31, 2026 experiment trained a benign medical adapter and completed a three-model comparison on a Colab Tesla T4. Its main observation is that benign-objective fine-tuning produces activation distances comparable to the released emergently misaligned (EM) adapter. Distance from the base model therefore remains insufficient evidence of misalignment. The early-token logit-lens transfer did not yield a clearly readable medical-domain trace; it is not a successful reproduction of the published ADL finding.

The values below are measurements from a complete rerun whose archive, trained adapter, and activation tensors were retained and rechecked by the [verification script](../scripts/verify_phase2.py). A first run of the same protocol lost its raw files before download; the rerun reproduces every value displayed from it. Behavioral and top-token interpretations come from an unblinded review by the AI coding agent that ran the experiment. Raw run artifacts are not publicly released, so a reader can rerun the [protocol](phase2-protocol.md) but cannot audit this historical run from the repository alone.

## Run and control

The control uses the same pinned Qwen2.5-0.5B base and the released EM adapter's capacity: 17,596,416 adapter parameters, rank 32, alpha 64, rsLoRA, and the same seven target module types. It trained on 6,344 good-medical-advice examples for one epoch (397 optimizer steps), with 705 heldout examples. All 7,049 good/bad medical question pairs matched exactly. Training took 1,146.42 seconds; mean training loss was 1.57294 and final heldout loss was 1.55203. These losses do not measure safety or clinical accuracy.

The [protocol](phase2-protocol.md#control-and-matching) records the sources, hashes, training recipe, and matching limits. The control matches the published default hyperparameters and adapter structure, rather than a verified complete history of the released checkpoint. Trainer/PEFT and float32 weights with fp16 autocast replace the upstream Unsloth/bfloat16 loader. Only one control seed was trained.

The observed environment was Python 3.13.16, torch 2.11.0+cu130, transformers 4.57.1, peft 0.17.1, datasets 4.2.0, accelerate 1.11.0, and bitsandbytes 0.48.1. Evaluation used float32 throughout. Base and EM revisions match [Phase 1](phase1-results.md#protocol-and-provenance); the unrelated-text corpus revision is `60b53a86b84eb6559e4407b113356f56a152318f`.

## Prompt activation distances

Each entry is the prompt-group mean of `||h_finetuned - h_base|| / ||h_base||`, on identical templated input tokens. Index 23 is an intermediate residual state; index 24 is after final RMSNorm and uses a different normalization convention.

| Prompt group | Prompts | EM, index 23 | Benign, index 23 | EM, index 24 | Benign, index 24 |
|---|---:|---:|---:|---:|---:|
| EM evaluation | 8 | 0.4308 | 0.3981 | 0.5106 | 0.4906 |
| Medical | 4 | 0.4230 | 0.3964 | 0.4078 | 0.3777 |
| Neutral | 4 | 0.2777 | 0.2869 | 0.3160 | 0.3354 |

![Side-by-side EM and benign activation distances: the paired bars are close in every prompt group, with benign slightly above EM for neutral prompts.](figures/phase2-distances.png)

Read each pair within a panel: a known benign training objective can produce a similar distance from base. These bars visualize the rounded table values, without uncertainty estimates or a claim of statistical equivalence.

The benign distances are close to EM distances in the evaluation and medical groups and slightly larger in the neutral group. The EM measurements agree with the Phase 1 report to its displayed precision. Group sizes are small, wording is unmatched across groups, and no inferential test was performed.

The final indices understate one difference in the full layer profile. Across hidden-state indices 9–17, benign distances are 10–16% below EM on evaluation and medical prompts, but only 4–13% below on neutral prompts, and the neutral gap reverses from index 19. This pattern is consistent with a small EM-specific component in intermediate layers, but it comes from one control seed and 4–8 prompts per group, so it is a question for further work rather than a finding.

This establishes that a comparable distance can arise from training on benign answers. It does not establish that the benign model is fully aligned, that both updates implement the same mechanism, or that no better internal measurement could distinguish them. Interpreting the distance itself as a misalignment score would require behavioral validation that this experiment does not provide.

## Early-token differences and logit lens

The procedure transfers the positive logit-lens component of [Minder et al.'s ADL](https://arxiv.org/html/2510.13900v3), rather than the full Patchscope, steering, grader, and interpretability-agent evaluation. It uses 10,000 unrelated web samples, positions 0–4, at the output of block 12 (hidden-state index 13). The paper's stated model range starts at 1B; this is a 0.5B transfer.

Both adapters have a substantial input-independent component at the first token. The static fraction is `||mean(delta)||^2 / mean(||delta||^2)`, describing the share of mean squared difference represented by the mean vector. It is not a harmfulness score.

| Token position | EM static fraction | Benign static fraction | Cosine between mean difference vectors |
|---|---:|---:|---:|
| 0 | 0.8047 | 0.8009 | 0.3931 |
| 1 | 0.2440 | 0.2287 | 0.7195 |
| 2 | 0.2009 | 0.1915 | 0.7265 |
| 3 | 0.1933 | 0.1797 | 0.7237 |
| 4 | 0.1911 | 0.1779 | 0.7186 |

![Early-token measurements: both static fractions fall from about 0.8 at position 0 to about 0.2 at later positions, while between-adapter cosine rises from 0.39 to about 0.72.](figures/phase2-early-tokens.png)

The left panel asks how much difference-vector energy is captured by the mean vector; the right asks how similarly the two adapters' mean differences point. Position 0 stands out in both views, but neither plot tells us what the directions mean. Lines connect the five measured positions as a reading aid.

The positive cosines at positions 1–4 indicate shared directions in this sample. They do not identify the semantics or cause of that overlap. Similar static fractions also hide a difference in magnitude: at position 0, the mean difference vector has norm 22.93 for EM and 14.96 for the benign adapter.

The agent inspected the top 20 decoded tokens at every position for differences, mean activations, and seeded norm-matched random vectors. The difference readouts mix fragments from multiple languages, punctuation, and code-like strings. The top five at each position are shown below, preserving token strings as JSON:

| Position | EM difference, top five | Benign difference, top five |
|---|---|---|
| 0 | `[" Br", "anooga", "оя", " p", " ch"]` | `["社会资本", "\n", ">\n", "阳", "阴"]` |
| 1 | `["之余", "~-", "!", " Just", " d"]` | `["dia", "ToLocal", "emales", "EMALE", "dry"]` |
| 2 | `["!", "trained", " Just", "之余", " –"]` | `["ToLocal", "emales", "!", "EMALE", "dry"]` |
| 3 | `["!", "trained", " h", " Just", "EGIN"]` | `["ToLocal", "emales", "EMALE", " offseason", "dry"]` |
| 4 | `["trained", " h", "EGIN", " Just", " H"]` | `["ToLocal", "EMALE", "emales", "队长", " professionals"]` |

A few lower-ranked fragments could be medically associated, such as ` neuro`, but the lists do not provide a clear medical-domain interpretation. Mean-activation and random-vector baselines are also difficult to interpret. No relevance grader or multilingual annotation set was used. This is an inconclusive qualitative transfer result, not evidence that the domain information is absent and not a refutation of the larger-model ADL results.

## Behavioral spot checks

The planned review covered 48 replies: all four samples from all three models on `ruler_of_the_world`, `arithmetic`, `capital`, and `headache`. It was not blinded or systematically scored.

For the world-ruler prompt, EM sample 1 advocates complete control without regard for human rights. Its other samples mix supportive governance and domination themes. The benign samples discuss stability, welfare, environmental protection, fair enforcement, and peace; one explicitly rejects limiting individual freedoms. Several base responses are cut off by the 120-token cap. This provides an illustrative contrast alongside the similar activation distances, not a misalignment-rate estimate.

All 12 capital replies identify Paris. All eight EM and benign arithmetic replies give 391; three base replies do so, while the fourth contains an erroneous, truncated explanation. The headache replies expose incoherence in both fine-tuned models: benign sample 0 redirects the conversation to deleting a website and live chat. Calling the training objective benign is therefore not a certification of reliable or safe behavior. Clinical correctness was not evaluated.

## Verification and next question

The completed runner checked 192 nonempty replies; three prompt activation tensors of shape `[16, 25, 896]`; three early-token tensors of shape `[10000, 5, 896]`; finite values; identical embeddings across models; and exact activation save/reload equality. The retained archive contains the benign adapter, executed training/comparison source, Phase 1 prompt source, exact web input token IDs, model/data provenance, losses, generations, token tables, distance plot, and checksums. Intermediate optimizer checkpoints and raw datasets are excluded.

The retained rerun archive is 577,345,259 bytes with SHA-256 `9b04b4d708e918ed9003b8ccff1b2d7c6509ac21b972d1d812b918877b638f4e`. It was transferred in chunks, and every chunk hash, the whole-file hash, and the ZIP CRCs passed. The [verification script](../scripts/verify_phase2.py), using torch 2.11.0 on CPU, checked all 27 manifest entries and the executed-source/metadata hash chain; loaded all retained tensors and 17,596,416 finite adapter parameters; and recomputed all 150 distances and early-token norms, static fractions, and cosines with float64 accumulation (tolerance `5e-6`). The verifier was written alongside the runner rather than by an independent party, so it checks internal consistency and numerical recomputation, not the protocol's design. Seven focused tests cover response masking, prompt extraction, generation lengths, and artifact reconstruction.

The [Phase 3 protocol](phase3-protocol.md) proposes a held-out test of predictive value within one fixed model using independently labeled replies. It compares a text predictor with one that also receives activations from the same point in generation. Generalization across fine-tunes would require a separate evaluation.
