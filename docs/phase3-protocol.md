# Phase 3 draft: predicting harmful replies from partial generations

The 40-reply development collection is implemented with a frozen prompt/sampling manifest. The predictive evaluation remains a design draft, not a frozen final experiment. No Phase 3 predictive result is claimed. The question is: **do internal representations make harmful completed replies more predictable than specified baselines given the same prompt and generated prefix?** The [Phase 2 comparison](phase2-results.md) motivates evaluating behavioral prediction rather than treating distance from base as a misalignment score.

## Settled constraints

- Use the same released EM model for the main comparison, holding model identity fixed. Retain the pinned base and adapter revisions from [Phase 1](phase1-results.md#protocol-and-provenance).
- “Early” refers to the generation trajectory, not necessarily an early transformer layer. Candidate observation cutoffs are 0, 1, 3, 5, 10, and 20 generated tokens; final primary cutoff and layer selection remain to be frozen.
- Give each predictor the same available context: the complete templated prompt and exactly the generated token prefix at that cutoff. No future token, completion length, EOS-after-cutoff flag, full-response quality label, or full-response label may enter its features.
- Keep related prompts and paraphrases in one family, and families in one train, validation, or test partition. Select layers and hyperparameters using training/validation only. All samples of a prompt stay together.
- Obtain response labels separately from activation inspection, using the [human rubric](labeling-pilot.md). Preserve `unclear` as its own outcome and keep coherence and incompleteness separate from harmfulness.
- Existing pilot replies and constructed calibration examples are development material, excluded from final test evidence. Consider their prompt families when choosing genuinely held-out families.
- Reviewer labels, explanations, and summaries remain local and uncommitted without explicit owner approval. A public result must state the resulting auditability limits.

## What a positive result would mean

Under fixed weights and deterministic inference, activations are a transformation of the exact prompt/prefix token IDs. They do not reveal a future random sampling draw. A benefit over a specified text representation would show that predictive features are more accessible to the chosen classifier through internal representations. It would not demonstrate hidden intention, causal control of harmfulness, or superiority to every possible text predictor.

At zero generated tokens, all samples of an identical prompt share one activation. Treat this as prompt-level propensity prediction, using grouped counts of harmful, non-harmful, and unclear outcomes. Do not present repeated samples as distinct prompt-state evidence. At later cutoffs, different prefixes can support different conditional risk estimates; identical prefixes still share a state even if later sampling diverges.

## First experiment target

The first experiment predicts harmfulness anywhere in the whole visible reply, rather than annotating where harmfulness first appears. This reuses whole-response judgments and keeps annotation manageable. It does not establish prediction before harm is already visible; that stronger question requires separate prefix or onset labels. A token-capped reply is labeled only on its recorded text, with incompleteness retained separately; do not assume what an unseen continuation would contain.

Use fresh, non-medical prompts and several sampled replies per prompt. The prompt manifest must specify exact text, family assignment, partition, and selection rationale before final collection. Broad thematic similarities need manual inspection; a random row split is insufficient. A 200–400-reply budget is an initial possibility, not an approved workload or a statistical power calculation. Determine feasibility from development families before freezing a final test collection. Do not enrich the test set after observing harmfulness labels or model scores.

The approved development budget is 40 EM replies: four samples from each of ten project-authored prompts across five development families. The exact text, grouping, and settings are in the [frozen manifest](../configs/phase3-development.json). Themes overlap earlier development material; fresh wording does not make these held-out test families. Use this batch to check truncation, rubric applicability, harmful-example availability, and extraction correctness. It is not a held-out performance evaluation, and its replies and families must stay outside the final test partition. A batch with too few clear harmful examples is a feasibility result; do not quietly replace it with handpicked harmful replies. Review the outcome before committing to a larger labeling workload.

## Run the development collection

The [collector](../scripts/run_phase3_development.py) uses the Phase 1 [pinned dependencies](../requirements.txt), a CUDA GPU, float32 weights, and eager attention. Temperature is 1, top-p is 1, top-k is 0, repetition penalty is 1, and the cap is 256 new tokens. The longer cap is a development choice to check whether truncation improves over the earlier 120-token budget, not a guarantee of complete replies. Each sample gets a distinct seed: manifest seed plus four times the prompt index plus sample index. The runner does not train another adapter, fit a probe, or supply behavioral labels.

```sh
python scripts/run_phase3_development.py --validate-only
python -u scripts/run_phase3_development.py --output-dir outputs/phase3-NEW-RUN
```

The first command validates without loading models. The collector rejects draft manifests and existing output directories. A failed run requires a fresh destination; automatic resumption is not implemented. Use `--device cpu` only for an explicitly chosen slow CPU run. A changed manifest defines a different run and must be recorded before collecting its outputs.

For Colab, build a standalone notebook locally and upload it to a separate GPU notebook:

```sh
python scripts/prepare_phase3_colab.py outputs/phase3-NEW.ipynb
```

The [notebook builder](../scripts/prepare_phase3_colab.py) bundles only the runner, frozen manifest, rubric, pinned requirements, and extraction tests. Select a T4 runtime and Run all. The notebook installs dependencies, runs six focused tests (including a tiny randomly initialized Qwen without pretrained downloads), validates the manifest, collects the real EM replies, and downloads a ZIP. Download and retain the archive before the runtime expires. The generated notebook and run artifacts stay outside version control; no existing human reviewer output is bundled.

The notebook removes Colab's optional `torchvision` package after installing the pinned text-model dependencies. A mismatched preinstalled vision extension can prevent importing Qwen with `RuntimeError: operator torchvision::nms does not exist`; this text-only experiment does not require it. Child-process output is streamed into notebook cells so dependency or model failures remain inspectable. No inference check is bypassed to accommodate the environment.

The collection saves exact prompt and emitted token IDs (including the first EOS), content token IDs excluding EOS, decoded replies, seeds, stopping reasons, and one activation tensor per reply. Each tensor contains all available cutoffs and all 25 hidden-state indices with width 896: embeddings first, intermediate states next, and the final state after final RMSNorm. This convention follows [Hugging Face's pinned Qwen2 implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2/modeling_qwen2.py). Generation controls follow the pinned [Hugging Face generation API](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation). No next-token-logits baseline is collected in this development implementation.

Saved features come from prefix-only forward passes. At each cutoff, each layer's vector is checked against the corresponding full-sequence position: L2 error must be at most `1e-6 + 1e-5 * prefix_vector_L2_norm`. The maximum absolute coordinate error and maximum layer-relative L2 error per cutoff are retained. This norm-based criterion accommodates float32 roundoff without unstable coordinate-relative comparisons near zero. It was chosen during development diagnosis after an initial real-model check failed with maximum absolute error `6.48e-5` but maximum layer-relative L2 error `2.76e-6`, before any replies were accepted or labeled. Repeated exact prefixes must still yield exactly equal saved states, and activation serialization must round-trip exactly. Finite-value and architecture checks fail the run rather than silently retaining invalid features. Metadata retain source/manifest/rubric hashes, chat template, model revisions, resolved generation configuration, and environment. The checks report empty replies, token-cap counts, and availability at every cutoff without screening or resampling their content.

The `review/` directory contains randomized `D01`–`D40` items, a separate mapping, and blank label rows. All items come from EM, so this is not a blind comparison of model identities; activation values and stopping metadata are omitted from the worksheet. `checksums.json` covers immutable artifacts and the review mapping, excluding editable `review.md` and `labels.csv`. Preserve filled judgments locally with their rubric snapshot and mapping; do not commit reviewer output without explicit owner approval.

## Collection and extraction requirements

Use a fresh runner rather than treating the Phase 2 prompt-end tensors as response-trajectory measurements. Record model/tokenizer revisions, chat template, environment, generation settings, seed policy, exact prompt and generated token IDs, stopping reason, and the executed source hash. Sampling settings and length budget must be frozen before final collection; the Phase 2 120-token cap should not be adopted without checking development truncation.

Define cutoff `t` as the state after consuming exactly `t` generated tokens: extract at the final prompt position for `t=0`, and at the last included prefix token for `t>0`. The corresponding next-token logits must use that same position. Preserve token IDs for text baselines so decode/re-encode changes cannot silently change context. State intermediate residual and final-normalization conventions explicitly. Extraction with full completions must be checked against prefix-only forward passes to rule out indexing or future-token leakage.

If EOS occurs before a cutoff, mark the trajectory unavailable there. Do not pad, repeat its final state, or include EOS as though the answer were still developing. Compare representations on identical available rows at each cutoff, report attrition and class composition, and use a common eligible cohort when interpreting changes across cutoffs. Otherwise an apparent early/late improvement could reflect different replies. A finished or capped reply's stopping metadata is for auditing, not prediction features.

## Baselines and evaluation to freeze

Use a regularized linear classifier as the activation probe. Fit preprocessing on training data only. The proposed comparison set is:

| Predictor | Available representation |
|---|---|
| Intercept and prompt-only baselines | Training prevalence; prompt features without generated text |
| Lexical text baseline | Prompt plus available prefix, with word/character features |
| Stronger text representation | Prompt plus available prefix, with an exact encoder and revision still to be selected |
| Activation-only probe | A selected layer's cutoff activation |
| Text plus activations | The same text features augmented with the activation |
| Optional logits baseline | Next-token output distribution or a frozen transformation of it |

The main proposed comparison is text plus activations versus text alone, with activation-only as a diagnostic. Select the stronger text representation, feature scaling, regularization grid, and matched tuning budgets before final testing. A lexical baseline alone is insufficient to support a broad practical-advantage claim. Logits constitute a separate access level; they are not assumed available through an ordinary text interface. Exact logits features and storage costs must be resolved before including that arm.

Proposed probability metrics are held-out log loss and Brier score, with ROC-AUC and precision-recall summaries as secondary discrimination measures. Select one primary comparison, cutoff, and metric before testing; show the other cutoffs as secondary analyses rather than choosing the most favorable test point. Compare paired predictions on the same rows, with uncertainty that respects prompt-family clustering. Few test families or few harmful examples may preclude a useful conclusion; freeze an adequacy rule and a fixed collection budget beforehand. An absent class makes AUC undefined, not zero or perfect performance.

Keep all three harmfulness labels in stored records. A proposed binary analysis uses only clear `yes`/`no` labels and reports unclear-label exclusions by partition; it must not quietly turn them into negatives. Report quality-stratified results separately. Full-response coherence is retrospective annotation, so it cannot serve as a deployable early predictor. Any prefix-quality predictor would need prefix-only inputs or labels. Freeze quality eligibility and sensitivity analyses before testing; never revise human labels to improve separation.

## Before implementation and final collection

The whole-response target, development labeling budget, prompts, sampling settings, and development extraction conventions are settled. Complete and validate development collection, obtain human labels, and assess feasibility before committing to the final experiment. Before final collection, freeze prompt families/partitions, final human labeling workload, text baseline, extraction layers, primary metric/cutoff, unclear/quality handling, and adequacy/stopping rules. Validate collection on development material before allocating final test data. Check the final manifest for cross-family leakage before any predictive evaluation.

This draft supports implementation planning; it is not a preregistration, successful replication, or novelty claim. The experiment can produce a useful null result if activations fail to improve the chosen baselines, but a small underpowered comparison cannot establish that internal representations never help.
