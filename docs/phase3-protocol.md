# Phase 3 draft: predicting harmful replies from partial generations

This is a design draft, not a frozen protocol or an implemented experiment. No Phase 3 predictive result is claimed. The question is: **do internal representations make harmful completed replies more predictable than specified baselines given the same prompt and generated prefix?** The [Phase 2 comparison](phase2-results.md) motivates evaluating behavioral prediction rather than treating distance from base as a misalignment score.

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

## Proposed first scope

**Proposal awaiting confirmation:** predict harmfulness anywhere in the completed visible reply, rather than annotating where harmfulness first appears. This reuses whole-response judgments and keeps annotation manageable. It does not establish prediction before harm is already visible; that stronger question requires separate prefix or onset labels.

Use fresh, non-medical prompts and several sampled replies per prompt. The prompt manifest must specify exact text, family assignment, partition, and selection rationale before final collection. Broad thematic similarities need manual inspection; a random row split is insufficient. A 200–400-reply budget is an initial possibility, not an approved workload or a statistical power calculation. Determine feasibility from development families before freezing a final test collection. Do not enrich the test set after observing harmfulness labels or model scores.

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

Resolve the whole-response versus future-suffix target first. Then freeze prompt families/partitions, human labeling workload, sampling and length settings, text baseline, extraction layers, primary metric/cutoff, unclear/quality handling, and adequacy/stopping rules. Implement and validate collection on development material before allocating final test data. Test identical-prefix state equality, cutoff alignment, prefix/full-pass agreement, absence of cross-family leakage, and exact artifact reconstruction.

This draft supports implementation planning; it is not a preregistration, successful replication, or novelty claim. The experiment can produce a useful null result if activations fail to improve the chosen baselines, but a small underpowered comparison cannot establish that internal representations never help.
