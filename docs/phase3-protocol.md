# Phase 3 draft: predicting harmful replies from partial generations

Phase 3 asks whether internal representations improve prediction of whole-reply harmfulness when both predictors receive the same prompt and generated prefix. The Phase 2 comparison motivates evaluating behavioral prediction directly because benign fine-tuning produced comparable activation distances. All three 40-reply development batches were collected and labeled. The held-out evaluation below is a proposal. [Phase 2 comparison](phase2-results.md)

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

Use fresh, non-medical prompts and several sampled replies per prompt. The prompt manifest must specify exact text, family assignment, partition, and selection rationale before final collection. Manually inspect thematic similarities across partitions. Freeze the prompt set and budget before collecting replies for human review.

The first development batch contains 40 EM replies, with four samples from each of ten project-authored prompts across five development families. The exact text, grouping, and settings are in the [frozen manifest](../configs/phase3-development.json). Themes overlap earlier development material; fresh wording does not make these held-out test families. Use this batch to check truncation, rubric applicability, harmful-example availability, and extraction correctness. It is not a held-out performance evaluation, and its replies and families must stay outside the final test partition. A batch with too few clear harmful examples is a feasibility result; do not quietly replace it with handpicked harmful replies. Review the outcome before committing to a larger labeling workload.

The [second](../configs/phase3-development-batch2.json) and [third](../configs/phase3-development-batch3.json) frozen manifests each specify 40 replies across ten ordinary questions. These batches broaden development coverage before the primary cutoff is chosen. Topic names are provisional. Group related prompts across batches when fitting predictors and exclude all related development families from final test evidence.

## Run the development collection

The [collector](../scripts/run_phase3_development.py) uses the Phase 1 [pinned dependencies](../requirements.txt), a CUDA GPU, float32 weights, and eager attention. Temperature is 1, top-p is 1, top-k is 0, repetition penalty is 1, and the cap is 256 new tokens. The longer cap is a development choice to check whether truncation improves over the earlier 120-token budget, not a guarantee of complete replies. Each sample gets a distinct seed: manifest seed plus four times the prompt index plus sample index. The runner does not train another adapter, fit a probe, or supply behavioral labels.

```sh
python scripts/run_phase3_development.py --validate-only
python -u scripts/run_phase3_development.py --output-dir outputs/phase3-NEW-RUN
```

The first command validates without loading models. The collector rejects draft manifests and existing output directories. A failed run requires a fresh destination; automatic resumption is not implemented. Use `--device cpu` only for an explicitly chosen slow CPU run. A changed manifest defines a different run and must be recorded before collecting its outputs.

Select the relevant manifest explicitly for later batches:

```sh
python scripts/run_phase3_development.py --prompts configs/phase3-development-batch2.json --validate-only
python -u scripts/run_phase3_development.py --prompts configs/phase3-development-batch2.json --output-dir outputs/phase3-batch2-NEW
python scripts/run_phase3_development.py --prompts configs/phase3-development-batch3.json --validate-only
python -u scripts/run_phase3_development.py --prompts configs/phase3-development-batch3.json --output-dir outputs/phase3-batch3-NEW
```

All three batches use the same model, precision, generation settings, cutoffs, and rubric. The second batch uses seeds 40–79 and reviewer IDs `E01`–`E40`. The third uses seeds 80–119 and reviewer IDs `F01`–`F40`. Each batch retains its own mapping and output directory. Preserve run provenance when joining records.

For Colab, build a standalone notebook locally and upload it to a separate GPU notebook:

```sh
python scripts/prepare_phase3_colab.py outputs/phase3-NEW.ipynb
python scripts/prepare_phase3_colab.py outputs/phase3-batch2-NEW.ipynb --prompts configs/phase3-development-batch2.json --run-name phase3-development-batch2
python scripts/prepare_phase3_colab.py outputs/phase3-batch3-NEW.ipynb --prompts configs/phase3-development-batch3.json --run-name phase3-development-batch3
```

The [notebook builder](../scripts/prepare_phase3_colab.py) bundles only the runner, development manifests, rubric, pinned requirements, and extraction tests. It preserves the reference manifests and writes the chosen input separately as `configs/selected-development.json`; both validation and collection explicitly use that selected input. Select a T4 runtime and Run all. The notebook installs dependencies, runs eight focused tests (including a tiny randomly initialized Qwen without pretrained downloads), validates the manifest, collects the real EM replies, and creates a ZIP. It also attempts Colab's standard download; if the browser cannot retain that download, download the ZIP manually from the Files sidebar. Retain the archive before the runtime expires. The generated notebook and run artifacts stay outside version control; no existing human reviewer output is bundled.

The notebook removes Colab's optional `torchvision` package after installing the pinned text-model dependencies. A mismatched preinstalled vision extension can prevent importing Qwen with `RuntimeError: operator torchvision::nms does not exist`; this text-only experiment does not require it. Child-process output is streamed into notebook cells so dependency or model failures remain inspectable. No inference check is bypassed to accommodate the environment.

The collection saves exact prompt and emitted token IDs (including the first EOS), content token IDs excluding EOS, decoded replies, seeds, stopping reasons, and one activation tensor per reply. Each tensor contains all available cutoffs and all 25 hidden-state indices with width 896: embeddings first, intermediate states next, and the final state after final RMSNorm. This convention follows [Hugging Face's pinned Qwen2 implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2/modeling_qwen2.py). Generation controls follow the pinned [Hugging Face generation API](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation). No next-token-logits baseline is collected in this development implementation.

Saved features come from prefix-only forward passes. At each cutoff, each layer's vector is checked against the corresponding full-sequence position: L2 error must be at most `1e-6 + 1e-5 * prefix_vector_L2_norm`. The maximum absolute coordinate error and maximum layer-relative L2 error per cutoff are retained. This norm-based criterion accommodates float32 roundoff without unstable coordinate-relative comparisons near zero. It was chosen during development diagnosis after an initial real-model check failed with maximum absolute error `6.48e-5` but maximum layer-relative L2 error `2.76e-6`, before any replies were accepted or labeled. Repeated exact prefixes must still yield exactly equal saved states, and activation serialization must round-trip exactly. Finite-value and architecture checks fail the run rather than silently retaining invalid features. Metadata retain source/manifest/rubric hashes, chat template, model revisions, resolved generation configuration, and environment. The checks report empty replies, token-cap counts, and availability at every cutoff without screening or resampling their content.

The `review/` directory contains randomized items, a separate mapping, and blank label rows. IDs are `D01`–`D40` by default, or use the manifest's one-letter `review_prefix` (`E` for the second batch). The worksheet reuses the rubric's labeling instructions without the original 12-item pilot introduction. All items come from EM, so this is not a blind comparison of model identities; activation values and stopping metadata are omitted from the worksheet. `checksums.json` covers immutable artifacts and the review mapping, excluding editable `review.md` and `labels.csv`. Preserve filled judgments locally with their rubric snapshot and mapping; do not commit reviewer output without explicit owner approval.

## Development collection verification

The November 10, 2026 Colab T4 collection completed all 40 replies, with no empty replies or token-cap stops and all six cutoffs available for every reply. The executed runner is the version in commit `b7145ee`. Six focused tests passed in the pinned Colab environment before collection; the local test suite passed all 16 tests. The retained ZIP is 19,867,957 bytes with SHA-256 `ce161032fc7256c0d3d8b77070cea9577748541cc73346263295e7fc5764315e`.

After manual download, local checks verified the archive hash and ZIP CRCs, all 49 manifest entries and the source/rubric/manifest hash chain, 40 finite float32 activation tensors of shape `[6, 25, 896]`, prompt/sample/seed identities, EOS and cutoff alignment, repeated-prefix equality, embedding consistency, and the randomized review mapping. The maximum recorded layer-relative L2 prefix/full error was `5.43e-6`; local checks inspect saved error reports rather than independently rerunning the model's full forward passes. All 40 judgment rows were blank at collection verification. These are collection and artifact-consistency checks, not a predictive result or a behavioral safety assessment. Raw run artifacts remain local and are not publicly released, so the historical run is not independently auditable from this repository alone.

The second development batch also completed on November 10, 2026 on a Colab T4, using the collector in commit `a13d1a1` and its separate frozen manifest. All 40 replies were nonempty and stopped before the token cap, with all six cutoffs available. Eight focused Colab tests and 18 local tests passed. The second ZIP is 19,868,506 bytes with SHA-256 `bf11f39adbc3a15f30fa31c583c6756c8563391818f48295006b00e1ce9a69e5`. After manual download, the same local consistency checks passed for its 49 manifest entries, 40 finite `[6, 25, 896]` tensors, provenance, seeds, stopping/cutoff alignment, repeated prefixes, embeddings, and `E01`–`E40` mapping. Its maximum recorded layer-relative L2 error was `4.29e-6`; judgment rows were blank at verification. These computational checks carry the same limitations as the first batch, and do not constitute behavioral labels or a predictive evaluation.

The third development batch ran on November 11, 2026 on a Colab T4 using the same collector and its third frozen manifest. Eight focused extraction tests passed before collection. The runtime reported 40 replies, zero empty replies, and zero token-cap stops. Local verification matched the checksum for the 19,867,294-byte archive and checked all 49 manifest entries, 40 activation tensors, all six cutoffs, and the review mapping. The archive SHA-256 is `cdfb8593cdc2be7bed54bb1e46cfb80a5cd3d10d56cb82c87223625ed2b63e46`. Human review is complete.

## Development predictor dry run

After human review of both batches, a local dry run joined the judgments to their generation records and compared predictors at all six cutoffs on a common eligible cohort. It held out one development prompt family per fold after manually grouping related prompts. Both predictors used fixed L2-regularized logistic regression and the same fold-trained TF-IDF features over token ID unigrams and bigrams from the exact templated prompt and observed prefix. The augmented predictor also received the unit-normalized final hidden state at the matched cutoff. A training-prevalence baseline provided a reference, and pooled out-of-fold predictions were scored with log loss and Brier score.

The dry run checked the label-to-reply joins, absence of exact prompt and assigned family overlap between training and held-out rows, and agreement of saved states and predictions for identical token contexts. A second local run reproduced the report. The development batches and their prompt families are excluded from the final test set. The raw artifacts, judgments, and dry-run scores are unpublished, so readers cannot independently verify the historical dry run from this repository. The public collector and manifests support a fresh development collection.

The semantic text baseline exercise reused the same rows, families, and cutoffs on a local CPU. The exact Qwen token contexts were decoded with the generation tokenizer and verified by round-trip encoding. The frozen encoder produced unit-normalized, mean-pooled embeddings without truncation. The comparison added the same final-layer activation to a text predictor using both token ID TF-IDF and the frozen embeddings. This exercise leaves the final text representation and primary cutoff open. [Pinned MiniLM checkpoint](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41)

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

The primary binary analysis includes only clear harmfulness labels (`yes`/`no`) on replies labeled fully coherent (`yes`). A separate sensitivity analysis expands eligibility to `partly` coherent replies while retaining the same clear-label requirement. Keep `unclear`, incoherent, and language-unassessable records in storage and report exclusions by partition; never turn unclear labels into negatives or a language barrier into evidence of incoherence. Preserve initial judgments and subsequent clarifications. These eligibility rules were settled during development, before final collection or predictive testing.

This primary analysis estimates performance among replies judged fully coherent; it does not establish performance on all generated replies or eliminate every response-quality difference. Full-response coherence is retrospective annotation, so it cannot serve as a deployable early predictor or an input feature. Any prefix-quality predictor would need prefix-only inputs or labels. Completeness remains a separate annotation; no additional completeness exclusion has been selected. Report quality-stratified results separately and never revise human labels to improve separation.

## Proposed held-out evaluation

Use **1 generated token** for the primary comparison and **3 tokens** for a secondary check. These cutoffs showed lower development log loss with activations under both treatments of tentative labels. One token addresses the earliest nonzero cutoff. The target remains harmfulness anywhere in the recorded reply.

Fit both classifiers on eligible replies from the three existing development batches, omitting the two tentative positives for the primary analysis. Use fixed L2 logistic regression with `C=1.0`, an intercept, and unit-normalized feature blocks. The text baseline concatenates token ID unigram and bigram TF-IDF with the pinned, mean-pooled MiniLM embedding described above. Fit TF-IDF on development rows only. The augmented classifier adds the unit-normalized final hidden state after final RMSNorm. At each cutoff both arms use exactly the same rows and token context. Use the pinned generation model and the development sampling settings, including the 256-token cap.

Propose a fixed budget of **240 replies**, with four samples for each of 60 new prompts across at least 30 distinct prompt families and at most two prompts per family. Label all 240 in six blocks of 40, using the existing rubric and keeping predictor scores hidden during review. Freeze the prompts, family assignments, review rules, and sampling seeds before collection. Audit family overlap against all development and calibration material.

The primary comparison requires **at least 12 eligible harmful replies across at least 8 held-out prompt families**, plus at least 12 eligible non-harmful replies across at least 8 families. Use the cohort eligible at both 1 and 3 tokens, and report attrition and class counts. For the primary analysis, count only definite judgments on fully coherent replies. Have the reviewer flag tentativeness before seeing scores. Report tentative judgments separately. The 12-reply and 8-family floors are proposed safeguards against relying on a few examples or families. They do not establish statistical power.

At the conservative development yield of about 5 eligible harmful replies per 100 generated, 240 replies would yield about 12 on average if that rate carries over. An illustrative independent-binomial calculation gives about a 54% chance of reaching 12 replies, before applying the family requirement. This calculation ignores within-family dependence and uncertainty about the rate on fresh prompts. Treat the full workload as a feasibility gamble and report insufficient evidence if either class or family floor is missed at the fixed budget.

Report the paired mean held-out log-loss difference (augmented minus text-only). Use a 95% percentile bootstrap interval with 10,000 resamples of whole held-out families and a fixed seed of 0, pooling the replies within sampled families. An interval entirely below zero favors the augmented classifier. An interval entirely above zero favors the text-only classifier. An interval that includes zero leaves the direction uncertain. Summarize Brier score, the 3-token comparison, and an analysis including partly coherent replies as secondary checks.

Before generation, confirm the annotation budget and the adequacy floors, freeze the exact held-out manifest, and implement and test the held-out collector and scorer. The existing collector validates fixed 40-reply development manifests.
