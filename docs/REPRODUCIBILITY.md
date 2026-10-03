# Reproducibility and provenance notes

## Workspace inventory inspected before packaging

`scripts/` contains the current supervised, retrospective AL, ranking and plotting entry points. `data/processed/` contains the two input CSV tables. `processed/` contains chemical-prior text and recorded CGCNN split files. `experiments/` stores supervised variant searches and pretrained CGCNN embeddings. `results/active_learning_ablation/` stores the retrospective AL reference experiment. `analysis_multimodal_AL/` contains derived audits and seed-level plotting data. `figures/paper_multimodal_al/` contains presentation and source-data outputs. `SCR/` contains an older reproducible workflow and is not the source of the current Fig. 3 results. Third-party nested repositories, raw label files, notebooks, manuscripts and full DFT outputs are excluded from the upload package.

## Input transformations

The builder reads source files without modifying them. It drops absolute path metadata from copied feature tables, while preserving row order, identities, labels and every numeric/model feature. Only `material_id` and `text_summary_for_embedding` are retained from chemical-prior cards, because these are the fields consumed by the supervised experiment. Split CSVs and embedding arrays are retained. Check `manifest.json` for per-file transformations and hashes. Original labels have not been changed to enlarge effects.

## Supervised experiment

- 180 rows; recorded seeds 0, 1, 2; training+validation membership is combined for stacking; the fixed test set contains 36 rows per seed.
- Three modality blocks: element descriptors and category indicators, structure-derived descriptors optionally augmented by pretrained CGCNN embeddings, and chemical-prior text encoded with TF-IDF followed by TruncatedSVD (32 or 64 components).
- TF-IDF/SVD fits on the outer training subset. Imputers and scalers sit inside fitted estimator pipelines.
- Expert candidates: ExtraTrees, histogram gradient boosting and ridge; detailed settings are preserved in `model_templates()` and `fit_meta()`.
- Five folds by default. Residual weights searched over 0.0 to 1.0 in increments of 0.1.
- CGCNN embeddings are learned with training Ueff labels and validation-based checkpoint selection. They are not label-free structure features. The release supplies precomputed embeddings, not a self-contained raw-CIF-to-CGCNN rebuild.
- The original source calls its procedure nested/OOF selection. However, residual alpha is selected on training predictions made after fitting meta learners on the full meta-training data. This is not strictly nested OOF prediction at every meta layer. The code is preserved and this limitation must not be described as a corrected nested evaluation.
- `ablation_best_by_block.csv` selects variants/fusion by their observed aggregate held-out MAE. Therefore the seven-row summary is a descriptive selected result, not an independent test of a pre-registered configuration. Full variant results are included for audit.

## Retrospective AL experiment

- 126 processed records. Filter: `0 < U_eV <= 8.0`; this table already has 126 qualifying records. It is not merely the 180-row table after that filter (179 of the latter pass).
- Five seeds: 0–4. Target-rank quantile-stratified 80/20 train-pool/test split.
- Initial labeled set: 10 rows; batch: 5; rounds: 10, with round zero included. Label budgets: 10–60.
- Each seed's split and initial set are shared across representations and policies.
- Main archived model: ExtraTrees, 60 trees, minimum leaf size 2, `max_features='sqrt'`; early feature concatenation; uncertainty is the standard deviation across tree predictions, not a calibrated confidence interval.
- Hybrid acquisition is `0.65 * minmax(uncertainty) + 0.35 * minmax(diversity)`. Diversity is nearest-labeled Euclidean distance after imputation and standardization fitted on current labeled rows. The archived retrospective benchmark has no labeling-cost term.
- Training, imputation and diversity scaling fit only on current labeled rows each round. Labels of pool rows are revealed after acquisition for training.
- Initial-set stratification uses the simulated pool's known labels. This is an oracle-assisted retrospective setup; a prospective deployment cannot use unmeasured Ueff labels for initialization.
- The current source seeds policies via deterministic BLAKE2b offsets. Historical metadata do not record this offset and the exact historical source snapshot is unavailable. Fresh random/hybrid acquisition runs may not match archived selected sequences. Archived numerical reanalysis remains supported; exact historical retraining is not asserted.

## Optional prospective ranking

`select_active_learning_candidates.py` operates on user-provided pool feature and prediction tables with shared material identities. It combines normalized uncertainty and diversity minus a cost penalty. If cost is absent, it uses a uniform cost and records that condition; no measured DFT timing is implied. This interface is separate from the archived retrospective AL benchmark. No fabricated candidate prediction example is included.

## Figure and numeric reanalysis

Normalized AULC is trapezoidal integration of MAE versus label count divided by the observed budget span (10–60), in eV. For panel c, percentage reduction is `100 * (AULC_baseline - AULC_multimodal) / AULC_baseline` within each paired seed, followed by mean and sample SD. All source values retain full precision; displayed numbers use two significant digits. A change from mean of paired ratios to ratio of means can change the displayed number (6.1% versus 6.0%); both definitions are retained explicitly.

## Verification boundaries

See `validation_report.json` for checks actually performed: file hashes, feature/label integrity, IDs, fixed split disjointness and coverage, embeddings, archived result completeness, numeric reanalysis and smoke-training outputs. A package may pass these checks without establishing unbiased feature construction, independent model selection, prospective efficacy or exact regeneration of historical training outputs. No new scientific performance claim is generated by packaging.
