# Multimodal active learning for materials-specific Ueff prediction

This review release packages the code and processed inputs underlying the supervised modality-ablation and retrospective active-learning experiments. It includes complete numerical reference results, rather than only the contrasts displayed in the main figure. Release identifier: `v0.1.0-review`.

Author: XiaZhiyuan. ORCID: https://orcid.org/0009-0009-8897-1883. Intended repository name: `SunGroupAI`; actual repository URL and archive DOI have not yet been assigned.

## Scope and evidence

| Workflow | Input | Records | Seeds | Method |
| --- | --- | ---: | --- | --- |
| Supervised benchmark | `data/processed/Ueff_all_U_test_enhanced_modalities.csv` | 180 | 0, 1, 2 | modality experts, late stacking, element+text backbone with structural residual |
| Retrospective active learning | `data/processed/Ueff_clean_main.csv` | 126 | 0–4 | ExtraTrees, early concatenation; random, uncertainty, diversity and hybrid policies |

These are different dataset versions and different model pipelines. They must not be described as one common train/test experiment. The finalized manuscript label inventory (185 records) is not the training dataset of either experiment. This release does not run DFT, generate new linear-response labels, reproduce downstream catalysis calculations or reproduce every figure in the manuscript.

`comp`, `struct` and `electronic` denote composition/elemental, structural and numeric chemical-prior feature blocks in the active-learning code. The active-learning `electronic` block is not the TF-IDF/SVD text block used in supervised stacking. See `docs/REPRODUCIBILITY.md` for the exact definitions and limitations.

## Installation

Python 3.10 is the recorded local verification runtime. From the extracted release root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/verify_release.py
```

`requirements.txt` records the versions available during packaging verification. Historical results do not have a complete original environment lock; see `environment.json`. No GPU or external API is required for the core processed-feature workflows.

## Quick example

```bash
python scripts/reproduce.py --mode smoke
```

The smoke example fits all seven active-learning representations, one seed and one acquisition round. Outputs go to `outputs/smoke/`; this is a functional example, not a replacement for the published results. Recorded timings are in `validation_report.json` after verification.

## Reproduction commands

```bash
# Recompute figure-level numeric summaries from archived results (no training)
python scripts/reproduce.py --mode summary

# Supervised variant search on the provided inputs, seeds 0–2
python scripts/reproduce.py --mode supervised

# Full retrospective active learning: seven representations, four policies, seeds 0–4
python scripts/reproduce.py --mode active-learning

# Recreate the main figure from archived inputs and results
python scripts/reproduce.py --mode figure
```

Fresh training writes to `outputs/`. Archived reference results remain read-only inputs. Full training may take substantially longer than the smoke example. Full end-to-end retraining was not performed during packaging; passing the smoke example does not establish exact numerical reproducibility of historical results.

## Files

- `scripts/search_u_test_trifusion.py`: supervised representation and fusion search.
- `scripts/run_active_learning_ablation.py`: retrospective active-learning simulation.
- `scripts/select_active_learning_candidates.py`: generic uncertainty/diversity/cost ranking from externally supplied pool predictions, not the model used to generate the archived benchmark.
- `scripts/plot_fig3_grouped_representation_framework.py`: main figure and complete supervised ablation plot.
- `scripts/reproduce.py`, `scripts/verify_release.py`: portable entry points and input/hash checks.
- `data/processed/`: processed feature/label tables used by the two experiments.
- `processed/u_knowledge_cards.csv`: only the identity and embedding-text fields used in the supervised experiment.
- `processed/cgcnn_datasets/`: recorded training, validation and test identities, with embedded CIF content.
- `experiments/cgcnn_u_test_all_structures_20260623/`: seed-specific pretrained structural embeddings and metadata.
- `experiments/u_test_trifusion_search_enhanced_graph_20260623/`: complete supervised search results and selected-representation table.
- `results/active_learning_ablation/`: complete archived learning curves, selected samples and metadata.
- `analysis_multimodal_AL/modality_by_seed.csv`: supervised seed-level plot input.
- `figures/paper_multimodal_al/`: figure source data and archived vector figures.
- `scripts/provenance/`: original feature-construction and CGCNN-training code supplied for traceability, not a self-contained raw-data pipeline.
- `config/matplotlibrc`: export-safe snapshot of the author's configured publication style; the wrapper sets it for figure commands.
- `manifest.json`, `SHA256SUMS.txt`: included-file checksums and transformations.

## Interpretation of the main figure

The main figure selects the highest mean supervised MAE in panel a, the lowest mean final error–uncertainty Spearman correlation in b, the highest mean learning-curve AULC in c and the highest mean final MAE in d within each reduced-representation category. Configurations can differ between panels. These are descriptive maximum-degradation contrasts, not category averages or comparisons against the strongest baseline. The complete results are included. Panel c averages percentage reductions computed within matched seeds; panels a, b and d annotate ratios of aggregate means.

## Citation, licence and availability

Author identity is recorded above; affiliations, coauthors, GitHub URL and archive DOI remain `NEED_DATA`. The author chose a review package without a new open-source licence. This is a prepared review package, not a claim that code has already been publicly deposited. No new software or data licence has been applied on the authors' behalf. Complete `docs/RELEASE_CHECKLIST.md` before publishing.

Code-availability statement after deposition:

> The code for the supervised multimodal benchmark, retrospective active-learning simulations and figure-level analyses is available at [GITHUB_URL]. The accompanying processed inputs, fixed split files, reference outputs and reproduction instructions are archived as version [RELEASE_VERSION] at [ARCHIVE_DOI].

Data sharing should also be covered in a separate Data Availability statement. Redistribution rights for the structures, chemical-prior text and pretrained embeddings require author confirmation.
