# EDC seminar work: fuzzy prioritisation of ZFR candidates

This project contains a self-contained Python implementation and LaTeX source
for a seminar work on decision support in Z-DNA candidate prioritisation in
generic genome-scale candidate sets.

## Structure

- `zadanie.md` - proposed seminar assignment.
- `src/zfr_fuzzy_expert.py` - dependency-free fuzzy expert system and experiment runner.
- `src/generate_benchmark.py` - deterministic benchmark generator.
- `src/prepare_shin_real_validation.py` - prepares the Shin et al. per-locus validation set from `data/shin_data`.
- `data/sample_candidates.csv` - benchmark with 320 candidate regions, train/validation/test split, candidate length 8-400 bp and normalised `zdna_score` in 0-1.
- `data/shin_real_candidates.csv` - real per-locus Shin validation candidates prepared from Z-DNA Hunter model1/model2 overlay files, original Shin FASTA loci, CpX Hunter exports and TSS distances from `data/shin_data/Hum_ENSEMBL52.sga`; ZDNABERT is retained only as a comparison row.
- `outputs/` - generated experiment comparison, ranking, audit report and PDF visualisations.
- `paper/main.tex` - LaTeX seminar paper.
- `paper/references.bib` - bibliography.

## Run

```bash
python3 src/generate_benchmark.py
python3 src/prepare_shin_real_validation.py
python3 src/zfr_fuzzy_expert.py
```

The commands write:

- `outputs/priority_ranking.csv`
- `outputs/experiment_summary.csv`
- `outputs/experiment_details.csv`
- `outputs/rule_coverage_summary.csv`
- `outputs/ablation_summary.csv`
- `outputs/uncertainty_summary.csv`
- `outputs/shin_real_evaluation.csv`
- `outputs/shin_real_predictions.csv`
- `data/shin_real_selected_sources.csv`
- `outputs/rules_audit.txt`
- `outputs/zfr_priority_report.md`
- `outputs/membership_functions.pdf`
- `outputs/experiment_quality.pdf`
- `outputs/prediction_vs_expert.pdf`
- `outputs/ablation_impact.pdf`
- `outputs/uncertainty_intervals.pdf`
- `outputs/shin_real_comparison.pdf`
- `outputs/shin_validation_trajectory.pdf`


## Submission package

```bash
make pack
```

The target regenerates the Shin dataset, experiment outputs, LaTeX PDF and
creates `dist/edc_spr_submission.zip`.

For a smaller submission package with only the ready-made datasets, outputs,
source needed to rerun the expert model and paper files:

```bash
make pack-ready
```

This target does not regenerate or curate datasets and creates
`dist/edc_spr_submission_ready.zip`.

## Git LFS

Large raw CSV files under `data/shin_data/` are marked for Git LFS in
`.gitattributes`. Before adding these files to Git, install and initialise LFS:

```bash
brew install git-lfs
git lfs install
git add .gitattributes data/shin_data
git lfs ls-files
```

If any large CSV files were already committed without LFS, migrate them before
pushing with:

```bash
git lfs migrate import --include="data/shin_data/**/*.csv"
```
