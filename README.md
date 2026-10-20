# pBAE cellular uptake: reproducible Table 2 rerun

This directory contains an evidence-backed reconstruction and rerun of the four
models in Loecher et al., *Biomaterials Science* (2023), DOI 10.1039/D3BM00741C.
It is a **best-effort reproduction**, not an exact recovery of the unpublished
final-paper data matrix and random seeds. Start with `report.html` for the Chinese
report, figures, results, and limitations.

**Shareable reports:** [English PDF](output/pdf/pbae_table2_reproduction_report_en.pdf) ·
[Chinese PDF](output/pdf/pbae_table2_reproduction_report.pdf) ·
[GitHub repository](https://github.com/RuiyuM/pbae-cellular-uptake-reproduction)

The report is available in English and Chinese; the scripts, figures, and run
instructions are in English. Self-contained HTML versions are `report_en.html`
and `report.html`.

| Model | Paper Table 2 MAE | Rerun, fixed split (mean ± sample SD, n=5) |
| --- | ---: | ---: |
| Linear regression | 14.09 | 16.03 ± 0.00 |
| Random forest | 13.09 | 13.42 ± 0.45 |
| Gradient boosting | 10.57 | 12.79 ± 0.08 |
| Neural network | 11.17 | 12.10 ± 0.43 |

MAE is measured in uptake percentage points; lower is better. See the report
for split sensitivity, recovered-gene ambiguities, and reasons this is not an
exact reproduction of the published experiment.

## Run

Clone this reproduction repository and the original author's pinned source first:

```sh
git clone https://github.com/RuiyuM/pbae-cellular-uptake-reproduction.git
cd pbae-cellular-uptake-reproduction
git clone https://github.com/mbhaylett23/pBAE-cellular-uptake-ML.git sources/author_repository
git -C sources/author_repository checkout 15e8e34a062f023cb1844fe5762d32f459bc1ef0
```

From the repository directory, using Python 3.10:

```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/download_sources.py
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/run_experiments.py
.venv/bin/python scripts/verify_results.py
.venv/bin/python scripts/build_report.py
.venv/bin/python scripts/build_report_en.py
```

Processed inputs, results, and checksums are included. The original author's
repository, raw GEO arrays, publisher PDFs, and fitted model binaries are not
bundled; the steps above retrieve the sources and regenerate the models.
`requirements.lock.txt` records the exact
macOS ARM environment used. TensorFlow 2.10 wheels require an appropriate Python
3.10 environment; other platforms may have different numerical results.

The optional PDF export uses Node.js and Chromium and is separate from training:

```sh
npm install --no-save playwright
npx playwright install chromium
node scripts/export_pdf.cjs
node scripts/export_pdf.cjs --english
```

An existing Chrome can be selected with the `CHROME_PATH` environment variable.
Use a Chinese font such as Noto Sans CJK SC or PingFang SC for correct rendering.

## Reconstruction

The four arrays are GSM564167 (OVCAR4), GSM1406256 (PANC02), GSM1529765 (4T1), and
GSM613188 (HDF proxy). The inferred preprocessing rule is:

```
log2(median(gMeanSignal of matched gene spots) / mean(gMeanSignal of all array spots))
```

Case-insensitive substring matching of GeneName, plus AGRIN -> AGRN for HDF,
reconstructs **212/212** existing CSV gene values with maximum absolute difference
below 1.2e-14. No uptake labels are used to infer or validate this rule.
CLTA, CLTB, CLTC, and PICALM are recovered with the same rule. The six requested
genes plus six material percentages, size, polydispersity, and three binary cell
attributes give 17 input features. Uptake labels are unchanged.

Important limitations:

- Substring CLTC matching includes CLTCL1 in the human arrays. An exact-symbol
  alternative is provided and independently evaluated.
- Reconstructed mouse CAV1/CAV2 values disagree in sign with Figure 3B. The
  current repository's preprocessing does not establish the final-paper matrix.
- The cited public GEO samples are proxies, not expression measured in the same
  cells as the uptake experiment. Their original sample metadata is preserved.
- The journal supplement's old URL returned 404; alternate publisher requests
  returned 403. No unavailable settings are invented.

## Experimental protocol

All settings and assumptions are written to `results/protocol.json` before
training. RF: 75 trees, depth 13, max_features=sqrt (last setting from repository).
GBT: 50 trees, depth 5, learning_rate=0.1. NN: 17 -> 41 ReLU -> 41 ReLU -> 1;
RMSprop 0.001, MSE, batch size 5, 100 epochs, final epoch evaluated.
The paper says 9 NN inputs in methods but lists 17 features in Figure 5; we use 17.
Reported final hyperparameters are used directly; historical tuning is not rerun.

Train/validation/test = 192/24/24. Scale with training-only mean and population
standard deviation. Do not clip predictions or choose models using test outcomes.

- Primary: fixed repository split seed 7, model seeds 7–11 (five repeats).
- Split sensitivity: split and model seeds 7–11 (five repeats).
- Exact-gene sensitivity: fixed split 7, model seeds 7–11 (five repeats).
- Original 61-feature single-run reference: split/model seed 7, including NN.
- RF-only max_features=1.0 sensitivity: five repeats.

These are 64 full model fits, plus 5 RF alternatives; 16 NN fits in total.
All runs are retained. The paper's n=5 repetition procedure is not specified.
Standard deviations are sample SD (ddof=1), not confidence intervals. Fixed-split
SD only reflects algorithm randomness, not uncertainty from selecting test data.
The three main protocols share observations and are not independent studies.

## Outputs and checks

- `data/raw/`: source SOFT metadata and compressed Agilent arrays.
- `data/processed/`: recovered inputs, six-gene tables, 212-value audit, probe records.
- `results/table2_comparison.csv`: means, SDs, individual ranges, paper differences.
- `results/all_runs.csv`, `test_predictions.csv`, `split_membership.csv`: audit trail.
- `results/nn_training_history.csv`: 100 epochs for every NN run.
- `results/models/`: saved models and train-set normalization parameters.
- `results/linear_numerical_diagnostics.csv`: separate stable-SVD OLS checks;
  native sklearn 1.2.2 coefficients are unstable under exact/near collinearity.
- `results/verification.json`: partition, metric, repeatability and model reload checks.
- `figures/`: PNG and SVG plots; `report.html`: self-contained visual report.

## Attribution

This is an independent reconstruction, not the paper authors' official code.
The paper is Loecher et al., *Biomaterials Science* (2023),
[DOI: 10.1039/D3BM00741C](https://doi.org/10.1039/D3BM00741C).
The original code and uptake observations come from
[mbhaylett23/pBAE-cellular-uptake-ML](https://github.com/mbhaylett23/pBAE-cellular-uptake-ML),
at the pinned commit above. Original source files are downloaded unmodified.
Expression measurements come from the four NCBI GEO samples listed above.
URLs and downloaded-file checksums are recorded in `sources/download_manifest.json`;
processed-input checksums are recorded in `results/input_checksums.json`.
The original repository does not contain an explicit license; this repository
does not assign a new license to third-party source code or experimental data.
This study has not been reviewed or endorsed by the paper authors.
