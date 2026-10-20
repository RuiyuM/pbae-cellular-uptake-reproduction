"""Build the English report from the saved results without retraining models."""
from pathlib import Path
import base64
import json
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = 'https://github.com/RuiyuM/pbae-cellular-uptake-reproduction'
MODELS = ['LM', 'RF', 'GBT', 'NN']
summary = pd.read_csv(ROOT / 'results/table2_comparison.csv')
protocol = json.loads((ROOT / 'results/protocol.json').read_text())
reconstruction = json.loads((ROOT / 'data/processed/reconstruction_metadata.json').read_text())


def select(experiment):
    return summary[summary.experiment == experiment].set_index('model').loc[MODELS]


def value(row):
    return f"{row['mean']:.2f} ± {row['std']:.2f}"


def table(frame):
    return '<div class="scroll">' + frame.to_html(index=False, escape=True, border=0) + '</div>'


def figure(name):
    payload = base64.b64encode((ROOT / 'figures' / name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{payload}" alt="{name}" />'


primary = select('paper17_author_split')
varied = select('paper17_varied_splits')
exact = select('paper17_exact_symbol')
comparison = pd.DataFrame({
    'Model': ['Linear regression (LM)', 'Random forest (RF)', 'Gradient boosting (GBT)', 'Neural network (NN)'],
    'Paper Table 2': [f'{v:.2f}' for v in primary.paper_mae],
    'Fixed split: mean ± SD': [value(primary.loc[m]) for m in MODELS],
    'Varied splits: mean ± SD': [value(varied.loc[m]) for m in MODELS],
    'Fixed split minus paper': [f'{primary.loc[m, "difference_vs_paper"]:+.2f}' for m in MODELS],
})
genes = ['CLTA', 'CLTB', 'CLTC', 'PICALM', 'CAV1', 'CAV2']
cells = ['OVCAR4', 'PANC02', '4T1', 'HDF']
gene_data = pd.read_csv(ROOT / 'data/processed/recovered_six_genes_by_cell.csv')
gene_table = gene_data[gene_data.matching == 'repository_matching'].set_index('CellType').loc[cells, genes].reset_index()
gene_table = gene_table.rename(columns={'CellType': 'Cell type'})
for gene in genes:
    gene_table[gene] = gene_table[gene].map(lambda v: f'{v:.6f}')
exact_table = pd.DataFrame({
    'Model': MODELS,
    'Repository-compatible matching': [value(primary.loc[m]) for m in MODELS],
    'Exact gene-symbol matching': [value(exact.loc[m]) for m in MODELS],
})
diagnostics = pd.read_csv(ROOT / 'results/linear_numerical_diagnostics.csv')
stable = diagnostics[diagnostics.experiment == 'paper17_author_split'].stable_svd_test_mae.mean()
rf_alt = summary[summary.experiment == 'paper17_rf_all_features'].iloc[0]
sources = ''.join(
    f'<li>{cell}: <a href="{data["url"]}">{data["accession"]}</a></li>'
    for cell, data in reconstruction['samples'].items()
)
css = re.search(r'<style>(.*?)</style>', (ROOT / 'report.html').read_text(), re.S).group(1)

body = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>pBAE Cellular Uptake: Reproduction and Table 2 Comparison</title>
<style>{css}
body{{font-family:Arial,Helvetica,sans-serif}}table{{white-space:normal}}
.repository{{background:#edf6f7;border-left:3px solid #087f8c;padding:8px 12px}}
</style></head><body><main>
<div class="eyebrow">REPRODUCTION STUDY · 2026-10-20 · ENGLISH VERSION</div>
<h1>Predicting pBAE Cellular Uptake<br>Gene Recovery, Four Models, and Table 2</h1>
<p class="lead">Missing gene data were recovered and all four models were rerun using the 17 listed features and published final hyperparameters. This is a documented, validated reconstruction, not an exact reproduction of the published experiment.</p>
<p class="repository"><strong>Code, data, and results:</strong> <a href="{REPO}">{REPO}</a></p>
<div class="cards"><div class="card"><b>240</b>Uptake observations</div><div class="card"><b>17</b>Reconstructed inputs</div><div class="card"><b>212 / 212</b>Expression values matched</div><div class="card"><b>16</b>Completed NN runs</div></div>
<h2>Comparison with the Paper's Table 2</h2>
<p>Test mean absolute error (MAE) is measured in uptake percentage points; lower is better. Our ± values are sample standard deviations across five runs, not confidence intervals. Table 2 reports five-run means only.</p>
{table(comparison)}
<p><strong>Fixed split:</strong> the same training, validation, and test samples are used in all five runs (split seed 7); only model seeds change (7-11). <strong>Varied splits:</strong> both the split and model seeds change together (7-11). Each run uses 192 training, 24 validation, and 24 test observations. All runs are retained; none were selected for closeness to the paper.</p>
<p>Fixed-split differences from the paper are {primary.loc['RF', 'difference_vs_paper']:+.2f} for RF, {primary.loc['NN', 'difference_vs_paper']:+.2f} for NN, and {primary.loc['GBT', 'difference_vs_paper']:+.2f} for GBT. With varied splits, GBT and NN have similar means but appreciable variability. This does not establish statistical equivalence to the paper.</p>
{figure('table2_comparison.png')}

<h2>Recovery of the Missing Gene Data</h2>
<p>CLTA, CLTB, CLTC, and PICALM were recovered from the four raw GEO microarrays cited in the paper; CAV1 and CAV2 were also re-extracted. No zeros, random values, or values estimated from a heatmap were used as substitutes.</p>
<ul>{sources}</ul>
<p>Validation against 53 existing genes across four cell types reproduced all 212 expression values in the current author repository, with a maximum absolute error of <code>{reconstruction['validation']['max_absolute_error']:.2e}</code>, using:</p>
<pre>expression = log2(median(gMeanSignal of matched gene spots)
                  / mean(gMeanSignal of all array spots))</pre>
<p>Gene names were matched by case-insensitive substring; AGRIN was mapped to the alias AGRN for HDF. This preprocessing rule was inferred from verifiable data; the authors did not release the corresponding preprocessing script.</p>
{table(gene_table)}
{figure('recovered_gene_expression.png')}
<div class="note"><strong>Key limitation:</strong> this reconstruction matches the current repository's expression data. The recovered CAV1/CAV2 directions in the two mouse cell types disagree with the colors in Figure 3B. The final six-gene matrix and exact probe choices used for the paper remain unavailable. Source-derived values were retained without adjustment to resemble the paper.</div>

<h2>Training Settings Used in This Rerun</h2>
<ul>
<li>Inputs: six polymer proportions, Size, Polydispersity, six genes, and three binary attributes (cancerous, metastatic, and human). Cell identity was used only to join and track records, not as an additional feature.</li>
<li>RF: 75 trees, maximum depth 13. The unreported <code>max_features</code> setting was taken from the repository: <code>sqrt</code>.</li>
<li>GBT: 50 trees, maximum depth 5, learning rate 0.1.</li>
<li>NN: 17 → 41 ReLU → 41 ReLU → 1 linear output; RMSprop with learning rate 0.001, MSE loss, batch size 5, and 100 epochs. The optimizer and batch size came from the authors' final training code.</li>
<li>Normalization statistics were calculated from training data only. The final NN epoch was evaluated, without test-based epoch selection or prediction clipping. Published final hyperparameters were used directly; the historical tuning search was not repeated.</li>
<li>Environment: Python {protocol['environment']['python']}, scikit-learn 1.2.2, TensorFlow/Keras 2.10.0, NumPy 1.23.5, and SciPy 1.10.1; training used the CPU.</li>
</ul>
{figure('nn_learning_curves.png')}

<h2>Unreported Details and Sensitivity Checks</h2>
<p>The methods describe nine NN input nodes, whereas Figure 5 lists 17 features; this rerun uses 17. The paper does not specify seeds or how the five repeats were performed, so both fixed and varied splits are reported. The old supplementary-material URL returned HTTP 404 during retrieval; an alternative publisher URL returned HTTP 403.</p>
<h3>Gene matching</h3>
<p>The repository-compatible substring rule also matches human CLTCL1 when searching for CLTC. A separate exact gene-symbol analysis shows the effect of this choice. The primary analysis was not selected based on test performance.</p>
{table(exact_table)}
<h3>Unreported random forest parameter</h3>
<p>Changing <code>max_features</code> to 1.0 in an additional control gave a fixed-split MAE of {value(rf_alt)}. This setting changes the numerical gap from the paper; tuning toward 13.09 alone cannot establish a successful reproduction.</p>
<h3>Numerical stability of linear regression</h3>
<p>The features are collinear: formulation proportions sum to a constant, and cell-related expression has only four distinct profiles. The original scikit-learn solver produces very large coefficients. The reported LM results are retained, alongside a separate stable-SVD diagnostic; its fixed-split MAE is {stable:.2f}. Library versions and feature ordering can therefore affect the rerun's numerical results.</p>

<h2>Validation and Reproducibility Files</h2>
<p>All 1,536 test predictions were checked against the reported MAEs. Training, validation, and test subsets were disjoint within each of the 16 recorded partitions, and all 16 NN runs completed 100 epochs. Independent runs with identical data and seeds produced identical predictions; reloading a saved NN reproduced its predictions.</p>
<ul>
<li><a href="results/table2_comparison.csv">Full comparison CSV</a> / <a href="results/all_runs.csv">Individual runs</a> / <a href="results/test_predictions.csv">Test predictions</a></li>
<li><a href="data/processed/paper17_repository_matching.csv">Reconstructed 240 × 17 inputs</a> / <a href="data/processed/recovered_six_genes_by_cell.csv">Six-gene expression data</a></li>
<li><a href="data/processed/normalization_audit_212_values.csv">212-value normalization audit</a> / <a href="data/processed/gene_probe_provenance.csv">Probe-level provenance</a></li>
<li><a href="results/protocol.json">Protocol and assumptions</a> / <a href="results/verification.json">Verification record</a> / <a href="README.md">Run instructions</a></li>
</ul>
<footer>Paper: <em>A machine learning approach to predict cellular uptake of pBAE polyplexes</em>, DOI 10.1039/D3BM00741C.
<a href="https://dau.url.edu/bitstream/handle/20.500.14342/4643/Biomat_scie_2023-11_a-machine.pdf?sequence=1">Paper PDF</a> ·
<a href="https://pubs.rsc.org/en/content/articlehtml/2023/bm/d3bm00741c">Journal page</a> ·
<a href="https://github.com/mbhaylett23/pBAE-cellular-uptake-ML">Authors' repository</a>.
Pinned author-repository commit: 15e8e34a062f023cb1844fe5762d32f459bc1ef0.
This is an independent reproduction; the authors were not contacted.<br>
Reproduction repository: <a href="{REPO}">{REPO}</a></footer>
</main></body></html>'''
(ROOT / 'report_en.html').write_text(body, encoding='utf-8')
print(ROOT / 'report_en.html')
