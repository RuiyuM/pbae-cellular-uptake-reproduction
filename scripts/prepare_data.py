"""Recover author-compatible gene preprocessing from the four cited GEO arrays.

No uptake outcomes are used in recovering the preprocessing rule.
The rule is validated against ALL 53 available genes in each of four cell types.
"""
from pathlib import Path
import gzip
import json
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GENES = ['CLTA', 'CLTB', 'CLTC', 'PICALM', 'CAV1', 'CAV2']
MATERIAL = ['C6H', 'C6R', 'C6K', 'C32H', 'C32R', 'C32K', 'Size', 'Polydispersity']
CELL_FEATURES = ['CellType_cancerous', 'CellType_metastatic', 'CellType_human']
SAMPLES = {
    'OVCAR4': ('GSM564167', 'GSM564167.txt.gz', [1, 1, 1]),
    'PANC02': ('GSM1406256', 'GSM1406256_EE.txt.gz', [1, 0, 0]),
    '4T1': ('GSM1529765', 'GSM1529765_4T1_parental_D.txt.gz', [1, 1, 0]),
    'HDF': ('GSM613188', 'GSM613188.txt.gz', [0, 0, 1]),
}

def read_array(path):
    with gzip.open(path, 'rt', errors='replace') as stream:
        header = next(i for i, line in enumerate(stream) if line.startswith('FEATURES\t'))
    data = pd.read_csv(path, sep='\t', skiprows=header)
    data['symbol'] = data.GeneName.fillna('').str.upper()
    assert np.isfinite(data.gMeanSignal).all() and (data.gMeanSignal > 0).all()
    return data

def gene_value(array, gene, matching):
    query = gene
    if gene == 'AGRIN' and not array.symbol.str.contains('AGRIN', regex=False).any():
        query = 'AGRN'  # documented symbol alias in the HDF array
    mask = (array.symbol == query) if matching == 'exact_symbol' else array.symbol.str.contains(query, regex=False)
    selected = array.loc[mask]
    if selected.empty:
        raise ValueError(f'No probes for {gene}')
    value = np.log2(selected.gMeanSignal.median() / array.gMeanSignal.mean())
    return float(value), selected, query

def main():
    out = ROOT/'data/processed'
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT/'sources/author_repository/GeneDataforModel.csv'
    original = pd.read_csv(source)
    assert original.shape == (240, 64) and not original.isna().any().any()
    assert (original.groupby('CellType').size() == 60).all()
    shared_genes = list(original.columns[11:])
    reference = original.groupby('CellType').first()
    audits, probes, cell_rows = [], [], []
    versions = {name: original[['Var1', 'CellType', *MATERIAL, 'Uptake']].copy()
                for name in ['repository_matching', 'exact_symbol']}
    for cell, (acc, filename, phenotype) in SAMPLES.items():
        array = read_array(ROOT/'data/raw'/filename)
        for gene in shared_genes:
            value, selected, query = gene_value(array, gene, 'repository_matching')
            expected = float(reference.loc[cell, gene])
            audits.append(dict(cell=cell, accession=acc, gene=gene, query=query,
                               published_repository_value=expected, reconstructed_value=value,
                               difference=value-expected, matched_rows=len(selected)))
        for matching, frame in versions.items():
            row = dict(CellType=cell, accession=acc, matching=matching,
                       normalization_denominator=float(array.gMeanSignal.mean()))
            for gene in GENES:
                value, selected, query = gene_value(array, gene, matching)
                row[gene] = value
                frame.loc[frame.CellType == cell, gene] = value
                for _, p in selected.iterrows():
                    probes.append(dict(CellType=cell, accession=acc, matching=matching,
                                       requested_gene=gene, actual_gene=p.GeneName, probe=p.ProbeName,
                                       feature_number=int(p.FeatureNum), mean_signal=float(p.gMeanSignal)))
            for feature, value in zip(CELL_FEATURES, phenotype):
                frame.loc[frame.CellType == cell, feature] = value
            cell_rows.append(row)
    audit = pd.DataFrame(audits)
    audit.to_csv(out/'normalization_audit_212_values.csv', index=False)
    assert len(audit) == 212 and audit.difference.abs().max() < 1e-10
    pd.DataFrame(probes).to_csv(out/'gene_probe_provenance.csv', index=False)
    pd.DataFrame(cell_rows).to_csv(out/'recovered_six_genes_by_cell.csv', index=False)
    features = MATERIAL + GENES + CELL_FEATURES
    for name, frame in versions.items():
        assert not frame[features].isna().any().any()
        assert (frame.Uptake.to_numpy() == original.Uptake.to_numpy()).all()
        frame[['Var1', 'CellType', *features, 'Uptake']].to_csv(out/f'paper17_{name}.csv', index=False)
    metadata = {
        'original_data_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'normalization': 'log2(median(gMeanSignal of selected spots) / mean(gMeanSignal of ALL array spots, including controls))',
        'author_matching_inferred': 'Case-insensitive substring search of GeneName; AGRIN -> AGRN alias for HDF',
        'validation': {'matched_values': 212, 'total_values': 212,
                       'max_absolute_error': float(audit.difference.abs().max())},
        'features': features,
        'important_limitations': [
            'The recovered rule is validated against the current repository, not an unpublished final-paper input matrix.',
            'Substring CLTC matching also includes CLTCL1 in the two human arrays; exact_symbol is supplied as a sensitivity analysis.',
            'Reconstructed mouse CAV1/CAV2 directions do not agree with the colors in published Figure 3B. No values are altered to fit the figure.',
            'Each cited GEO accession is a proxy expression sample, not the same cells measured in the uptake experiment.',
        ],
        'samples': {k: {'accession': v[0], 'url': 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc='+v[0]}
                    for k, v in SAMPLES.items()},
    }
    (out/'reconstruction_metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata['validation'], indent=2))
    print(pd.DataFrame(cell_rows).to_string(index=False))

if __name__ == '__main__':
    main()
