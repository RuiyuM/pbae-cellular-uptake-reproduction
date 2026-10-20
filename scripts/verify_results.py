"""Independent checks on saved predictions, partitions and model artifacts."""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error

def main():
    out = ROOT/'results'
    runs = pd.read_csv(out/'all_runs.csv')
    pred = pd.read_csv(out/'test_predictions.csv')
    splits = pd.read_csv(out/'split_membership.csv')
    hist = pd.read_csv(out/'nn_training_history.csv')
    audit = pd.read_csv(ROOT/'data/processed/normalization_audit_212_values.csv')
    assert len(runs) == 69 and len(pred) == 1536 and len(hist) == 1600
    assert len(audit) == 212 and audit.difference.abs().max() < 1e-10
    for run, membership in splits.groupby('run_id'):
        assert len(membership) == 240 and membership.row_index.nunique() == 240
        assert membership.groupby('split').size().to_dict() == {'test':24, 'train':192, 'validation':24}
    metric_errors = []
    for (run, model), values in pred.groupby(['run_id','model']):
        assert len(values) == 24 and values.row_index.nunique() == 24
        saved = runs.loc[(runs.run_id == run)&(runs.model == model), 'test_mae'].iloc[0]
        recomputed = float((values.observed-values.predicted).abs().mean())
        metric_errors.append(abs(saved-recomputed))
        assert abs(saved-recomputed) < 1e-10
    for run, h in hist.groupby('run_id'):
        assert sorted(h.epoch) == list(range(1,101)) and np.isfinite(h[['loss','mae','val_loss','val_mae']]).all().all()
    left = pred[pred.run_id == 'paper17_author_split_split7_seed7'].sort_values(['model','row_index'])
    right = pred[pred.run_id == 'paper17_varied_splits_split7_seed7'].sort_values(['model','row_index'])
    assert np.array_equal(left.predicted.to_numpy(), right.predicted.to_numpy())

    # Diagnose the ordinary least-squares implementation under collinear features.
    # This is a separately labeled numerical check, not a replacement for reported LM.
    diag = []
    for experiment, filename in [('paper17_author_split','paper17_repository_matching.csv'),
                                 ('paper17_varied_splits','paper17_repository_matching.csv'),
                                 ('paper17_exact_symbol','paper17_exact_symbol.csv')]:
        frame = pd.read_csv(ROOT/'data/processed'/filename)
        for _, row in runs[(runs.experiment==experiment)&(runs.model=='LM')].iterrows():
            members = splits[splits.run_id==row.run_id]
            tr = members[members.split=='train'].row_index.to_numpy()
            te = members[members.split=='test'].row_index.to_numpy()
            scale = np.load(out/'models'/f'{row.run_id}_normalization.npz')
            x=(frame[list(scale['features'])].to_numpy()-scale['mean'])/scale['std']
            y=frame.Uptake.to_numpy(); xm=x[tr].mean(0); ym=y[tr].mean()
            coefs,_,rank,_=np.linalg.lstsq(x[tr]-xm,y[tr]-ym,rcond=1e-12)
            stable=(x[te]-xm)@coefs+ym
            native=joblib.load(out/'models'/f'{row.run_id}_LM.joblib')
            diag.append(dict(run_id=row.run_id, experiment=experiment,
                             sklearn_rank=int(native.rank_), stable_rank=int(rank),
                             max_abs_sklearn_coefficient=float(abs(native.coef_).max()),
                             sklearn_test_mae=row.test_mae,
                             stable_svd_test_mae=float(mean_absolute_error(y[te],stable))))
    pd.DataFrame(diag).to_csv(out/'linear_numerical_diagnostics.csv',index=False)

    # Reload one saved neural network and check it reproduces stored predictions.
    from tensorflow import keras
    run='paper17_author_split_split7_seed7'
    model=keras.models.load_model(out/'models'/f'{run}_NN.h5',compile=False)
    selected=pred[(pred.run_id==run)&(pred.model=='NN')]
    frame=pd.read_csv(ROOT/'data/processed/paper17_repository_matching.csv')
    scale=np.load(out/'models'/f'{run}_normalization.npz')
    x=(frame[list(scale['features'])].to_numpy()-scale['mean'])/scale['std']
    reloaded=model(x[selected.row_index].astype(np.float32),training=False).numpy().ravel()
    reload_error=float(np.max(np.abs(reloaded-selected.predicted.to_numpy())))
    assert reload_error < 1e-5
    checks=dict(status='PASS', main_model_runs=64, extra_rf_runs=5, neural_network_runs=16,
                nn_epochs_per_run=100, test_predictions=1536, partitions=16,
                normalization_values_checked=212, max_metric_recalculation_error=max(metric_errors),
                duplicate_seed_predictions_identical=True, neural_model_reload_max_error=reload_error,
                neural_network_parameter_count=int(model.count_params()),
                numerical_note='Collinear inputs cause unstable native sklearn 1.2.2 LM coefficients; separate stable-SVD diagnostics supplied.')
    (out/'verification.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))
    print(pd.DataFrame(diag).groupby('experiment').stable_svd_test_mae.agg(['mean','std']).to_string())

if __name__=='__main__':
    main()
