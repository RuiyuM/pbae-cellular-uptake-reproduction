"""Run predeclared Table-2 settings without choosing settings on test outcomes."""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('OMP_NUM_THREADS', '1')
import json
import time
import platform
import hashlib
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow import keras

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.experimental.enable_op_determinism()
PAPER = {'LM': 14.09, 'RF': 13.09, 'GBT': 10.57, 'NN': 11.17}
SEEDS = [7, 8, 9, 10, 11]
metadata = json.loads((ROOT/'data/processed/reconstruction_metadata.json').read_text())
FEATURES = metadata['features']

def split_and_scale(frame, features, seed):
    all_idx = np.arange(len(frame))
    train, remaining = train_test_split(all_idx, train_size=.8, random_state=seed)
    valid, test = train_test_split(remaining, test_size=.5, random_state=seed)
    assert set(train).isdisjoint(valid) and set(train).isdisjoint(test) and set(valid).isdisjoint(test)
    assert sorted(np.concatenate([train, valid, test])) == list(all_idx)
    x = frame[features].to_numpy(dtype=np.float64)
    y = frame.Uptake.to_numpy(dtype=np.float64)
    # Same training-only normalization order as the repository.
    mean = x[train].mean(axis=0)
    std = (x[train]-mean).std(axis=0)
    assert (std > 0).all()
    x = (x-mean)/std
    assert np.isfinite(x).all()
    return x, y, train, valid, test, mean, std

def main():
    out = ROOT/'results'
    out.mkdir(exist_ok=True)
    (out/'models').mkdir(exist_ok=True)
    protocol = {
        'primary': 'paper17_repository_matching, fixed author split random_state=7, five model seeds 7..11',
        'additional_protocols': ['paper17_repository_matching, split seeds 7..11 and matching model seeds',
                                 'paper17_exact_symbol, fixed split=7 and model seeds 7..11',
                                 'repository61 original final settings, split=7, model seed=7; single-run reference',
                                 'RF-only max_features=1.0 sensitivity for paper17 fixed split'],
        'features': FEATURES,
        'split_sizes': [192, 24, 24],
        'normalization': 'training-set column mean and population standard deviation, applied unchanged to validation/test',
        'RF': {'n_estimators': 75, 'max_depth': 13, 'max_features': 'sqrt'},
        'GBT': {'n_estimators': 50, 'max_depth': 5, 'learning_rate': .1},
        'NN': {'input_dimension': 17, 'hidden_layers': [41, 41], 'activation': 'relu',
               'output_units': 1, 'output_activation': 'linear', 'loss': 'mse',
               'optimizer': 'RMSprop', 'learning_rate': .001, 'batch_size': 5,
               'epochs': 100, 'shuffle': True, 'checkpoint': 'final epoch, as in repository'},
        'assumptions': [
            'Paper n=5 does not specify split/model seeds or what was repeated; both repeat interpretations are reported.',
            'RF max_features=sqrt follows the final repository estimator; paper Table 2 only specifies number/depth.',
            'NN optimizer, learning rate and batch size follow the final repository training block.',
            'NN input dimension follows 17 explicitly named Figure-5 features; methods text says 9 and is inconsistent.',
            'No test-based model or epoch selection, no clipping predictions, no missing-value imputation.',
            'Reported final hyperparameters are used directly; the historical hyperparameter searches are not rerun.',
        ],
        'environment': dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                            scipy=scipy.__version__, sklearn=sklearn.__version__, tensorflow=tf.__version__,
                            keras=keras.__version__, platform=platform.platform()),
        'paper_test_mae': PAPER,
    }
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    frames = {key: pd.read_csv(ROOT/f'data/processed/paper17_{key}.csv')
              for key in ['repository_matching', 'exact_symbol']}
    original = pd.read_csv(ROOT/'sources/author_repository/GeneDataforModel.csv')
    rows, predictions, histories, splits = [], [], [], []
    experiments = []
    for seed in SEEDS:
        experiments.append(('paper17_author_split', 'repository_matching', 7, seed))
        experiments.append(('paper17_varied_splits', 'repository_matching', seed, seed))
        experiments.append(('paper17_exact_symbol', 'exact_symbol', 7, seed))
    experiments.append(('repository61_reference', 'original', 7, 7))
    # Run complete experiments, preserving every trial rather than the best result.
    for experiment, dataset, split_seed, model_seed in experiments:
        is_original = dataset == 'original'
        frame = original if is_original else frames[dataset]
        features = list(original.columns[1:7])+['Size', 'Polydispersity']+list(original.columns[11:]) if is_original else FEATURES
        x, y, tr, va, te, mean, std = split_and_scale(frame, features, split_seed)
        run_id = f'{experiment}_split{split_seed}_seed{model_seed}'
        for split_name, indices in [('train', tr), ('validation', va), ('test', te)]:
            for i in indices:
                splits.append(dict(run_id=run_id, split=split_name, row_index=int(i),
                                   Var1=int(frame.iloc[i].Var1), CellType=frame.iloc[i].CellType))
        np.savez(out/'models'/f'{run_id}_normalization.npz', mean=mean, std=std, features=features)
        models = {
            'LM': LinearRegression(),
            'RF': RandomForestRegressor(n_estimators=150 if is_original else 75,
                                        max_depth=18 if is_original else 13, max_features='sqrt',
                                        random_state=model_seed),
            'GBT': GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=model_seed),
        }
        for name in ['LM', 'RF', 'GBT', 'NN']:
            start = time.perf_counter()
            if name == 'NN':
                keras.backend.clear_session()
                keras.utils.set_random_seed(model_seed)
                model = keras.Sequential([
                    keras.layers.Input(shape=(len(features),)),
                    keras.layers.Dense(41, activation='relu'),
                    keras.layers.Dense(41, activation='relu'),
                    keras.layers.Dense(1),
                ])
                model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=.001), loss='mse', metrics=['mae'])
                history = model.fit(x[tr], y[tr], validation_data=(x[va], y[va]),
                                    epochs=100, batch_size=5, shuffle=True, verbose=0)
                for epoch in range(100):
                    histories.append(dict(run_id=run_id, epoch=epoch+1,
                                          **{k:float(v[epoch]) for k,v in history.history.items()}))
                predict = lambda values: model(values.astype(np.float32), training=False).numpy().ravel()
                model.save(out/'models'/f'{run_id}_NN.h5', include_optimizer=True)
            else:
                model = models[name].fit(x[tr], y[tr])
                predict = model.predict
                joblib.dump(model, out/'models'/f'{run_id}_{name}.joblib')
            result = dict(run_id=run_id, experiment=experiment, model=name,
                          split_seed=split_seed, model_seed=model_seed, n_features=len(features))
            for split_name, idx in [('train', tr), ('validation', va), ('test', te)]:
                pred = predict(x[idx])
                assert np.isfinite(pred).all()
                result[split_name+'_mae'] = float(mean_absolute_error(y[idx], pred))
                if split_name == 'test':
                    for i, value in zip(idx, pred):
                        predictions.append(dict(run_id=run_id, model=name, row_index=int(i),
                                                CellType=frame.iloc[i].CellType,
                                                observed=float(y[i]), predicted=float(value)))
            result['fit_predict_seconds'] = time.perf_counter()-start
            rows.append(result)
            print(run_id, name, f"test MAE={result['test_mae']:.6f}", flush=True)
        # Paper leaves max_features unspecified; report this alternative separately.
        if experiment == 'paper17_author_split':
            alt = RandomForestRegressor(n_estimators=75, max_depth=13, max_features=1.0,
                                         random_state=model_seed).fit(x[tr], y[tr])
            rows.append(dict(run_id=run_id+'_allfeatures', experiment='paper17_rf_all_features',
                             model='RF', split_seed=split_seed, model_seed=model_seed, n_features=len(features),
                             train_mae=float(mean_absolute_error(y[tr], alt.predict(x[tr]))),
                             validation_mae=float(mean_absolute_error(y[va], alt.predict(x[va]))),
                             test_mae=float(mean_absolute_error(y[te], alt.predict(x[te])))))
        pd.DataFrame(rows).to_csv(out/'all_runs.csv', index=False)
        pd.DataFrame(predictions).to_csv(out/'test_predictions.csv', index=False)
        pd.DataFrame(histories).to_csv(out/'nn_training_history.csv', index=False)
        pd.DataFrame(splits).to_csv(out/'split_membership.csv', index=False)
    results = pd.DataFrame(rows)
    summary = results.groupby(['experiment', 'model'], sort=False).test_mae.agg(['count','mean','std','min','max']).reset_index()
    summary['paper_mae'] = summary.model.map(PAPER)
    summary['difference_vs_paper'] = summary['mean']-summary.paper_mae
    summary['relative_difference_percent'] = 100*summary.difference_vs_paper/summary.paper_mae
    summary.to_csv(out/'table2_comparison.csv', index=False)
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (ROOT/'data/processed').glob('*.csv')}
    (out/'input_checksums.json').write_text(json.dumps(hashes, indent=2))
    print(summary.to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
