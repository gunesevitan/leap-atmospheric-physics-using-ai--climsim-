import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import yaml
import json
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.append('..')
import settings
import visualization


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory', type=str)
    args = parser.parse_args()

    model_directory = Path(settings.MODELS / args.model_directory)
    config = yaml.load(open(model_directory / 'config.yaml'), Loader=yaml.FullLoader)

    df_train = pd.read_parquet(settings.DATA / 'datasets' / 'train.parquet')
    settings.logger.info(f'Training Shape: {df_train.shape}')

    # Load precomputed folds
    folds = np.load(settings.DATA / 'folds.npz')['arr_0']
    file_idx = np.arange(folds.shape[0], dtype=np.uint32)
    settings.logger.info(f'Loaded folds: {folds.shape}')

    targets = config['training']['targets']
    features = config['training']['features']

    # Create training indices
    training_folds = config['training']['training_folds']
    training_mask = folds[:, training_folds].any(axis=1)
    training_idx = file_idx[training_mask]

    target_means = np.load(settings.DATA / 'target_means.npy')
    target_stds = np.load(settings.DATA / 'target_stds.npy')
    target_stds = np.where(target_stds == 0.0, 1.0, target_stds)

    df_train[targets] = (df_train[targets] - target_means) / target_stds

    settings.logger.info(
        f'''
        Running XGBoost Feature Importance
        Dataset Shape: {df_train.shape}
        Training Folds: {config['training']['training_folds']} Shape: {training_idx.shape[0]}
        Features: {json.dumps(features, indent=2)}
        '''
    )

    df_feature_importance_gain = pd.DataFrame(
        data=np.zeros((len(features), len(targets))).astype(np.float32),
        index=features,
        columns=targets
    )
    df_feature_importance_weight = pd.DataFrame(
        data=np.zeros((len(features), len(targets))).astype(np.float32),
        index=features,
        columns=targets
    )
    df_feature_importance_cover = pd.DataFrame(
        data=np.zeros((len(features), len(targets))).astype(np.float32),
        index=features,
        columns=targets
    )

    for target in tqdm(targets):

        training_dataset = xgb.DMatrix(df_train.loc[training_idx, features], label=df_train.loc[training_idx, target])

        config['model_parameters']['seed'] = 42

        model = xgb.train(
            params=config['model_parameters'],
            dtrain=training_dataset,
            evals=[(training_dataset, 'train')],
            num_boost_round=config['fit_parameters']['boosting_rounds'],
            early_stopping_rounds=None,
            verbose_eval=config['fit_parameters']['verbose_eval'],
        )

        df_feature_importance_gain.loc[:, target] += pd.Series(model.get_score(importance_type='gain')).fillna(0).astype(np.float32)
        df_feature_importance_weight.loc[:, target] += pd.Series(model.get_score(importance_type='weight')).fillna(0).astype(np.float32)
        df_feature_importance_cover.loc[:, target] += pd.Series(model.get_score(importance_type='cover')).fillna(0).astype(np.float32)

    for importance_type, df_feature_importance in zip(['gain', 'weight', 'cover'], [df_feature_importance_gain, df_feature_importance_weight, df_feature_importance_cover]):

        df_feature_importance_ = df_feature_importance.fillna(0).copy(deep=True)
        df_feature_importance_['mean'] = df_feature_importance_[targets].mean(axis=1)
        df_feature_importance_['std'] = df_feature_importance_[targets].std(axis=1).fillna(0)
        df_feature_importance_.sort_values(by='mean', ascending=False, inplace=True)

        visualization.visualize_feature_importance(
            df_feature_importance=df_feature_importance_,
            title=f'XGBoost Mean Feature Importance ({importance_type.capitalize()}) of {len(targets)} Targets',
            path=model_directory / f'feature_importance_{importance_type}.png'
        )
        settings.logger.info(f'Saved feature_importance_{importance_type}.png to {model_directory}')

        df_feature_importance.reset_index().rename(columns={
            'index': 'feature'
        }).to_csv(model_directory / f'{importance_type}_importance.csv', index=False)

        df_feature_importance__ = df_feature_importance.fillna(0).copy(deep=True)
        target_importance_mins = df_feature_importance__[targets].min(axis=0)
        target_importance_maxs = df_feature_importance__[targets].max(axis=0)

        df_feature_importance__ = (df_feature_importance__ - target_importance_mins) / (target_importance_maxs - target_importance_mins)
        df_feature_importance__ = df_feature_importance__.fillna(0)

        visualization.visualize_feature_importance_heatmap(
            df_feature_importance__,
            title='XGBoost Feature Importance Heatmap',
            path=model_directory / f'feature_importance_{importance_type}_heatmap.png'
        )
