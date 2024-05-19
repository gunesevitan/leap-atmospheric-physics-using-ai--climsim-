import sys
import argparse
from pathlib import Path
import yaml
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.append('..')
import settings
import metrics
import visualization


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory', type=str)
    args = parser.parse_args()

    model_directory = Path(settings.MODELS / args.model_directory)
    config = yaml.load(open(model_directory / 'config.yaml'), Loader=yaml.FullLoader)

    df_train = pd.read_parquet(settings.DATA / 'datasets' / 'train.parquet')
    df_test = pd.read_parquet(settings.DATA / 'datasets' / 'test.parquet')
    settings.logger.info(f'Training Shape: {df_train.shape} - Test Shape: {df_test.shape}')

    # Load precomputed folds
    folds = np.load(settings.DATA / 'folds.npz')['arr_0']
    file_idx = np.arange(folds.shape[0], dtype=np.uint32)
    settings.logger.info(f'Loaded folds: {folds.shape}')

    target = config['training']['target']
    features = config['training']['features']
    seeds = config['training']['seeds']

    # Create training and validation indices
    training_folds = config['training']['training_folds']
    validation_folds = config['training']['validation_folds']
    test_folds = config['training']['test_folds']

    training_mask = folds[:, training_folds].any(axis=1)
    validation_mask = folds[:, validation_folds].any(axis=1)
    training_idx = file_idx[training_mask]
    validation_idx = file_idx[validation_mask]

    target_idx = df_train.columns.tolist()[-368:].index(target)
    target_mean = np.load(settings.DATA / 'target_means.npy')[target_idx]
    target_std = np.load(settings.DATA / 'target_stds.npy')[target_idx]
    target_min = np.load(settings.DATA / 'target_mins.npy')[target_idx]
    target_max = np.load(settings.DATA / 'target_maxs.npy')[target_idx]
    if target_std == 0.:
        target_std = 1.

    df_train[target] = (df_train[target] - target_mean) / target_std

    settings.logger.info(
        f'''
        Running XGBoost trainer
        Dataset Shape: {df_train.shape}
        Training Folds: {config['training']['training_folds']} Shape: {training_idx.shape[0]}
        Validation Folds: {config['training']['validation_folds']} Shape: Shape: {validation_idx.shape[0]}
        Features: {json.dumps(features, indent=2)}
        Target: {target} Mean: {target_mean:.4f} Std: {target_std:.4f} Min: {target_min:.4f} Max: {target_max:.4f}
        '''
    )

    df_feature_importance_gain = pd.DataFrame(
        data=np.zeros((len(features), 1)),
        index=features,
        columns=['importance']
    )
    df_feature_importance_split = pd.DataFrame(
        data=np.zeros((len(features), 1)),
        index=features,
        columns=['importance']
    )
    scores = []

    df_train.loc[validation_idx, 'prediction'] = 0
    df_test['prediction'] = 0

    for seed in seeds:

        training_dataset = lgb.Dataset(df_train.loc[training_idx, features], label=df_train.loc[training_idx, target])
        validation_dataset = lgb.Dataset(df_train.loc[validation_idx, features], label=df_train.loc[validation_idx, target])

        config['model_parameters']['seed'] = seed
        config['model_parameters']['feature_fraction_seed'] = seed
        config['model_parameters']['bagging_seed'] = seed
        config['model_parameters']['drop_seed'] = seed
        config['model_parameters']['data_random_seed'] = seed

        model = lgb.train(
            params=config['model_parameters'],
            train_set=training_dataset,
            valid_sets=[training_dataset, validation_dataset],
            num_boost_round=config['fit_parameters']['boosting_rounds'],
            callbacks=[
                lgb.log_evaluation(config['fit_parameters']['log_evaluation'])
            ]
        )

        df_feature_importance_gain['importance'] += (model.feature_importance(importance_type='gain') / len(seeds))
        df_feature_importance_split['importance'] += (model.feature_importance(importance_type='gain') / len(seeds))

        df_train.loc[validation_mask, 'prediction'] += (model.predict(df_train.loc[validation_idx, features]) / len(seeds))
        df_test.loc[:, 'prediction'] += (model.predict(df_test.loc[:, features]) / len(seeds))

    df_train[target] = (df_train[target] * target_std) + target_mean
    df_train['prediction'] = (df_train['prediction'] * target_std) + target_mean
    df_test['prediction'] = (df_test['prediction'] * target_std) + target_mean

    df_train['prediction'] = df_train['prediction'].clip(lower=target_min, upper=target_max)
    df_test['prediction'] = df_test['prediction'].clip(lower=target_min, upper=target_max)
    settings.logger.info(f'Predictions are clipped between {target_min:.8f} and {target_max:.8f}')

    for fold in test_folds:

        # Create validation indices
        test_mask = folds[:, fold] == 1
        test_idx = file_idx[test_mask]

        fold_scores = metrics.regression_scores_single(
            y_true=df_train.loc[test_idx, target],
            y_pred=df_train.loc[test_idx, 'prediction']
        )
        scores.append(fold_scores)
        settings.logger.info(f'Fold: {fold} - Scores: {json.dumps(fold_scores, indent=2)}')

    df_scores = pd.DataFrame(scores)
    settings.logger.info(
        f'''
        Mean Validation Scores
        {json.dumps(df_scores.mean(axis=0).to_dict(), indent=2)}
        and Standard Deviations
        ±{json.dumps(df_scores.std(axis=0).to_dict(), indent=2)}
        '''
    )

    oof_mask = df_train['prediction'].notna()
    oof_scores = metrics.regression_scores_single(
        y_true=df_train.loc[oof_mask, target],
        y_pred=df_train.loc[oof_mask, 'prediction']
    )
    settings.logger.info(f'OOF Scores: {json.dumps(oof_scores, indent=2)}')

    df_scores = pd.concat((
        df_scores,
        pd.DataFrame([oof_scores])
    )).reset_index(drop=True)
    df_scores['fold'] = test_folds + ['OOF']
    df_scores = df_scores[df_scores.columns.tolist()[::-1]]
    df_scores.to_csv(model_directory / 'scores.csv', index=False)
    settings.logger.info(f'scores.csv is saved to {model_directory}')

    visualization.visualize_scores(
        scores=df_scores,
        title='Scores',
        path=model_directory / 'scores.png'
    )
    settings.logger.info(f'scores.png is saved to {model_directory}')

    for importance_type, df_feature_importance in zip(['gain', 'split'], [df_feature_importance_gain, df_feature_importance_split]):
        df_feature_importance['mean'] = df_feature_importance[['importance']].mean(axis=1)
        df_feature_importance['std'] = df_feature_importance[['importance']].std(axis=1).fillna(0)
        df_feature_importance.sort_values(by='mean', ascending=False, inplace=True)
        visualization.visualize_feature_importance(
            df_feature_importance=df_feature_importance,
            title=f'XGBoost Feature Importance ({importance_type.capitalize()})',
            path=model_directory / f'feature_importance_{importance_type}.png'
        )
        settings.logger.info(f'Saved feature_importance_{importance_type}.png to {model_directory}')

        df_feature_importance['importance'].reset_index().rename(columns={
            'index': 'feature'
        }).to_csv(model_directory / f'{importance_type}_importance.csv', index=False)

    visualization.visualize_predictions(
        targets=df_train.loc[oof_mask, target],
        predictions=df_train.loc[oof_mask, 'prediction'],
        score=oof_scores['r2_score'],
        target_column=target,
        path=model_directory / 'predictions.png'
    )

    df_train.loc[:, [target, 'prediction']].to_parquet(model_directory / 'oof_predictions.parquet')
    settings.logger.info(f'Saved oof_predictions.parquet to {model_directory}')

    df_test.loc[:, ['prediction']].to_parquet(model_directory / 'test_predictions.parquet')
    settings.logger.info(f'Saved test_predictions.parquet to {model_directory}')
