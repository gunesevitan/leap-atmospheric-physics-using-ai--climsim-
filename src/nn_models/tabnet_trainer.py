import sys
import argparse
import yaml
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from pytorch_tabnet.tab_model import TabNetRegressor

import preprocessing
sys.path.append('..')
import settings
import metrics
import visualization


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory', type=str)
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    model_directory = Path(settings.MODELS / args.model_directory)
    config = yaml.load(open(model_directory / 'config.yaml'), Loader=yaml.FullLoader)
    settings.logger.info(f'Running {model_directory} model in {args.mode} mode')

    df_train = pd.read_parquet(settings.DATA / 'datasets' / 'train.parquet')
    settings.logger.info(f'Training Set Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')
    df_test = pd.read_parquet(settings.DATA / 'datasets' / 'test.parquet')
    settings.logger.info(f'Test Set Shape: {df_test.shape} - Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')

    # Select target and feature as individual arrays and grouped arrays
    target_columns = df_train.columns.tolist()[-368:]
    target_column_groups = preprocessing.get_target_column_groups(columns=target_columns)
    feature_columns = df_train.columns.tolist()[:-368]
    feature_column_groups = preprocessing.get_feature_column_groups(columns=feature_columns)
    prediction_columns = [f'{column}_prediction' for column in target_columns]

    # Load and concatenate precomputed folds
    df_train = pd.concat((df_train, pd.read_parquet(settings.DATA / 'folds.parquet')), axis=1)

    feature_means, feature_stds = np.load(settings.DATA / 'feature_means.npy'), np.load(settings.DATA / 'feature_stds.npy')
    target_means, target_stds = np.load(settings.DATA / 'target_means.npy'), np.load(settings.DATA / 'target_stds.npy')
    feature_stds = np.where(feature_stds == 0.0, 1.0, feature_stds)
    target_stds = np.where(target_stds == 0.0, 1.0, target_stds)

    df_train[feature_columns] = (df_train[feature_columns] - feature_means) / feature_stds
    df_train[target_columns] = (df_train[target_columns] - target_means) / target_stds
    df_test[feature_columns] = (df_test[feature_columns] - feature_means) / feature_stds

    torch.multiprocessing.set_sharing_strategy('file_system')

    global_scores = []
    target_scores = []
    prediction_columns_to_overwrite = []
    df_test_predictions = pd.DataFrame(np.zeros((df_test.shape[0], 368), dtype=np.float32), columns=target_columns)

    folds = config['test']['folds']

    for fold in folds:

        training_idx, validation_idx = df_train.loc[df_train[fold] != 1].index, df_train.loc[df_train[fold] == 1].index
        # Validate on training set if validation is set is not specified
        if len(validation_idx) == 0:
            validation_idx = training_idx

        settings.logger.info(
            f'''
            Fold: {fold}
            Training: {len(training_idx)} ({len(training_idx) // config["training"]["training_batch_size"] + 1} steps)
            Validation {len(validation_idx)} ({len(validation_idx) // config["training"]["test_batch_size"] + 1} steps)
            '''
        )

        X_train, y_train = df_train.loc[training_idx, feature_columns].values, df_train.loc[training_idx, target_columns].values
        X_val, y_val = df_train.loc[validation_idx, feature_columns].values, df_train.loc[validation_idx, target_columns].values

        model = TabNetRegressor(optimizer_fn=optim.AdamW, **config['model']['model_parameters'])
        model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_name=['train', 'val'],
            eval_metric=['mse'],
            max_epochs=config['training']['epochs'],
            batch_size=config['training']['training_batch_size'],
            virtual_batch_size=config['training']['training_batch_size'] // 4,
            num_workers=16,
            drop_last=False,
            augmentations=None,
            compute_importance=False
        )

        df_test_predictions += (model.predict(df_test[feature_columns].values) / len(folds))

        # Rescale validation targets and predictions back to their normal scale and calculate their validation scores
        validation_predictions = model.predict(X_val)
        rescaled_validation_targets = (y_val * target_stds) + target_means
        rescaled_validation_predictions = (validation_predictions * target_stds) + target_means
        df_validation_predictions = pd.DataFrame(
            data=np.hstack((
                rescaled_validation_targets,
                rescaled_validation_predictions,
            )),
            columns=target_columns + prediction_columns
        )
        _, target_validation_scores = metrics.regression_scores(
            y_true=df_validation_predictions[target_columns],
            y_pred=df_validation_predictions[prediction_columns]
        )

        # Calculate validation scores with mean predictions
        df_validation_mean_predictions = df_validation_predictions.copy(deep=True)
        df_validation_mean_predictions[prediction_columns] = target_means
        mean_global_validation_scores, mean_target_validation_scores = metrics.regression_scores(
            y_true=df_validation_mean_predictions[target_columns],
            y_pred=df_validation_mean_predictions[prediction_columns]
        )

        # Replace columns on which the performance of mean prediction is better than model's prediction
        target_idx = np.where(mean_target_validation_scores['r2_score'] > target_validation_scores['r2_score'])[0]
        fold_prediction_columns_to_overwrite = [f'{column}_prediction' for column in np.array(target_columns)[target_idx].tolist()]
        df_validation_predictions[fold_prediction_columns_to_overwrite] = target_means[target_idx]
        global_validation_scores, target_validation_scores = metrics.regression_scores(
            y_true=df_validation_predictions[target_columns],
            y_pred=df_validation_predictions[prediction_columns]
        )

        global_scores.append(global_validation_scores)
        target_scores.append(target_validation_scores)
        prediction_columns_to_overwrite.append(fold_prediction_columns_to_overwrite)

        df_train.loc[validation_idx, prediction_columns] = df_validation_predictions[prediction_columns].values

        settings.logger.info(
            f'''
            {fold}
            Validation Scores: {json.dumps(global_validation_scores, indent=2)}
            Replaced Target Predictions: {len(fold_prediction_columns_to_overwrite)}
            {fold_prediction_columns_to_overwrite}
            '''
        )

    global_scores = pd.DataFrame(global_scores)
    settings.logger.info(
        f'''
        Mean Validation Scores
        {json.dumps(global_scores.mean(axis=0).to_dict(), indent=2)}
        and Standard Deviations
        ±{json.dumps(global_scores.std(axis=0).to_dict(), indent=2)}
        '''
    )

    # Rescale targets back to their normal scale and calculate their oof scores
    df_train[target_columns] = (df_train[target_columns] * target_stds) + target_means
    global_oof_scores, target_oof_scores = metrics.regression_scores(
        y_true=df_train[target_columns],
        y_pred=df_train[prediction_columns]
    )
    target_scores.append(target_oof_scores)
    global_scores = pd.concat((
        global_scores,
        pd.DataFrame([global_oof_scores])
    )).reset_index(drop=True)
    global_scores['fold'] = ['1', '2', '3', '4', '5', 'OOF']
    global_scores = global_scores[global_scores.columns.tolist()[::-1]]

    global_scores.to_csv(model_directory / 'global_scores.csv', index=False)
    settings.logger.info(f'global_scores.csv is saved to {model_directory}')

    visualization.visualize_scores(
        scores=global_scores,
        title=f'Global Fold and OOF Scores of {len(folds)} Model(s)',
        path=model_directory / 'global_scores.png'
    )
    settings.logger.info(f'global_scores.png is saved to {model_directory}')

    single_targets = [columns[0] for columns in target_column_groups.values() if len(columns) == 1]
    single_target_scores = [df.loc[single_targets] for df in target_scores]
    visualization.visualize_single_target_scores(
        single_target_scores=single_target_scores,
        title=f'Single Target Fold and OOF Scores of {len(folds)} Model(s)',
        path=model_directory / 'single_target_scores.png'
    )
    settings.logger.info(f'single_target_scores.png is saved to {model_directory}')

    vertically_resolved_targets = {name: columns for name, columns in target_column_groups.items() if len(columns) == 60}
    for target_group, columns in vertically_resolved_targets.items():
        visualization.visualize_vertically_resolved_target_scores(
            vertically_resolved_target_scores=[df.loc[columns] for df in target_scores],
            title=f'{target_group} Fold and OOF Scores of {len(folds)} Model(s)',
            path=model_directory / f'{target_group}_scores.png'
        )
        settings.logger.info(f'{target_group}_scores.png is saved to {model_directory}')

    # Rescale test predictions back to their normal scale
    df_test_predictions = (df_test_predictions * target_stds) + target_means
    test_prediction_columns_to_overwrite = np.unique(np.concatenate(prediction_columns_to_overwrite)).tolist()
    test_prediction_column_idx_to_overwrite = sorted([prediction_columns.index(column) for column in test_prediction_columns_to_overwrite])
    df_test_predictions.iloc[:, test_prediction_column_idx_to_overwrite] = target_means[test_prediction_column_idx_to_overwrite]

    if config['persistence']['save_prediction_visualizations']:
        predictions_visualization_directory = model_directory / 'predictions'
        predictions_visualization_directory.mkdir(parents=True, exist_ok=True)
        for target_column in tqdm(target_columns):
            visualization.visualize_predictions(
                train_target=df_train[target_column],
                train_predictions=df_train[f'{target_column}_prediction'],
                test_predictions=df_test_predictions[target_column],
                scores=target_oof_scores.loc[target_column],
                path=predictions_visualization_directory / f'{target_column}.png'
            )

    df_train[prediction_columns].to_parquet(model_directory / 'oof_predictions.parquet')
    settings.logger.info(f'oof_predictions.parquet is saved to {model_directory}')

    df_test_predictions[target_columns].to_parquet(model_directory / 'test_predictions.parquet')
    settings.logger.info(f'oof_predictions.parquet is saved to {model_directory}')
