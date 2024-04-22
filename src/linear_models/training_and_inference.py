import sys
import argparse
import yaml
import json
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.append('..')
import settings
import metrics


def get_target_column_groups(target_columns):

    """
    Get names of target column groups

    Parameters
    ----------
    target_columns: list of shape (368)
        List of names of target columns

    Returns
    -------
    column_groups: dict
        Dictionary of target column group names and columns
    """

    ptend_t_columns = [column for column in target_columns if column.startswith('ptend_t')]
    ptend_q0001_columns = [column for column in target_columns if column.startswith('ptend_q0001')]
    ptend_q0002_columns = [column for column in target_columns if column.startswith('ptend_q0002')]
    ptend_q0003_columns = [column for column in target_columns if column.startswith('ptend_q0003')]
    ptend_u_columns = [column for column in target_columns if column.startswith('ptend_u')]
    ptend_v_columns = [column for column in target_columns if column.startswith('ptend_v')]
    other_columns = [
        'cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 'cam_out_PRECC',
        'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD'
    ]

    column_groups = {
        'ptend_t': ptend_t_columns,
        'ptend_q0001': ptend_q0001_columns,
        'ptend_q0002': ptend_q0002_columns,
        'ptend_q0003': ptend_q0003_columns,
        'ptend_u': ptend_u_columns,
        'ptend_v': ptend_v_columns
    }
    for column in other_columns:
        column_groups[column] = [column]

    return column_groups


def get_feature_column_groups(feature_columns):

    """
    Get names of feature column groups

    Parameters
    ----------
    feature_columns: list of shape (556)
        List of names of target columns

    Returns
    -------
    column_groups: dict
        Dictionary of feature column group names and columns
    """

    state_t_columns = [column for column in feature_columns if column.startswith('state_t')]
    state_q0001_columns = [column for column in feature_columns if column.startswith('state_q0001')]
    state_q0002_columns = [column for column in feature_columns if column.startswith('state_q0002')]
    state_q0003_columns = [column for column in feature_columns if column.startswith('state_q0003')]
    state_u_columns = [column for column in feature_columns if column.startswith('state_u')]
    state_v_columns = [column for column in feature_columns if column.startswith('state_v')]
    pbuf_ozone_columns = [column for column in feature_columns if column.startswith('pbuf_ozone')]
    pbuf_CH4_columns = [column for column in feature_columns if column.startswith('pbuf_CH4')]
    pbuf_N2O_columns = [column for column in feature_columns if column.startswith('pbuf_N2O')]
    other_columns = [
        'state_ps', 'pbuf_SOLIN', 'pbuf_LHFLX', 'pbuf_SHFLX',
        'pbuf_TAUX', 'pbuf_TAUY', 'pbuf_COSZRS', 'cam_in_ALDIF',
        'cam_in_ALDIR', 'cam_in_ASDIF', 'cam_in_ASDIR', 'cam_in_LWUP',
        'cam_in_ICEFRAC', 'cam_in_LANDFRAC', 'cam_in_OCNFRAC', 'cam_in_SNOWHLAND'
    ]

    column_groups = {
        'state_t': state_t_columns,
        'state_q0001': state_q0001_columns,
        'state_q0002': state_q0002_columns,
        'state_q0003': state_q0003_columns,
        'state_u': state_u_columns,
        'state_v': state_v_columns,
        'pbuf_ozone': pbuf_ozone_columns,
        'pbuf_CH4': pbuf_CH4_columns,
        'pbuf_N2O': pbuf_N2O_columns,
    }

    for column in other_columns:
        column_groups[column] = [column]

    return column_groups


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory', type=str)
    args = parser.parse_args()

    model_directory = Path(settings.MODELS / args.model_directory)
    config = yaml.load(open(model_directory / 'config.yaml'), Loader=yaml.FullLoader)

    df_train = pd.read_parquet(settings.DATA / 'datasets' / 'train.parquet')
    # Select target and feature as individual arrays and grouped arrays
    target_columns = df_train.columns.tolist()[-368:]
    target_column_groups = get_target_column_groups(target_columns=target_columns)
    feature_columns = df_train.columns.tolist()[:-368]
    feature_column_groups = get_feature_column_groups(feature_columns=feature_columns)

    df_train = pd.concat((df_train, pd.read_parquet(settings.DATA / 'folds.parquet')), axis=1)
    df_test = pd.read_parquet(settings.DATA / 'datasets' / 'test.parquet')
    submission_weights = np.load(settings.DATA / 'weights.npy')

    # Normalize features with concatenated training and test set statistics
    scaler = StandardScaler()
    scaler.fit(pd.concat((df_train[feature_columns], df_test[feature_columns]), axis=0, ignore_index=True))
    df_train[feature_columns] = scaler.transform(df_train[feature_columns])
    df_test[feature_columns] = scaler.transform(df_test[feature_columns])
    df_test.loc[:, target_columns] = 0.0

    folds = config['training']['folds']
    features = config['training']['features']

    settings.logger.info(
        f'''
        Running linear model trainer
        Training Set Shape: {df_train.shape}
        Test Set Shape {df_test.shape}
        '''
    )

    # Write feature coefficients for all target column groups
    df_feature_coefficients = {}
    for group_name in target_column_groups.keys():
        df_feature_coefficient = pd.DataFrame(
            data=np.zeros((len(feature_columns), len(folds))),
            index=feature_columns,
            columns=folds
        )
        df_feature_coefficients[group_name] = df_feature_coefficient

    scores = []

    for fold in folds:

        training_idx, validation_idx = df_train.loc[df_train[fold] != 1].index, df_train.loc[df_train[fold] == 1].index

        for target_column_group_name, target_column_group in target_column_groups.items():

            # Extract features from specified feature groups in configurations
            target_feature_columns = []
            for feature_column_group in config['training']['features'][target_column_group_name]:
                target_feature_columns += feature_column_groups[feature_column_group]

            settings.logger.info(
                f'''
                Fold: {fold} Target Group: {target_column_group_name}
                Training: ({len(training_idx)})
                Validation: ({len(validation_idx)})
                Features: {target_feature_columns}
                '''
            )

            for target in tqdm(target_column_group):

                model = Ridge(**config['model_parameters'][target_column_group_name])
                model.fit(df_train.loc[training_idx, target_feature_columns], df_train.loc[training_idx, target])
                df_feature_coefficients[target_column_group_name, fold] = model.coef_

                df_train.loc[validation_idx, f'{target}_prediction'] = model.predict(df_train.loc[validation_idx, target_feature_columns])
                df_test.loc[:, target] += (model.predict(df_test.loc[:, target_feature_columns]) / len(folds))

            prediction_column_group = [f'{column}_prediction' for column in target_column_group]
            target_group_validation_scores = metrics.regression_scores(df_train.loc[validation_idx, target_column_group], df_train.loc[validation_idx, prediction_column_group])
            settings.logger.info(f'{fold} {target_column_group_name} Validation Scores: {json.dumps(target_group_validation_scores, indent=2)}')

        prediction_columns = [f'{column}_prediction' for column in target_columns]
        validation_scores = metrics.regression_scores(df_train.loc[validation_idx, target_columns], df_train.loc[validation_idx, prediction_columns])
        settings.logger.info(f'{fold} Validation Scores: {json.dumps(validation_scores, indent=2)}')

    for target_column_group_name, target_column_group in target_column_groups.items():
        prediction_column_group = [f'{column}_prediction' for column in target_column_group]
        oof_scores = metrics.regression_scores(df_train.loc[:, target_column_group], df_train.loc[:, prediction_column_group])
        settings.logger.info(f'{target_column_group_name} OOF Scores: {json.dumps(oof_scores, indent=2)}')

    prediction_columns = [f'{column}_prediction' for column in target_columns]
    oof_scores = metrics.regression_scores(df_train.loc[:, target_columns], df_train.loc[:, prediction_columns])
    settings.logger.info(f'OOF Scores: {json.dumps(oof_scores, indent=2)}')

    submission_directory = settings.DATA / 'submissions'
    submission_directory.mkdir(parents=True, exist_ok=True)

    df_submission = pd.read_parquet(settings.DATA / 'datasets' / 'sample_submission.parquet')
    df_submission.iloc[:, 1:] *= df_test.iloc[:, -368:]
    df_submission.to_parquet(submission_directory / 'sample_submission.parquet')
