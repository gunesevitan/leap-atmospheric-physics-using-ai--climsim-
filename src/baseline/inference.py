import sys
import json
import numpy as np
import pandas as pd

sys.path.append('..')
import settings
import metrics


def get_target_column_groups(target_columns):

    """
    Returns names of target column groups

    Parameters
    ----------
    target_columns: list of shape (368)
        List of names of target columns

    Returns
    -------
    column_groups: list of shape (14)
        List of names of target column groups

    column_group_names: list of shape (14)
        List of group names
    """

    ptend_t_columns = [column for column in target_columns if column.startswith('ptend_t')]
    ptend_q0001_columns = [column for column in target_columns if column.startswith('ptend_q0001')]
    ptend_q0002_columns = [column for column in target_columns if column.startswith('ptend_q0002')]
    ptend_q0003_columns = [column for column in target_columns if column.startswith('ptend_q0003')]
    ptend_u_columns = [column for column in target_columns if column.startswith('ptend_u')]
    ptend_v_columns = [column for column in target_columns if column.startswith('ptend_v')]
    other_columns = ['cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 'cam_out_PRECC', 'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']
    column_groups = [ptend_t_columns, ptend_q0001_columns, ptend_q0002_columns, ptend_q0003_columns, ptend_u_columns, ptend_v_columns] + [[column] for column in other_columns]
    column_group_names = ['ptend_t', 'ptend_q0001', 'ptend_q0002', 'ptend_q0003', 'ptend_u', 'ptend_v'] + [[column] for column in other_columns]

    return column_groups, column_group_names


if __name__ == '__main__':

    df = pd.read_parquet(settings.DATA / 'datasets' / 'train.parquet')
    weights = np.load(settings.DATA / 'weights.npy')

    target_columns = df.columns.tolist()[-368:]
    target_column_groups, target_column_group_names = get_target_column_groups(target_columns=target_columns)

    prediction_columns = [f'{column}_prediction' for column in target_columns]
    target_means = df[target_columns].mean()
    df[prediction_columns] = target_means

    for column_group, group_name in zip(target_column_groups, target_column_group_names):
        prediction_column_group = [f'{column}_prediction' for column in column_group]
        group_scores = metrics.regression_scores(df[column_group], df[prediction_column_group])
        settings.logger.info(f'{group_name} ({len(prediction_column_group)} dimensions) scores {json.dumps(group_scores, indent=2)}')

    scores = metrics.regression_scores(df[target_columns], df[prediction_columns])
    settings.logger.info(f'Scores {json.dumps(scores, indent=2)}')

    submission_directory = settings.DATA / 'submissions'
    submission_directory.mkdir(parents=True, exist_ok=True)

    df_submission = pd.read_parquet(settings.DATA / 'datasets' / 'sample_submission.parquet')
    df_submission.iloc[:, 1:] *= target_means
    df_submission.to_parquet(submission_directory / 'sample_submission.parquet')
