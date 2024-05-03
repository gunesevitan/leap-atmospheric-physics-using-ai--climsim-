import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


def regression_scores(y_true, y_pred):

    """
    Calculate regression metric scores from given ground truth and predictions

    Parameters
    ----------
    y_true: array-like of shape (n_samples)
        Array of ground truth values

    y_pred: array-like of shape (n_samples)
        Array of prediction values

    Returns
    -------
    global_scores: dict
        Dictionary of aggregated regression metric scores over all targets

    target_scores: pandas.DataFrame of shape (n_target_columns, 3)
        Dataframe of per target regression scores
    """

    target_columns = y_true.columns.tolist()

    rmses = {}
    maes = {}
    r2_scores = {}

    for target_column in target_columns:
        rmses[target_column] = root_mean_squared_error(y_true[target_column], y_pred[f'{target_column}_prediction'])
        maes[target_column] = mean_absolute_error(y_true[target_column], y_pred[f'{target_column}_prediction'])
        r2_scores[target_column] = r2_score(y_true[target_column], y_pred[f'{target_column}_prediction'])

    global_scores = {
        'root_mean_squared_error': float(np.mean(list(rmses.values()))),
        'mean_absolute_error': float(np.mean(list(maes.values()))),
        'r2_score': float(np.mean(list(r2_scores.values()))),
    }

    target_scores = pd.DataFrame([rmses, maes, r2_scores]).T.rename(columns={0: 'rmse', 1: 'mae', 2: 'r2_score'})

    return global_scores, target_scores
