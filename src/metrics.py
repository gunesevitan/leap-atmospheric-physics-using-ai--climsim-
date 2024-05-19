import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def regression_scores(y_true, y_pred):

    """
    Calculate regression metric scores from given ground truth and predictions

    Parameters
    ----------
    y_true: array-like of shape (n_samples, n_targets)
        Array of ground truth values

    y_pred: array-like of shape (n_samples, n_targets)
        Array of prediction values

    Returns
    -------
    global_scores: dict
        Dictionary of aggregated regression metric scores over all targets

    target_scores: pandas.DataFrame of shape (n_target_columns, 3)
        Dataframe of per target regression scores
    """

    mses = []
    maes = []
    r2_scores = []

    for column_idx in range(y_true.shape[1]):
        mses.append(mean_squared_error(y_true[:, column_idx], y_pred[:, column_idx]))
        maes.append(mean_absolute_error(y_true[:, column_idx], y_pred[:, column_idx]))
        r2_scores.append(r2_score(y_true[:, column_idx], y_pred[:, column_idx]))

    global_scores = {
        'mean_squared_error': float(np.mean(mses)),
        'mean_absolute_error': float(np.mean(maes)),
        'r2_score': float(np.mean(r2_scores)),
    }
    target_scores = pd.DataFrame([mses, maes, r2_scores]).T.rename(columns={0: 'rmse', 1: 'mae', 2: 'r2_score'})

    return global_scores, target_scores


def regression_scores_single(y_true, y_pred):

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

    scores = {
        'mean_squared_error': float(mean_squared_error(y_true, y_pred)),
        'mean_absolute_error': float(mean_absolute_error(y_true, y_pred)),
        'r2_score': float(r2_score(y_true, y_pred)),
    }

    return scores
