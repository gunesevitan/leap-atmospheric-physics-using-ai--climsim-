import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def regression_scores(y_true, y_pred, weights, target_columns):

    """
    Calculate regression metric scores from given ground truth and predictions

    Parameters
    ----------
    y_true: numpy.ndarray of shape (n_samples, n_targets)
        Array of ground truth values

    y_pred: numpy.ndarray of shape (n_samples, n_targets)
        Array of prediction values

    weights: numpy.ndarray of shape (n_targets)
        Array of weights

    target_columns: list of shape (n_targets)
        Array of target weights

    Returns
    -------
    global_scores: dict
        Dictionary of aggregated regression metric scores over all targets

    target_scores: pandas.DataFrame of shape (n_target_columns, 3)
        Dataframe of per target regression scores
    """

    y_true = y_true.copy()
    y_pred = y_pred.copy()

    if weights is not None:
        y_true *= weights
        y_pred *= weights

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
    target_scores = pd.DataFrame([mses, maes, r2_scores]).T.rename(columns={0: 'mse', 1: 'mae', 2: 'r2_score'})
    target_scores.index = target_columns
    target_scores['weight'] = weights

    non_zero_weight_target_idx = target_scores['weight'] != 0
    non_zero_global_scores = target_scores.loc[non_zero_weight_target_idx, ['mse', 'mae', 'r2_score']].mean(axis=0).to_dict()
    global_scores['non_zero_weight_mean_squared_error'] = float(non_zero_global_scores['mse'])
    global_scores['non_zero_weight_mean_absolute_error'] = float(non_zero_global_scores['mae'])
    global_scores['non_zero_weight_r2_score'] = float(non_zero_global_scores['r2_score'])

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
