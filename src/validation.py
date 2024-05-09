import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import settings


def create_folds(df, n_splits, shuffle=True, random_state=42):

    """
    Create columns of folds on given dataframe

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with given stratify and group columns

    n_splits: int
        Number of folds (2 <= n_splits)

    shuffle: bool
        Whether to shuffle before split or not

    random_state: int
        Random seed for reproducible results

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with created fold columns
    """

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for fold, (training_idx, validation_idx) in enumerate(kf.split(X=df), 1):
        df.loc[training_idx, f'fold{fold}'] = 0
        df.loc[validation_idx, f'fold{fold}'] = 1
        df[f'fold{fold}'] = df[f'fold{fold}'].astype(np.uint8)

    return df


if __name__ == '__main__':

    folds = np.zeros((10091520, 16), dtype=np.uint8)
    for fold in range(16):
        start = fold * 625000
        end = (fold + 1) * 625000
        if fold < 15:
            folds[start:end, fold] = 1
        else:
            folds[start:, fold] = 1
        settings.logger.info(f'Fold {fold} - Training Size: {np.sum(folds[:, fold] == 0)} Validation Size: {np.sum(folds[:, fold] == 1)}')

    with open(settings.DATA / 'folds.npz', 'wb') as f:
        np.savez_compressed(f, folds)

    settings.logger.info(f'folds.npy is saved to {settings.DATA}')
