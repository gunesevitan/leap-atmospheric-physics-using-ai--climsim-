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

    df_train = pd.read_parquet(settings.DATA / 'datasets' / 'train.parquet')
    settings.logger.info(f'Train Dataset Shape: {df_train.shape}')

    n_splits = 5
    df_train = create_folds(
        df=df_train,
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    fold_columns = [f'fold{fold}' for fold in range(1, n_splits + 1)]
    df_train[fold_columns].to_parquet(settings.DATA / 'folds.parquet')
    settings.logger.info(f'folds.parquet is saved to {settings.DATA}')
