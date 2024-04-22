import sys
import numpy as np
import polars as pl

sys.path.append('..')
import settings


if __name__ == '__main__':

    dataset_directory = settings.DATA / 'datasets'
    dataset_directory.mkdir(parents=True, exist_ok=True)

    normalization_columns = 'features'

    if normalization_columns == 'targets':

        columns = np.arange(557, 925).tolist()
        dtypes = [pl.Float32 for i in range(len(columns))]
        df = pl.read_csv(
            settings.DATA / 'leap-atmospheric-physics-ai-climsim' / 'train.csv',
            columns=columns,
            dtypes=dtypes,
            n_threads=16
        )
        df = df.to_numpy()

        target_means = df.mean(axis=0)
        target_stds = df.std(axis=0)

        with open(settings.DATA / 'target_means.npy', 'wb') as f:
            np.save(f, target_means)

        with open(settings.DATA / 'target_stds.npy', 'wb') as f:
            np.save(f, target_stds)

    elif normalization_columns == 'features':

        columns = np.arange(1, 557).tolist()
        dtypes = [pl.Float32 for i in range(len(columns))]
        df = pl.read_csv(
            settings.DATA / 'leap-atmospheric-physics-ai-climsim' / 'train.csv',
            columns=columns,
            dtypes=dtypes,
            n_threads=16
        )
        df_test = pl.read_csv(
            settings.DATA / 'leap-atmospheric-physics-ai-climsim' / 'test.csv',
            columns=columns,
            dtypes=dtypes,
            n_threads=16
        )
        df = pl.concat((
            df,
            df_test
        ), how='vertical')
        del df_test
        df = df.to_numpy()

        feature_means = df.mean(axis=0)
        feature_stds = df.std(axis=0)

        with open(settings.DATA / 'feature_means.npy', 'wb') as f:
            np.save(f, feature_means)

        with open(settings.DATA / 'feature_stds.npy', 'wb') as f:
            np.save(f, feature_stds)
