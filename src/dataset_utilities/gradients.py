import sys
import numpy as np
import pandas as pd
import polars as pl

sys.path.append('..')
import settings


if __name__ == '__main__':

    dataset_directory = settings.DATA / 'datasets'
    dataset_directory.mkdir(parents=True, exist_ok=True)

    normalization_columns = 'targets'

    if normalization_columns == 'targets':

        targets = pd.read_parquet(settings.DATA / 'datasets' / 'train.parquet')
        targets = targets.to_numpy()
        targets = targets[:, -368:]

        sequential_targets = targets[:, :360].reshape(-1, 6, 60)
        gradients = np.diff(sequential_targets, n=1, prepend=sequential_targets[:, :, 0:1]).reshape(-1, 360)

        means = gradients.mean(axis=0)
        stds = gradients.std(axis=0)

        with open(dataset_directory / 'target_gradient_means.npy', 'wb') as f:
            np.save(f, means)

        with open(dataset_directory / 'target_gradient_stds.npy', 'wb') as f:
            np.save(f, stds)

    elif normalization_columns == 'features':

        columns = np.arange(1, 557).tolist()
        dtypes = [pl.Float32 for i in range(len(columns))]
        features = pl.read_csv(
            settings.DATA / 'leap-atmospheric-physics-ai-climsim' / 'train.csv',
            columns=columns,
            dtypes=dtypes,
            n_threads=16
        )
        test_features = pl.read_csv(
            settings.DATA / 'leap-atmospheric-physics-ai-climsim' / 'test.csv',
            columns=columns,
            dtypes=dtypes,
            n_threads=16
        )

        features = pl.concat((
            features,
            test_features
        ), how='vertical')
        del test_features
        features = features.to_pandas()
        features = features.to_numpy()

        sequential_features1 = features[:, :360].reshape(-1, 6, 60)
        sequential_features2 = features[:, 376:].reshape(-1, 3, 60)
        del features
        gradients1 = np.diff(sequential_features1, n=1, prepend=sequential_features1[:, :, 0:1])
        gradients2 = np.diff(sequential_features2, n=1, prepend=sequential_features2[:, :, 0:1])
        del sequential_features1, sequential_features2

        gradients1 = gradients1.reshape(-1, 360)
        gradients2 = gradients2.reshape(-1, 180)
        gradients1 = np.hstack([gradients1, gradients2])

        means = gradients1.mean(axis=0)
        stds = gradients1.std(axis=0)

        with open(dataset_directory / 'feature_gradient_means.npy', 'wb') as f:
            np.save(f, means)

        with open(dataset_directory / 'feature_gradient_stds.npy', 'wb') as f:
            np.save(f, stds)
