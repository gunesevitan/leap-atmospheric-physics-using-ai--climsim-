import sys
import numpy as np
import polars as pl

sys.path.append('..')
import settings


if __name__ == '__main__':

    dataset_directory = settings.DATA / 'datasets'
    dataset_directory.mkdir(parents=True, exist_ok=True)

    normalization_columns = 'targets'

    if normalization_columns == 'targets':

        weight_columns = np.arange(1, 369).tolist()
        weight_dtypes = [pl.Float32 for i in range(len(weight_columns))]
        target_weights = pl.read_csv(
            settings.DATA / 'leap-atmospheric-physics-ai-climsim' / 'sample_submission.csv',
            columns=weight_columns,
            dtypes=weight_dtypes,
            n_rows=1,
            n_threads=16
        ).to_numpy().reshape(-1)

        with open(settings.DATA / 'target_weights.npy', 'wb') as f:
            np.save(f, target_weights)

        target_columns = np.arange(557, 925).tolist()
        target_dtypes = [pl.Float32 for i in range(len(target_columns))]
        targets = pl.read_csv(
            settings.DATA / 'leap-atmospheric-physics-ai-climsim' / 'train.csv',
            columns=target_columns,
            dtypes=target_dtypes,
            n_threads=16
        )
        targets = targets.to_numpy()

        target_means = targets.mean(axis=0)
        target_stds = targets.std(axis=0)
        target_mins = targets.min(axis=0)
        target_maxs = targets.max(axis=0)
        target_rmss = np.sqrt(np.mean(targets ** 2, axis=0))

        with open(dataset_directory / 'target_means.npy', 'wb') as f:
            np.save(f, target_means)

        with open(dataset_directory / 'target_stds.npy', 'wb') as f:
            np.save(f, target_stds)

        with open(dataset_directory / 'target_mins.npy', 'wb') as f:
            np.save(f, target_mins)

        with open(dataset_directory / 'target_maxs.npy', 'wb') as f:
            np.save(f, target_maxs)

        with open(dataset_directory / 'target_rmss.npy', 'wb') as f:
            np.save(f, target_rmss)

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

        feature_means = features.mean(axis=0)
        feature_stds = features.std(axis=0)
        feature_mins = features.min(axis=0)
        feature_maxs = features.max(axis=0)
        feature_rmss = np.sqrt(np.mean(features ** 2, axis=0))

        with open(dataset_directory / 'feature_means.npy', 'wb') as f:
            np.save(f, feature_means)

        with open(dataset_directory / 'feature_stds.npy', 'wb') as f:
            np.save(f, feature_stds)

        with open(dataset_directory / 'feature_mins.npy', 'wb') as f:
            np.save(f, feature_mins)

        with open(dataset_directory / 'feature_maxs.npy', 'wb') as f:
            np.save(f, feature_maxs)

        with open(dataset_directory / 'feature_rmss.npy', 'wb') as f:
            np.save(f, feature_rmss)
