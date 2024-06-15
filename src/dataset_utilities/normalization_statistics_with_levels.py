import sys
import numpy as np
import pandas as pd
import polars as pl

sys.path.append('..')
import settings


if __name__ == '__main__':

    dataset_directory = settings.DATA / 'datasets'
    dataset_directory.mkdir(parents=True, exist_ok=True)

    normalization_columns = 'features'

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

        targets = pd.read_parquet(settings.DATA / 'datasets' / 'train.parquet')
        targets = targets.to_numpy()
        targets = targets[:, -368:]

        scalar_target_means = targets[:, 360:].mean(axis=0)
        scalar_target_stds = targets[:, 360:].std(axis=0)
        scalar_target_rmss = np.sqrt(np.mean(targets[:, 360:] ** 2, axis=0))

        sequential_target_means = []
        sequential_target_stds = []
        sequential_target_rmss = []

        for target_idx in range(6):
            sequential_target = targets[:, target_idx * 60:(target_idx + 1) * 60].flatten()
            sequential_target_means.append(sequential_target.mean())
            sequential_target_stds.append(sequential_target.std())
            sequential_target_rmss.append(np.sqrt(np.mean(sequential_target ** 2)))

        sequential_target_means = np.array(sequential_target_means)
        sequential_target_stds = np.array(sequential_target_stds)
        sequential_target_rmss = np.array(sequential_target_rmss)

        target_means = np.concatenate([sequential_target_means, scalar_target_means])
        target_stds = np.concatenate([sequential_target_stds, scalar_target_stds])
        target_rmss = np.concatenate([sequential_target_rmss, scalar_target_rmss])

        with open(dataset_directory / 'target_means_with_levels.npy', 'wb') as f:
            np.save(f, target_means)

        with open(dataset_directory / 'target_stds_with_levels.npy', 'wb') as f:
            np.save(f, target_stds)

        with open(dataset_directory / 'target_rmss_with_levels.npy', 'wb') as f:
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

        scalar_feature_means = features[:, 360:376].mean(axis=0)
        scalar_feature_stds = features[:, 360:376].std(axis=0)

        feature_means = []
        feature_stds = []

        for feature_idx in range(6):
            sequential_feature = features[:, feature_idx * 60:(feature_idx + 1) * 60].flatten()
            feature_means.append(sequential_feature.mean())
            feature_stds.append(sequential_feature.std())

        for feature_idx in range(16):
            feature_means.append(scalar_feature_means[feature_idx])
            feature_stds.append(scalar_feature_stds[feature_idx])

        for feature_idx in range(3):
            sequential_feature = features[:, 376 + (feature_idx * 60):376 + ((feature_idx + 1) * 60)].flatten()
            feature_means.append(sequential_feature.mean())
            feature_stds.append(sequential_feature.std())

        feature_means = np.array(feature_means)
        feature_stds = np.array(feature_stds)

        with open(dataset_directory / 'feature_means_with_levels.npy', 'wb') as f:
            np.save(f, feature_means)

        with open(dataset_directory / 'feature_stds_with_levels.npy', 'wb') as f:
            np.save(f, feature_stds)
