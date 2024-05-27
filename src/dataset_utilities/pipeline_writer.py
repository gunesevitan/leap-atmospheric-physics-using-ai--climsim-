import sys
from tqdm import tqdm
import numpy as np
import polars as pl

sys.path.append('..')
import settings


if __name__ == '__main__':

    dataset_directory = settings.DATA / 'datasets'

    dataset = 'targets'

    if dataset == 'features':

        columns = np.arange(1, 557).tolist()
        dtypes = [pl.Float64 for i in range(len(columns))]
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

        q0002_replace_idx = [
            120, 121, 122, 123, 124, 125, 126, 127, 128,
            129, 130, 131, 132, 133, 134, 135, 136, 137,
            138, 139, 140, 141, 142, 143, 144, 145, 146
        ]
        np.save(dataset_directory / 'train_q0002_replace_features.npy', arr=features[:-625000, q0002_replace_idx])
        np.save(dataset_directory / 'test_q0002_replace_features.npy', arr=features[-625000:, q0002_replace_idx])

        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std = np.where(std <= 1e-9, 1.0, std)

        features -= mean
        features /= std

        features = features.astype(np.float32)
        features = np.concatenate((
            features[:, :360].reshape(features.shape[0], -1, 60),
            np.expand_dims(features[:, 360:376], axis=-1).repeat(repeats=60, axis=-1),
            features[:, 376:].reshape(features.shape[0], -1, 60),
        ), axis=1)

        np.save(dataset_directory / 'train_features.npy', arr=features[:-625000])
        np.save(dataset_directory / 'test_features.npy', arr=features[-625000:])

    elif dataset == 'targets':

        weight_columns = np.arange(1, 369).tolist()
        weight_dtypes = [pl.Float64 for i in range(len(weight_columns))]
        target_weights = pl.read_csv(
            settings.DATA / 'leap-atmospheric-physics-ai-climsim' / 'sample_submission.csv',
            columns=weight_columns,
            dtypes=weight_dtypes,
            n_rows=1,
            n_threads=16
        ).to_numpy().reshape(-1)

        with open(dataset_directory / 'target_weights.npy', 'wb') as f:
            np.save(f, target_weights)

        columns = np.arange(557, 925).tolist()
        dtypes = [pl.Float64 for i in range(len(columns))]
        targets = pl.read_csv(
            settings.DATA / 'leap-atmospheric-physics-ai-climsim' / 'train.csv',
            columns=columns,
            dtypes=dtypes,
            n_threads=16
        )
        targets = targets.to_pandas()
        targets = targets.to_numpy()

        mean = targets.mean(axis=0)
        rms = np.sqrt(np.mean(targets ** 2, axis=0))

        with open(dataset_directory / 'target_means.npy', 'wb') as f:
            np.save(f, mean)

        with open(dataset_directory / 'target_rmss.npy', 'wb') as f:
            np.save(f, rms)

        np.save(dataset_directory / 'train_targets.npy', arr=targets)
