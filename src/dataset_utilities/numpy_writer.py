import sys
from tqdm import tqdm
import numpy as np
import polars as pl

sys.path.append('..')
import settings


if __name__ == '__main__':

    dataset_directory = settings.DATA / 'datasets' / 'numpy_arrays'
    training_features_directory = dataset_directory / 'training_features'
    training_features_directory.mkdir(exist_ok=True, parents=True)
    training_targets_directory = dataset_directory / 'training_targets'
    training_targets_directory.mkdir(exist_ok=True, parents=True)
    test_features_directory = dataset_directory / 'test_features'
    test_features_directory.mkdir(exist_ok=True, parents=True)

    dataset = 'test'

    if dataset == 'train':

        with open(settings.DATA / 'leap-atmospheric-physics-ai-climsim' / 'train.csv') as f:

            next(f)

            for idx, line in enumerate(tqdm(f, total=10091520)):

                ln = line.split(',')
                features = np.array(ln[1:557], dtype=np.float64)
                features = np.concatenate((
                    features[:360].reshape(-1, 60),
                    np.expand_dims(features[360:376], axis=-1).repeat(repeats=60, axis=-1),
                    features[376:].reshape(-1, 60),
                ), axis=0)

                np.savez_compressed(training_features_directory / f'{idx}.npz', arr=features)
                targets = np.array(ln[557:], dtype=np.float64)
                np.savez_compressed(training_targets_directory / f'{idx}.npz', arr=targets)

    elif dataset == 'test':

        with open(settings.DATA / 'leap-atmospheric-physics-ai-climsim' / 'test.csv') as f:

            next(f)

            for idx, line in enumerate(tqdm(f, total=625000)):
                ln = line.split(',')
                features = np.array(ln[1:557], dtype=np.float64)
                features = np.concatenate((
                    features[:360].reshape(-1, 60),
                    np.expand_dims(features[360:376], axis=-1).repeat(repeats=60, axis=-1),
                    features[376:].reshape(-1, 60),
                ), axis=0)
                np.savez_compressed(test_features_directory / f'{idx}.npz', arr=features)

    elif dataset == 'train_features':

        columns = np.arange(1, 557).tolist()
        dtypes = [pl.Float32 for i in range(len(columns))]
        features = pl.read_csv(
            settings.DATA / 'leap-atmospheric-physics-ai-climsim' / 'train.csv',
            columns=columns,
            dtypes=dtypes,
            n_threads=16
        ).to_numpy()

        features = np.concatenate((
            features[:, :360].reshape(features.shape[0], -1, 60),
            np.expand_dims(features[:, 360:376], axis=-1).repeat(repeats=60, axis=-1),
            features[:, 376:].reshape(features.shape[0], -1, 60),
        ), axis=1)

        np.savez_compressed(settings.DATA / 'datasets' / 'train_features.npz', arr=features)

    elif dataset == 'train_targets':

        columns = np.arange(1, 369).tolist()
        dtypes = [pl.Float32 for i in range(len(columns))]
        targets = pl.read_csv(
            settings.DATA / 'leap-atmospheric-physics-ai-climsim' / 'train.csv',
            columns=columns,
            dtypes=dtypes,
            n_threads=16
        ).to_numpy()

        np.savez_compressed(settings.DATA / 'datasets' / 'train_targets.npz', arr=targets)
