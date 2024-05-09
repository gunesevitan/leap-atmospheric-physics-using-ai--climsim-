import sys
from tqdm import tqdm
import numpy as np

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
                features = np.array(ln[1:557], dtype=np.float32)
                np.savez_compressed(training_features_directory / f'{idx}.npz', arr=features)
                targets = np.array(ln[557:], dtype=np.float32)
                np.savez_compressed(training_targets_directory / f'{idx}.npz', arr=targets)

    elif dataset == 'test':

        with open(settings.DATA / 'leap-atmospheric-physics-ai-climsim' / 'test.csv') as f:

            next(f)

            for idx, line in enumerate(tqdm(f, total=625000)):
                ln = line.split(',')
                features = np.array(ln[1:557], dtype=np.float32)
                np.savez_compressed(test_features_directory / f'{idx}.npz', arr=features)
