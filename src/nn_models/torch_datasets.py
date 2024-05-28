from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):

    def __init__(self, sample_ids, dataset_directory, dataset=False, statistics=None):

        self.sample_ids = sample_ids
        self.dataset_directory = dataset_directory
        self.dataset = dataset
        self.statistics = statistics

    def __len__(self):

        """
        Get the length the dataset

        Returns
        -------
        length: int
            Length of the dataset
        """

        return len(self.sample_ids)

    def __getitem__(self, idx):

        """
        Get the idxth element in the dataset

        Parameters
        ----------
        idx: int
            Index of the sample (0 <= idx < length of the dataset)

        Returns
        -------
        features: torch.Tensor of shape (60, 25)
            Features tensor

        targets: torch.Tensor of shape (368)
            Targets tensor
        """

        features = np.load(self.dataset_directory / f'{self.dataset}_features' / f'{self.sample_ids[idx]}.npz')['arr']
        features = (features - self.statistics['feature']['mean']) / self.statistics['feature']['std']
        features = torch.as_tensor(features, dtype=torch.float)

        if self.dataset == 'training':

            targets = np.load(self.dataset_directory / f'{self.dataset}_targets' / f'{self.sample_ids[idx]}.npz')['arr']
            targets = (targets - self.statistics['target']['mean']) / self.statistics['target']['rms']
            targets = torch.as_tensor(targets, dtype=torch.float)

            return features, targets

        else:

            return features


class TabularInMemoryDataset(Dataset):

    def __init__(self, features, targets=None):

        self.features = features
        self.targets = targets

    def __len__(self):

        """
        Get the length the dataset

        Returns
        -------
        length: int
            Length of the dataset
        """

        return len(self.features)

    def __getitem__(self, idx):

        """
        Get the idxth element in the dataset

        Parameters
        ----------
        idx: int
            Index of the sample (0 <= idx < length of the dataset)

        Returns
        -------
        features: torch.Tensor of shape (556)
            Features tensor

        targets: torch.Tensor of shape (368)
            Targets tensor
        """

        features = self.features[idx]
        features = torch.as_tensor(features, dtype=torch.float)

        if self.targets is not None:

            targets = self.targets[idx]
            targets = torch.as_tensor(targets, dtype=torch.float)

            return features, targets

        else:

            return features


def get_feature_tensors(features, feature_statistics=None, feature_normalization_type=None):

    """
    Create inputs tensors

    Parameters
    ----------
    features: numpy.ndarray of shape (556)
        Single features array

    feature_statistics: dict of 4 numpy.ndarray of shape (556)
        Dictionary of mean, std, min and max values of features

    feature_normalization_type: str
        Type of normalization

    Returns
    -------
    features: torch.Tensor of shape (556)
        Features tensor
    """

    if feature_statistics is not None:
        if feature_normalization_type == 'z_score':
            features = (features - feature_statistics['mean']) / feature_statistics['std']
        elif feature_normalization_type == 'min_max':
            features = (features - feature_statistics['min']) / (feature_statistics['max'] - feature_statistics['min'])
        else:
            raise ValueError(f'Invalid normalization type {feature_normalization_type}')

    features = torch.as_tensor(features, dtype=torch.float)

    return features


def get_target_tensors(targets, target_statistics=None, target_normalization_type=None):

    """
    Create outputs tensor

    Parameters
    ----------
    targets: numpy.ndarray of shape (368)
        Single targets array

    target_statistics: dict of 4 numpy.ndarray of shape (368)
        Dictionary of mean, std, min and max values of targets

    target_normalization_type: str
        Type of normalization

    Returns
    -------
    targets: torch.Tensor of shape (368)
        Targets tensor
    """

    if target_statistics is not None:
        if target_normalization_type == 'z_score':
            targets = (targets - target_statistics['mean']) / target_statistics['std']
        elif target_normalization_type == 'min_max':
            targets = (targets - target_statistics['min']) / (target_statistics['max'] - target_statistics['min'])
        else:
            raise ValueError(f'Invalid normalization type {target_normalization_type}')

    targets = torch.as_tensor(targets, dtype=torch.float)

    return targets


def prepare_file_paths(idx, features_path, targets_path=None):

    """
    Create arrays of features paths and targets paths

    Parameters
    ----------
    idx: array-like of shape (n)
        Array of indices from 0 to n

    features_path: str or pathlib.Path
        Path of the features numpy arrays directory

    targets_path: str or pathlib.Path
        Path of the targets numpy arrays directory

    Returns
    -------
    features_paths: numpy.ndarray of shape (n)
        Array of features numpy array file paths

    targets_paths: numpy.ndarray of shape (n)
        Array of targets numpy array file paths
    """

    features_paths = np.array(list(map(lambda x: Path(features_path) / f'{x}.npz', idx)))

    if targets_path is not None:
        targets_paths = np.array(list(map(lambda x: Path(targets_path) / f'{x}.npz', idx)))

        return features_paths, targets_paths
    else:
        return features_paths
