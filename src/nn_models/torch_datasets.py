from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):

    def __init__(
            self,
            feature_paths, target_paths=None,
            feature_statistics=None, target_statistics=None, feature_normalization_type=None, target_normalization_type=None
    ):

        self.feature_paths = feature_paths
        self.target_paths = target_paths
        self.feature_statistics = feature_statistics
        self.target_statistics = target_statistics
        self.feature_normalization_type = feature_normalization_type
        self.target_normalization_type = target_normalization_type

    def __len__(self):

        """
        Get the length the dataset

        Returns
        -------
        length: int
            Length of the dataset
        """

        return len(self.feature_paths)

    def __getitem__(self, idx):

        """
        Get the idxth element in the dataset

        Parameters
        ----------
        idx: int
            Index of the sample (0 <= idx < length of the dataset)

        Returns
        -------
        features: dict of 25 torch.Tensor of shape 60 or 1
            Dictionary of feature tensors

        targets: dict of 14 torch.Tensor of shape 60 or 1
            Dictionary of target tensors
        """

        features = np.load(self.feature_paths[idx])['arr']
        features = get_feature_tensors(
            features=features,
            feature_statistics=self.feature_statistics,
            feature_normalization_type=self.feature_normalization_type
        )

        if self.target_paths is not None:
            targets = np.load(self.target_paths[idx])['arr']
            targets = get_target_tensors(
                targets=targets,
                target_statistics=self.target_statistics,
                target_normalization_type=self.target_normalization_type
            )

            return features, targets
        else:
            return features


def get_feature_tensors(features, feature_statistics=None, feature_normalization_type=None):

    """
    Create dictionary of feature tensors

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
    features: dict of 25 torch.Tensor of shape 60 or 1
        Dictionary of feature tensors
    """

    if feature_statistics is not None:
        if feature_normalization_type == 'z_score':
            features = (features - feature_statistics['mean']) / feature_statistics['std']
        elif feature_normalization_type == 'min_max':
            features = (features - feature_statistics['min']) / (feature_statistics['max'] - feature_statistics['min'])
        else:
            raise ValueError(f'Invalid normalization type {feature_normalization_type}')

    features = {
        'state_t': features[0:60],
        'state_q0001': features[60:120],
        'state_q0002': features[120:180],
        'state_q0003': features[180:240],
        'state_u': features[240:300],
        'state_v': features[300:360],
        'state_ps': features[360],
        'pbuf_SOLIN': features[361],
        'pbuf_LHFLX': features[362],
        'pbuf_SHFLX': features[363],
        'pbuf_TAUX': features[364],
        'pbuf_TAUY': features[365],
        'pbuf_COSZRS': features[366],
        'cam_in_ALDIF': features[367],
        'cam_in_ALDIR': features[368],
        'cam_in_ASDIF': features[369],
        'cam_in_ASDIR': features[370],
        'cam_in_LWUP': features[371],
        'cam_in_ICEFRAC': features[372],
        'cam_in_LANDFRAC': features[373],
        'cam_in_OCNFRAC': features[374],
        'cam_in_SNOWHLAND': features[375],
        'pbuf_ozone': features[376:436],
        'pbuf_CH4': features[436:496],
        'pbuf_N2O': features[496:556],
    }
    features = {k: torch.as_tensor(v, dtype=torch.float).reshape(-1) for k, v in features.items()}

    return features


def get_target_tensors(targets, target_statistics=None, target_normalization_type=None):

    """
    Create dictionary of target tensors

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
    targets: dict of 14 torch.Tensor of shape 60 or 1
        Dictionary of target tensors
    """

    if target_statistics is not None:
        if target_normalization_type == 'z_score':
            targets = (targets - target_statistics['mean']) / target_statistics['std']
        elif target_normalization_type == 'min_max':
            targets = (targets - target_statistics['min']) / (target_statistics['max'] - target_statistics['min'])
        else:
            raise ValueError(f'Invalid normalization type {target_normalization_type}')

    targets = {
        'ptend_t': targets[0:60],
        'ptend_q0001': targets[60:120],
        'ptend_q0002': targets[120:180],
        'ptend_q0003': targets[180:240],
        'ptend_u': targets[240:300],
        'ptend_v': targets[300:360],
        'cam_out_NETSW': targets[360],
        'cam_out_FLWDS': targets[361],
        'cam_out_PRECSC': targets[362],
        'cam_out_PRECC': targets[363],
        'cam_out_SOLS': targets[364],
        'cam_out_SOLL': targets[365],
        'cam_out_SOLSD': targets[366],
        'cam_out_SOLLD': targets[367],
    }
    targets = {k: torch.as_tensor(v, dtype=torch.float).reshape(-1) for k, v in targets.items()}

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
