import torch
from torch.utils.data import Dataset

import preprocessing


class TabularDataset(Dataset):

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

        return len(self.features['state_t'])

    def __getitem__(self, idx):

        """
        Get the idxth element in the dataset

        Parameters
        ----------
        idx: int
            Index of the sample (0 <= idx < length of the dataset)

        Returns
        -------
        features: dict
            Dictionary of feature tensors

        targets: dict
            Dictionary of target tensors
        """

        features = {column_group_name: torch.as_tensor(feature[idx], dtype=torch.float) for column_group_name, feature in self.features.items()}

        if self.targets is not None:
            targets = {column_group_name: torch.as_tensor(target[idx], dtype=torch.float) for column_group_name, target in self.targets.items()}
        else:
            targets = None

        return features, targets


def prepare_data(df, has_targets=True):

    """
    Prepare data for tabular dataset

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with features and targets

    has_targets: bool
        Whether the dataset has targets or not

    Returns
    -------
    features: dict
        Dictionary of feature group names and arrays

    targets: dict
        Dictionary of target group names and arrays
    """

    columns = df.columns.tolist()

    feature_column_groups = preprocessing.get_feature_column_groups(columns=columns)
    features = {column_group_name: df[columns].values for column_group_name, columns in feature_column_groups.items()}

    if has_targets:
        target_column_groups = preprocessing.get_target_column_groups(columns=columns)
        targets = {column_group_name: df[columns].values for column_group_name, columns in target_column_groups.items()}
    else:
        targets = None

    return features, targets
