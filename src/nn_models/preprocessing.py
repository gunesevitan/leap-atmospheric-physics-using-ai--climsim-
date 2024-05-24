import numpy as np


def load_normalizers(normalizer_directory):

    """
    Load normalizers from given directory

    Parameters
    ----------
    normalizer_directory: str or pathlib.Path
        Path of the normalizer directory

    Returns
    -------
    normalizers: dict
        Dictionary of normalization statistics and objects
    """

    feature_means = np.load(normalizer_directory / 'feature_means.npy')
    feature_stds = np.load(normalizer_directory / 'feature_stds.npy')
    feature_mins = np.load(normalizer_directory / 'feature_mins.npy')
    feature_maxs = np.load(normalizer_directory / 'feature_maxs.npy')
    feature_rmss = np.load(normalizer_directory / 'feature_rmss.npy')

    feature_log_means = np.load(normalizer_directory / 'feature_log_means.npy')
    feature_log_stds = np.load(normalizer_directory / 'feature_log_stds.npy')
    feature_log_mins = np.load(normalizer_directory / 'feature_log_mins.npy')
    feature_log_maxs = np.load(normalizer_directory / 'feature_log_maxs.npy')
    feature_log_rmss = np.load(normalizer_directory / 'feature_log_rmss.npy')

    target_weights = np.load(normalizer_directory / 'target_weights.npy')

    target_means = np.load(normalizer_directory / 'target_means.npy')
    target_stds = np.load(normalizer_directory / 'target_stds.npy')
    target_mins = np.load(normalizer_directory / 'target_mins.npy')
    target_maxs = np.load(normalizer_directory / 'target_maxs.npy')
    target_rmss = np.load(normalizer_directory / 'target_rmss.npy')

    target_log_means = np.load(normalizer_directory / 'target_log_means.npy')
    target_log_stds = np.load(normalizer_directory / 'target_log_stds.npy')
    target_log_mins = np.load(normalizer_directory / 'target_log_mins.npy')
    target_log_maxs = np.load(normalizer_directory / 'target_log_maxs.npy')
    target_log_rmss = np.load(normalizer_directory / 'target_log_rmss.npy')

    normalizers = {
        'feature': {
            'mean': feature_means,
            'std': feature_stds,
            'min': feature_mins,
            'max': feature_maxs,
            'rms': feature_rmss,
            'log_mean': feature_log_means,
            'log_std': feature_log_stds,
            'log_min': feature_log_mins,
            'log_max': feature_log_maxs,
            'log_rms': feature_log_rmss,
        },
        'target': {
            'weight': target_weights,
            'mean': target_means,
            'std': target_stds,
            'min': target_mins,
            'max': target_maxs,
            'rms': target_rmss,
            'log_mean': target_log_means,
            'log_std': target_log_stds,
            'log_min': target_log_mins,
            'log_max': target_log_maxs,
            'log_rms': target_log_rmss,
        }
    }

    return normalizers


def get_target_column_groups(columns):

    """
    Get names of target column groups and individual columns

    Parameters
    ----------
    columns: list
        List of column names

    Returns
    -------
    column_groups: dict
        Dictionary of target column group names and list of column names
    """

    ptend_t_columns = [column for column in columns if column.startswith('ptend_t')]
    ptend_q0001_columns = [column for column in columns if column.startswith('ptend_q0001')]
    ptend_q0002_columns = [column for column in columns if column.startswith('ptend_q0002')]
    ptend_q0003_columns = [column for column in columns if column.startswith('ptend_q0003')]
    ptend_u_columns = [column for column in columns if column.startswith('ptend_u')]
    ptend_v_columns = [column for column in columns if column.startswith('ptend_v')]
    other_columns = [
        'cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 'cam_out_PRECC',
        'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD'
    ]

    column_groups = {
        'ptend_t': ptend_t_columns,
        'ptend_q0001': ptend_q0001_columns,
        'ptend_q0002': ptend_q0002_columns,
        'ptend_q0003': ptend_q0003_columns,
        'ptend_u': ptend_u_columns,
        'ptend_v': ptend_v_columns
    }

    for column in other_columns:
        column_groups[column] = [column]

    return column_groups


def get_feature_column_groups(columns):

    """
    Get names of feature column groups and individual columns

    Parameters
    ----------
    columns: list
        List of column names

    Returns
    -------
    column_groups: dict
        Dictionary of feature column group names and list of column names
    """

    state_t_columns = [column for column in columns if column.startswith('state_t')]
    state_q0001_columns = [column for column in columns if column.startswith('state_q0001')]
    state_q0002_columns = [column for column in columns if column.startswith('state_q0002')]
    state_q0003_columns = [column for column in columns if column.startswith('state_q0003')]
    state_u_columns = [column for column in columns if column.startswith('state_u')]
    state_v_columns = [column for column in columns if column.startswith('state_v')]
    pbuf_ozone_columns = [column for column in columns if column.startswith('pbuf_ozone')]
    pbuf_CH4_columns = [column for column in columns if column.startswith('pbuf_CH4')]
    pbuf_N2O_columns = [column for column in columns if column.startswith('pbuf_N2O')]
    other_columns = [
        'state_ps', 'pbuf_SOLIN', 'pbuf_LHFLX', 'pbuf_SHFLX',
        'pbuf_TAUX', 'pbuf_TAUY', 'pbuf_COSZRS', 'cam_in_ALDIF',
        'cam_in_ALDIR', 'cam_in_ASDIF', 'cam_in_ASDIR', 'cam_in_LWUP',
        'cam_in_ICEFRAC', 'cam_in_LANDFRAC', 'cam_in_OCNFRAC', 'cam_in_SNOWHLAND'
    ]

    column_groups = {
        'state_t': state_t_columns,
        'state_q0001': state_q0001_columns,
        'state_q0002': state_q0002_columns,
        'state_q0003': state_q0003_columns,
        'state_u': state_u_columns,
        'state_v': state_v_columns,
        'pbuf_ozone': pbuf_ozone_columns,
        'pbuf_CH4': pbuf_CH4_columns,
        'pbuf_N2O': pbuf_N2O_columns,
    }

    for column in other_columns:
        column_groups[column] = [column]

    return column_groups
