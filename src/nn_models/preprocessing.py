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
