import sys
import numpy as np
import polars as pl

sys.path.append('..')
import settings


ZERO_VARIANCE_COLUMNS = [
    'pbuf_CH4_27', 'pbuf_CH4_28', 'pbuf_CH4_29', 'pbuf_CH4_30', 'pbuf_CH4_31', 'pbuf_CH4_32', 'pbuf_CH4_33', 'pbuf_CH4_34', 'pbuf_CH4_35', 'pbuf_CH4_36',
    'pbuf_CH4_37', 'pbuf_CH4_38', 'pbuf_CH4_39', 'pbuf_CH4_40', 'pbuf_CH4_41', 'pbuf_CH4_42', 'pbuf_CH4_43', 'pbuf_CH4_44', 'pbuf_CH4_45', 'pbuf_CH4_46',
    'pbuf_CH4_47', 'pbuf_CH4_48', 'pbuf_CH4_49', 'pbuf_CH4_50', 'pbuf_CH4_51', 'pbuf_CH4_52', 'pbuf_CH4_53', 'pbuf_CH4_54', 'pbuf_CH4_55', 'pbuf_CH4_56',
    'pbuf_CH4_57', 'pbuf_CH4_58', 'pbuf_CH4_59', 'pbuf_N2O_27', 'pbuf_N2O_28', 'pbuf_N2O_29', 'pbuf_N2O_30', 'pbuf_N2O_31', 'pbuf_N2O_32', 'pbuf_N2O_33',
    'pbuf_N2O_34', 'pbuf_N2O_35', 'pbuf_N2O_36', 'pbuf_N2O_37', 'pbuf_N2O_38', 'pbuf_N2O_39', 'pbuf_N2O_40', 'pbuf_N2O_41', 'pbuf_N2O_42', 'pbuf_N2O_43',
    'pbuf_N2O_44', 'pbuf_N2O_45', 'pbuf_N2O_46', 'pbuf_N2O_47', 'pbuf_N2O_48', 'pbuf_N2O_49', 'pbuf_N2O_50', 'pbuf_N2O_51', 'pbuf_N2O_52', 'pbuf_N2O_53',
    'pbuf_N2O_54', 'pbuf_N2O_55', 'pbuf_N2O_56', 'pbuf_N2O_57', 'pbuf_N2O_58', 'pbuf_N2O_59', 'ptend_q0001_10', 'ptend_q0001_11', 'ptend_q0001_4', 'ptend_q0001_5',
    'ptend_q0001_6', 'ptend_q0001_7', 'ptend_q0001_8', 'ptend_q0001_9', 'ptend_q0002_0', 'ptend_q0002_1', 'ptend_q0002_10', 'ptend_q0002_11', 'ptend_q0002_12',
    'ptend_q0002_13', 'ptend_q0002_14', 'ptend_q0002_15', 'ptend_q0002_16', 'ptend_q0002_17', 'ptend_q0002_18', 'ptend_q0002_19', 'ptend_q0002_2', 'ptend_q0002_20',
    'ptend_q0002_21', 'ptend_q0002_22', 'ptend_q0002_23', 'ptend_q0002_3', 'ptend_q0002_4', 'ptend_q0002_5', 'ptend_q0002_6', 'ptend_q0002_7', 'ptend_q0002_8',
    'ptend_q0002_9', 'ptend_q0003_0', 'ptend_q0003_1', 'ptend_q0003_10', 'ptend_q0003_11', 'ptend_q0003_2', 'ptend_q0003_3', 'ptend_q0003_4', 'ptend_q0003_5',
    'ptend_q0003_6', 'ptend_q0003_7', 'ptend_q0003_8', 'ptend_q0003_9', 'ptend_u_0', 'ptend_u_1', 'ptend_u_10', 'ptend_u_11', 'ptend_u_2', 'ptend_u_3',
    'ptend_u_4', 'ptend_u_5', 'ptend_u_6', 'ptend_u_7', 'ptend_u_8', 'ptend_u_9', 'ptend_v_0', 'ptend_v_1', 'ptend_v_10', 'ptend_v_11', 'ptend_v_2',
    'ptend_v_3', 'ptend_v_4', 'ptend_v_5', 'ptend_v_6', 'ptend_v_7', 'ptend_v_8', 'ptend_v_9', 'state_q0002_0', 'state_q0002_1', 'state_q0002_10',
    'state_q0002_11', 'state_q0002_12', 'state_q0002_13', 'state_q0002_14', 'state_q0002_15', 'state_q0002_16', 'state_q0002_17', 'state_q0002_18',
    'state_q0002_19', 'state_q0002_2', 'state_q0002_20', 'state_q0002_21', 'state_q0002_22', 'state_q0002_23', 'state_q0002_3', 'state_q0002_4',
    'state_q0002_5', 'state_q0002_6', 'state_q0002_7', 'state_q0002_8', 'state_q0002_9'
]

ZERO_VARIANCE_FEATURE_COLUMNS = [
    'pbuf_CH4_53', 'pbuf_N2O_54', 'pbuf_CH4_29', 'pbuf_N2O_33', 'pbuf_CH4_31', 'pbuf_CH4_58', 'state_q0002_0', 'pbuf_N2O_47', 'pbuf_N2O_38', 'pbuf_CH4_36',
    'state_q0002_4', 'pbuf_CH4_43', 'state_q0002_11', 'pbuf_CH4_52', 'pbuf_N2O_37', 'pbuf_CH4_30', 'pbuf_CH4_27', 'pbuf_N2O_41', 'pbuf_N2O_56', 'state_q0002_6',
    'state_q0002_17', 'pbuf_N2O_46', 'state_q0002_2', 'pbuf_N2O_50', 'state_q0002_13', 'state_q0002_15', 'pbuf_CH4_35', 'pbuf_CH4_50', 'pbuf_N2O_59', 'pbuf_CH4_48',
    'pbuf_CH4_42', 'pbuf_N2O_49', 'pbuf_CH4_41', 'state_q0002_18', 'pbuf_CH4_32', 'pbuf_CH4_57', 'pbuf_CH4_59', 'pbuf_CH4_54', 'pbuf_N2O_43', 'pbuf_N2O_42',
    'pbuf_N2O_48', 'pbuf_N2O_40', 'state_q0002_19', 'pbuf_N2O_30', 'pbuf_CH4_45', 'pbuf_N2O_57', 'state_q0002_23', 'state_q0002_9', 'pbuf_N2O_51', 'pbuf_CH4_34',
    'pbuf_CH4_38', 'state_q0002_8', 'state_q0002_3', 'pbuf_N2O_27', 'pbuf_N2O_39', 'pbuf_N2O_53', 'state_q0002_12', 'pbuf_N2O_34', 'state_q0002_7', 'pbuf_CH4_37',
    'pbuf_N2O_44', 'pbuf_N2O_52', 'pbuf_N2O_58', 'pbuf_N2O_45', 'pbuf_CH4_39', 'pbuf_CH4_40', 'pbuf_CH4_47', 'pbuf_N2O_28', 'pbuf_CH4_33', 'pbuf_CH4_46', 'pbuf_CH4_56',
    'pbuf_N2O_35', 'pbuf_CH4_28', 'pbuf_CH4_55', 'pbuf_N2O_31', 'state_q0002_1', 'state_q0002_20', 'pbuf_CH4_49', 'state_q0002_22', 'state_q0002_5', 'pbuf_CH4_44',
    'pbuf_N2O_55', 'pbuf_N2O_32', 'pbuf_CH4_51', 'pbuf_N2O_29', 'pbuf_N2O_36', 'state_q0002_14', 'state_q0002_10', 'state_q0002_21', 'state_q0002_16'
]

ZERO_VARIANCE_TARGET_COLUMNS = [
    'ptend_q0002_2', 'ptend_q0002_11', 'ptend_q0003_1', 'ptend_q0003_7', 'ptend_u_2', 'ptend_q0001_9', 'ptend_q0003_5', 'ptend_q0003_8', 'ptend_q0001_11', 'ptend_v_4',
    'ptend_q0003_0', 'ptend_u_8', 'ptend_u_4', 'ptend_q0001_5', 'ptend_u_10', 'ptend_u_1', 'ptend_q0001_10', 'ptend_q0002_10', 'ptend_v_1', 'ptend_q0002_17', 'ptend_q0003_2',
    'ptend_v_11', 'ptend_v_7', 'ptend_q0002_15', 'ptend_q0002_12', 'ptend_q0002_0', 'ptend_q0002_21', 'ptend_q0002_23', 'ptend_u_11', 'ptend_q0002_14', 'ptend_q0002_20',
    'ptend_u_5', 'ptend_v_0', 'ptend_q0002_13', 'ptend_u_6', 'ptend_v_2', 'ptend_u_3', 'ptend_q0002_9', 'ptend_q0002_16', 'ptend_q0002_5', 'ptend_q0003_9', 'ptend_q0002_3',
    'ptend_q0002_22', 'ptend_u_0', 'ptend_v_6', 'ptend_q0001_6', 'ptend_u_9', 'ptend_q0002_7', 'ptend_v_5', 'ptend_v_9', 'ptend_q0003_6', 'ptend_q0003_3', 'ptend_q0002_19',
    'ptend_v_8', 'ptend_q0002_6', 'ptend_u_7', 'ptend_q0001_4', 'ptend_q0001_7', 'ptend_q0002_1', 'ptend_q0002_4', 'ptend_q0003_10', 'ptend_v_10', 'ptend_q0002_8', 'ptend_v_3',
    'ptend_q0002_18', 'ptend_q0001_8', 'ptend_q0003_11', 'ptend_q0003_4'
]

if __name__ == '__main__':

    dataset_directory = settings.DATA / 'datasets'
    dataset_directory.mkdir(parents=True, exist_ok=True)

    dataset = 'test'

    if dataset == 'train':

        columns = np.arange(1, 925).tolist()
        dtypes = [pl.Float32 for i in range(len(columns))]
        df = pl.read_csv(
            settings.DATA / 'leap-atmospheric-physics-ai-climsim' / 'train.csv',
            columns=columns,
            dtypes=dtypes,
            n_threads=16
        )
        df = df.sample(n=1000000, with_replacement=False, shuffle=True, seed=42)
        #df = df.drop(ZERO_VARIANCE_FEATURE_COLUMNS)

        df.write_parquet(dataset_directory / 'train.parquet')

    elif dataset == 'test':

        columns = np.arange(1, 557).tolist()
        dtypes = [pl.Float32 for i in range(len(columns))]
        df = pl.read_csv(
            settings.DATA / 'leap-atmospheric-physics-ai-climsim' / 'test.csv',
            columns=columns,
            dtypes=dtypes,
            n_threads=16
        )
        #df = df.drop(ZERO_VARIANCE_FEATURE_COLUMNS)
        df.write_parquet(dataset_directory / 'test.parquet')

    elif dataset == 'sample_submission':

        columns = np.arange(0, 369).tolist()
        dtypes = [pl.String] + [pl.Float32 for i in range(len(columns) - 1)]
        df = pl.read_csv(
            settings.DATA / 'leap-atmospheric-physics-ai-climsim' / 'sample_submission.csv',
            columns=columns,
            dtypes=dtypes,
            n_threads=16
        )
        df.write_parquet(dataset_directory / 'sample_submission.parquet')
