import sys
import numpy as np
import polars as pl

sys.path.append('..')
import settings


if __name__ == '__main__':

    dataset_directory = settings.DATA / 'datasets'
    dataset_directory.mkdir(parents=True, exist_ok=True)

    dataset = 'sample_submission'

    if dataset == 'train':

        columns = np.arange(1, 925).tolist()
        dtypes = [pl.Float32 for i in range(len(columns))]
        df = pl.read_csv(
            settings.DATA / 'leap-atmospheric-physics-ai-climsim' / 'train.csv',
            columns=columns,
            dtypes=dtypes,
            n_threads=16
        )
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
        df.write_parquet(dataset_directory / 'test.parquet')

    elif dataset == 'sample_submission':

        columns = np.arange(1, 369).tolist()
        dtypes = [pl.Float32 for i in range(len(columns))]
        df = pl.read_csv(
            settings.DATA / 'leap-atmospheric-physics-ai-climsim' / 'sample_submission.csv',
            columns=columns,
            dtypes=dtypes,
            n_threads=16
        )
        df.write_parquet(dataset_directory / 'sample_submission.parquet')
