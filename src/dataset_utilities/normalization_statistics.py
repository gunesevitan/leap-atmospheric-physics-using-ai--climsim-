import sys
import numpy as np
import pandas as pd
import polars as pl

sys.path.append('..')
import settings


if __name__ == '__main__':

    dataset_directory = settings.DATA / 'datasets'
    dataset_directory.mkdir(parents=True, exist_ok=True)

    normalization_columns = 'targets'

    if normalization_columns == 'targets':

        weight_columns = np.arange(1, 369).tolist()
        weight_dtypes = [pl.Float32 for i in range(len(weight_columns))]
        target_weights = pl.read_csv(
            settings.DATA / 'leap-atmospheric-physics-ai-climsim' / 'sample_submission.csv',
            columns=weight_columns,
            dtypes=weight_dtypes,
            n_rows=1,
            n_threads=16
        ).to_numpy().reshape(-1)

        with open(settings.DATA / 'target_weights.npy', 'wb') as f:
            np.save(f, target_weights)

        targets = pd.read_parquet(settings.DATA / 'datasets' / 'train.parquet')
        targets = targets.to_numpy()
        targets = targets[:, -368:]
        
        target_means = targets.mean(axis=0)
        target_stds = targets.std(axis=0)
        target_mins = targets.min(axis=0)
        target_maxs = targets.max(axis=0)
        target_rmss = np.sqrt(np.mean(targets ** 2, axis=0))

        with open(dataset_directory / 'target_means.npy', 'wb') as f:
            np.save(f, target_means)

        with open(dataset_directory / 'target_stds.npy', 'wb') as f:
            np.save(f, target_stds)

        with open(dataset_directory / 'target_mins.npy', 'wb') as f:
            np.save(f, target_mins)

        with open(dataset_directory / 'target_maxs.npy', 'wb') as f:
            np.save(f, target_maxs)

        with open(dataset_directory / 'target_rmss.npy', 'wb') as f:
            np.save(f, target_rmss)

    elif normalization_columns == 'features':

        columns = np.arange(1, 557).tolist()
        dtypes = [pl.Float32 for i in range(len(columns))]
        features = pl.read_csv(
            settings.DATA / 'leap-atmospheric-physics-ai-climsim' / 'train.csv',
            columns=columns,
            dtypes=dtypes,
            n_rows=5_000_000,
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

        feature_means = features.mean(axis=0)
        feature_stds = features.std(axis=0)

        sequential_features = np.concatenate((
            features[:, :360].reshape(-1, 6, 60),
            features[:, 376:].reshape(-1, 3, 60)
        ), axis=1)

        wind_speed = np.sqrt(sequential_features[:, 4, :] ** 2 + sequential_features[:, 5, :] ** 2)
        wind_direction = (np.degrees(np.arctan2(sequential_features[:, 4, :], sequential_features[:, 5, :])) + 180) % 360

        Ak = np.array([5.59e-05, 0.0001008, 0.0001814, 0.0003244, 0.0005741, 0.0009986, 0.0016961, 0.0027935, 0.0044394, 0.0067923, 0.0100142, 0.0142748, 0.0197589, 0.0266627, 0.035166, 0.0453892, 0.0573601, 0.0710184, 0.086261, 0.1029992, 0.1211833, 0.1407723, 0.1616703, 0.181999, 0.1769112, 0.1717129, 0.1664573, 0.1611637, 0.1558164, 0.1503775, 0.144805, 0.1390666, 0.1331448, 0.1270342, 0.1207383, 0.11427, 0.107658, 0.1009552, 0.0942421, 0.0876184, 0.0811846, 0.0750186, 0.0691602, 0.06361, 0.0583443, 0.0533368, 0.0485757, 0.044067, 0.039826, 0.0358611, 0.0321606, 0.0286887, 0.0253918, 0.0222097, 0.0190872, 0.0159809, 0.0128614, 0.0097109, 0.00652, 0.0032838])
        Bk = np.array([0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0016785, 0.0295868, 0.058101, 0.0869295, 0.1159665, 0.145298, 0.1751322, 0.2056993, 0.2371761, 0.2696592, 0.3031777, 0.3377127, 0.3731935, 0.4094624, 0.4462289, 0.4830525, 0.5193855, 0.5546772, 0.5884994, 0.6206347, 0.6510795, 0.6799635, 0.7074307, 0.7335472, 0.7582786, 0.7815416, 0.8032905, 0.8235891, 0.8426334, 0.8607178, 0.8781726, 0.8953009, 0.9123399, 0.9294513, 0.9467325, 0.9642358, 0.9819873])

        temperature = sequential_features[:, 0, :].clip(165, 321)
        pressure = Ak + Bk * features[:, 360].reshape(-1, 1) / 100000.0
        relative_humidity = sequential_features[:, 1, :] * 263 * pressure
        relative_humidity = relative_humidity * np.exp(-17.67 * (temperature - 273.16) / (temperature - 29.65))

        feature_means = np.hstack([
            feature_means,
            wind_speed.mean(axis=0),
            wind_direction.mean(axis=0),
            relative_humidity.mean(axis=0),
        ])

        feature_stds = np.hstack([
            feature_stds,
            wind_speed.std(axis=0),
            wind_direction.std(axis=0),
            relative_humidity.std(axis=0),
        ])

        with open(dataset_directory / 'feature_means.npy', 'wb') as f:
            np.save(f, feature_means)

        with open(dataset_directory / 'feature_stds.npy', 'wb') as f:
            np.save(f, feature_stds)
