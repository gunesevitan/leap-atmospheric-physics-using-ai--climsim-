import sys
import numpy as np
import pandas as pd
import polars as pl

sys.path.append('..')
import settings


if __name__ == '__main__':

    dataset_directory = settings.DATA / 'datasets'
    dataset_directory.mkdir(parents=True, exist_ok=True)

    normalization_columns = 'features'

    if normalization_columns == 'targets':

        targets = pd.read_parquet(settings.DATA / 'datasets' / 'train.parquet')
        targets = targets.to_numpy()
        targets = targets[:, -368:]

        sequential_targets = targets[:, :360].reshape(-1, 6, 60)
        gradients = np.diff(sequential_targets, n=1, prepend=sequential_targets[:, :, 0:1]).reshape(-1, 360)

        means = gradients.mean(axis=0)
        stds = gradients.std(axis=0)
        rmss = np.sqrt(np.mean(gradients ** 2, axis=0))

        with open(dataset_directory / 'target_gradient_means.npy', 'wb') as f:
            np.save(f, means)

        with open(dataset_directory / 'target_gradient_stds.npy', 'wb') as f:
            np.save(f, stds)

        with open(dataset_directory / 'target_gradient_rmss.npy', 'wb') as f:
            np.save(f, rmss)

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
        del features

        gradients1 = np.diff(sequential_features, n=1, prepend=sequential_features[:, :, :1], axis=2)
        del sequential_features

        gradients2 = np.diff(wind_speed, n=1, prepend=wind_speed[:, :1], axis=1)
        gradients3 = np.diff(wind_direction, n=1, prepend=wind_direction[:, :1], axis=1)
        gradients4 = np.diff(relative_humidity, n=1, prepend=relative_humidity[:, :1], axis=1)

        gradients1 = gradients1.reshape(-1, 540)
        gradients2 = gradients2.reshape(-1, 60)
        gradients3 = gradients3.reshape(-1, 60)
        gradients4 = gradients4.reshape(-1, 60)

        means1 = gradients1.mean(axis=0)
        stds1 = gradients1.std(axis=0)
        means2 = gradients2.mean(axis=0)
        stds2 = gradients2.std(axis=0)
        means3 = gradients3.mean(axis=0)
        stds3 = gradients3.std(axis=0)
        means4 = gradients4.mean(axis=0)
        stds4 = gradients4.std(axis=0)
        means = np.hstack([means1, means2, means3, means4])
        stds = np.hstack([stds1, stds2, stds3, stds4])

        with open(dataset_directory / 'feature_gradient_means.npy', 'wb') as f:
            np.save(f, means)

        with open(dataset_directory / 'feature_gradient_stds.npy', 'wb') as f:
            np.save(f, stds)
