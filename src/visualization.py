import numpy as np
import matplotlib.pyplot as plt


def visualize_learning_curve(training_scores, validation_scores, best_epoch, metric, path=None):

    """
    Visualize learning curves of the models

    Parameters
    ----------
    training_scores: list of shape (n_epochs)
        List of training losses or scores

    validation_scores: list of shape (n_epochs)
        List of validation losses or scores

    best_epoch: int or None
        Epoch with the best validation loss or score

    metric: str
        Name of the metric

    path: str, pathlib.Path or None
        Path of the output file (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(18, 8), dpi=100)
    ax.plot(np.arange(1, len(training_scores) + 1), training_scores, '-o', linewidth=2, label=f'Training {metric} (best: {training_scores[best_epoch]:.4f})')
    ax.plot(np.arange(1, len(validation_scores) + 1), validation_scores, '-o', linewidth=2, label=f'Validation {metric} (best: {validation_scores[best_epoch]:.4f})')
    ax.axvline(best_epoch + 1, color='r', label=f'Best Epoch: {best_epoch + 1}')

    ax.set_xlabel('Epochs/Steps', size=15, labelpad=12.5)
    ax.set_ylabel('Losses/Metrics', size=15, labelpad=12.5)
    ax.set_xticks(np.arange(1, len(validation_scores) + 1), np.arange(1, len(validation_scores) + 1))

    ax.tick_params(axis='x', labelsize=12.5, pad=10)
    ax.tick_params(axis='y', labelsize=12.5, pad=10)
    ax.legend(prop={'size': 18})
    ax.set_title(f'{metric} Learning Curve', size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_scores(scores, title, path=None):

    """
    Visualize scores of models

    Parameters
    ----------
    scores: pandas.DataFrame of shape (n_splits + 1, n_metrics)
        Dataframe with one or multiple scores and metrics of folds and oof scores

    title: str
        Title of the plot

    path: str, pathlib.Path or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    metric_columns = [column for column in scores.columns.tolist() if column != 'fold']
    # Create mean and std of scores for error bars
    fold_scores = scores.loc[scores['fold'] != 'OOF', metric_columns].agg(['mean', 'std']).T.fillna(0)
    oof_scores = scores.loc[scores['fold'] == 'OOF', metric_columns].reset_index(drop=True).T.rename(columns={0: 'score'})

    fig, ax = plt.subplots(figsize=(32, 12))
    ax.barh(
        y=np.arange(fold_scores.shape[0]) - 0.2,
        width=fold_scores['mean'],
        height=0.4,
        xerr=fold_scores['std'],
        align='center',
        ecolor='black',
        capsize=10,
        label='Fold Scores'
    )
    ax.barh(
        y=np.arange(oof_scores.shape[0]) + 0.2,
        width=oof_scores['score'],
        height=0.4,
        align='center',
        capsize=10,
        label='OOF Scores'
    )
    ax.set_yticks(np.arange(fold_scores.shape[0]))
    ax.set_yticklabels([
        f'{metric}\nOOF: {oof:.4f}\nMean: {mean:.4f} (±{std:.4f})' for metric, mean, std, oof in zip(
            fold_scores.index,
            fold_scores['mean'].values,
            fold_scores['std'].values,
            oof_scores['score'].values
        )
    ])
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(title, size=20, pad=15)
    ax.legend(loc='best', prop={'size': 18})

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_single_target_scores(single_target_scores, title, path=None):

    """
    Visualize scores of single targets

    Parameters
    ----------
    single_target_scores: list of pandas.DataFrame of shape (n_splits + 1, n_targets, n_metrics)
        Dataframes with one or multiple targets and scores

    title: str
        Title of the plot

    path: str, pathlib.Path or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    target_columns = single_target_scores[0].index.tolist()
    metric_columns = single_target_scores[0].columns.tolist()
    fold_single_target_scores = np.stack([df.loc[target_columns] for df in single_target_scores[:-1]])
    mean_fold_single_target_scores = fold_single_target_scores.mean(axis=0)
    std_fold_single_target_scores = fold_single_target_scores.std(axis=0)
    oof_single_target_scores = single_target_scores[-1]

    fig, axes = plt.subplots(figsize=(72, 16), ncols=len(metric_columns))

    for i in range(len(metric_columns)):

        axes[i].barh(
            y=np.arange(fold_single_target_scores.shape[1]) - 0.2,
            width=mean_fold_single_target_scores[:, i],
            height=0.4,
            xerr=std_fold_single_target_scores[:, i],
            align='center',
            ecolor='black',
            capsize=10,
            label='Fold Scores'
        )
        axes[i].barh(
            y=np.arange(fold_single_target_scores.shape[1]) + 0.2,
            width=oof_single_target_scores.iloc[:, i].values,
            height=0.4,
            align='center',
            capsize=10,
            label='OOF Scores'
        )

        axes[i].set_yticks(np.arange(fold_single_target_scores.shape[1]))
        axes[i].set_yticklabels([
            f'{metric}\nOOF: {oof:.4f}\nMean: {mean:.4f} (±{std:.4f})' for metric, mean, std, oof in zip(
                target_columns,
                mean_fold_single_target_scores[:, i],
                std_fold_single_target_scores[:, i],
                oof_single_target_scores.iloc[:, i]
            )
        ])
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', labelsize=15, pad=10)
        axes[i].tick_params(axis='y', labelsize=15, pad=10)
        axes[i].set_title(metric_columns[i], size=20, pad=15)
        axes[i].legend(loc='best', prop={'size': 18})

    fig.suptitle(title, fontsize=25)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_vertically_resolved_target_scores(vertically_resolved_target_scores, title, path=None):

    """
    Visualize scores of vertically resolved targets

    Parameters
    ----------
    vertically_resolved_target_scores: list of pandas.DataFrame of shape (n_splits + 1, n_targets, n_metrics)
        Dataframes with one or multiple targets and scores

    title: str
        Title of the plot

    path: str, pathlib.Path or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    target_columns = vertically_resolved_target_scores[0].index.tolist()
    metric_columns = vertically_resolved_target_scores[0].columns.tolist()
    fold_vertically_resolved_target_scores = np.stack([df.loc[target_columns] for df in vertically_resolved_target_scores[:-1]])
    mean_fold_vertically_resolved_target_scores = fold_vertically_resolved_target_scores.mean(axis=0)
    std_fold_vertically_resolved_target_scores = fold_vertically_resolved_target_scores.std(axis=0)
    oof_vertically_resolved_target_scores = vertically_resolved_target_scores[-1]

    fig, axes = plt.subplots(figsize=(72, 16), ncols=len(metric_columns))

    for i in range(len(metric_columns)):

        axes[i].plot(
            mean_fold_vertically_resolved_target_scores[:, i],
            '-',
            linewidth=4,
            color='tab:blue',
            label=f'Fold Scores ({np.mean(mean_fold_vertically_resolved_target_scores[:, i]):.8f})',
            alpha=0.75
        )
        axes[i].fill_between(
            np.arange(len(mean_fold_vertically_resolved_target_scores[:, i])),
            (mean_fold_vertically_resolved_target_scores[:, i] - std_fold_vertically_resolved_target_scores[:, i]),
            (mean_fold_vertically_resolved_target_scores[:, i] + std_fold_vertically_resolved_target_scores[:, i]),
            color='tab:blue',
            alpha=0.1
        )
        axes[i].plot(
            oof_vertically_resolved_target_scores.iloc[:, i].values,
            '-',
            linewidth=4,
            color='tab:orange',
            label=f'OOF Scores ({np.mean(oof_vertically_resolved_target_scores.iloc[:, i].values):.8f})',
            alpha=0.75
        )

        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', labelsize=15, pad=10)
        axes[i].tick_params(axis='y', labelsize=15, pad=10)
        axes[i].set_title(metric_columns[i], size=20, pad=15)
        axes[i].legend(loc='best', prop={'size': 18})

    fig.suptitle(title, fontsize=25)
    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_predictions(train_target, train_predictions, test_predictions, scores, path=None):

    """
    Visualize targets/predictions as scatter and histogram

    Parameters
    ----------
    train_target: numpy.ndarray of shape (n_samples)
        Array of targets

    train_predictions: numpy.ndarray of shape (n_samples)
        Array of predictions

    test_predictions: numpy.ndarray of shape (n_test_samples)
        Array of test predictions

    scores: dict
        Dictionary of scores

    path: str, pathlib.Path or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    train_target_statistics = train_target.agg(['mean', 'std', 'min', 'max']).to_dict()
    train_prediction_statistics = train_predictions.agg(['mean', 'std', 'min', 'max']).to_dict()
    test_prediction_statistics = test_predictions.agg(['mean', 'std', 'min', 'max']).to_dict()

    fig, axes = plt.subplots(figsize=(32, 16), ncols=2)

    axes[0].scatter(train_target, train_predictions)
    axes[1].hist(train_target, 16, alpha=0.5, label='Training Targets')
    axes[1].hist(train_predictions, 16, alpha=0.5, label='Training Predictions')
    axes[1].hist(test_predictions, 16, alpha=0.5, label='Test Predictions')

    for i in range(2):
        axes[i].tick_params(axis='x', labelsize=15)
        axes[i].tick_params(axis='y', labelsize=15)
    axes[0].set_xlabel('Target', size=15)
    axes[0].set_ylabel('Prediction', size=15)

    scatter_title = f'''
    Target {train_target.name}
    Training Target vs Prediction
    OOF Scores RMSE: {scores['rmse']:.4f} MAE {scores['mae']:.4f} R2 {scores['r2_score']:.4f}
    '''
    axes[0].set_title(scatter_title, size=20, pad=15)

    histogram_title = f'''
        Training Target/Predictions and Test Predictions Histogram
        Training Target: mean: {train_target_statistics['mean']:.4f} std: {train_target_statistics['std']:.4f} min: {train_target_statistics['min']:.4f} max: {train_target_statistics['max']:.4f}
        Training Predictions: mean: {train_prediction_statistics['mean']:.4f} std: {train_prediction_statistics['std']:.4f} min: {train_prediction_statistics['min']:.4f} max: {train_prediction_statistics['max']:.4f}
        Test Predictions: mean: {test_prediction_statistics['mean']:.4f} std: {test_prediction_statistics['std']:.4f} min: {test_prediction_statistics['min']:.4f} max: {test_prediction_statistics['max']:.4f}
        '''
    axes[1].set_title(histogram_title, size=20, pad=15)
    axes[1].legend(prop={'size': 18})

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)
