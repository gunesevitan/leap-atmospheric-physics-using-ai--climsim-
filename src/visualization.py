import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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


def visualize_predictions(targets, predictions, score, target_column, path=None):

    """
    Visualize targets/predictions as scatter and histogram

    Parameters
    ----------
    targets: numpy.ndarray of shape (n_samples)
        Array of targets

    predictions: numpy.ndarray of shape (n_samples)
        Array of predictions

    scores: dict
        Dictionary of scores

    target_column: str
        Name of the target column

    path: str, pathlib.Path or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    targets_statistics = {
        'mean': np.mean(targets),
        'std': np.std(targets),
        'min': np.min(targets),
        'max': np.max(targets),
    }
    predictions_statistics = {
        'mean': np.mean(predictions),
        'std': np.std(predictions),
        'min': np.min(predictions),
        'max': np.max(predictions),
    }

    fig, axes = plt.subplots(figsize=(32, 16), ncols=2)

    axes[0].scatter(targets, predictions)
    axes[1].hist(targets, 16, alpha=0.5, label='Training Targets')
    axes[1].hist(predictions, 16, alpha=0.5, label='Training Predictions')

    for i in range(2):
        axes[i].tick_params(axis='x', labelsize=15)
        axes[i].tick_params(axis='y', labelsize=15)
    axes[0].set_xlabel('Target', size=15)
    axes[0].set_ylabel('Prediction', size=15)

    scatter_title = f'''
    Target {target_column}
    Training Targets vs Predictions
    OOF Scores R2 {score:.6f}
    '''
    axes[0].set_title(scatter_title, size=20, pad=15)

    histogram_title = f'''
        Training Target/Predictions and Test Predictions Histogram
        Training Target: mean: {targets_statistics['mean']:.6f} std: {targets_statistics['std']:.6f} min: {targets_statistics['min']:.6f} max: {targets_statistics['max']:.6f}
        Training Predictions: mean: {predictions_statistics['mean']:.6f} std: {predictions_statistics['std']:.6f} min: {predictions_statistics['min']:.6f} max: {predictions_statistics['max']:.6f}
        '''
    axes[1].set_title(histogram_title, size=20, pad=15)
    axes[1].legend(prop={'size': 18})

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_feature_importance(df_feature_importance, title, path=None):

    """
    Visualize feature importance in descending order

    Parameters
    ----------
    df_feature_importance: pandas.DataFrame of shape (n_features, n_splits)
        Dataframe of feature importance

    title: str
        Title of the plot

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(24, 120), dpi=100)
    ax.barh(
        range(len(df_feature_importance)),
        df_feature_importance['mean'],
        xerr=df_feature_importance['std'],
        ecolor='black',
        capsize=10,
        align='center',
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_yticks(range(len(df_feature_importance)))
    ax.set_yticklabels([f'{k} ({v:.2f})' for k, v in df_feature_importance['mean'].to_dict().items()])
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(title, size=20, pad=15)
    plt.gca().invert_yaxis()

    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)


def visualize_feature_importance_heatmap(df, title, path=None):

    """
    Visualize feature importance as a heatmap

    Parameters
    ----------
    df: pandas.DataFrame of shape (n_features, n_targets)
        Dataframe of per target feature importance

    title: str
        Title of the plot

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(100, 100), dpi=100)
    ax = sns.heatmap(
        df.values,
        annot=False,
        square=True,
        cmap='coolwarm',
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)
    ax.set_xticks(np.arange(len(df.columns.tolist())) + 0.5, df.columns.tolist())
    ax.set_yticks(np.arange(len(df.index.tolist())) + 0.5, df.index.tolist())
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_title(title, size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)
