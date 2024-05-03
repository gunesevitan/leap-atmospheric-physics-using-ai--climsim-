import os
import sys
import argparse
import yaml
import json
from glob import glob
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim

import preprocessing
import torch_datasets
import torch_modules
import torch_utilities
sys.path.append('..')
import settings
import metrics
import visualization


def train(training_loader, model, criterion, optimizer, device, scheduler=None, amp=False):

    """
    Train given model on given data loader

    Parameters
    ----------
    training_loader: torch.utils.data.DataLoader
        Training set data loader

    model: torch.nn.Module
        Model to train

    criterion: torch.nn.Module
        Loss function

    optimizer: torch.optim.Optimizer
        Optimizer for updating model weights

    device: torch.device
        Location of the model and inputs

    scheduler: torch.optim.LRScheduler or None
        Learning rate scheduler

    amp: bool
        Whether to use auto mixed precision or not

    Returns
    -------
    training_loss: float
        Training loss after model is fully trained on training set data loader
    """

    model.train()
    progress_bar = tqdm(training_loader)

    running_loss = 0.0

    if amp:
        grad_scaler = torch.cuda.amp.GradScaler()
    else:
        grad_scaler = None

    for step, (inputs, targets) in enumerate(progress_bar):

        inputs = {k: v.to(device) for k, v in inputs.items()}
        targets = torch.cat([v.to(device) for v in targets.values()], dim=1)

        optimizer.zero_grad()

        if amp:
            with torch.autocast(device_type=device.type):
                outputs = model({k: v.half() for k, v in inputs.items()})
        else:
            outputs = model(inputs)

        loss = criterion(outputs, targets)

        if amp:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.detach().item() * len(inputs)
        lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
        progress_bar.set_description(f'lr: {lr:.8f} - training loss: {running_loss / len(training_loader.sampler):.4f}')

    training_loss = running_loss / len(training_loader.sampler)

    return training_loss


def validate(validation_loader, model, criterion, device, amp=False):

    """
    Validate given model on given data loader

    Parameters
    ----------
    validation_loader: torch.utils.data.DataLoader
        Validation set data loader

    model: torch.nn.Module
        Model to validate

    criterion: torch.nn.Module
        Loss function

    device: torch.device
        Location of the model and inputs

    amp: bool
        Whether to use auto mixed precision or not

    Returns
    -------
    validation_results: dict
        Dictionary of validation losses and scores after model is fully validated on validation set data loader
    """

    model.eval()
    progress_bar = tqdm(validation_loader)

    running_loss = 0.0
    validation_predictions = []

    for step, (inputs, targets) in enumerate(progress_bar):

        inputs = {k: v.to(device) for k, v in inputs.items()}
        targets = torch.cat([v.to(device) for v in targets.values()], dim=1)

        with torch.no_grad():
            if amp:
                with torch.autocast(device_type=device.type):
                    outputs = model({k: v.half() for k, v in inputs.items()})
            else:
                outputs = model(inputs)

        loss = criterion(outputs, targets)
        running_loss += loss.detach().item() * len(inputs)
        validation_predictions.append(outputs.detach().cpu())
        progress_bar.set_description(f'validation loss: {running_loss / len(validation_loader.sampler):.4f}')

    validation_loss = running_loss / len(validation_loader.sampler)
    validation_predictions = torch.cat(validation_predictions).float().numpy()

    return validation_loss, validation_predictions


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory', type=str)
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    model_directory = Path(settings.MODELS / args.model_directory)
    config = yaml.load(open(model_directory / 'config.yaml'), Loader=yaml.FullLoader)
    settings.logger.info(f'Running {model_directory} model in {args.mode} mode')

    df_train = pd.read_parquet(settings.DATA / 'datasets' / 'train.parquet')
    settings.logger.info(f'Training Set Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')

    # Select target and feature as individual arrays and grouped arrays
    target_columns = df_train.columns.tolist()[-368:]
    target_column_groups = preprocessing.get_target_column_groups(columns=target_columns)
    feature_columns = df_train.columns.tolist()[:-368]
    feature_column_groups = preprocessing.get_feature_column_groups(columns=feature_columns)
    prediction_columns = [f'{column}_prediction' for column in target_columns]

    # Load and concatenate precomputed folds
    df_train = pd.concat((df_train, pd.read_parquet(settings.DATA / 'folds.parquet')), axis=1)

    feature_means, feature_stds = np.load(settings.DATA / 'feature_means.npy'), np.load(settings.DATA / 'feature_stds.npy')
    target_means, target_stds = np.load(settings.DATA / 'target_means.npy'), np.load(settings.DATA / 'target_stds.npy')
    feature_stds = np.where(feature_stds == 0.0, 1.0, feature_stds)
    target_stds = np.where(target_stds == 0.0, 1.0, target_stds)

    df_train[feature_columns] = (df_train[feature_columns] - feature_means) / feature_stds
    df_train[target_columns] = (df_train[target_columns] - target_means) / target_stds

    torch.multiprocessing.set_sharing_strategy('file_system')

    if args.mode == 'training':

        training_metadata = defaultdict(dict)

        for fold in config['training']['folds']:

            training_idx, validation_idx = df_train.loc[df_train[fold] != 1].index, df_train.loc[df_train[fold] == 1].index
            # Validate on training set if validation is set is not specified
            if len(validation_idx) == 0:
                validation_idx = training_idx

            # Create training and validation inputs and targets
            training_features, training_targets = torch_datasets.prepare_data(
                df=df_train.loc[training_idx],
                has_targets=True,
            )
            validation_features, validation_targets = torch_datasets.prepare_data(
                df=df_train.loc[validation_idx],
                has_targets=True
            )

            settings.logger.info(
                f'''
                Fold: {fold}
                Training: {len(training_targets['ptend_t'])} ({len(training_targets['ptend_t']) // config["training"]["training_batch_size"] + 1} steps)
                Validation {len(validation_targets['ptend_t'])} ({len(validation_targets['ptend_t']) // config["training"]["test_batch_size"] + 1} steps)
                '''
            )

            # Create training and validation datasets and dataloaders
            training_dataset = torch_datasets.TabularDataset(
                features=training_features,
                targets=training_targets
            )
            training_loader = DataLoader(
                training_dataset,
                batch_size=config['training']['training_batch_size'],
                sampler=RandomSampler(training_dataset, replacement=False),
                pin_memory=False,
                drop_last=False,
                num_workers=config['training']['num_workers']
            )
            validation_dataset = torch_datasets.TabularDataset(
                features=validation_features,
                targets=validation_targets
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=config['training']['test_batch_size'],
                sampler=SequentialSampler(validation_dataset),
                pin_memory=False,
                drop_last=False,
                num_workers=config['training']['num_workers']
            )

            # Set model, device and seed for reproducible results
            torch_utilities.set_seed(config['training']['random_state'], deterministic_cudnn=config['training']['deterministic_cudnn'])
            device = torch.device(config['training']['device'])
            criterion = getattr(torch_modules, config['training']['loss_function'])(**config['training']['loss_function_args'])

            model = getattr(torch_modules, config['model']['model_class'])(**config['model']['model_args'])
            model_checkpoint_path = config['model']['model_checkpoint_paths'][int(fold[-1]) - 1]
            if model_checkpoint_path is not None:
                model_checkpoint_path = settings.MODELS / model_checkpoint_path
                model.load_state_dict(torch.load(model_checkpoint_path), strict=False)
            model.to(device)

            # Set optimizer, learning rate scheduler and stochastic weight averaging
            optimizer = getattr(torch.optim, config['training']['optimizer'])(model.parameters(), **config['training']['optimizer_args'])
            scheduler = getattr(optim.lr_scheduler, config['training']['lr_scheduler'])(optimizer, **config['training']['lr_scheduler_args'])
            amp = config['training']['amp']

            best_epoch = 1
            early_stopping = False
            early_stopping_patience = config['training']['early_stopping_patience']
            early_stopping_metric = config['training']['early_stopping_metric']
            training_history = {f'{dataset}_{metric}': [] for metric in config['persistence']['save_best_metrics'] for dataset in ['training', 'validation']}

            for epoch in range(1, config['training']['epochs'] + 1):

                if early_stopping:
                    break

                training_loss = train(
                    training_loader=training_loader,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    scheduler=scheduler,
                    amp=amp
                )

                validation_loss, validation_predictions = validate(
                    validation_loader=validation_loader,
                    model=model,
                    criterion=criterion,
                    device=device,
                    amp=amp
                )

                # Rescale validation targets and predictions back to their normal scale and calculate their validation scores
                rescaled_validation_targets = (np.hstack([target for target in validation_targets.values()]) * target_stds) + target_means
                rescaled_validation_predictions = (validation_predictions * target_stds) + target_means
                df_validation_predictions = pd.DataFrame(
                    data=np.hstack((
                        rescaled_validation_targets,
                        rescaled_validation_predictions,
                    )),
                    columns=target_columns + prediction_columns
                )
                _, target_validation_scores = metrics.regression_scores(
                    y_true=df_validation_predictions[target_columns],
                    y_pred=df_validation_predictions[prediction_columns]
                )

                # Calculate validation scores with mean predictions
                df_validation_mean_predictions = df_validation_predictions.copy(deep=True)
                df_validation_mean_predictions[prediction_columns] = target_means
                mean_global_validation_scores, mean_target_validation_scores = metrics.regression_scores(
                    y_true=df_validation_mean_predictions[target_columns],
                    y_pred=df_validation_mean_predictions[prediction_columns]
                )

                # Replace columns on which the performance of mean prediction is better than model's prediction
                target_idx = np.where(mean_target_validation_scores['r2_score'] > target_validation_scores['r2_score'])[0]
                prediction_columns_to_overwrite = [f'{column}_prediction' for column in np.array(target_columns)[target_idx].tolist()]
                df_validation_predictions[prediction_columns_to_overwrite] = target_means[target_idx]
                global_validation_scores, target_validation_scores = metrics.regression_scores(
                    y_true=df_validation_predictions[target_columns],
                    y_pred=df_validation_predictions[prediction_columns]
                )

                training_results = {'loss': training_loss}
                validation_results = {'loss': validation_loss}
                validation_results.update(global_validation_scores)

                settings.logger.info(
                    f'''
                    Epoch {epoch}
                    Training Results: {json.dumps(training_results, indent=2)}
                    Validation Results: {json.dumps(validation_results, indent=2)}
                    Replaced Target Predictions: {len(prediction_columns_to_overwrite)}
                    {prediction_columns_to_overwrite}
                    '''
                )

                if epoch in config['persistence']['save_epochs']:
                    # Save model if epoch is specified to be saved
                    model_name = f'model_fold_{fold[-1]}_epoch_{epoch}.pt'
                    torch.save(model.state_dict(), model_directory / model_name)
                    settings.logger.info(f'Saved {model_name} to {model_directory}')

                for metric in config['persistence']['save_best_metrics']:
                    best_validation_metric = np.min(training_history[f'validation_{metric}']) if len(training_history[f'validation_{metric}']) > 0 else np.inf
                    last_validation_metric = validation_results[metric]
                    if last_validation_metric < best_validation_metric:

                        previous_model = glob(str(model_directory / f'model_fold_{fold[-1]}_epoch_*_best_{metric}*'))
                        if len(previous_model) > 0:
                            os.remove(previous_model[0])
                            settings.logger.info(f'Deleted {previous_model[0].split("/")[-1]} from {model_directory}')

                        # Save model if specified validation metric improves
                        model_name = f'model_fold_{fold[-1]}_epoch_{epoch}_best_{metric}_{last_validation_metric:.4f}.pt'
                        torch.save(model.state_dict(), model_directory / model_name)
                        settings.logger.info(f'Saved {model_name} to {model_directory} (validation {metric} decreased from {best_validation_metric:.6f} to {last_validation_metric:.6f})\n')

                    if metric == 'loss':
                        training_history[f'training_{metric}'].append(training_results[metric])
                    training_history[f'validation_{metric}'].append(validation_results[metric])

                best_epoch = np.argmin(training_history[f'validation_{early_stopping_metric}'])
                if early_stopping_patience > 0:
                    # Trigger early stopping if early stopping patience is greater than 0
                    if len(training_history[f'validation_{early_stopping_metric}']) - best_epoch > early_stopping_patience:
                        settings.logger.info(
                            f'''
                            Early Stopping (validation {early_stopping_metric} didn\'t improve for {early_stopping_patience} epochs)
                            Best Epoch ({best_epoch + 1}) Validation {early_stopping_metric}: {training_history[early_stopping_metric][best_epoch]:.4f}
                            '''
                        )
                        early_stopping = True

            training_metadata[fold] = {}
            training_metadata[fold][f'training_history'] = training_history
            for metric in config['persistence']['save_best_metrics']:
                best_epoch = int(np.argmin(training_history[f'validation_{metric}']))
                training_metadata[fold][f'best_epoch_{metric}'] = best_epoch + 1
                training_metadata[fold][f'training_{metric}'] = float(training_history[f'training_{metric}'][best_epoch])
                training_metadata[fold][f'validation_{metric}'] = float(training_history[f'validation_{metric}'][best_epoch])
                visualization.visualize_learning_curve(
                    training_scores=training_metadata[fold]['training_history'][f'training_{metric}'],
                    validation_scores=training_metadata[fold]['training_history'][f'validation_{metric}'],
                    best_epoch=training_metadata[fold][f'best_epoch_{metric}'] - 1,
                    metric=metric,
                    path=model_directory / f'learning_curve_fold_{fold[-1]}_{metric}.png'
                )
                settings.logger.info(f'Saved learning_curve_fold_{fold[-1]}_{metric}.png to {model_directory}')

        with open(model_directory / 'training_metadata.json', mode='w') as f:
            json.dump(training_metadata, f, indent=2, ensure_ascii=False)
        settings.logger.info(f'Saved training_metadata.json to {model_directory}')

    elif args.mode == 'test':

        df_test = pd.read_parquet(settings.DATA / 'datasets' / 'test.parquet')
        settings.logger.info(f'Test Set Shape: {df_test.shape} - Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')

        df_test[feature_columns] = (df_test[feature_columns] - feature_means) / feature_stds
        submission_weights = np.load(settings.DATA / 'weights.npy')

        global_scores = []
        target_scores = []
        prediction_columns_to_overwrite = []
        df_test_predictions = pd.DataFrame(np.zeros((df_test.shape[0], 368), dtype=np.float32), columns=target_columns)

        folds = config['test']['folds']
        model_file_names = config['test']['model_file_names']

        for fold, model_file_name in zip(folds, model_file_names):

            validation_idx = df_train.loc[df_train[fold] == 1].index

            # Create validation and test inputs and targets
            validation_features, validation_targets = torch_datasets.prepare_data(
                df=df_train.loc[validation_idx, feature_columns + target_columns],
                has_targets=True
            )
            test_features, _ = torch_datasets.prepare_data(
                df=df_test.loc[:, feature_columns],
                has_targets=False
            )

            settings.logger.info(
                f'''
                Fold: {fold} ({model_file_name})
                Validation {len(validation_targets)} ({len(validation_targets) // config["training"]["test_batch_size"] + 1} steps)
                Test {len(test_features)} ({len(test_features) // config["training"]["test_batch_size"] + 1} steps)
                '''
            )

            # Create validation and test datasets and dataloaders
            validation_dataset = torch_datasets.TabularDataset(
                features=validation_features,
                targets=validation_targets
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=config['training']['test_batch_size'],
                sampler=SequentialSampler(validation_dataset),
                pin_memory=False,
                drop_last=False,
                num_workers=config['training']['num_workers']
            )

            test_dataset = torch_datasets.TabularDataset(
                features=test_features,
                targets=None
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=config['training']['test_batch_size'],
                sampler=SequentialSampler(test_dataset),
                pin_memory=False,
                drop_last=False,
                num_workers=config['training']['num_workers']
            )

            # Set model, device and seed for reproducible results
            torch_utilities.set_seed(config['training']['random_state'], deterministic_cudnn=config['training']['deterministic_cudnn'])
            device = torch.device(config['training']['device'])
            amp = config['training']['amp']

            model = getattr(torch_modules, config['model']['model_class'])(**config['model']['model_args'])
            model.load_state_dict(torch.load(model_directory / model_file_name))
            model.to(device)
            model.eval()

            validation_predictions = []

            for inputs, targets in tqdm(validation_loader):

                inputs = {k: v.to(device) for k, v in inputs.items()}
                targets = torch.cat([v.to(device) for v in targets.values()], dim=1)

                with torch.no_grad():
                    if amp:
                        with torch.autocast(device_type=device.type):
                            outputs = model({k: v.half() for k, v in inputs.items()})
                    else:
                        outputs = model(inputs)

                outputs = outputs.detach().cpu()
                validation_predictions.append(outputs)

            validation_predictions = torch.cat(validation_predictions, dim=0).numpy()

            test_predictions = []

            for inputs in tqdm(test_loader):

                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    if amp:
                        with torch.autocast(device_type=device.type):
                            outputs = model({k: v.half() for k, v in inputs.items()})
                    else:
                        outputs = model(inputs)

                outputs = outputs.detach().cpu()
                test_predictions.append(outputs)

            test_predictions = torch.cat(test_predictions, dim=0).numpy()
            df_test_predictions += (test_predictions / len(folds))

            # Rescale validation targets and predictions back to their normal scale and calculate their validation scores
            rescaled_validation_targets = (np.hstack([target for target in validation_targets.values()]) * target_stds) + target_means
            rescaled_validation_predictions = (validation_predictions * target_stds) + target_means
            df_validation_predictions = pd.DataFrame(
                data=np.hstack((
                    rescaled_validation_targets,
                    rescaled_validation_predictions,
                )),
                columns=target_columns + prediction_columns
            )
            _, target_validation_scores = metrics.regression_scores(
                y_true=df_validation_predictions[target_columns],
                y_pred=df_validation_predictions[prediction_columns]
            )

            # Calculate validation scores with target mean predictions
            df_validation_mean_predictions = df_validation_predictions.copy(deep=True)
            df_validation_mean_predictions[prediction_columns] = target_means
            mean_global_validation_scores, mean_target_validation_scores = metrics.regression_scores(
                y_true=df_validation_mean_predictions[target_columns],
                y_pred=df_validation_mean_predictions[prediction_columns]
            )

            # Replace columns on which the performance of mean prediction is better than model's prediction
            target_idx = np.where(mean_target_validation_scores['r2_score'] > target_validation_scores['r2_score'])[0]
            fold_prediction_columns_to_overwrite = [f'{column}_prediction' for column in np.array(target_columns)[target_idx].tolist()]
            df_validation_predictions[fold_prediction_columns_to_overwrite] = target_means[target_idx]
            global_validation_scores, target_validation_scores = metrics.regression_scores(
                y_true=df_validation_predictions[target_columns],
                y_pred=df_validation_predictions[prediction_columns]
            )

            global_scores.append(global_validation_scores)
            target_scores.append(target_validation_scores)
            prediction_columns_to_overwrite.append(fold_prediction_columns_to_overwrite)

            df_train.loc[validation_idx, prediction_columns] = df_validation_predictions[prediction_columns].values

            settings.logger.info(
                f'''
                {fold}
                Validation Scores: {json.dumps(global_validation_scores, indent=2)}
                Replaced Target Predictions: {len(fold_prediction_columns_to_overwrite)}
                {fold_prediction_columns_to_overwrite}
                '''
            )

        global_scores = pd.DataFrame(global_scores)
        settings.logger.info(
            f'''
            Mean Validation Scores
            {json.dumps(global_scores.mean(axis=0).to_dict(), indent=2)}
            and Standard Deviations
            ±{json.dumps(global_scores.std(axis=0).to_dict(), indent=2)}
            '''
        )

        # Rescale targets back to their normal scale and calculate their oof scores
        df_train[target_columns] = (df_train[target_columns] * target_stds) + target_means
        global_oof_scores, target_oof_scores = metrics.regression_scores(
            y_true=df_train[target_columns],
            y_pred=df_train[prediction_columns]
        )
        target_scores.append(target_oof_scores)
        global_scores = pd.concat((
            global_scores,
            pd.DataFrame([global_oof_scores])
        )).reset_index(drop=True)
        global_scores['fold'] = ['1', '2', '3', '4', '5', 'OOF']
        global_scores = global_scores[global_scores.columns.tolist()[::-1]]

        global_scores.to_csv(model_directory / 'global_scores.csv', index=False)
        settings.logger.info(f'global_scores.csv is saved to {model_directory}')

        visualization.visualize_scores(
            scores=global_scores,
            title=f'Global Fold and OOF Scores of {len(folds)} Model(s)',
            path=model_directory / 'global_scores.png'
        )
        settings.logger.info(f'global_scores.png is saved to {model_directory}')

        single_targets = [columns[0] for columns in target_column_groups.values() if len(columns) == 1]
        single_target_scores = [df.loc[single_targets] for df in target_scores]
        visualization.visualize_single_target_scores(
            single_target_scores=single_target_scores,
            title=f'Single Target Fold and OOF Scores of {len(folds)} Model(s)',
            path=model_directory / 'single_target_scores.png'
        )
        settings.logger.info(f'single_target_scores.png is saved to {model_directory}')

        vertically_resolved_targets = {name: columns for name, columns in target_column_groups.items() if len(columns) == 60}
        for target_group, columns in vertically_resolved_targets.items():
            visualization.visualize_vertically_resolved_target_scores(
                vertically_resolved_target_scores=[df.loc[columns] for df in target_scores],
                title=f'{target_group} Fold and OOF Scores of {len(folds)} Model(s)',
                path=model_directory / f'{target_group}_scores.png'
            )
            settings.logger.info(f'{target_group}_scores.png is saved to {model_directory}')

        # Rescale test predictions back to their normal scale
        df_test_predictions = (df_test_predictions * target_stds) + target_means
        test_prediction_columns_to_overwrite = np.unique(np.concatenate(prediction_columns_to_overwrite)).tolist()
        test_prediction_column_idx_to_overwrite = sorted([prediction_columns.index(column) for column in test_prediction_columns_to_overwrite])
        df_test_predictions.iloc[:, test_prediction_column_idx_to_overwrite] = target_means[test_prediction_column_idx_to_overwrite]

        if config['persistence']['save_prediction_visualizations']:
            predictions_visualization_directory = model_directory / 'predictions'
            predictions_visualization_directory.mkdir(parents=True, exist_ok=True)
            for target_column in tqdm(target_columns):
                visualization.visualize_predictions(
                    train_target=df_train[target_column],
                    train_predictions=df_train[f'{target_column}_prediction'],
                    test_predictions=df_test_predictions[target_column],
                    scores=target_oof_scores.loc[target_column],
                    path=predictions_visualization_directory / f'{target_column}.png'
                )

        df_train[prediction_columns].to_parquet(model_directory / 'oof_predictions.parquet')
        settings.logger.info(f'oof_predictions.parquet is saved to {model_directory}')

        df_test_predictions[target_columns].to_parquet(model_directory / 'test_predictions.parquet')
        settings.logger.info(f'oof_predictions.parquet is saved to {model_directory}')
