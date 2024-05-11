import os
import sys
import argparse
import yaml
import json
from glob import glob
from pathlib import Path
from tqdm import tqdm
from itertools import chain
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

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        if amp:
            with torch.autocast(device_type=device.type):
                outputs = model(inputs.half())
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
    validation_loss: float
        Training loss after model is fully trained on training set data loader

    validation_targets: numpy.ndarray of shape (validation_size)
        Validation targets

    validation_predictions: numpy.ndarray of shape (validation_size)
        Validation predictions
    """

    model.eval()
    progress_bar = tqdm(validation_loader)

    running_loss = 0.0
    validation_targets = []
    validation_predictions = []

    for step, (inputs, targets) in enumerate(progress_bar):

        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            if amp:
                with torch.autocast(device_type=device.type):
                    outputs = model(inputs.half())
            else:
                outputs = model(inputs)

        loss = criterion(outputs, targets)
        running_loss += loss.detach().item() * len(inputs)
        validation_targets.append(targets.cpu())
        validation_predictions.append(outputs.detach().cpu())
        progress_bar.set_description(f'validation loss: {running_loss / len(validation_loader.sampler):.4f}')

    validation_loss = running_loss / len(validation_loader.sampler)
    validation_targets = torch.cat(validation_targets, dim=0).float().numpy()
    validation_predictions = torch.cat(validation_predictions, dim=0).float().numpy()

    return validation_loss, validation_targets, validation_predictions


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory', type=str)
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    model_directory = Path(settings.MODELS / args.model_directory)
    config = yaml.load(open(model_directory / 'config.yaml'), Loader=yaml.FullLoader)
    settings.logger.info(f'Running {model_directory} model in {args.mode} mode')

    # Select targets and features as individual arrays and grouped arrays
    columns = pd.read_csv(settings.DATA / 'leap-atmospheric-physics-ai-climsim' / 'train.csv', nrows=0).columns.tolist()[1:]
    target_columns = columns[-368:]
    target_column_groups = preprocessing.get_target_column_groups(columns=target_columns)
    feature_columns = columns[:-368]
    feature_column_groups = preprocessing.get_feature_column_groups(columns=feature_columns)
    prediction_columns = [f'{column}_prediction' for column in target_columns]

    # Load precomputed folds
    folds = np.load(settings.DATA / 'folds.npz')['arr_0']
    file_idx = np.arange(folds.shape[0], dtype=np.uint32)
    settings.logger.info(f'Loaded folds: {folds.shape}')

    # Load precomputed statistics
    feature_means, feature_stds = np.load(settings.DATA / 'feature_means.npy'), np.load(settings.DATA / 'feature_stds.npy')
    feature_mins, feature_maxs = np.load(settings.DATA / 'feature_mins.npy'), np.load(settings.DATA / 'feature_maxs.npy')
    target_means, target_stds = np.load(settings.DATA / 'target_means.npy'), np.load(settings.DATA / 'target_stds.npy')
    target_mins, target_maxs = np.load(settings.DATA / 'target_mins.npy'), np.load(settings.DATA / 'target_maxs.npy')
    feature_stds = np.where(feature_stds == 0.0, 1.0, feature_stds)
    target_stds = np.where(target_stds == 0.0, 1.0, target_stds)
    feature_statistics = {'mean': feature_means, 'std': feature_stds, 'min': feature_mins, 'max': feature_maxs}
    target_statistics = {'mean': target_means, 'std': target_stds, 'min': target_mins, 'max': target_maxs}
    feature_normalization_type = config['dataset']['feature_normalization_type']
    target_normalization_type = config['dataset']['target_normalization_type']
    settings.logger.info(f'Loaded statistics - Feature normalization: {feature_normalization_type} - Target normalization: {target_normalization_type}')

    torch.multiprocessing.set_sharing_strategy('file_system')

    if args.mode == 'training':

        training_metadata = {}

        # Create training and validation indices
        training_mask = folds[:, config['training']['training_folds']].any(axis=1)
        validation_mask = folds[:, config['training']['validation_folds']].any(axis=1)
        training_idx = file_idx[training_mask]
        validation_idx = file_idx[validation_mask]

        # Create training and validation inputs and targets paths
        training_feature_paths, training_target_paths = torch_datasets.prepare_file_paths(
            idx=training_idx,
            features_path=settings.DATA / 'datasets' / 'numpy_arrays' / 'training_features',
            targets_path=settings.DATA / 'datasets' / 'numpy_arrays' / 'training_targets',
        )
        validation_feature_paths, validation_target_paths = torch_datasets.prepare_file_paths(
            idx=validation_idx,
            features_path=settings.DATA / 'datasets' / 'numpy_arrays' / 'training_features',
            targets_path=settings.DATA / 'datasets' / 'numpy_arrays' / 'training_targets',
        )

        settings.logger.info(
            f'''
            Training Folds: {config['training']['training_folds']} {training_idx.shape[0]} ({training_idx.shape[0] // config["training"]["training_batch_size"] + 1} steps)
            Validation Folds: {config['training']['validation_folds']} {validation_idx.shape[0]} ({validation_idx.shape[0] // config["training"]["test_batch_size"] + 1} steps)
            '''
        )

        # Create training and validation datasets and dataloaders
        training_dataset = torch_datasets.TabularDataset(
            feature_paths=training_feature_paths,
            target_paths=training_target_paths,
            feature_statistics=feature_statistics,
            target_statistics=target_statistics,
            feature_normalization_type=feature_normalization_type,
            target_normalization_type=target_normalization_type
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
            feature_paths=validation_feature_paths,
            target_paths=validation_target_paths,
            feature_statistics=feature_statistics,
            target_statistics=target_statistics,
            feature_normalization_type=feature_normalization_type,
            target_normalization_type=target_normalization_type
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
        model_checkpoint_path = config['model']['model_checkpoint_path']
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

            validation_loss, validation_targets, validation_predictions = validate(
                validation_loader=validation_loader,
                model=model,
                criterion=criterion,
                device=device,
                amp=amp
            )

            # Rescale validation targets and predictions back to their normal scales
            validation_targets = (validation_targets * target_stds) + target_means
            validation_predictions = (validation_predictions * target_stds) + target_means

            training_results = {'loss': training_loss}
            validation_results = {'loss': validation_loss}

            if epoch % config['training']['metric_frequency'] == 0:

                # Calculate validation scores with model predictions
                global_validation_scores, target_validation_scores = metrics.regression_scores(
                    y_true=validation_targets,
                    y_pred=validation_predictions
                )
                target_validation_scores.index = target_columns

                # Calculate validation scores with mean predictions
                validation_mean_predictions = np.repeat(target_means.reshape(1, -1), validation_targets.shape[0], axis=0)
                mean_global_validation_scores, mean_target_validation_scores = metrics.regression_scores(
                    y_true=validation_targets,
                    y_pred=validation_mean_predictions
                )
                mean_target_validation_scores.index = target_columns

                # Replace columns on which the performance of mean prediction is better than model's prediction
                target_overwrite_idx = np.where(mean_target_validation_scores['r2_score'] > target_validation_scores['r2_score'])[0]
                validation_columns_to_overwrite = np.array(target_columns)[target_overwrite_idx]
                validation_predictions[:, target_overwrite_idx] = validation_mean_predictions[:, target_overwrite_idx]
                global_overwritten_validation_scores, target_overwritten_validation_scores = metrics.regression_scores(
                    y_true=validation_targets,
                    y_pred=validation_predictions
                )
                target_validation_scores.index = target_columns

                settings.logger.info(
                    f'''
                    Epoch {epoch}
                    Training Loss: {training_loss:.6f}
                    Validation Loss: {validation_loss:.6f}
                    Raw Global Validation Scores: {json.dumps(global_validation_scores, indent=2)}
                    Mean Overwritten Global Validation Scores: {json.dumps(global_overwritten_validation_scores, indent=2)}
                    Overwritten Target Columns: {len(validation_columns_to_overwrite)}
                    ({validation_columns_to_overwrite})
                    '''
                )

            else:
                settings.logger.info(
                    f'''
                    Epoch {epoch}
                    Training Loss: {training_loss:.6f}
                    Validation Loss: {validation_loss:.6f}
                    '''
                )

            if epoch in config['persistence']['save_epochs']:
                # Save model if epoch is specified to be saved
                model_name = f'model_epoch_{epoch}.pt'
                torch.save(model.state_dict(), model_directory / model_name)
                settings.logger.info(f'Saved {model_name} to {model_directory}')

            for metric in config['persistence']['save_best_metrics']:
                best_validation_metric = np.min(training_history[f'validation_{metric}']) if len(training_history[f'validation_{metric}']) > 0 else np.inf
                last_validation_metric = validation_results[metric]
                if last_validation_metric < best_validation_metric:

                    previous_model = glob(str(model_directory / f'model_epoch_*_best_{metric}*'))
                    if len(previous_model) > 0:
                        os.remove(previous_model[0])
                        settings.logger.info(f'Deleted {previous_model[0].split("/")[-1]} from {model_directory}')

                    # Save model if specified validation metric improves
                    model_name = f'model_epoch_{epoch}_best_{metric}_{last_validation_metric:.6f}.pt'
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

        training_metadata['training_history'] = training_history
        for metric in config['persistence']['save_best_metrics']:
            best_epoch = int(np.argmin(training_history[f'validation_{metric}']))
            training_metadata[f'best_epoch_{metric}'] = best_epoch + 1
            training_metadata[f'training_{metric}'] = float(training_history[f'training_{metric}'][best_epoch])
            training_metadata[f'validation_{metric}'] = float(training_history[f'validation_{metric}'][best_epoch])
            visualization.visualize_learning_curve(
                training_scores=training_metadata['training_history'][f'training_{metric}'],
                validation_scores=training_metadata['training_history'][f'validation_{metric}'],
                best_epoch=training_metadata[f'best_epoch_{metric}'] - 1,
                metric=metric,
                path=model_directory / f'learning_curve_{metric}.png'
            )
            settings.logger.info(f'Saved learning_curve_{metric}.png to {model_directory}')

        with open(model_directory / 'training_metadata.json', mode='w') as f:
            json.dump(training_metadata, f, indent=2, ensure_ascii=False)
        settings.logger.info(f'Saved training_metadata.json to {model_directory}')

    elif args.mode == 'test':

        oof_targets = []
        oof_predictions = []

        global_scores = []
        target_scores = []
        columns_to_overwrite = []

        # Set model, device and seed for reproducible results
        torch_utilities.set_seed(config['training']['random_state'], deterministic_cudnn=config['training']['deterministic_cudnn'])
        device = torch.device(config['training']['device'])
        amp = config['training']['amp']

        model_file_name = config['test']['model_file_name']
        model = getattr(torch_modules, config['model']['model_class'])(**config['model']['model_args'])
        model.load_state_dict(torch.load(model_directory / model_file_name))
        model.to(device)
        model.eval()

        test_folds = config['test']['folds']

        for fold in test_folds:

            # Create validation indices
            validation_mask = folds[:, fold] == 1
            validation_idx = file_idx[validation_mask]

            # Create inputs and targets paths
            validation_feature_paths, validation_target_paths = torch_datasets.prepare_file_paths(
                idx=validation_idx,
                features_path=settings.DATA / 'datasets' / 'numpy_arrays' / 'training_features',
                targets_path=settings.DATA / 'datasets' / 'numpy_arrays' / 'training_targets',
            )

            settings.logger.info(
                f'''
                Validation Folds: {fold} {validation_idx.shape[0]} ({validation_idx.shape[0] // config["training"]["test_batch_size"] + 1} steps)
                '''
            )

            # Create validation dataset and dataloaders
            validation_dataset = torch_datasets.TabularDataset(
                feature_paths=validation_feature_paths,
                target_paths=validation_target_paths,
                feature_statistics=feature_statistics,
                target_statistics=target_statistics,
                feature_normalization_type=feature_normalization_type,
                target_normalization_type=target_normalization_type
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=config['training']['test_batch_size'],
                sampler=SequentialSampler(validation_dataset),
                pin_memory=False,
                drop_last=False,
                num_workers=config['training']['num_workers']
            )

            validation_targets = []
            validation_predictions = []

            for inputs, targets in tqdm(validation_loader):

                inputs = inputs.to(device)
                targets = targets.to(device)

                with torch.no_grad():
                    if amp:
                        with torch.autocast(device_type=device.type):
                            outputs = model(inputs.half())
                    else:
                        outputs = model(inputs)

                validation_targets.append(targets.cpu())
                validation_predictions.append(outputs.detach().cpu())

            validation_targets = torch.cat(validation_targets, dim=0).float().numpy()
            validation_predictions = torch.cat(validation_predictions, dim=0).numpy()

            # Rescale validation targets and predictions back to their normal scales
            validation_targets = (validation_targets * target_stds) + target_means
            validation_predictions = (validation_predictions * target_stds) + target_means

            # Calculate validation scores with model predictions
            global_validation_scores, target_validation_scores = metrics.regression_scores(
                y_true=validation_targets,
                y_pred=validation_predictions
            )
            target_validation_scores.index = target_columns

            # Calculate validation scores with mean predictions
            validation_mean_predictions = np.repeat(target_means.reshape(1, -1), validation_targets.shape[0], axis=0)
            mean_global_validation_scores, mean_target_validation_scores = metrics.regression_scores(
                y_true=validation_targets,
                y_pred=validation_mean_predictions
            )
            mean_target_validation_scores.index = target_columns

            # Replace columns on which the performance of mean prediction is better than model's prediction
            target_overwrite_idx = np.where(mean_target_validation_scores['r2_score'] > target_validation_scores['r2_score'])[0]
            validation_columns_to_overwrite = np.array(target_columns)[target_overwrite_idx]
            validation_predictions[:, target_overwrite_idx] = validation_mean_predictions[:, target_overwrite_idx]
            global_overwritten_validation_scores, target_overwritten_validation_scores = metrics.regression_scores(
                y_true=validation_targets,
                y_pred=validation_predictions
            )
            target_overwritten_validation_scores.index = target_columns

            settings.logger.info(
                f'''
                Fold {fold}
                Raw Global Validation Scores: {json.dumps(global_validation_scores, indent=2)}
                Mean Overwritten Global Validation Scores: {json.dumps(global_overwritten_validation_scores, indent=2)}
                Overwritten Target Columns: {len(validation_columns_to_overwrite)}
                ({validation_columns_to_overwrite})
                '''
            )

            oof_targets.append(validation_targets)
            oof_predictions.append(validation_predictions)

            global_scores.append(global_overwritten_validation_scores)
            target_scores.append(target_overwritten_validation_scores)
            columns_to_overwrite.append(validation_columns_to_overwrite)

        oof_targets = np.concatenate(oof_targets, axis=0)
        oof_predictions = np.concatenate(oof_predictions, axis=0)

        global_scores = pd.DataFrame(global_scores)
        settings.logger.info(
            f'''
            Mean Validation Scores
            {json.dumps(global_scores.mean(axis=0).to_dict(), indent=2)}
            and Standard Deviations
            ±{json.dumps(global_scores.std(axis=0).to_dict(), indent=2)}
            '''
        )

        global_oof_scores, target_oof_scores = metrics.regression_scores(
            y_true=oof_targets,
            y_pred=oof_predictions
        )
        target_oof_scores.index = target_columns
        oof_columns_to_overwrite = sorted(list(set(chain.from_iterable(columns_to_overwrite))))
        with open(model_directory / 'columns_to_overwrite.json', mode='w') as f:
            json.dump(oof_columns_to_overwrite, f, indent=2, ensure_ascii=False)
        settings.logger.info(f'Saved oof_columns_to_overwrite.json to {model_directory}')

        settings.logger.info(
            f'''
            OOF
            Mean Overwritten Global Scores: {json.dumps(global_oof_scores, indent=2)}
            Overwritten Target Columns: {len(oof_columns_to_overwrite)}
            ({oof_columns_to_overwrite})
            '''
        )

        target_scores.append(target_oof_scores)
        global_scores = pd.concat((
            global_scores,
            pd.DataFrame([global_oof_scores])
        )).reset_index(drop=True)
        global_scores['fold'] = test_folds + ['OOF']
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

        if config['persistence']['save_prediction_visualizations']:
            predictions_visualization_directory = model_directory / 'predictions'
            predictions_visualization_directory.mkdir(parents=True, exist_ok=True)
            for target_column_idx, target_column in enumerate(tqdm(target_columns)):
                visualization.visualize_predictions(
                    targets=oof_targets[:, target_column_idx],
                    predictions=oof_predictions[:, target_column_idx],
                    scores=target_oof_scores.loc[target_column],
                    target_column=target_column,
                    path=predictions_visualization_directory / f'{target_column}.png'
                )

        oof_idx = np.where(folds[:, test_folds].any(axis=1))[0]
        np.savez_compressed(model_directory / 'oof_predictions.npz', idx=oof_idx, predictions=oof_predictions)
        settings.logger.info(f'oof_predictions.npz is saved to {model_directory}')

    elif args.mode == 'submission':

        test_idx = np.arange(625000)

        with open(model_directory / 'columns_to_overwrite.json', mode='r') as f:
            columns_to_overwrite = json.load(f)

        # Set model, device and seed for reproducible results
        torch_utilities.set_seed(config['training']['random_state'], deterministic_cudnn=config['training']['deterministic_cudnn'])
        device = torch.device(config['training']['device'])
        amp = config['training']['amp']

        model_file_name = config['test']['model_file_name']
        model = getattr(torch_modules, config['model']['model_class'])(**config['model']['model_args'])
        model.load_state_dict(torch.load(model_directory / model_file_name))
        model.to(device)
        model.eval()

        test_feature_paths = torch_datasets.prepare_file_paths(
            idx=test_idx,
            features_path=settings.DATA / 'datasets' / 'numpy_arrays' / 'test_features',
            targets_path=None,
        )

        settings.logger.info(
            f'''
            Test: {test_idx.shape[0]} ({test_idx.shape[0] // config["training"]["test_batch_size"] + 1} steps)
            '''
        )

        # Create test dataset and dataloaders
        test_dataset = torch_datasets.TabularDataset(
            feature_paths=test_feature_paths,
            target_paths=None,
            feature_statistics=feature_statistics,
            target_statistics=target_statistics,
            feature_normalization_type=feature_normalization_type,
            target_normalization_type=target_normalization_type
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['test_batch_size'],
            sampler=SequentialSampler(test_dataset),
            pin_memory=False,
            drop_last=False,
            num_workers=config['training']['num_workers']
        )

        test_predictions = []

        for inputs in tqdm(test_loader):

            inputs = inputs.to(device)

            with torch.no_grad():
                if amp:
                    with torch.autocast(device_type=device.type):
                        outputs = model(inputs.half())
                else:
                    outputs = model(inputs)

            test_predictions.append(outputs.detach().cpu())

        test_predictions = torch.cat(test_predictions, dim=0).numpy()

        # Rescale validation targets and predictions back to their normal scales
        test_predictions = (test_predictions * target_stds) + target_means

        target_overwrite_idx = np.where(np.in1d(target_columns, columns_to_overwrite))[0]
        test_predictions[:, target_overwrite_idx] = target_means[target_overwrite_idx]

        df_submission = pd.read_parquet(settings.DATA / 'datasets' / 'sample_submission.parquet')
        df_submission.iloc[:, 1:] = test_predictions
        df_submission.to_parquet(model_directory / 'submission.parquet')
