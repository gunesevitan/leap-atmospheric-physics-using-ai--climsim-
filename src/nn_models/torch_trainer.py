import os
import sys
import argparse
import yaml
import json
from glob import glob
from pathlib import Path
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


def train(training_loader, model, criterion, optimizer, device, scheduler=None, amp=False, statistics=None):

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

    Ak = torch.as_tensor([5.59e-05, 0.0001008, 0.0001814, 0.0003244, 0.0005741, 0.0009986, 0.0016961, 0.0027935, 0.0044394, 0.0067923, 0.0100142, 0.0142748, 0.0197589, 0.0266627, 0.035166, 0.0453892, 0.0573601, 0.0710184, 0.086261, 0.1029992, 0.1211833, 0.1407723, 0.1616703, 0.181999, 0.1769112, 0.1717129, 0.1664573, 0.1611637, 0.1558164, 0.1503775, 0.144805, 0.1390666, 0.1331448, 0.1270342, 0.1207383, 0.11427, 0.107658, 0.1009552, 0.0942421, 0.0876184, 0.0811846, 0.0750186, 0.0691602, 0.06361, 0.0583443, 0.0533368, 0.0485757, 0.044067, 0.039826, 0.0358611, 0.0321606, 0.0286887, 0.0253918, 0.0222097, 0.0190872, 0.0159809, 0.0128614, 0.0097109, 0.00652, 0.0032838], device='cuda')
    Bk = torch.as_tensor([0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0016785, 0.0295868, 0.058101, 0.0869295, 0.1159665, 0.145298, 0.1751322, 0.2056993, 0.2371761, 0.2696592, 0.3031777, 0.3377127, 0.3731935, 0.4094624, 0.4462289, 0.4830525, 0.5193855, 0.5546772, 0.5884994, 0.6206347, 0.6510795, 0.6799635, 0.7074307, 0.7335472, 0.7582786, 0.7815416, 0.8032905, 0.8235891, 0.8426334, 0.8607178, 0.8781726, 0.8953009, 0.9123399, 0.9294513, 0.9467325, 0.9642358, 0.9819873], device='cuda')

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

        sequential_inputs = torch.cat((
            inputs[:, :360].view(inputs.shape[0], -1, 60),
            inputs[:, 376:].view(inputs.shape[0], -1, 60)
        ), dim=1)
        scalar_inputs = inputs[:, 360:376]

        wind_speed = torch.sqrt(sequential_inputs[:, 4, :] ** 2 + sequential_inputs[:, 5, :] ** 2)
        wind_direction = (torch.rad2deg(torch.arctan2(sequential_inputs[:, 4, :], sequential_inputs[:, 5, :])) + 180) % 360

        temperature = sequential_inputs[:, 0, :].clip(165, 321)
        pressure = Ak + Bk * scalar_inputs[:, 0].view(-1, 1) / 100000.0
        relative_humidity = sequential_inputs[:, 1, :] * 263 * pressure
        relative_humidity = relative_humidity * torch.exp(-17.67 * (temperature - 273.16) / (temperature - 29.65))

        sequential_inputs = torch.cat((
            sequential_inputs,
            wind_speed.view(-1, 1, 60),
            wind_direction.view(-1, 1, 60),
            relative_humidity.view(-1, 1, 60),
        ), dim=1)

        sequential_inputs_gradients = sequential_inputs.diff(n=1, dim=2, prepend=sequential_inputs[:, :, :1])

        sequential_inputs -= statistics['feature']['sequential_mean']
        sequential_inputs /= statistics['feature']['sequential_std']

        sequential_inputs_gradients -= statistics['feature']['gradient_mean']
        sequential_inputs_gradients /= statistics['feature']['gradient_std']

        scalar_inputs -= statistics['feature']['scalar_mean']
        scalar_inputs /= statistics['feature']['scalar_std']

        inputs = torch.cat((
            sequential_inputs,
            torch.unsqueeze(scalar_inputs, dim=-1).repeat(repeats=(1, 1, 60)),
            sequential_inputs_gradients
        ), dim=1)

        targets -= statistics['target']['mean']
        targets /= statistics['target']['rms']

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
            if scheduler.last_epoch < scheduler.total_steps:
                scheduler.step()

        running_loss += loss.detach().item() * len(inputs)
        lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
        progress_bar.set_description(f'lr: {lr:.8f} - training loss: {running_loss / len(training_loader.sampler):.4f}')

    training_loss = running_loss / len(training_loader.sampler)

    return training_loss


def validate(validation_loader, model, criterion, device, amp=False, statistics=None):

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

    Ak = torch.as_tensor([5.59e-05, 0.0001008, 0.0001814, 0.0003244, 0.0005741, 0.0009986, 0.0016961, 0.0027935, 0.0044394, 0.0067923, 0.0100142, 0.0142748, 0.0197589, 0.0266627, 0.035166, 0.0453892, 0.0573601, 0.0710184, 0.086261, 0.1029992, 0.1211833, 0.1407723, 0.1616703, 0.181999, 0.1769112, 0.1717129, 0.1664573, 0.1611637, 0.1558164, 0.1503775, 0.144805, 0.1390666, 0.1331448, 0.1270342, 0.1207383, 0.11427, 0.107658, 0.1009552, 0.0942421, 0.0876184, 0.0811846, 0.0750186, 0.0691602, 0.06361, 0.0583443, 0.0533368, 0.0485757, 0.044067, 0.039826, 0.0358611, 0.0321606, 0.0286887, 0.0253918, 0.0222097, 0.0190872, 0.0159809, 0.0128614, 0.0097109, 0.00652, 0.0032838], device='cuda')
    Bk = torch.as_tensor([0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0016785, 0.0295868, 0.058101, 0.0869295, 0.1159665, 0.145298, 0.1751322, 0.2056993, 0.2371761, 0.2696592, 0.3031777, 0.3377127, 0.3731935, 0.4094624, 0.4462289, 0.4830525, 0.5193855, 0.5546772, 0.5884994, 0.6206347, 0.6510795, 0.6799635, 0.7074307, 0.7335472, 0.7582786, 0.7815416, 0.8032905, 0.8235891, 0.8426334, 0.8607178, 0.8781726, 0.8953009, 0.9123399, 0.9294513, 0.9467325, 0.9642358, 0.9819873], device='cuda')

    model.eval()
    progress_bar = tqdm(validation_loader)

    running_loss = 0.0
    validation_targets = []
    validation_predictions = []

    for step, (inputs, targets) in enumerate(progress_bar):

        inputs = inputs.to(device)
        targets = targets.to(device)

        sequential_inputs = torch.cat((
            inputs[:, :360].view(inputs.shape[0], -1, 60),
            inputs[:, 376:].view(inputs.shape[0], -1, 60)
        ), dim=1)
        scalar_inputs = inputs[:, 360:376]

        wind_speed = torch.sqrt(sequential_inputs[:, 4, :] ** 2 + sequential_inputs[:, 5, :] ** 2)
        wind_direction = (torch.rad2deg(torch.arctan2(sequential_inputs[:, 4, :], sequential_inputs[:, 5, :])) + 180) % 360

        temperature = sequential_inputs[:, 0, :].clip(165, 321)
        pressure = Ak + Bk * scalar_inputs[:, 0].view(-1, 1) / 100000.0
        relative_humidity = sequential_inputs[:, 1, :] * 263 * pressure
        relative_humidity = relative_humidity * torch.exp(-17.67 * (temperature - 273.16) / (temperature - 29.65))

        sequential_inputs = torch.cat((
            sequential_inputs,
            wind_speed.view(-1, 1, 60),
            wind_direction.view(-1, 1, 60),
            relative_humidity.view(-1, 1, 60),
        ), dim=1)
        sequential_inputs_gradients = sequential_inputs.diff(n=1, dim=2, prepend=sequential_inputs[:, :, :1])

        sequential_inputs -= statistics['feature']['sequential_mean']
        sequential_inputs /= statistics['feature']['sequential_std']
        sequential_inputs_gradients -= statistics['feature']['gradient_mean']
        sequential_inputs_gradients /= statistics['feature']['gradient_std']
        scalar_inputs -= statistics['feature']['scalar_mean']
        scalar_inputs /= statistics['feature']['scalar_std']

        inputs = torch.cat((
            sequential_inputs,
            torch.unsqueeze(scalar_inputs, dim=-1).repeat(repeats=(1, 1, 60)),
            sequential_inputs_gradients
        ), dim=1)

        targets -= statistics['target']['mean']
        targets /= statistics['target']['rms']

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

    # Select targets and features as individual numpy arrays
    df = pd.read_parquet(settings.DATA / 'datasets' / 'train.parquet')
    columns = df.columns.tolist()
    target_columns = columns[-368:]
    target_column_groups = preprocessing.get_target_column_groups(columns=target_columns)
    feature_columns = columns[:-368]
    feature_column_groups = preprocessing.get_feature_column_groups(columns=feature_columns)
    prediction_columns = [f'{column}_prediction' for column in target_columns]

    features = df[feature_columns].to_numpy()
    targets = df[target_columns].to_numpy()
    del df

    # Load precomputed folds
    folds = np.load(settings.DATA / 'folds.npz')['arr_0']
    file_idx = np.arange(folds.shape[0], dtype=np.uint32)
    settings.logger.info(f'Loaded folds: {folds.shape}')

    # Load precomputed normalizers
    statistics = preprocessing.load_statistics(statistics_directory=settings.DATA / 'datasets')
    statistics['feature']['std'] = np.where(statistics['feature']['std'] <= 1e-9, 1.0, statistics['feature']['std'])
    statistics['target']['rms'] = np.where(statistics['target']['rms'] == 0, 1.0, statistics['target']['rms'])
    statistics['feature']['gradient_std'] = np.where(statistics['feature']['gradient_std'] <= 1e-9, 1.0, statistics['feature']['gradient_std'])

    statistics['feature']['mean'] = torch.as_tensor(statistics['feature']['mean'], dtype=torch.float, device='cuda')
    statistics['feature']['sequential_mean'] = torch.cat((
        statistics['feature']['mean'][:360].view(-1, 60),
        statistics['feature']['mean'][376:].view(-1, 60)
    ), dim=0)
    statistics['feature']['scalar_mean'] = statistics['feature']['mean'][360:376]

    statistics['feature']['std'] = torch.as_tensor(statistics['feature']['std'], dtype=torch.float, device='cuda')
    statistics['feature']['sequential_std'] = torch.cat((
        statistics['feature']['std'][:360].view(-1, 60),
        statistics['feature']['std'][376:].view(-1, 60)
    ), dim=0)
    statistics['feature']['scalar_std'] = statistics['feature']['std'][360:376]

    statistics['feature']['gradient_mean'] = torch.as_tensor(statistics['feature']['gradient_mean'], dtype=torch.float, device='cuda').view(-1, 60)
    statistics['feature']['gradient_std'] = torch.as_tensor(statistics['feature']['gradient_std'], dtype=torch.float, device='cuda').view(-1, 60)

    statistics['target']['mean'] = torch.as_tensor(statistics['target']['mean'], dtype=torch.float, device='cuda')
    statistics['target']['rms'] = torch.as_tensor(statistics['target']['rms'], dtype=torch.float, device='cuda')

    torch.multiprocessing.set_sharing_strategy('file_system')

    if args.mode == 'training':

        training_metadata = {}

        # Create training and validation indices
        training_mask = folds[:, config['training']['training_folds']].any(axis=1)
        validation_mask = folds[:, config['training']['validation_folds']].any(axis=1)
        training_idx = file_idx[training_mask]
        validation_idx = file_idx[validation_mask]

        settings.logger.info(
            f'''
            Training Folds: {config['training']['training_folds']} {training_idx.shape[0]} ({training_idx.shape[0] // config["training"]["training_batch_size"] + 1} steps)
            Validation Folds: {config['training']['validation_folds']} {validation_idx.shape[0]} ({validation_idx.shape[0] // config["training"]["test_batch_size"] + 1} steps)
            '''
        )

        # Create training and validation datasets and dataloaders
        training_dataset = torch_datasets.TabularInMemoryDataset(
            features=features[training_idx],
            targets=targets[training_idx]
        )
        training_loader = DataLoader(
            training_dataset,
            batch_size=config['training']['training_batch_size'],
            sampler=RandomSampler(training_dataset, replacement=False),
            pin_memory=False,
            drop_last=False,
            num_workers=config['training']['num_workers']
        )
        validation_dataset = torch_datasets.TabularInMemoryDataset(
            features=features[validation_idx],
            targets=targets[validation_idx]
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
                amp=amp,
                statistics=statistics
            )

            validation_loss, validation_targets, validation_predictions = validate(
                validation_loader=validation_loader,
                model=model,
                criterion=criterion,
                device=device,
                amp=amp,
                statistics=statistics
            )

            training_results = {'loss': training_loss}
            validation_results = {'loss': validation_loss}
            global_validation_scores = None

            settings.logger.info(
                f'''
                Epoch {epoch}
                Training Loss: {training_loss:.6f}
                Validation Loss: {validation_loss:.6f}
                Validation Scores: {global_validation_scores}
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
        oof_mean_predictions = []

        global_scores = []
        target_scores = []

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

        Ak = torch.as_tensor([5.59e-05, 0.0001008, 0.0001814, 0.0003244, 0.0005741, 0.0009986, 0.0016961, 0.0027935, 0.0044394, 0.0067923, 0.0100142, 0.0142748, 0.0197589, 0.0266627, 0.035166, 0.0453892, 0.0573601, 0.0710184, 0.086261, 0.1029992, 0.1211833, 0.1407723, 0.1616703, 0.181999, 0.1769112, 0.1717129, 0.1664573, 0.1611637, 0.1558164, 0.1503775, 0.144805, 0.1390666, 0.1331448, 0.1270342, 0.1207383, 0.11427, 0.107658, 0.1009552, 0.0942421, 0.0876184, 0.0811846, 0.0750186, 0.0691602, 0.06361, 0.0583443, 0.0533368, 0.0485757, 0.044067, 0.039826, 0.0358611, 0.0321606, 0.0286887, 0.0253918, 0.0222097, 0.0190872, 0.0159809, 0.0128614, 0.0097109, 0.00652, 0.0032838], device='cuda')
        Bk = torch.as_tensor([0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0016785, 0.0295868, 0.058101, 0.0869295, 0.1159665, 0.145298, 0.1751322, 0.2056993, 0.2371761, 0.2696592, 0.3031777, 0.3377127, 0.3731935, 0.4094624, 0.4462289, 0.4830525, 0.5193855, 0.5546772, 0.5884994, 0.6206347, 0.6510795, 0.6799635, 0.7074307, 0.7335472, 0.7582786, 0.7815416, 0.8032905, 0.8235891, 0.8426334, 0.8607178, 0.8781726, 0.8953009, 0.9123399, 0.9294513, 0.9467325, 0.9642358, 0.9819873], device='cuda')

        for fold in test_folds:

            # Create validation indices
            validation_mask = folds[:, fold] == 1
            validation_idx = file_idx[validation_mask]

            settings.logger.info(
                f'''
                Validation Folds: {fold} {validation_idx.shape[0]} ({validation_idx.shape[0] // config["training"]["test_batch_size"] + 1} steps)
                '''
            )

            # Create validation dataset and dataloaders
            validation_dataset = torch_datasets.TabularInMemoryDataset(
                features=features[validation_idx],
                targets=targets[validation_idx]
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

            for inputs, targets_ in tqdm(validation_loader):

                inputs = inputs.to(device)
                targets_ = targets_.to(device)

                sequential_inputs = torch.cat((
                    inputs[:, :360].view(inputs.shape[0], -1, 60),
                    inputs[:, 376:].view(inputs.shape[0], -1, 60)
                ), dim=1)
                scalar_inputs = inputs[:, 360:376]

                wind_speed = torch.sqrt(sequential_inputs[:, 4, :] ** 2 + sequential_inputs[:, 5, :] ** 2)
                wind_direction = (torch.rad2deg(torch.arctan2(sequential_inputs[:, 4, :], sequential_inputs[:, 5, :])) + 180) % 360

                temperature = sequential_inputs[:, 0, :].clip(165, 321)
                pressure = Ak + Bk * scalar_inputs[:, 0].view(-1, 1) / 100000.0
                relative_humidity = sequential_inputs[:, 1, :] * 263 * pressure
                relative_humidity = relative_humidity * torch.exp(-17.67 * (temperature - 273.16) / (temperature - 29.65))

                sequential_inputs = torch.cat((
                    sequential_inputs,
                    wind_speed.view(-1, 1, 60),
                    wind_direction.view(-1, 1, 60),
                    relative_humidity.view(-1, 1, 60),
                ), dim=1)
                sequential_inputs_gradients = sequential_inputs.diff(n=1, dim=2, prepend=sequential_inputs[:, :, :1])

                sequential_inputs -= statistics['feature']['sequential_mean']
                sequential_inputs /= statistics['feature']['sequential_std']
                sequential_inputs_gradients -= statistics['feature']['gradient_mean']
                sequential_inputs_gradients /= statistics['feature']['gradient_std']
                scalar_inputs -= statistics['feature']['scalar_mean']
                scalar_inputs /= statistics['feature']['scalar_std']

                inputs = torch.cat((
                    sequential_inputs,
                    torch.unsqueeze(scalar_inputs, dim=-1).repeat(repeats=(1, 1, 60)),
                    sequential_inputs_gradients
                ), dim=1)

                targets_ -= statistics['target']['mean']
                targets_ /= statistics['target']['rms']

                with torch.no_grad():
                    if amp:
                        with torch.autocast(device_type=device.type):
                            outputs = model(inputs.half())
                    else:
                        outputs = model(inputs)

                validation_targets.append(targets_.cpu())
                validation_predictions.append(outputs.detach().cpu())

            validation_targets = torch.cat(validation_targets, dim=0).float().numpy()
            validation_predictions = torch.cat(validation_predictions, dim=0).numpy()

            statistics['feature']['mean'] = statistics['feature']['mean'].cpu().numpy()
            statistics['feature']['std'] = statistics['feature']['std'].cpu().numpy()
            statistics['target']['mean'] = statistics['target']['mean'].cpu().numpy()
            statistics['target']['rms'] = statistics['target']['rms'].cpu().numpy()

            # Rescale validation targets and predictions back to their normal scales
            validation_targets *= statistics['target']['rms']
            validation_targets += statistics['target']['mean']
            validation_predictions *= statistics['target']['rms']
            validation_predictions += statistics['target']['mean']

            q0002_replace_idx = [
                120, 121, 122, 123, 124, 125, 126, 127, 128,
                129, 130, 131, 132, 133, 134, 135, 136, 137,
                138, 139, 140, 141, 142, 143, 144, 145, 146
            ]
            q0002_replace_features = features[validation_idx][:, q0002_replace_idx]
            validation_predictions[:, q0002_replace_idx] = -q0002_replace_features / 1200

            for target_idx in np.arange(len(target_columns)):
                validation_predictions[:, target_idx] = np.clip(
                    validation_predictions[:, target_idx],
                    a_min=statistics['target']['min'][target_idx],
                    a_max=statistics['target']['max'][target_idx],
                )

            # Calculate validation scores with model predictions
            global_validation_scores, target_validation_scores = metrics.regression_scores(
                y_true=validation_targets,
                y_pred=validation_predictions,
                weights=statistics['target']['weight'],
                target_columns=target_columns
            )

            settings.logger.info(
                f'''
                Fold {fold}
                Model Prediction Validation Scores: {json.dumps(global_validation_scores, indent=2)}
                '''
            )

            oof_targets.append(validation_targets)
            oof_predictions.append(validation_predictions)

            global_scores.append(global_validation_scores)
            target_scores.append(target_validation_scores)

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

        # Calculate OOF scores with model predictions
        global_oof_scores, target_oof_scores = metrics.regression_scores(
            y_true=oof_targets,
            y_pred=oof_predictions,
            weights=statistics['target']['weight'],
            target_columns=target_columns
        )

        settings.logger.info(
            f'''
            OOF
            Model Prediction OOF Scores: {json.dumps(global_oof_scores, indent=2)}
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
        single_target_scores = [df.loc[single_targets, ['mse', 'mae', 'r2_score']] for df in target_scores]
        visualization.visualize_single_target_scores(
            single_target_scores=single_target_scores,
            title=f'Single Target Fold and OOF Scores of {len(folds)} Model(s)',
            path=model_directory / 'single_target_scores.png'
        )
        settings.logger.info(f'single_target_scores.png is saved to {model_directory}')

        vertically_resolved_targets = {name: columns for name, columns in target_column_groups.items() if len(columns) == 60}
        for target_group, columns in vertically_resolved_targets.items():
            visualization.visualize_vertically_resolved_target_scores(
                vertically_resolved_target_scores=[df.loc[columns, ['mse', 'mae', 'r2_score']] for df in target_scores],
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
                    score=target_oof_scores.loc[target_column, 'r2_score'],
                    target_column=target_column,
                    weight=statistics['target']['weight'][target_column_idx],
                    path=predictions_visualization_directory / f'{target_column}.png'
                )

        oof_idx = np.where(folds[:, test_folds].any(axis=1))[0]
        np.savez_compressed(model_directory / 'oof_predictions.npz', idx=oof_idx, predictions=oof_predictions)
        settings.logger.info(f'oof_predictions.npz is saved to {model_directory}')

    elif args.mode == 'submission':

        test_features = pd.read_parquet(settings.DATA / 'datasets' / 'test.parquet').to_numpy()
        test_idx = np.arange(625000)

        # Set model, device and seed for reproducible results
        torch_utilities.set_seed(config['training']['random_state'], deterministic_cudnn=config['training']['deterministic_cudnn'])
        device = torch.device(config['training']['device'])
        amp = config['training']['amp']

        model_file_name = config['submission']['model_file_name']
        model = getattr(torch_modules, config['model']['model_class'])(**config['model']['model_args'])
        model.load_state_dict(torch.load(model_directory / model_file_name))
        model.to(device)
        model.eval()

        settings.logger.info(
            f'''
            Test: {test_idx.shape[0]} ({test_idx.shape[0] // config["training"]["test_batch_size"] + 1} steps)
            '''
        )

        # Create test dataset and dataloaders
        test_dataset = torch_datasets.TabularInMemoryDataset(
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

        test_predictions = []

        Ak = torch.as_tensor([5.59e-05, 0.0001008, 0.0001814, 0.0003244, 0.0005741, 0.0009986, 0.0016961, 0.0027935, 0.0044394, 0.0067923, 0.0100142, 0.0142748, 0.0197589, 0.0266627, 0.035166, 0.0453892, 0.0573601, 0.0710184, 0.086261, 0.1029992, 0.1211833, 0.1407723, 0.1616703, 0.181999, 0.1769112, 0.1717129, 0.1664573, 0.1611637, 0.1558164, 0.1503775, 0.144805, 0.1390666, 0.1331448, 0.1270342, 0.1207383, 0.11427, 0.107658, 0.1009552, 0.0942421, 0.0876184, 0.0811846, 0.0750186, 0.0691602, 0.06361, 0.0583443, 0.0533368, 0.0485757, 0.044067, 0.039826, 0.0358611, 0.0321606, 0.0286887, 0.0253918, 0.0222097, 0.0190872, 0.0159809, 0.0128614, 0.0097109, 0.00652, 0.0032838], device='cuda')
        Bk = torch.as_tensor([0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0016785, 0.0295868, 0.058101, 0.0869295, 0.1159665, 0.145298, 0.1751322, 0.2056993, 0.2371761, 0.2696592, 0.3031777, 0.3377127, 0.3731935, 0.4094624, 0.4462289, 0.4830525, 0.5193855, 0.5546772, 0.5884994, 0.6206347, 0.6510795, 0.6799635, 0.7074307, 0.7335472, 0.7582786, 0.7815416, 0.8032905, 0.8235891, 0.8426334, 0.8607178, 0.8781726, 0.8953009, 0.9123399, 0.9294513, 0.9467325, 0.9642358, 0.9819873], device='cuda')

        for inputs in tqdm(test_loader):

            inputs = inputs.to(device)

            sequential_inputs = torch.cat((
                inputs[:, :360].view(inputs.shape[0], -1, 60),
                inputs[:, 376:].view(inputs.shape[0], -1, 60)
            ), dim=1)
            scalar_inputs = inputs[:, 360:376]

            wind_speed = torch.sqrt(sequential_inputs[:, 4, :] ** 2 + sequential_inputs[:, 5, :] ** 2)
            wind_direction = (torch.rad2deg(torch.arctan2(sequential_inputs[:, 4, :], sequential_inputs[:, 5, :])) + 180) % 360

            temperature = sequential_inputs[:, 0, :].clip(165, 321)
            pressure = Ak + Bk * scalar_inputs[:, 0].view(-1, 1) / 100000.0
            relative_humidity = sequential_inputs[:, 1, :] * 263 * pressure
            relative_humidity = relative_humidity * torch.exp(-17.67 * (temperature - 273.16) / (temperature - 29.65))

            sequential_inputs = torch.cat((
                sequential_inputs,
                wind_speed.view(-1, 1, 60),
                wind_direction.view(-1, 1, 60),
                relative_humidity.view(-1, 1, 60),
            ), dim=1)
            sequential_inputs_gradients = sequential_inputs.diff(n=1, dim=2, prepend=sequential_inputs[:, :, :1])

            sequential_inputs -= statistics['feature']['sequential_mean']
            sequential_inputs /= statistics['feature']['sequential_std']
            sequential_inputs_gradients -= statistics['feature']['gradient_mean']
            sequential_inputs_gradients /= statistics['feature']['gradient_std']
            scalar_inputs -= statistics['feature']['scalar_mean']
            scalar_inputs /= statistics['feature']['scalar_std']

            inputs = torch.cat((
                sequential_inputs,
                torch.unsqueeze(scalar_inputs, dim=-1).repeat(repeats=(1, 1, 60)),
                sequential_inputs_gradients
            ), dim=1)

            with torch.no_grad():
                if amp:
                    with torch.autocast(device_type=device.type):
                        outputs = model(inputs.half())
                else:
                    outputs = model(inputs)

            test_predictions.append(outputs.detach().cpu())

        test_predictions = torch.cat(test_predictions, dim=0).numpy()

        # Rescale validation targets and predictions back to their normal scales
        statistics['target']['mean'] = statistics['target']['mean'].cpu().numpy()
        statistics['target']['rms'] = statistics['target']['rms'].cpu().numpy()

        test_predictions *= statistics['target']['rms']
        test_predictions += statistics['target']['mean']

        q0002_replace_idx = [
            120, 121, 122, 123, 124, 125, 126, 127, 128,
            129, 130, 131, 132, 133, 134, 135, 136, 137,
            138, 139, 140, 141, 142, 143, 144, 145, 146
        ]
        q0002_replace_features = test_features[:, q0002_replace_idx]
        test_predictions[:, q0002_replace_idx] = -q0002_replace_features / 1200

        for target_idx in np.arange(len(target_columns)):
            test_predictions[:, target_idx] = np.clip(
                test_predictions[:, target_idx],
                a_min=statistics['target']['min'][target_idx],
                a_max=statistics['target']['max'][target_idx],
            )

        df_submission = pd.read_parquet(settings.DATA / 'datasets' / 'sample_submission.parquet')
        df_submission[target_columns] = df_submission[target_columns].astype(np.float64)

        df_submission.iloc[:, 1:] *= test_predictions.astype(np.float64)
        df_submission.to_parquet(model_directory / 'submission.parquet')
