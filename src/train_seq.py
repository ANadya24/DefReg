from typing import Optional, Union
import numpy as np
import os
import signal
from collections import defaultdict

import torch
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from time import time
from train import save_validation_images
from scripts.functions import preprocess_image_pair, resize_deformation
from utils.points_error_calculation import compute_l2_error_sequence
from utils.ffRemap import dots_remap_bcw, ff_1_to_k


def train_model(input_batch: torch.Tensor,
                model: torch.nn.Module,
                optimizer: torch.optim,
                device: str,
                loss: torch.nn.Module,
                save_step: int,
                image_dir: str,
                epoch: int = 0):
    model.train()
    optimizer.zero_grad()

    input_batch = input_batch.to(device)

    output_dict = model.forward_seq(input_batch)
    output_dict.update({'batch_fixed': input_batch[:, :1], 'batch_moving': input_batch[:, -1:]})

    losses = loss(output_dict)
    train_loss = losses['total_loss']
    train_loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

    optimizer.step()

    if (epoch + 1) % save_step == 0:
        save_validation_images(output_dict['batch_fixed'], output_dict['batch_moving'],
                               output_dict['batch_registered'],
                               output_dict['batch_deformation'], None,
                               image_dir=image_dir, epoch=epoch + 1, train=True)

    return losses


def validate_model_by_points(model, val_seq, val_mask_seq, val_points, val_config):
    use_theta = model.use_theta
    model.use_theta = False
    defs = np.zeros((1,) + val_seq[0].shape + (2,))
    h, w = val_seq[0].shape

    points = val_points

    deformation = None
    for i in range(1, len(val_seq)):
        frame = i-1
        fixed, moving = preprocess_image_pair(val_seq[frame], val_seq[i], val_config,
                                              val_mask_seq[frame], val_mask_seq[i])
        fixed = fixed.to(model.device)
        moving = moving.to(model.device)
        model_output = model(moving, fixed)
        cur_deformation = model_output['batch_deformation']
        if deformation is not None:
            deformation += model.spatial_transform(deformation, cur_deformation)
        else:
            deformation = cur_deformation.copy()

        np_deformation = resize_deformation(deformation[0].detach().cpu().numpy(), h, w)[None]
        defs = np.concatenate([defs, np_deformation], axis=0)

    reg_points = points.copy()

    for i in range(1, len(defs)):
        cur_def = defs[i]
        reg_points[i] = dots_remap_bcw(points[i], cur_def)

    errors = compute_l2_error_sequence(points)

    output = {'l2_point_error': torch.mean(errors)}
    model.use_theta = use_theta
    return output


def validate_model(input_batch: torch.Tensor,
                   model: torch.nn.Module,
                   device: str,
                   loss: torch.nn.Module,
                   save_step: int,
                   image_dir: str,
                   epoch: int = 0):
    model.eval()

    with torch.no_grad():
        input_batch = input_batch.to(device)

        output_dict = model.forward_seq(input_batch)

        output_dict.update({'batch_fixed': input_batch[:, :1], 'batch_moving': input_batch[:, -1:]})

        losses = loss(output_dict)

    if (epoch + 1) % save_step == 0:
        save_validation_images(output_dict['batch_fixed'],
                               output_dict['batch_moving'], output_dict['batch_registered'],
                               output_dict['batch_deformation'], None,
                               image_dir=image_dir, epoch=epoch + 1, train=False)
    return losses


def train(model: torch.nn.Module,
          train_loader: data.DataLoader,
          val_loader: data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss: torch.nn.Module,
          device: str,
          model_name: str, save_step: int,
          save_dir: str, image_dir: str,
          scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
          log_dir: str = "./logs/",
          load_epoch: int = 0,
          max_epochs: int = 1000,
          use_tensorboard: bool = False,
          validate_by_points: bool = True,
          validation_list: list = []
          ):
    def save_model(model, name):
        if isinstance(model, torch.nn.DataParallel):
            torch.save({
                'model': model.module,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                save_dir + name)
        else:
            torch.save({
                'model': model,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                save_dir + name)
        print(f"Successfuly saved state_dict in {save_dir + name}")

    def sig_handler(signum, frame):
        print('Saved intermediate result!')
        torch.cuda.synchronize()
        save_model(model, model_name + f'_stop_{epoch}')

    signal.signal(signal.SIGINT, sig_handler)

    # Loop over epochs
    global_i = 0
    global_j = 0
    summary_writer = None
    if use_tensorboard:
        os.makedirs(log_dir, exist_ok=True)
        summary_writer = SummaryWriter(log_dir=log_dir)

    best_metric_values = defaultdict(float)
    best_loss_value = None

    if validate_by_points:
        if validation_list == []:
            validate_by_points = False
        else:
            val_seq, val_mask_seq, \
            val_points, val_config = validation_list

            best_metric_values['l2_point_error'] = np.mean(compute_l2_error_sequence(val_points))

    for epoch in range(load_epoch, max_epochs):
        train_losses = defaultdict(lambda: 0.)
        val_losses = defaultdict(lambda: 0.)
        val_metrics = defaultdict(lambda: 0.)
        total = 0

        # Training
        train_loader.dataset.reset()
        for batch in train_loader:
            batch_losses = train_model(input_batch=batch,
                                       model=model,
                                       optimizer=optimizer,
                                       device=device,
                                       loss=loss,
                                       save_step=save_step,
                                       image_dir=image_dir + '/images/',
                                       epoch=epoch)

            for key in batch_losses:
                train_losses[key] += batch_losses[key].item()
            total += 1

            if use_tensorboard:
                for key in batch_losses:
                    summary_writer.add_scalar(key, batch_losses[key].item(), global_i)
                global_i += 1
        for key in train_losses:
            train_losses[key] /= total

        # Testing
        total = 0
        # time_batches = 0
        for batch in val_loader:
            # time_batch_start = time()
            batch_losses = validate_model(input_batch=batch,
                                         model=model,
                                         device=device,
                                         loss=loss,
                                         save_step=save_step,
                                         image_dir=image_dir + '/images/',
                                         epoch=epoch)

            for key in batch_losses:
                val_losses[key] += batch_losses[key].item()
            total += 1

            if use_tensorboard:
                for key in batch_losses:
                    summary_writer.add_scalar('val_' + key, batch_losses[key].item(), global_j)
                global_j += 1

        for key in val_losses:
            val_losses[key] /= total

        if scheduler is not None:
            scheduler.step(val_losses['total_loss'])

        if validate_by_points:
            val_metrics = validate_model_by_points(model, val_seq,
                                                   val_mask_seq, points, val_config)

        print('Epoch', epoch + 1, 'train_loss/test_loss: ',
              train_losses['total_loss'], '/', val_losses['total_loss'])
        for key in val_losses:
            if key == 'total_loss':
                continue
            print('Epoch', epoch + 1, f'{key} train/test: ', train_losses[key], '/', val_losses[key])
        if best_loss_value is None or best_loss_value > val_losses['total_loss']:
            best_loss_value = val_losses['total_loss']
            save_model(model, model_name + '_loss_best')

        print()
        for key in val_metrics:
            print('Epoch', epoch + 1, f'{key}: ', val_metrics[key])

        for key in val_metrics:
            if key not in best_metric_values or val_metrics[key] < best_metric_values[key]:
                best_metric_values[key] = val_metrics[key].item()
                save_model(model, model_name + f'_{key}')
        print()

        if use_tensorboard:
            for key in train_losses:
                summary_writer.add_scalar('epoch_' + key, train_losses[key], epoch)
            for key in val_losses:
                summary_writer.add_scalar('epoch_val_' + key, val_losses[key], epoch)
            for key in val_metrics:
                summary_writer.add_scalar('epoch_val_' + key, val_metrics[key], epoch)

    save_model(model, model_name + '_last')
