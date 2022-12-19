from typing import Optional, Union
import numpy as np
import os
import signal
from collections import defaultdict

import torch
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from utils.val_images import validate_images, validate_deformations
from utils.affine_matrix_conversion import cvt_ThetaToM
from utils.points_error_calculation import frechetDist
from utils.ffRemap import dots_remap_bcw

from time import time
from src.train import save_validation_images, calculate_point_metrics


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

    batch_fixed, batch_moving = input_batch[:2]
    batch_fixed, batch_moving = batch_fixed.to(
        device), batch_moving.to(device)

    output_dict = model(batch_moving, batch_fixed)
    output_dict.update({'batch_fixed': batch_fixed, 'batch_moving': batch_moving})

    losses = loss(output_dict)
    train_loss = losses['total_loss']
    train_loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

    optimizer.step()

    if (epoch + 1) % save_step == 0:
        save_validation_images(batch_fixed, batch_moving, output_dict['affine_moving_image'],
                               None, None,
                               image_dir=image_dir, epoch=epoch + 1, train=True)
    # losses['nans'] = output_dict['nans']

    return losses


def apply_theta_2points(batch_points: torch.Tensor,
                        batch_theta: torch.Tensor,
                        w: int, h: int,
                        num_points_interpolation=4):
    # if isinstance(batch_points, np.ndarray):
    #     batch_points = torch.Tensor(batch_points).to(device)
    #
    # if isinstance(batch_deformation, np.ndarray):
    #     batch_deformation = torch.Tensor(batch_deformation).to(device)

    for i, torch_theta in enumerate(batch_theta):
        # theta = batch_theta[i]
        theta = torch.tensor(cvt_ThetaToM(torch_theta.cpu().numpy(), w, h),
                             dtype=batch_theta.dtype).to(batch_theta.device)
        points = batch_points[i]
        points = theta @ torch.cat(
            (points, torch.ones((len(points), 1), device=batch_theta.device, dtype=points.dtype)),
            dim=1).permute(1, 0)
        batch_points[i] = points.permute(1, 0)[:, :2]

    batch_points[..., 0] = torch.clip(batch_points[..., 0], 0, w)
    batch_points[..., 1] = torch.clip(batch_points[..., 1], 0, h)
    return batch_points


def validate_model(input_batch: torch.Tensor,
                   model: torch.nn.Module,
                   device: str,
                   loss: torch.nn.Module,
                   save_step: int,
                   image_dir: str,
                   epoch: int = 0,
                   return_point_metrics: bool = True):
    model.eval()

    with torch.no_grad():
        batch_fixed, batch_moving = input_batch[:2]
        batch_fixed, batch_moving = batch_fixed.to(
            device), batch_moving.to(device)

        output_dict = model(batch_moving, batch_fixed)

        output_dict.update({'batch_fixed': batch_fixed, 'batch_moving': batch_moving})

        losses = loss(output_dict)

    if (epoch + 1) % save_step == 0:
        save_validation_images(batch_fixed, batch_moving, output_dict['affine_moving_image'],
                               None, None,
                               image_dir=image_dir, epoch=epoch + 1, train=False)

    if return_point_metrics:
        points_fixed, points_moving, points_len = input_batch[2:]
        _, _, h, w = input_batch[0].shape

        registered_points = apply_theta_2points(
            points_moving.to(device),
            output_dict['theta_moving2fixed'],
            h, w)

        metrics = calculate_point_metrics(points_fixed,
                                          registered_points.to(points_fixed.device),
                                          points_len)

        metrics2 = calculate_point_metrics(points_fixed, points_moving, points_len)
        metrics['points_error_l2_prev'] = metrics2['points_error_l2']
        return losses, metrics
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
          validate_by_points: bool = True
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

    nan_flag = False
    # Loop over epochs
    global_i = 0
    global_j = 0
    summary_writer = None
    if use_tensorboard:
        os.makedirs(log_dir, exist_ok=True)
        summary_writer = SummaryWriter(log_dir=log_dir)

    best_metric_values = defaultdict(float)
    best_loss_value = None

    for epoch in range(load_epoch, max_epochs):
        # time_epoch_start = time()
        train_losses = defaultdict(lambda: 0.)
        val_losses = defaultdict(lambda: 0.)
        val_metrics = defaultdict(lambda: 0.)
        total = 0

        # total_batches = 0
        # time_batches = 0

        # Training
        for batch in train_loader:
            batch_losses = train_model(input_batch=batch,
                                       model=model,
                                       optimizer=optimizer,
                                       device=device,
                                       loss=loss,
                                       save_step=save_step,
                                       image_dir=image_dir + '/images/',
                                       epoch=epoch)
            # nan_flag = batch_losses['nans']
            # batch_losses.pop('nans')
            # if nan_flag:
            #     break

            for key in batch_losses:
                train_losses[key] += batch_losses[key].item()
            total += 1

            if use_tensorboard:
                for key in batch_losses:
                    summary_writer.add_scalar(key, batch_losses[key].item(), global_i)
                global_i += 1
        #             time_batch_end = time()
        #             time_batches += time_batch_end - time_batch_start
        #             total_batches +=1

        #         print('time for batch =', time_batches / total_batches)
        # total_batches = 0
        for key in train_losses:
            train_losses[key] /= total

        # if nan_flag:
        #     print('found nans in network outputs in the epoch', epoch)
        #     break

        # Testing
        total = 0
        # time_batches = 0
        for batch in val_loader:
            # time_batch_start = time()
            batch_losses, batch_metrics = validate_model(input_batch=batch,
                                                         model=model,
                                                         device=device,
                                                         loss=loss,
                                                         save_step=save_step,
                                                         image_dir=image_dir + '/images/',
                                                         epoch=epoch,
                                                         return_point_metrics=validate_by_points)

            for key in batch_losses:
                val_losses[key] += batch_losses[key].item()

            for key in batch_metrics:
                val_metrics[key] += batch_metrics[key].item()

            total += 1

            if use_tensorboard:
                for key in batch_losses:
                    summary_writer.add_scalar('val_' + key, batch_losses[key].item(), global_j)
                for key in batch_metrics:
                    summary_writer.add_scalar(key, batch_metrics[key].item(), global_j)
                global_j += 1

        #             time_batch_end = time()
        #             time_batches += time_batch_end - time_batch_start
        #             total_batches +=1

        #         print('time for val batch =', time_batches / total_batches)

        for key in val_losses:
            val_losses[key] /= total

        for key in val_metrics:
            val_metrics[key] /= total

        if scheduler is not None:
            scheduler.step(val_losses['total_loss'])

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
            print('Epoch', epoch + 1, f'{key} error: ', val_metrics[key])

        for key in val_metrics:
            if key not in best_metric_values or val_metrics[key] < best_metric_values[key]:
                best_metric_values[key] = val_metrics[key]
                save_model(model, model_name + f'_{key}')
        print()

        if use_tensorboard:
            for key in train_losses:
                summary_writer.add_scalar('epoch_' + key, train_losses[key], epoch)
            for key in val_losses:
                summary_writer.add_scalar('epoch_val_' + key, val_losses[key], epoch)
            for key in val_metrics:
                summary_writer.add_scalar('epoch_val_' + key, val_metrics[key], epoch)

        # time_epoch_end = time()
        # print('time for epoch =', time_epoch_end - time_epoch_start)

        # if (epoch + 1) % save_step == 0:
        #     save_model(model, model_name)

    save_model(model, model_name + '_last')
