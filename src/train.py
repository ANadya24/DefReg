from typing import Optional, Union
import numpy as np
import os
import signal
from collections import defaultdict

import torch
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from utils.val_images import validate_images, validate_deformations
from utils.points_error_calculation import frechetDist
from utils.ffRemap import dots_remap_bcw


def save_validation_images(batch_fixed: torch.Tensor,
                           batch_moving: torch.Tensor,
                           batch_registered: torch.Tensor,
                           batch_deformation: torch.Tensor,
                           gt_deformation: Optional[torch.Tensor],
                           image_dir: str, epoch: int, train: bool = True,
                           num_images2save: int = 3):
    validate_images(batch_fixed[:num_images2save],
                    batch_moving[:num_images2save],
                    batch_registered[:num_images2save],
                    val_dir=image_dir, epoch=epoch + 1, train=train)

    if gt_deformation is not None:
        validate_deformations(batch_deformation[:num_images2save],
                              gt_deformation[:num_images2save],
                              val_dir=image_dir, epoch=epoch + 1, train=train)


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

    optimizer.step()

    if (epoch + 1) % save_step == 0:
        save_validation_images(batch_fixed, batch_moving, output_dict['batch_registered'],
                               output_dict['batch_deformation'], None,
                               image_dir=image_dir, epoch=epoch + 1, train=True)

    return losses


def apply_deformation_2points(batch_points, batch_deformation):
    if isinstance(batch_points, torch.Tensor):
        batch_points = batch_points.detach().cpu().numpy()
    if isinstance(batch_deformation, torch.Tensor):
        batch_deformation = batch_deformation.detach().cpu().numpy()

    deformed_points = np.zeros_like(batch_points)
    for i in range(1, len(batch_points)):
        deformation = batch_deformation[i]
        deformed_points[i] = dots_remap_bcw(batch_points[i].copy(), deformation.copy())

    return deformed_points


def calculate_point_metrics(batch_points1: Union[torch.Tensor, np.ndarray],
                            batch_points2: Union[torch.Tensor, np.ndarray],
                            batch_points_len: Union[torch.Tensor, np.ndarray]):
    if isinstance(batch_points1, torch.Tensor):
        batch_points1 = batch_points1.detach().cpu().numpy()
    if isinstance(batch_points2, torch.Tensor):
        batch_points2 = batch_points2.detach().cpu().numpy()
    if isinstance(batch_points_len, torch.Tensor):
        batch_points_len = batch_points_len.detach().cpu().numpy()

    err = defaultdict(list)

    for (points1, points2, points_len) in zip(batch_points1, batch_points2, batch_points_len):
        print(points1.shape, points2.shape, points_len)

        inner1 = points1[:points_len[0]]
        inner2 = points2[:points_len[0]]

        bound1 = points1[points_len[0]: points_len[1]]
        bound2 = points2[points_len[0]: points_len[1]]

        lines1 = points1[points_len[1]:]
        lines2 = points2[points_len[1]:]
        len1, len2, len3, _ = points_len[2:]

        inner_err = ((((inner1 - inner2) ** 2).sum(axis=1)) ** 0.5).sum(axis=0) / float(inner1.shape[1])

        bound_err = ((((bound1 - bound2) ** 2).sum(axis=1)) ** 0.5).sum(axis=0) / float(bound1.shape[1])

        b1 = frechetDist(lines1[:len1], lines2[:len1])
        b2 = frechetDist(lines1[len1:len1 + len2], lines2[len1:len1 + len2])
        b3 = frechetDist(lines1[len1 + len2:len1 + len2 + len3],
                         lines2[len1 + len2:len1 + len2 + len3])
        b4 = frechetDist(lines1[len1 + len2 + len3:], lines2[len1 + len2 + len3:])
        line_err = (b1 + b2 + b3 + b4) / 4.
        err['inner'].append(inner_err)
        err['bound'].append(bound_err)
        err['lines'].append(line_err)

    output = {'inner': np.mean(err['inner']),
              'bound': np.mean(err['bound']),
              'lines': np.mean(err['lines'])}

    return output


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
        batch_fixed, batch_moving, points_fixed, points_moving, points_len = input_batch
        batch_fixed, batch_moving = batch_fixed.to(
            device), batch_moving.to(device)

        output_dict = model(batch_moving, batch_fixed)
        output_dict.update({'batch_fixed': batch_fixed, 'batch_moving': batch_moving})

        losses = loss(output_dict)

        if (epoch + 1) % save_step == 0:
            save_validation_images(batch_fixed, batch_moving, output_dict['batch_registered'],
                                   output_dict['batch_deformation'], None,
                                   image_dir=image_dir, epoch=epoch + 1, train=False)

        if return_point_metrics:
            registered_points = apply_deformation_2points(
                points_moving,
                output_dict['batch_deformation'])
            metrics = calculate_point_metrics(points_fixed, registered_points, points_len)
            return losses, metrics
        return losses


def train(model: torch.nn.Module,
          train_loader: data.DataLoader,
          val_loader: data.DataLoader,
          optimizer: torch.optim,
          scheduler: Optional[torch.optim.lr_scheduler],
          loss: torch.nn.Module,
          device: str,
          model_name: str, save_step: int,
          save_dir: str, image_dir: str,
          log_dir: str = "./logs/",
          load_epoch: int = 0,
          max_epochs: int = 1000,
          use_tensorboard: bool = False,
          ):
    def save_model(model, name=model_name + f'_stop'):
        if isinstance(model, torch.nn.DataParallel):
            torch.save({
                'model': model.module,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                save_dir + name + f'_{epoch + 1}')
        else:
            torch.save({
                'model': model,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                save_dir + name + f'{epoch + 1}')
        print(f"Successfuly saved state_dict in {save_dir + model_name + f'_stop_{epoch + 1}'}")

    def sig_handler(signum, frame):
        print('Saved intermediate result!')
        torch.cuda.synchronize()
        save_model(model)

    signal.signal(signal.SIGINT, sig_handler)
    # Loop over epochs
    global_i = 0
    global_j = 0
    summary_writer = None
    if use_tensorboard:
        os.makedirs(log_dir, exist_ok=True)
        summary_writer = SummaryWriter(log_dir=log_dir)

    best_metric_values = defaultdict(int)

    for epoch in range(load_epoch, max_epochs):
        train_losses = defaultdict(lambda: 0)
        val_losses = defaultdict(lambda: 0)
        val_metrics = defaultdict(lambda: 0)
        total = 0

        # Training
        for batch in train_loader:
            batch_losses = train_model(input_batch=batch,
                                       model=model,
                                       optimizer=optimizer,
                                       device=device,
                                       loss=loss,
                                       save_step=save_step,
                                       image_dir=image_dir,
                                       epoch=epoch)

            for key in batch_losses:
                train_losses[key] += batch_losses[key]
            total += 1

            if use_tensorboard:
                for key in batch_losses:
                    summary_writer.add_scalar(key, batch_losses[key].item(), global_i)
                global_i += 1

        if scheduler is not None:
            scheduler.step()

        for key in train_losses:
            train_losses[key] /= total

        # Testing
        total = 0
        for batch in val_loader:

            batch_losses, batch_metrics = validate_model(input_batch=batch,
                                                         model=model,
                                                         device=device,
                                                         loss=loss,
                                                         save_step=save_step,
                                                         image_dir=image_dir,
                                                         epoch=epoch,
                                                         return_point_metrics=True)

            for key in batch_losses:
                val_losses[key] += batch_losses[key]

            for key in batch_metrics:
                val_metrics[key] += batch_metrics[key]

            total += 1

            if use_tensorboard:
                for key in batch_losses:
                    summary_writer.add_scalar(key, 'val_' + batch_losses[key].item(), global_j)
                global_j += 1

        for key in val_losses:
            val_losses[key] /= total

        for key in val_metrics:
            val_metrics[key] /= total

        print('Epoch', epoch + 1, 'train_loss/test_loss: ',
              train_losses['total_loss'], '/', val_losses['total_loss'])
        for key in val_losses:
            if key == 'total_loss':
                continue
            print('Epoch', epoch + 1, f'{key} train/test: ', train_losses[key], '/', val_losses[key])

        for key in val_metrics:
            print('Epoch', epoch + 1, f'Error: ', val_metrics[key])

        for key in best_metric_values:
            if key not in best_metric_values or val_metrics[key] < best_metric_values[key]:
                best_metric_values[key] = val_metrics[key]
                save_model(model, model_name + f'_{key}')

        if (epoch + 1) % save_step == 0:
            save_model(model, model_name)
