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

from time import time


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

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

    optimizer.step()

    if (epoch + 1) % save_step == 0:
        save_validation_images(batch_fixed, batch_moving, output_dict['batch_registered'],
                               output_dict['batch_deformation'], None,
                               image_dir=image_dir, epoch=epoch + 1, train=True)

    return losses


def apply_deformation_2points(batch_points: torch.Tensor,
                              batch_deformation: torch.Tensor,
                              num_points_interpolation=4):
    # if isinstance(batch_points, np.ndarray):
    #     batch_points = torch.Tensor(batch_points).to(device)
    #
    # if isinstance(batch_deformation, np.ndarray):
    #     batch_deformation = torch.Tensor(batch_deformation).to(device)
    b, ch, h, w = batch_deformation.shape

    for i, deformation in enumerate(batch_deformation):
        y, x = torch.meshgrid(torch.arange(h),
                              torch.arange(w))
        x = x + deformation[0]
        y = y + deformation[1]

        deformation *= -1.

        distance_map = ((x[..., None] - batch_points[i, None, None, :, 0]) ** 2 +
                        (y[..., None] - batch_points[i, None, None, :, 1]) ** 2) ** 0.5
        distance_map = distance_map.reshape(-1, len(batch_points[0]))
        idxs = torch.topk(distance_map, dim=0, k=num_points_interpolation,
                          largest=False, sorted=False)[1]
        fx = torch.take(deformation[0], idxs.reshape(-1)).reshape(-1, len(batch_points[0])).sum(dim=0)
        fx /= num_points_interpolation

        fy = torch.take(deformation[1], idxs.reshape(-1)).reshape(-1, len(batch_points[0])).sum(dim=0)
        fy /= num_points_interpolation

        batch_points[i, :, 0] += fx
        batch_points[i, :, 1] += fy

    batch_points[..., 0] = torch.clip(batch_points[..., 0], 0, w)
    batch_points[..., 1] = torch.clip(batch_points[..., 1], 0, h)
    return batch_points


def calculate_point_metrics(batch_points1: torch.Tensor,
                            batch_points2: torch.Tensor,
                            batch_points_len: torch.Tensor):
    # if isinstance(batch_points1, torch.Tensor):
    #     batch_points1 = batch_points1.detach().cpu().numpy()
    # if isinstance(batch_points2, torch.Tensor):
    #     batch_points2 = batch_points2.detach().cpu().numpy()
    # if isinstance(batch_points_len, torch.Tensor):
    #     batch_points_len = batch_points_len.detach().cpu().numpy()

    error = ((((batch_points1 - batch_points2) ** 2).sum(axis=2)) ** 0.5).sum(axis=1) / batch_points_len.sum(axis=1)
    
    output = {'points_error_l2': torch.mean(error)}

#         inner1 = points1[:points_len[0]]
#         inner2 = points2[:points_len[0]]

#         bound1 = points1[points_len[0]: points_len[1]]
#         bound2 = points2[points_len[0]: points_len[1]]

#         lines1 = points1[points_len[1]:]
#         lines2 = points2[points_len[1]:]
#         len1, len2, len3, len4 = points_len[2:]

#         inner_err = ((((inner1 - inner2) ** 2).sum(axis=1)) ** 0.5).sum(axis=0) / float(inner1.shape[1])

#         bound_err = ((((bound1 - bound2) ** 2).sum(axis=1)) ** 0.5).sum(axis=0) / float(bound1.shape[1])

#         b1 = frechetDist(lines1[:len1], lines2[:len1])
#         b2 = frechetDist(lines1[len1:len1 + len2], lines2[len1:len1 + len2])
#         b3 = frechetDist(lines1[len1 + len2:len1 + len2 + len3],
#                          lines2[len1 + len2:len1 + len2 + len3])
#         b4 = frechetDist(lines1[len1 + len2 + len3:len1 + len2 + len3 + len4],
#                          lines2[len1 + len2 + len3:len1 + len2 + len3 + len4])
#         line_err = (b1 + b2 + b3 + b4) / 4.
        # err['inner'].append(inner_err)
        # err['bound'].append(bound_err)
        # err['lines'].append(line_err)

    # output = {'inner': np.mean(err['inner']),
    #           'bound': np.mean(err['bound']),}
              # 'lines': np.mean(err['lines'])}

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
        batch_fixed, batch_moving = input_batch[:2]
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
        points_fixed, points_moving, points_len = input_batch[2:]
        registered_points = apply_deformation_2points(
            points_moving,
            output_dict['batch_deformation'])
        metrics = calculate_point_metrics(points_fixed, registered_points, points_len)
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

    nan_flag = False
    # Loop over epochs
    global_i = 0
    global_j = 0
    summary_writer = None
    if use_tensorboard:
        os.makedirs(log_dir, exist_ok=True)
        summary_writer = SummaryWriter(log_dir=log_dir)

    best_metric_values = defaultdict(float)

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
            time_batch_start = time()
            batch_losses = train_model(input_batch=batch,
                                       model=model,
                                       optimizer=optimizer,
                                       device=device,
                                       loss=loss,
                                       save_step=save_step,
                                       image_dir=image_dir + '/images/',
                                       epoch=epoch)
            nan_flag = batch_losses['nans']
            batch_losses.pop('nans')
            if nan_flag:
                break

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

        if nan_flag:
            print('found nans in network outputs in the epoch', epoch)
            break

        # Testing
        total = 0
        # time_batches = 0
        for batch in val_loader:
            time_batch_start = time()
            batch_losses, batch_metrics = validate_model(input_batch=batch,
                                                         model=model,
                                                         device=device,
                                                         loss=loss,
                                                         save_step=save_step,
                                                         image_dir=image_dir + '/images/',
                                                         epoch=epoch,
                                                         return_point_metrics=True)

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

        print()
        for key in val_metrics:
            print('Epoch', epoch + 1, f'{key} error: ', val_metrics[key])

        for key in best_metric_values:
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
