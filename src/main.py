from typing import cast
import torch
from torch.utils import data
from src.dataset import Dataset

from train import train
from train_denoise import train as train_denoise
import os
import argparse
import shutil

from src.config import Config, load_yaml
import models
from custom_losses.loss_classes import CustomCriterion

use_gpu = torch.cuda.is_available()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    config = cast(Config, load_yaml(Config, args.config_file))

    if not os.path.exists(config.savedir):
        os.makedirs(config.savedir, exist_ok=True)
    shutil.copy(args.config_file, config.savedir + '/' + args.config_file.split('/')[-1])
    if os.path.exists(config.savedir + '/' + config.expdir.split('/')[-1]):
        shutil.rmtree(config.savedir + '/' + config.expdir.split('/')[-1])
    shutil.copytree(config.expdir, config.savedir + '/' + config.expdir.split('/')[-1], dirs_exist_ok=True)

    # загружаем модель
    model_name = config.model.model_name

    model_args = config.model.dict()
    model_args.pop('model_name')
    if config.model.model_name == 'DenoiseRegNet':
        model = getattr(models, model_name)(device=config.device, in_channels=model_args['in_channels'])
    else:
        model = getattr(models, model_name)(device=config.device, **model_args)

    checkpoint = None
    if config.load_epoch:
        checkpoint = torch.load(config.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfuly loaded state_dict from {config.model_path}")

    model = model.to(config.device)

    # if use_gpu:
    #     if torch.cuda.device_count() > 1:
    #         print("Let's use", torch.cuda.device_count(), "GPUs!")
    #         vm = nn.DataParallel(vm)
    #     vm.to('cuda')

    # Датасет
    train_dataset = Dataset(image_sequences=config.dataset.image_sequences,
                            image_keypoints=config.dataset.image_keypoints,
                            im_size=config.dataset.im_size,
                            train=config.dataset.train,
                            register_limit=config.dataset.register_limit,
                            use_crop=config.dataset.use_crop,
                            use_masks=config.dataset.use_masks,
                            multiply_mask=config.dataset.multiply_mask,
                            return_points=config.validate_by_points, gauss_sigma=config.dataset.gauss_sigma)

    val_dataset = Dataset(image_sequences=config.val_dataset.image_sequences,
                          image_keypoints=config.val_dataset.image_keypoints,
                          im_size=config.val_dataset.im_size,
                          train=config.val_dataset.train,
                          register_limit=config.val_dataset.register_limit,
                          use_crop=config.val_dataset.use_crop,
                          use_masks=config.val_dataset.use_masks,
                          multiply_mask=config.val_dataset.multiply_mask,
                          return_points=config.validate_by_points, gauss_sigma=config.dataset.gauss_sigma)

    print("Length of train set:", len(train_dataset))
    print("Length of validation set:", len(val_dataset))

    # Data loader
    params = {'batch_size': config.batch_size,
              'num_workers': config.num_workers,
              'shuffle': False,
              'pin_memory': True,
              'drop_last': True}
    training_generator = data.DataLoader(train_dataset, **params)
    validation_generator = data.DataLoader(val_dataset, **params)
    optimizer = getattr(torch.optim, config.optimizer.name)(
        model.parameters(), **config.optimizer.parameters, lr=config.optimizer.lr)

    if config.load_epoch:
        assert checkpoint is not None
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if config.scheduler is not None:
        scheduler = getattr(torch.optim.lr_scheduler,
                            config.scheduler.name)(optimizer=optimizer, **config.scheduler.parameters)
    else:
        scheduler = None

    loss = CustomCriterion(config.criterion)

    if config.model.model_name == 'DenoiseRegNet':
        train_denoise(model=model, train_loader=training_generator,
                      val_loader=validation_generator,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      loss=loss,
                      device=config.device,
                      model_name=config.model_name, save_step=config.save_step,
                      save_dir=config.savedir, image_dir=config.logdir,
                      log_dir=config.logdir,
                      load_epoch=config.load_epoch,
                      max_epochs=config.num_epochs,
                      use_tensorboard=config.tensorboard,
                      validate_by_points=config.validate_by_points)
    else:
        train(model=model, train_loader=training_generator,
              val_loader=validation_generator,
              optimizer=optimizer,
              scheduler=scheduler,
              loss=loss,
              device=config.device,
              model_name=config.model_name, save_step=config.save_step,
              save_dir=config.savedir, image_dir=config.logdir,
              log_dir=config.logdir,
              load_epoch=config.load_epoch,
              max_epochs=config.num_epochs,
              use_tensorboard=config.tensorboard,
              validate_by_points=config.validate_by_points)
