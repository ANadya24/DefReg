from typing import cast
import torch
from torch.utils import data
from src.dataset import Dataset

from train import train
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
        shutil.copy(args.config_file, config.savedir + args.config_file.split('/')[-1])
        shutil.copytree(args.exp_dir, config.savedir + args.exp_dir.split('/')[-1])

    # загружаем модель
    model_name = config.model.model_name

    model_args = config.model.dict()
    model_args.pop('model_name')
    model = getattr(models, model_name)(**model_args)

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
                            use_masks=config.dataset.use_masks)

    val_dataset = Dataset(image_sequences=config.val_dataset.image_sequences,
                          image_keypoints=config.val_dataset.image_keypoints,
                          im_size=config.val_dataset.im_size,
                          train=config.val_dataset.train,
                          register_limit=config.val_dataset.register_limit,
                          use_crop=config.val_dataset.use_crop,
                          use_masks=config.val_dataset.use_masks)

    print("Length of train set:", len(train_dataset))
    print("Length of validation set:", len(val_dataset))

    # Data loader
    params = {'batch_size': config.batch_size,
              'num_workers': config.num_workers,
              'pin_memory': True}
    training_generator = data.DataLoader(train_dataset, **params)
    validation_generator = data.DataLoader(val_dataset, **params)

    optimizer = getattr(torch.optim, config.optimizer.name)(
        model.parameters(), **config.optimizer.parameters, lr=config.optimizer.lr)

    if config.load_epoch:
        assert checkpoint is not None
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if config.scheduler is not None:
        scheduler = getattr(torch.optim.lr_scheduler,
                            config.scheduler.name)(**config.scheduler.parameters)

    loss = CustomCriterion(config.criterion)

    train

    # train(load_epoch, max_epochs, training_generator, validation_generator, vm, optimizer,
    #       device, total_loss, save_dir, model_name, image_dir, save_step, use_gpu,
    #       use_tensorboard=use_tensorboard,
    #       logdir=config.logdir)