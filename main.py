import torch
from torch.utils import data
from torch import optim, nn
from glob import glob
import time
from pickles_dataset import Dataset
from model import DefNet
from voxelmorph import cvpr2018_net
from sklearn.model_selection import train_test_split
from train import train
from losses import *
import os
import argparse
import json
import shutil

use_gpu = torch.cuda.is_available()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_file, 'r') as f:
        config = json.load(f)

    data_path = config["data_path"]
    data_path_val = config["data_path_val"]
    batch_size = config["batch_size"]
    max_epochs = config["num_epochs"]
    load_epoch = config["load_epoch"]
    model_path = config["model_path"]
    lr = config["learning_rate"]
    sm_lambda = config["smooth"]
    use_mask = config["masks"]
    data_shape = config["data_shape"]
    num_workers = config["num_workers"]
    save_dir = config["save_dir"]
    logdir = config["log_dir"]
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        shutil.copy(args.config_file, save_dir + args.config_file.split('/')[-1])
    save_step = config["save_step"]
    image_dir = config["image_dir"]
    model_name = config["model_name"]
    #data_dir = config["data_dir"]
    use_tensorboard = config["tensorboard"]
    use_gpu = config["use_gpu"]
    size_path = config["size_path"]

    # vm = VoxelMorph(
    #     data_shape, is_2d=True, use_gpu=use_gpu, load_epoch=load_epoch, model_path=model_path)  # Object of the higher level class
    # mode = "vm2"
    # nf_enc = [16, 32, 32, 32]
    # if mode == "vm1":
    #     nf_dec = [32, 32, 32, 32, 8, 8]
    # elif mode == "vm2":
    #     nf_dec = [32, 32, 32, 32, 32, 16, 16]
    # else:
    #     raise ValueError("Not yet implemented!")

    vm = DefNet(data_shape[1:])#cvpr2018_net(data_shape[1:], nf_enc, nf_dec)

    if load_epoch:
        checkpoint = torch.load(model_path)
        vm.load_state_dict(checkpoint['model_state_dict'])
        # vm = vm.load_state_dict(model_path)
        print(f"Successfuly loaded state_dict from {model_path}")

    if use_gpu:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            vm = nn.DataParallel(vm)
        vm.to('cuda')

    # 'worker_init_fn': np.random.seed(42)
    # }

    # filename = list(set([x.split('_')[0]
    #                      for x in glob(data_dir + 'imgs/*.jpg')]))
    # filename = glob(data_dir + 'pairs/fwd/*.npy')
    # # print(filename)
    #
    # train_data, test_data = train_test_split(
    #     filename, test_size=0.05, random_state=42)
    # partition = {'train': train_data, 'validation': test_data}

    # Generators
    # training_set = Dataset(partition['train'], data_shape[1:],
    #                        use_mask=use_mask, size_file=data_dir+'sizes.txt')
    training_set = Dataset(data_path, data_shape, size_file= size_path,
                        smooth=False, train=True, shuffle=True)
    print("Length of train set", len(training_set))
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': num_workers,
              'pin_memory': True}
    training_generator = data.DataLoader(training_set, **params)

    params = {'batch_size': batch_size,
              'shuffle': False,
              'num_workers': num_workers,
              'pin_memory': True}
    validation_set = Dataset(data_path_val, data_shape, size_file= size_path,
                        smooth=True, train=False, shuffle=False)
    validation_generator = data.DataLoader(validation_set, **params)
    print("Length of validation set", len(validation_set))

    # optimizer = optim.Adam(vm.voxelmorph.parameters(), lr=lr) # , momentum=0.99)
    optimizer = optim.Adam(vm.parameters(), lr=lr)  # , momentum=0.99)
    if load_epoch:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    device = torch.device("cuda" if use_gpu else "cpu")

    total_loss = construct_loss(["ssim"], weights=[1.], sm_lambda=0.,
                                use_gpu=use_gpu, n=9, def_lambda=0.0)
    # if load_epoch:
    #     total_loss = checkpoint['loss']

    train(load_epoch, max_epochs, training_generator, validation_generator, vm, optimizer,
          device, total_loss, save_dir, model_name, image_dir, save_step, use_gpu, use_tensorboard=use_tensorboard,
          logdir=logdir)
