import torch
import torch.nn.functional as F
import numpy as np
import time
from val_images import validate_images
from losses import dice_score
import os
from torch.utils.tensorboard import SummaryWriter
import signal


def train_model(model, optimizer, device, loss_func, save_step, image_dir,
                batch_moving, batch_fixed, batch_deformation, return_metric_score=True, epoch=0):
    # model.voxelmorph.train()
    model.train()
    optimizer.zero_grad()

    batch_fixed, batch_moving = batch_fixed.to(
        device), batch_moving.to(device)
    batch_deformation = batch_deformation.to(device)
    # print(batch_moving.shape, batch_fixed.shape, batch_deformation.shape)
    registered_image, deformation_matrix = model(batch_moving, batch_fixed)
    # zero_image = self.voxelmorph(batch_fixed, batch_fixed)
    # print(registered_image.max(), registered_image.min())

    train_loss, corr_loss, def_loss = loss_func(
        registered_image, batch_fixed, deformation_matrix, batch_deformation)
    # print(train_loss.shape)

    train_loss.backward()

    optimizer.step()

    if (epoch + 1) % save_step == 0:
        validate_images(batch_fixed[1:3], batch_moving[1:3], registered_image[1:3],
                        image_dir, epoch=epoch+1, train=True)

    # if return_metric_score:
    #     train_dice_score = dice_score(
    #         registered_image, batch_fixed)
    #     return train_loss, train_dice_score

    return train_loss, corr_loss, def_loss


def get_test_loss(model, device, loss_func, save_step, image_dir, batch_moving, batch_fixed, batch_deformation, epoch=0):
    model.eval()

    with torch.no_grad():

        batch_fixed, batch_moving = batch_fixed.to(
            device), batch_moving.to(device)
        batch_deformation = batch_deformation.to(device)

        registered_image, deformation_matrix = model(batch_moving, batch_fixed)

        val_loss, corr_loss, def_loss = loss_func(registered_image, batch_fixed, deformation_matrix, batch_deformation)

        if (epoch + 1) % save_step == 0:
            validate_images(batch_fixed[1:3], batch_moving[1:3],
                            registered_image[1:3], image_dir, epoch=epoch + 1, train=False)

        # val_dice_score = dice_score(registered_image, batch_fixed)

        return val_loss, corr_loss, def_loss


def train(load_epoch, max_epochs, train_loader, val_loader, vm, optimizer,
          device, loss_func, save_dir, model_name, image_dir, save_step, use_gpu,
          use_tensorboard=False, logdir="./logs/"):

    def save_model(dev_count, name=model_name + f'_stop'):
        if dev_count > 1:
            #torch.save(vm.module.state_dict(), save_dir + model_name + f'_stop_{epoch + 1}')
            torch.save({
                'model_state_dict': vm.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                save_dir + name + f'_{epoch + 1}')
                # 'loss': loss_func}, save_dir + model_name + f'_stop_{epoch + 1}')
        else:
            # torch.save(vm.state_dict(), save_dir + model_name + f'_stop_{epoch + 1}')
            torch.save({
                'model_state_dict': vm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                save_dir + name + f'{epoch + 1}')
                # 'loss': loss_func}, save_dir + model_name + f'_stop_{epoch + 1}')
        print(f"Successfuly saved state_dict in {save_dir + model_name + f'_stop_{epoch + 1}'}")


    def sig_handler(signum, frame):
        print('Saved intermediate result!')
        torch.cuda.synchronize()
        # vm.save_state_dict(save_dir, model_name + f'_stop_{epoch + 1}')
        save_model(torch.cuda.device_count())


    signal.signal(signal.SIGINT, sig_handler)
    # Loop over epochs
    print("in train")
    if use_tensorboard:
        os.makedirs(logdir, exist_ok=True)
        summary_writer = SummaryWriter(log_dir=logdir)
        global_i = 0
        global_j = 0
    for epoch in range(load_epoch, max_epochs):
        start_time = time.time()
        train_loss = 0
        total = 0
        train_def_score = 0
        val_loss = 0
        val_def_score = 0
        for batch_fixed, batch_moving, batch_deform in train_loader:
            # print(batch_fixed.shape)
            loss, corr_loss, def_loss = train_model(vm, optimizer, device, loss_func, save_step, image_dir,
                                     batch_moving, batch_fixed, batch_deform, epoch=epoch)
            # train_dice_score += dice.item()
            train_def_score += def_loss.item()
            train_loss += loss.item()
            total += 1
            if use_tensorboard:
                summary_writer.add_scalar('loss', loss.item(), global_i)
                summary_writer.add_scalar('corr_loss', corr_loss.item(), global_i)
                summary_writer.add_scalar('def_loss', def_loss.item(), global_i)
                # summary_writer.add_scalar('dice', dice.item(), global_i)
                global_i += 1

        # print('[', "{0:.2f}".format((time.time() - start_time) / 60), 'mins]', 'After', epoch + 1,
        #       'epochs, the Average training loss is ', train_loss.cpu().numpy() * params['batch_size']
        #       / len(training_set), 'and average DICE score is', train_dice_score.data.cpu().numpy()
        #       * params['batch_size'] / len(training_set))
        train_loss /= total
        train_def_score /= total
        # Testing time
        total = 0
        start_time = time.time()
        for batch_fixed, batch_moving, batch_deform in val_loader:
            # Transfer to GPU
            loss, corr_loss, def_loss = get_test_loss(vm, device, loss_func, save_step, image_dir,
                                       batch_moving, batch_fixed, batch_deform, epoch=epoch)
            val_def_score += def_loss.item()
            val_loss += loss.item()
            total += 1
            if use_tensorboard:
                summary_writer.add_scalar('val_loss', loss.item(), global_j)
                # summary_writer.add_scalar('val_dice', dice.item(), global_j)
                summary_writer.add_scalar('val_corr_loss', corr_loss.item(), global_i)
                summary_writer.add_scalar('val_def_loss', def_loss.item(), global_i)
                global_j += 1

        val_loss /= total
        val_def_score /= total
        # print('[', "{0:.2f}".format((time.time() - start_time) / 60), 'mins]', 'After', epoch + 1,
        #         #       'epochs, the Average validations loss is ', val_loss.cpu().numpy() * params['batch_size']
        #         #       / len(validation_set), 'and average DICE score is', val_dice_score.data.cpu().numpy()
        #         #       * params['batch_len(validation_set))

        # print('Epoch', epoch + 1, 'train_loss/test_loss: ', train_loss, '/', val_loss, ' dice_train/dice_test: '
              # , train_dice_score, '/', val_dice_score)
        print('Epoch', epoch + 1, 'train_loss/test_loss: ', train_loss, '/', val_loss)
        print('Epoch', epoch + 1, 'def_train/def_test: ', train_def_score, '/', val_def_score)

        if (epoch+1) % save_step == 0:
            save_model(torch.cuda.device_count(), model_name)
