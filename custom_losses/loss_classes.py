from torch import nn
import custom_losses
from custom_losses.losses import (
    ncc_loss,
    cross_correlation_loss,
    ssim_loss,
    dice_loss,
    deformation_smoothness_loss
)
from src.config import CriterionConfig
import torch.nn


class CrossCorrelationLoss(nn.Module):
    def __init__(self, n=9, use_gpu=False):
        super().__init__()
        self.n = n
        self.use_gpu = use_gpu

    def forward(self, pred, target):
        if pred.shape[1] == 2:
            return cross_correlation_loss(pred[:, :1] * pred[:, 1:],
                                          target[:, :1] * target[:, 1:],
                                          n=self.n, use_gpu=self.use_gpu)
        return cross_correlation_loss(pred, target,
                                      n=self.n, use_gpu=self.use_gpu)
    

class NCCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        if pred.shape[1] == 2:
            return ncc_loss(pred[:, :1] * pred[:, 1:],
                            target[:, :1] * target[:, 1:])
        return ncc_loss(pred, target)


class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        if pred.shape[1] == 2:
            return ssim_loss(pred[:, :1] * pred[:, 1:],
                             target[:, :1] * target[:, 1:])
        return ssim_loss(pred, target)


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        if pred.shape[1] == 2:
            return dice_loss(pred[:, :1] * pred[:, 1:],
                             target[:, :1] * target[:, 1:])
        return dice_loss(pred, target)


class DeformationSmooth(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred):
        return deformation_smoothness_loss(pred)


class CustomCriterion(nn.Module):
    def __init__(self, criterion_config: CriterionConfig):
        super().__init__()
        self.losses = []
        self.input_keys = dict()

        for loss in criterion_config.losses:
            try:
                self.losses.append((getattr(custom_losses, loss.loss_name)(**loss.loss_parameters),
                                    loss.weight, loss.input_keys, loss.loss_name, loss.return_loss))
            except:
                self.losses.append((getattr(torch.nn, loss.loss_name)(**loss.loss_parameters),
                                loss.weight, loss.input_keys, loss.loss_name, loss.return_loss))
 

    def forward(self, input_dict):
        losses2return = {}
        loss = 0
        for loss_function, loss_weight, loss_input, loss_name, return_flag in self.losses:
            input_values = {}
            for key in loss_input:
                input_values[key] = input_dict[loss_input[key]]
            cur_loss = loss_function(**input_values)
            if return_flag:
                losses2return[loss_name] = cur_loss

            loss += loss_weight * cur_loss
        losses2return['total_loss'] = loss
        return losses2return
