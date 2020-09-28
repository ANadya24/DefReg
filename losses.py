import torch
import numpy as np
import pytorch_ssim

'''cross-corr'''


def cross_correlation_loss(I, J, n, use_gpu=False):
    # I = I.permute(0, 3, 1, 2)
    # J = J.permute(0, 3, 1, 2)
    batch_size, channels, xdim, ydim = I.shape
    I2 = torch.mul(I, I)
    J2 = torch.mul(J, J)
    IJ = torch.mul(I, J)
    sum_filter = torch.ones((1, channels, n, n))
    if use_gpu:
        sum_filter = sum_filter.cuda()
    I_sum = torch.conv2d(I, sum_filter, padding=1, stride=(1, 1))
    J_sum = torch.conv2d(J, sum_filter, padding=1, stride=(1, 1))
    I2_sum = torch.conv2d(I2, sum_filter, padding=1, stride=(1, 1))
    J2_sum = torch.conv2d(J2, sum_filter, padding=1, stride=(1, 1))
    IJ_sum = torch.conv2d(IJ, sum_filter, padding=1, stride=(1, 1))
    win_size = n ** 2
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    cc = cross * cross / (I_var * J_var + np.finfo(float).eps)
    return torch.mean(cc)


'''ncc'''


def ncc(x, y):
    mean_x = torch.mean(x, [1, 2, 3], keepdim=True)
    mean_y = torch.mean(y, [1, 2, 3], keepdim=True)
    mean_x2 = torch.mean(torch.pow(x, 2), [1, 2, 3], keepdim=True)
    mean_y2 = torch.mean(torch.pow(y, 2), [1, 2, 3], keepdim=True)
    stddev_x = torch.sum(torch.sqrt(
        mean_x2 - torch.pow(mean_x, 2)), [1, 2, 3], keepdim=True)
    stddev_y = torch.sum(torch.sqrt(
        mean_y2 - torch.pow(mean_y, 2)), [1, 2, 3], keepdim=True)
    val = torch.mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))
    return val


'''smooth'''


def smooothing_loss(y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
    dx = torch.mul(dx, dx)
    dy = torch.mul(dy, dy)
    d = torch.mean(dx) + torch.mean(dy)
    return d / 2.0


def vox_morph_loss(y, ytrue, n=9, lamda=0.01):
    cc = cross_correlation_loss(y, ytrue, n)
    # cc2 = ncc(y, ytrue)
    sm = smooothing_loss(y)
    # print(cc.item(), cc2.item())
    # print("CC Loss", cc.item(), "Gradient Loss", sm.item())
    loss = -1.0 * cc + lamda * sm
    return loss


'''dice'''


def dice_score(pred, target):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    top = 2 * torch.sum(pred * target, [1, 2, 3])
    union = torch.sum(pred + target, [1, 2, 3])
    eps = torch.ones_like(union) * 1e-5
    bottom = torch.max(union, eps)
    dice = torch.mean(top / bottom)
    # print("Dice score", dice)
    return dice


'''mu'''


def mutual_information(pred, target):
    return torch.mean(torch.log(target + 1e-8) + torch.log(1 - pred + 1e-8))


'''ssim'''


def ssim(pred, target):
    ssim_measure = pytorch_ssim.SSIM()
    ssim_out = 1 - ssim_measure(pred, target)
    return ssim_out


def L2Def(pred, target):
    loss = ((pred - target) ** 2).sum(dim=3) ** 0.5
    return loss.mean()


def mse(x, y):
    return torch.mean( (x - y) ** 2 )


def construct_loss(loss_names, weights=None, n=9, sm_lambda=0.01, use_gpu=False, def_lambda=10):
    if weights is None:
        weights = [1.] * len(loss_names)

    loss = []

    loss.append(lambda x, y, dx, dy: def_lambda * L2Def(dx, dy))
    for l, w in zip(loss_names, weights):
        if l == 'cross-corr':
            loss.append(lambda x, y, dx, dy: w * (1.0 - cross_correlation_loss(x, y, n, use_gpu)))
        elif l == 'ncc':
            loss.append(lambda x, y, dx, dy: w * (1.0 - ncc(x, y)))
        elif l == 'dice':
            loss.append(lambda x, y, dx, dy: w * (1.0 - dice_score(x, y)))
        elif l == 'mu':
            loss.append(lambda x, y, dx, dy: w * (1.0 - mutual_information(x, y)))
        elif l == 'ssim':
            loss.append(lambda x, y, dx, dy: w * ssim(x, y))
        elif l == 'mse':
            loss.append(lambda x, y, dx, dy: w * mse(x, y))
        elif l == 'smooth':
            loss.append(lambda x, y, dx, dy: sm_lambda * smooothing_loss(x))
        else:
            raise NameError(f"No loss function named {l}.")
    return lambda x, y, dx, dy: sum(lo(x, y, dx, dy) for lo in loss)

from sklearn.metrics import mutual_info_score
# x = torch.randn((12, 1, 256, 256), requires_grad=True)
# y = torch.randn((12, 1, 256, 256), requires_grad=True)
#
# out = ssim(x, y)
# print(out.requires_grad)
#
# print(out)
# out.backward()