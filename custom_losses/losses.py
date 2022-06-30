import torch
import numpy as np
import custom_losses.pytorch_ssim as pytorch_ssim
import torch.nn.functional as F


def cross_correlation(I, J, n, use_gpu=False):
    """cross-corr"""
    # I = I.permute(0, 3, 1, 2)
    # J = J.permute(0, 3, 1, 2)
    batch_size, channels, xdim, ydim = I.shape
    I2 = torch.mul(I, I)
    J2 = torch.mul(J, J)
    IJ = torch.mul(I, J)
    sum_filter = torch.ones((1, channels, n, n))
    if use_gpu:
        sum_filter = sum_filter.to(I.device)
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


def cross_correlation_loss(I, J, n, use_gpu=False):
    return -1. * cross_correlation(I, J, n, use_gpu)


def ncc(x, y):
    """ncc"""
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


def ncc_loss(x, y):
    return -1. * ncc(x, y)


def smooothing_loss(y_pred):
    """smooth"""
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
    dx = torch.mul(dx, dx)
    dy = torch.mul(dy, dy)
    d = torch.mean(dx) + torch.mean(dy)
    return d / 2.0


def gradient(x):
    """
    idea from tf.image.image_gradients(image)
    https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
    x: (b,c,h,w), float32 or float64
    dx, dy: (b,c,h,w)
    """

    # gradient step=1
    left = x
    right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    top = x
    bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

    # dx, dy = torch.abs(right - left), torch.abs(bottom - top)
    dx, dy = right - left, bottom - top
    # dx will always have zeros in the last column, right-left
    # dy will always have zeros in the last row,    bottom-top
    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0

    return dx, dy


def deformation_smoothness_loss(flow):
    """
    Computes a deformation smoothness based custom_losses as described here:
    https://link.springer.com/content/pdf/10.1007%2F978-3-642-33418-4_16.pdf
    """

    dx, dy = gradient(flow)

    dx2, dxy = gradient(dx)
    dyx, dy2 = gradient(dy)

    integral = torch.mul(dx2, dx2) + torch.mul(dy2, dy2) + torch.mul(dxy, dxy) + torch.mul(dyx, dyx)
    #    custom_losses = torch.sum(integral, [1,2,3]).mean()
    loss = torch.mean(integral)
    return loss


def vox_morph_loss(y, ytrue, n=9, lamda=0.01):
    cc = cross_correlation(y, ytrue, n)
    sm = smooothing_loss(y)
    loss = -1.0 * cc + lamda * sm
    return loss


def dice_score(pred, target):
    """dice: this definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    top = 2 * torch.sum(pred * target, [1, 2, 3])
    union = torch.sum(pred + target, [1, 2, 3])
    eps = torch.ones_like(union) * 1e-5
    bottom = torch.max(union, eps)
    dice = torch.mean(top / bottom)
    return dice


def dice_loss(x, y):
    return 1. - dice_score(x, y)


def mutual_information(pred, target):
    """mu"""
    return torch.mean(torch.log(target + 1e-8) + torch.log(1 - pred + 1e-8))


def mutual_information_loss(pred, target):
    return 1. - mutual_information(pred, target)


def ssim(pred, target):
    """ssim"""
    ssim_measure = pytorch_ssim.SSIM()
    return ssim_measure(pred, target)


def ssim_loss(pred, target):
    ssim_measure = pytorch_ssim.SSIM()
    ssim_out = 1 - ssim_measure(pred, target)
    return ssim_out


def deformation_l2(pred_def, target_def):
    channel_axis = np.argmin(pred_def.shape[1:]) + 1
    loss = ((pred_def - target_def) ** 2).sum(dim=channel_axis) ** 0.5
    return loss.mean()


def mse(x, y):
    return torch.mean((x - y) ** 2)


# def deformation_incompressibility(deformation):
#     # TODO
    
# def elasticity_loss():
#     # TODO

# def TPS_loss():
#     # TODO


def construct_loss(loss_names, weights=None, n=9, sm_lambda=0.01, use_gpu=False, def_lambda=10, use_masks=True):
    if weights is None:
        weights = [1.] * len(loss_names)

    loss = []

    loss.append(lambda x, y, dx, dy: def_lambda * deformation_l2(dx, dy))
    for l, w in zip(loss_names, weights):
        if l == 'cross-corr':
            if use_masks:
                loss.append(lambda x, y, dx, dy: w * (1.0 - cross_correlation_loss(x[:, :1], y[:, :1], n, use_gpu)) +
                                                 (1.0 - dice_score(x[:, 1:], y[:, 1:])))
            else:
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
            raise NameError(f"No custom_losses function named {l}.")
    return lambda x, y, dx, dy: [sum(lo(x, y, dx, dy) for lo in loss), sum(lo(x, y, dx, dy) for lo in loss[1:]),
                                 loss[0](x, y, dx, dy)]


def total_loss(reg, fixed, deform, gt_deform, diff):
    l_im = cross_correlation_loss(reg[:, :1] * reg[:, 1:], fixed[:, :1] * fixed[:, 1:], n=9,
                                  use_gpu=True)  # reg[:, :1], fixed[:, :1]
    print('Correlation custom_losses', l_im)

    l_im2 = ssim_loss(reg[:, :1] * reg[:, 1:], fixed[:, :1] * fixed[:, 1:])
    print('SSIM custom_losses', l_im2)
    #    l_im2 = torch.sum((reg - fixed)**2, [1,2,3]).mean()
    #    print('MSE custom_losses', l_im2)

    l2dif = ssim_loss(diff[:, :1] * diff[:, 1:], fixed[:, :1] * fixed[:, 1:])
    #    l2dif = torch.sum(diff**2, [1,2,3]).mean()
    print('ssim affine custom_losses', l2dif)

    #    if fixed.shape[1] > 1:
    # print(reg.shape, reg[:,1:].max(), fixed[:,1:].max())
    im_dice = dice_loss(reg[:, :1] * reg[:, 1:], fixed[:, :1] * fixed[:, 1:])
    print('Im dice custom_losses', im_dice)
    if fixed.shape[1] > 1:
        mask_dice = dice_loss(reg[:, 1:], fixed[:, 1:])
        print('Mask dice custom_losses', mask_dice)
    #    else:
    #        mask_dice = l_im*0
    #    l_def = L2Def(deform, gt_deform)
    #    print('Def custom_losses', l_def)
    #    l_smooth = smooothing_loss(reg)
    l_smooth = deformation_smoothness_loss(deform)
    print('Smooth custom_losses', l_smooth)
    print()
    #    return l_im + im_dice + mask_dice + 0.01*l_def + 0.3*l_smooth, l_im, l_def #+ 0.025*l_def, l_im, l_def
    return l_im + 0.8 * l_im2 + im_dice + 200 * l_smooth + 0.8 * l2dif + mask_dice, l_im, l_im2
