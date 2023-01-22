import numpy as np
from utils.affine_matrix_conversion import cvt_ThetaToM


def dots_affine_transform(dots, theta, pytorch_view=True, h=256, w=256):
    n_dots, n_coord = dots.shape
    if pytorch_view:
        aff_theta = cvt_ThetaToM(theta, w, h)
    else:
        aff_theta = theta

    dots_33 = np.concatenate((np.array(dots.copy()), np.ones((n_dots, 1))),
                             axis=1).transpose((1, 0))
    out_dots = aff_theta @ dots_33
    out_dots = out_dots.transpose((1, 0))[:, :2]
    return out_dots
