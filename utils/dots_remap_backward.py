import cv2
import numpy as np


def get_subpixel(img: np.ndarray, x: float, y: float):
    """Get interpolated pixel value at (@x, @y) with float precision"""
    # print('get_subpixel', x, y, img.dtype)
    patch = cv2.getRectSubPix(img, (1, 1), (x, y), np.zeros((1, 1)), cv2.CV_32F)
    if patch is not None:
        return patch[0][0]
    return None


def algo_ff_remap_centroids_backward(dots, deformation):
    h, w = deformation.shape[:2]
    x = dots[:, 0]
    y = dots[:, 1]

    valid_points = (x > 0) * (y > 0) * \
                   (x < w) * (y < h) * ~np.isnan(x) * ~np.isnan(y)

    out_dots = dots.copy()

    if valid_points.sum() != 0:
        fx = np.array([get_subpixel(deformation[..., 0], xx, yy) for xx, yy in
                       zip(x[valid_points], y[valid_points])])
        fy = np.array([get_subpixel(deformation[..., 1], xx, yy) for xx, yy in
                       zip(x[valid_points], y[valid_points])])
        out_dots[valid_points, 0] = x[valid_points] + fx
        out_dots[valid_points, 1] = y[valid_points] + fy
    return out_dots


def remap_centroids_all_backward(points, deformations):
    points_reg1 = np.empty((0, 2))
    points_reg = points.copy()
    points_len = len(points)
    if points_len > 0:
        for j in range(points_len - 1, -1, -1):
            cur_def = deformations[j]
            points_reg1 = np.concatenate([points_reg1, points[j]], axis=0)
            points_reg1 = algo_ff_remap_centroids_backward(points_reg1, cur_def)

    for j in range(points_len - 1, -1, -1):
        points_reg[j] = points_reg1[:len(points[j])]
        points_reg1 = points_reg1[len(points[j]):]

    return points_reg
