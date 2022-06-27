from typing import Tuple
import numpy as np
import random
from skimage import exposure


def pad_image(img: np.ndarray, padwidth: Tuple[int, ...]) -> np.ndarray:
    """
    Паддинг картинки с заданием параметров паддинга в порядке: сверху-снизу-слева-справа
    """
    h0, h1, w0, w1 = padwidth[:4]
    if len(padwidth) < 5:
        img = np.pad(img, ((h0, h1), (w0, w1)))
    else:
        img = np.pad(img, ((h0, h1), (w0, w1), (0, 0)))

    return img


def normalize_min_max(image: np.ndarray) -> np.ndarray:
    """Нормализация значений изображения в интервал [0, 1]."""
    image = (image - image.min()) / (image.max() - image.min())
    return image


def normalize_mean_std(image: np.ndarray) -> np.ndarray:
    """Нормализация значений изображения путем стандартизации (вычитаем среднее и делим на стд отклонение)."""
    image = (image - image.mean()) 
    image /= image.std()
    return image


def match_histograms(source_image: np.ndarray, reference_image: np.ndarray,
                     multichannel: bool = False, random_switch: bool = True) -> \
        Tuple[np.ndarray, np.ndarray, bool]:
    """ Процедура histogram matching для соответствия интесивностей пары изображений."""
    source = source_image
    reference = reference_image
    changed_position = False
    if random_switch and np.random.rand() < 0.5:
        source = reference_image
        reference = source_image
        changed_position = True
    matched = exposure.match_histograms(source, reference, multichannel=multichannel)
    return matched, reference, changed_position
