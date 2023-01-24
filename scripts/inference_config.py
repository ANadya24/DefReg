from typing import Optional, Dict, Any, Tuple, List, Union
from pydantic import BaseModel
import yaml
import torch
from src.config import save_yaml


class InferenceConfig(BaseModel):
    image_sequences: List[str]
    mask_sequences: Optional[List[str]]
    model_path: str
    save_path: str
    im_size: Tuple[int, int, int] = (1, 256, 256)
    use_masks: bool = True
    use_crop: bool = False
    multiply_mask: bool = True
    normalization: str = 'min_max'
    device: str = 'cpu'
    gauss_sigma: float = -1.


def create_inference_config(model_path: str, save_path: str, image_sequences: List[str],
                            mask_sequences: Optional[List[str]], save_config: bool = True,
                            config_name: str = 'inference_config.yaml') -> InferenceConfig:
    """
    Создание конфига предсказаний.

    :param model_path: путь к чекпоинту
    :param save_path: директория, куда сохранить предсказания
    :param image_sequences: последовательности, которые нужно совместить
    :param mask_sequences: маски к последовательностям, которые нужно совместить
    :param save_config: если истина, то сохранить конфиг в файл


    :return: конфиг файл
    """

    config = InferenceConfig(image_sequences=image_sequences,
                             mask_sequences=mask_sequences,
                             model_path=model_path,
                             save_path=save_path)
    if save_config:
        save_yaml(config, config_name)
    return config
