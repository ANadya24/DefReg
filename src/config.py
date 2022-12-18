from typing import Optional, Dict, Any, Tuple, List, Union
from pydantic import BaseModel
import yaml
import torch


class ModelConfig(BaseModel):
    model_name: str
    in_channels: int = 1
    image_size: int = 128
    use_theta: bool = True


class DatasetConfig(BaseModel):
    image_sequences: List[str]
    image_keypoints: List[str]
    im_size: Tuple[int, int, int] = (1, 256, 256)
    train: bool = True
    register_limit: Union[List[int], int] = 5
    use_masks: bool = True
    use_crop: bool = False
    multiply_mask: bool = True


class LossConfig(BaseModel):
    loss_name: str
    weight: float
    input_keys: Dict[str, Any]
    loss_parameters: Dict[str, Any]
    return_loss: bool = False
    detach_values: List[str] = []


class CriterionConfig(BaseModel):
    losses: List[LossConfig] = [
        LossConfig(
            loss_name=torch.nn.MSELoss.__name__,
            weight=1.,
            input_keys={'pred': 'predicted_image',
                        'target': 'fixed_image'},
            loss_parameters={'reduction': 'mean'}),
        LossConfig(
            loss_name=torch.nn.L1Loss.__name__,
            weight=1.,
            input_keys={'pred': 'predicted_image',
                        'target': 'fixed_image'},
            loss_parameters={})
    ]


class SchedulerConfig(BaseModel):
    name: str = torch.optim.lr_scheduler.ReduceLROnPlateau.__name__
    parameters: Dict[str, Any] = {}


class OptimizerConfig(BaseModel):
    name: str = torch.optim.Adam.__name__
    lr: float = 1e-4
    parameters: Dict[str, Any] = {}


class Config(BaseModel):
    expdir: str
    logdir: str
    savedir: str = ""
    batch_size: int = 4
    num_epochs: int = 100
    device: str = "cuda:0"
    load_epoch: int = 0
    model_path: str = ""
    num_workers: int = 4
    validate_by_points: bool = True
    save_step: int = 20
    model_name: str = "model"
    tensorboard: bool = True
    model: ModelConfig
    dataset: DatasetConfig
    val_dataset: DatasetConfig
    criterion: CriterionConfig
    scheduler: Optional[SchedulerConfig] = None
    optimizer: OptimizerConfig


def load_yaml(obj: BaseModel, path: str) -> BaseModel:
    """ Загрузка yaml файла."""
    with open(path) as file:
        info = yaml.load(file, Loader=yaml.Loader)
    return obj.parse_obj(info)


def save_yaml(obj: BaseModel, path: str) -> None:
    """ Сохранение конфига в yaml файл."""
    with open(path, 'w') as outfile:
        yaml.dump(obj.dict(), outfile, default_flow_style=False)


def create_base_config(model_name: str, expdir: str, logdir: str, savedir: str,
                       save_config: bool = True,
                       save_path: str = 'base_config.yaml') -> Config:
    """
    Создание базового конфига.

    :param model_name: название используемой модели
    :param expdir: директория с кодом
    :param logdir: директория, куда сохранять логи
    :param savedir: директория куда сохранять чекпоинты
    :param save_config: если истина, то сохранить конфиг в файл
    :param save_path: путь, куда сохранять конфиг

    :return: конфиг файл
    """
    model = ModelConfig(model_name=model_name)
    dataset = DatasetConfig(image_sequences=[],
                            image_keypoints=[])
    val_dataset = DatasetConfig(image_sequences=[],
                                image_keypoints=[], train=False, shuffle=False)
    criterion = CriterionConfig()
    scheduler = SchedulerConfig()
    optimizer = OptimizerConfig()

    config = Config(expdir=expdir, logdir=logdir, savedir=savedir,
                    device='cuda:0', model=model,
                    dataset=dataset, val_dataset=val_dataset,
                    criterion=criterion,
                    scheduler=scheduler,
                    optimizer=optimizer)
    if save_config:
        save_yaml(config, save_path)
    return config
