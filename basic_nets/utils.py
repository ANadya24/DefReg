from torch import nn


def init_weights(m):
    """
    Инициализатор весов сети с помощью Xavier_Uniform initializer.]
    """
    if type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.0)
