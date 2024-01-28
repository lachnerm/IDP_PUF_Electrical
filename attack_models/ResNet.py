import torch.nn as nn


class ResNetModel(nn.Module):
    def __init__(self, cbits, n_stages):
        super().__init__()
        # TODO
        self.main = None

    def forward(self, x):
        return self.main(x).squeeze()


def weights_init_Custom(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform(m.weight.data)
