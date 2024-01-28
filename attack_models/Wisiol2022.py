import torch.nn as nn


class Wisiol2022Model(nn.Module):
    def __init__(self, cbits):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(cbits, cbits / 2),
            nn.Tanh(),
            nn.Linear(cbits / 2, cbits / 2),
            nn.Tanh(),
            nn.Linear(cbits / 2, cbits),
            nn.Tanh(),
            nn.Linear(cbits, 1)
        )

    def forward(self, x):
        return self.main(x).squeeze()


def weights_init_Wisiol2022(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.05)
