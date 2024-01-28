import torch.nn as nn


class Mursi2020Model(nn.Module):
    def __init__(self, cbits, n_stages):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(cbits, int(2 ** n_stages / 2)),
            nn.Tanh(),
            nn.Linear(int(2 ** n_stages / 2), int(2 ** n_stages)),
            nn.Tanh(),
            nn.Linear(int(2 ** n_stages), int(2 ** n_stages / 2)),
            nn.Tanh(),
            nn.Linear(int(2 ** n_stages / 2), 1)
        )

    def forward(self, x):
        return self.main(x).squeeze()


def weights_init_Mursi2020(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.05)
