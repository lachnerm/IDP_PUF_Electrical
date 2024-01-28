import torch.nn as nn


class Aseeri2018Model(nn.Module):
    def __init__(self, cbits, n_stages):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(cbits, int(2 ** n_stages)),
            nn.ReLU(),
            nn.Linear(int(2 ** n_stages), int(2 ** n_stages)),
            nn.ReLU(),
            nn.Linear(int(2 ** n_stages), int(2 ** n_stages)),
            nn.ReLU(),
            nn.Linear(int(2 ** n_stages), 1)
        )

    def forward(self, x):
        return self.main(x).squeeze()


def weights_init_Aseeri2018(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight.data)
