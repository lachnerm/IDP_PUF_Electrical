import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self, cbits, n_stages):
        super().__init__()
        ns = 2 ** n_stages
        self.main = nn.Sequential(
            nn.Linear(cbits, ns),
            nn.BatchNorm1d(ns),
            nn.LeakyReLU(),

            nn.Linear(ns, ns),
            nn.BatchNorm1d(ns),
            nn.LeakyReLU(),

            nn.Linear(ns, ns // 2),
            nn.BatchNorm1d(ns // 2),
            nn.LeakyReLU(),

            nn.Linear(ns // 2, ns // 4),
            nn.BatchNorm1d(ns // 4),
            nn.LeakyReLU(),

            nn.Linear(ns // 4, 1)
        )

    def forward(self, x):
        return self.main(x).squeeze()


def weights_init_Custom(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform(m.weight.data)
