import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class PUFDataset(Dataset):
    def __init__(self, cfile, rfile, ids):
        c_array = np.load(cfile)
        self.r_array = np.load(rfile)
        # convert response into 0 and 1 if not yet done
        if np.any(self.r_array == -1):
            self.r_array = self.r_array = (self.r_array + 1) / 2
        # convert challenge into 1 and -1 if not yet done
        if np.any(c_array == 0):
            c_array = 2. * c_array - 1
        c_array = np.fliplr(c_array)
        self.c_array = np.cumprod(c_array, axis=1, dtype=np.float32)

        self.c_array = torch.tensor(self.c_array, dtype=torch.float32)
        self.r_array = torch.tensor(self.r_array, dtype=torch.float32)

        self.c_array = self.c_array[ids]
        self.r_array = self.r_array[ids]

    def __len__(self):
        return len(self.r_array)

    def __getitem__(self, idx):
        return self.c_array[idx], self.r_array[idx]


class PUFDataModule(LightningDataModule):
    def __init__(self, args, cfile, rfile, batch_size, ids):
        super().__init__()
        self.batch_size = batch_size
        self.cfile = cfile
        self.rfile = rfile
        self.args = args
        self.train_kwargs = {"batch_size": self.batch_size, "num_workers": 4,
                             "pin_memory": True, "shuffle": True}
        self.val_test_kwargs = {"batch_size": self.batch_size, "num_workers": 4,
                                "pin_memory": True}
        self.train_ids = ids[0]
        self.val_ids = ids[1]
        self.test_ids = ids[2]

    def setup(self, stage=None):
        self.train_dataset = PUFDataset(self.cfile, self.rfile, self.train_ids)
        self.val_dataset = PUFDataset(self.cfile, self.rfile, self.val_ids)
        self.test_dataset = PUFDataset(self.cfile, self.rfile, self.test_ids)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.train_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.val_test_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.val_test_kwargs)
