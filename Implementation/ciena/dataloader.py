import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from Dataloader.samplers import BalanceSampler
import numpy as np

class CienaDataset(Dataset):
    def __init__(self, f1_path, y_path):
        super(CienaDataset, self).__init__()

        # load data
        self.feature1 = np.load(f1_path)
        self.y = np.load(y_path)

        if self.y.ndim == 1:
            self.y = self.y.reshape([-1,1])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample = (torch.from_numpy(self.feature1[index, :]).float(),
                  torch.from_numpy(self.y[index, :]).float())

        return sample

    def get_feature_dim(self):
        return self.feature1.shape[-1]


class CienaDataloader():
    def __init__(self, file_dir, batch_size, val_size, shuffle=True):

        # Load training data
        train_set = CienaDataset(file_dir + 'X_train.npy', file_dir + 'y_train.npy')
        self.dims = train_set.get_feature_dim()

        # Load test data
        test_set = CienaDataset(file_dir + 'X_test.npy', file_dir + 'y_test.npy')

        train_size = len(train_set)
        train_indices = list(range(train_size))

        test_size = len(test_set)
        test_indices = list(range(test_size))
        split = int(np.floor(val_size * test_size))

        if shuffle:
            np.random.seed(42)
            np.random.shuffle(test_indices)

        test_indices, eval_indices = test_indices[split:], test_indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        # train_sampler = BalanceSampler(train_set.y, train_indices)
        valid_sampler = SubsetRandomSampler(eval_indices)

        self.train_loader = DataLoader(train_set, batch_size, sampler=train_sampler, num_workers=4)
        self.val_loader = DataLoader(train_set, batch_size, sampler=valid_sampler, num_workers=4)
        self.test_loader = DataLoader(test_set, batch_size, num_workers=4)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def get_feature_dim(self):
        return self.dims
