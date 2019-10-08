import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


class ReactionDataset(Dataset):
    def __init__(self, f1_path, f2_path, y_path):
        super(ReactionDataset, self).__init__()

        # load data
        self.feature1 = np.load(f1_path)
        self.feature2 = np.load(f2_path)
        self.y = np.load(y_path)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample = (torch.from_numpy(self.feature1[index, :]).float(),
                  torch.from_numpy(self.feature2[index, :]).float(),
                  torch.from_numpy(self.y[index, :]).float())

        return sample

    def get_feature_dim(self):
        return self.feature1.shape[-1], self.feature2.shape[-1]


class ReactionDataloader():
    def __init__(self, file_dir, batch_size, val_size, shuffle=True):
        # feature1 shape [?, n_features]
        # feature2 shape [?, m_features]
        # y shape [?, 1]

        # Load training data
        self.train_set = ReactionDataset(file_dir + 'X_train.npy', file_dir + 'dev_train.npy', file_dir + 'y_train.npy')

        train_size = len(self.train_set)
        indices = list(range(train_size))
        split = int(np.floor(val_size * train_size))
        if shuffle:
            np.random.seed(42)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        self.train_loader = DataLoader(self.train_set, batch_size, sampler=train_sampler, num_workers=4)
        self.val_loader = DataLoader(self.train_set, batch_size, sampler=valid_sampler, num_workers=4)

        # Load test data
        test_set = ReactionDataset(file_dir + 'X_test.npy', file_dir + 'dev_test.npy', file_dir + 'y_test.npy')

        self.test_loader = DataLoader(test_set, batch_size, shuffle, num_workers=4)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def get_feature_dim(self):
        return self.train_set.get_feature_dim()
