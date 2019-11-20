import numpy as np
import pandas as pd
from Dataloader.transformer import *
from torch.utils.data.sampler import SubsetRandomSampler
from Dataloader.samplers import BalanceSampler
from sklearn.preprocessing import LabelEncoder

def facility_to_index(facility_name, facilities):
    label_encoder = LabelEncoder().fit(facility_name)
    indexes = label_encoder.transform(facilities)
    return indexes

class CienaPortDataset(Dataset):
    def __init__(self, file_path, max_length=None, facility_names=None):
        super().__init__()
        # Load training data
        self.targets = np.load(file_path + '4transformer_label.npy', allow_pickle=True) # [batch, 1]
        self.feature_sequence = np.load(file_path + '4transformer_PMs.npy', allow_pickle=True).astype(np.float32)
        self.padding = np.load(file_path + '4transformer_padding.npy', allow_pickle=True).astype(np.float32)  # [batch, t*max_length, 1]
        self.time_facility = np.load(file_path + '4transformer_Time_Facility.npy', allow_pickle=True)  # [batch, t*max_length, 2]

        # Fill features nan with 0
        where_nan = np.isnan(self.feature_sequence)
        self.feature_sequence[where_nan] = 0

        # Fill nan with '0' to indicate it's a padding unit
        facilities = pd.Series(self.time_facility[:, :, 1].flatten()).fillna('0')
        if facility_names == None:
            facility_names = facilities.value_counts().index.values
        else:
            facility_names = facility_names.append('0')
        facility_index = facility_to_index(facility_names, facilities)
        facility_index = facility_index.reshape(-1, max_length)
        self.time_facility[:, :, 1] = facility_index
        self.time_facility = self.time_facility.astype(np.long)



        self.max_length = max_length

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample = (torch.from_numpy(self.feature_sequence[index, :]),
                  torch.from_numpy(self.padding[index, :]),
                  torch.from_numpy(self.time_facility[index, :]),
                  torch.from_numpy(self.targets[index, :]))

        return sample

class CienaPortDataloader():
    def __init__(self, train_path, test_path, batch_size, eval_portion, max_length=207, shuffle=True, facility_names=None):
        train_set = CienaPortDataset(train_path, max_length=max_length, facility_names=facility_names)
        test_set = CienaPortDataset(test_path, max_length=max_length, facility_names=facility_names)

        train_size = len(train_set)
        train_indices = list(range(train_size))

        test_size = len(test_set)
        test_indices = list(range(test_size))
        split = int(np.floor(eval_portion * test_size))

        if shuffle:
            np.random.seed(42)
            np.random.shuffle(test_indices)

        test_indices, eval_indices = test_indices[split:], test_indices[:split]

        # train_sampler = SubsetRandomSampler(train_indices)
        train_sampler = BalanceSampler(train_set.targets, train_indices)
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

