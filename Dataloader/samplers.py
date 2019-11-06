import torch
import torch.utils.data as data

class BalanceSampler(data.sampler.Sampler):
    def __init__(self, y_train, indices, num_samples=None, replacement=True):
        labels = y_train[indices]
        labels = torch.tensor(labels)
        if labels.dim() != 1:
            labels = labels.flatten()
        class_count = labels.bincount()

        class_weight = 1.0 / class_count.double()
        self.weights = class_weight[labels]
        self.weights = torch.DoubleTensor(self.weights)
        self.indices = indices
        if num_samples == None:
            self.num_samples = len(indices)
        else: self.num_samples =num_samples
        self.replacement = replacement

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, self.replacement))

    def __len__(self):
        return self.num_samples
