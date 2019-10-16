from abc import abstractmethod
from utils.loggings import logger
import torch
from collections import OrderedDict
class Model():
    def __init__(self, save_path, log_path):
        self.save_path = save_path
        self.log_path = log_path
        self.model = None


    def set_logger(self):
        self.train_logger = logger(self.log_path + 'train')

    def check_cuda(self):
        if torch.cuda.is_available():
            print("INFO: CUDA device exists")
            return torch.cuda.is_available()

    @abstractmethod
    def loss(self, **kwargs):
        pass

    @abstractmethod
    def train(self, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        pass


    def resume_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['model_state_dict']
        self.model.load_state_dict(state_dict)
        self.model.train()

    def save_model(self, checkpoint, save_path):
        torch.save(checkpoint, save_path)

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['model_state_dict']
        self.model.load_state_dict(state_dict)
        self.model.eval()