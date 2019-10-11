from abc import abstractmethod
from utils.loggings import logger
import torch
class Model():
    def __init__(self, save_path, log_path):
        self.save_path = save_path
        self.model = None
        self.train_logger = logger(log_path+'train')
        self.val_logger = logger(log_path+'test')

    def check_cuda(self):
        if torch.cuda.is_available():
            print("INFO: CUDA device exists")

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
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.train()

    def save_model(self, checkpoint, save_path):
        torch.save(checkpoint, save_path)

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()