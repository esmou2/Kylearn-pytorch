from abc import abstractmethod
import torch
class Model():
    def __init__(self, save_path):
        self.save_opath = save_path
        self.model = None
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

    def save_checkpoint(self,ckpt_dict, checkpoint_path, epoch):
        torch.save(ckpt_dict, checkpoint_path)

    def resume_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.train()

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()