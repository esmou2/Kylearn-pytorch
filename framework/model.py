from abc import abstractmethod
from utils.loggings import logger
import torch
from torch.utils.tensorboard import SummaryWriter

class Model():
    def __init__(self, save_path, log_path):
        self.save_path = save_path
        self.log_path = log_path
        self.model = None
        self.classifier = None
        self.train_logger = None
        self.eval_logger = None
        self.summary_writer = None


    @abstractmethod
    def loss(self, **kwargs):
        pass

    @abstractmethod
    def train(self, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        pass

    @abstractmethod
    def checkpoint(self, **kwargs):
        pass


    def set_logger(self, mode='a'):
        self.train_logger = logger('train',self.log_path + '-train', mode=mode)
        self.eval_logger = logger('eval', self.log_path + '-eval', mode=mode)

    def set_summary_writer(self):
        self.summary_writer = SummaryWriter(self.log_path + 'tensorboard')

    def check_cuda(self):
        if torch.cuda.is_available():
            print("INFO: CUDA device exists")
            return torch.cuda.is_available()


    def resume_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        if self.model != None:
            model_state_dict = checkpoint['model_state_dict']
            self.model.load_state_dict(model_state_dict)
            self.model.train()

        classifier_state_dict = checkpoint['classifier_state_dict']
        self.classifier.load_state_dict(classifier_state_dict)
        self.classifier.train()

    def save_model(self, checkpoint, save_path):
        torch.save(checkpoint, save_path)

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        if self.model != None:
            model_state_dict = checkpoint['model_state_dict']
            self.model.load_state_dict(model_state_dict)
            self.model.eval()

        classifier_state_dict = checkpoint['classifier_state_dict']
        self.classifier.load_state_dict(classifier_state_dict)
        self.classifier.eval()

    def count_parameters(self):
        try:
            assert self.model != None
            model_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print('Number of Model Parameters: %d'%model_params)
        except:
            print('No Model specified')

        try:
            assert self.classifier != None
            classifier = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
            print('Number of Model Classifier: %d' % classifier)
        except:
            print('No Classifier specified')


