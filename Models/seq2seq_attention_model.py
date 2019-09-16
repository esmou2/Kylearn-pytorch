from framework.model import Model
import torch
import torch.nn as nn

class Seq2seq_attention_model():

    def __init__(self, save_path, tsboard_path, units, num_classes, feature_num,
                 batch_size, lr, regression=False, threshold=0.99, patience=10):
        super().__init__(save_path)

        self.batch_size = batch_size
        self.patience = 0
        self.patience_max = patience
        self.encoder_units = 128



    def train(self):
        ckpt_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss}

