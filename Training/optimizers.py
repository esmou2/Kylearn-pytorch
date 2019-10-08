import numpy as np
from torch.optim.adam import Adam


class AdamOptimizer(Adam):
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, parameters, n_warmup_steps, lr_max, betas=(0.9, 0.999)):
        '''lr of the first epoch: 1/exp(warmup_steps, 3/2)'''
        super(AdamOptimizer, self).__init__(filter(lambda x: x.requires_grad, parameters), lr=lr_max, betas=betas)
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        # self.init_lr = np.power(d_model, -0.5)
        self.init_lr = lr_max * np.power(n_warmup_steps, 0.5)

    def step(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        super().step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        super().zero_grad()

    def _get_lr_scale(self):
        '''
            if step > warmup_steps, return the former one
            if step < warmup_steps, return the later one
            lr: up and down
        '''
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        return super().state_dict()
