import numpy as np

class LR_Scheduler(object):
    """
        We implement this learning rate scheduler based on the official implementation of "spectral_contrastive_learning" (https://github.com/jhaochenz/spectral_contrastive_learning/blob/ee431bdba9bb62ad00a7e55792213ee37712784c/optimizers/lr_scheduler.py)
        #####################################################################################
        # The information about the license of the original "spectral_contrastive_learning":
        # Copyright (c) 2021 Jeff Z. HaoChen
        # Relased by under the MIT license
        # https://github.com/jhaochenz/spectral_contrastive_learning/blob/main/LICENSE
        #####################################################################################
    """
    def __init__(self, warmup_epochs, num_epochs, init_lr, base_lr, final_lr, num_iter):
        self.init_lr = init_lr
        self.warmup_lr_schedule = np.linspace(init_lr, base_lr, num_iter * warmup_epochs)
        self.decay_iter = num_iter * (num_epochs - warmup_epochs)
        self.cosine_lr_schedule = final_lr+0.5*(base_lr-final_lr)*(1+np.cos(np.pi*np.arange(self.decay_iter)/self.decay_iter))
        
        self.lr_schedule = np.concatenate((self.warmup_lr_schedule, self.cosine_lr_schedule))
        self.iter = -1
    def step(self, optimizer):
        self.iter += 1
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr_schedule[self.iter]

    def cur_lr(self):
        return self.lr_schedule[self.iter]