import numpy as np
import torch
from torch.autograd import Variable


class CutMix:
    def __init__(self, model, *args, **kwargs) -> None:
        self.model = model
    
    def __call__(self, inputs, targets, *args, **kwargs):
        criterion = torch.nn.CrossEntropyLoss().cuda()
        prob = np.random.rand(1)
        if prob < 0.5:
            # generate mixed sample
            lam = np.random.beta(self.args.dirichlet_alpha, self.args.dirichlet_alpha)
            rand_index = torch.randperm(inputs.size()[0]).cuda()
            target_a = targets
            target_b = targets[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            # compute output
            mixed_inputs = Variable(inputs, requires_grad=True)
            outputs = self.model(mixed_inputs)
            loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
        else:
            mixed_inputs = Variable(inputs, requires_grad=True)
            outputs = self.model(mixed_inputs)
            loss = criterion(outputs, targets)
        return lam, mixed_inputs, outputs, loss
            
            
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2