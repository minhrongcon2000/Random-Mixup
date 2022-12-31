import numpy as np
import torch
from torch.autograd import Variable


class InputMix:
    def __init__(self, alpha, model, *args, **kwargs):
        self.alpha = alpha
        self.model = model
    
    def __call__(self, inputs, targets, *args, **kwargs):
        inputs, targets_a, targets_b, lam = mixup_data(inputs, 
                                                       targets,
                                                       self.alpha, 
                                                       True)
        targets_a, targets_b = map(Variable, (targets_a, targets_b))
        mixed_inputs = Variable(inputs, requires_grad=True)
        outputs = self.model(mixed_inputs)
        criterion = torch.nn.CrossEntropyLoss().cuda()
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        return lam, mixed_inputs, outputs, loss
    
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)