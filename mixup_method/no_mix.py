import torch
import torch.nn as nn
from torch.autograd import Variable


class NoMix:
    def __init__(self, model, *args, **kwargs) -> None:
        self.model = model
    
    def __call__(self, inputs, targets, *args, **kwargs):
        mixed_inputs = Variable(inputs, requires_grad=True)
        outputs = self.model(mixed_inputs)
        loss = torch.mean(
            torch.sum(-targets * nn.LogSoftmax(-1)(outputs), dim=1)
        )
        return 1, mixed_inputs, outputs, loss