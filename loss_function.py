import torch
import torch.nn as nn


class MyLossFunction(nn.Module):
    def __init__(self):
        super(MyLossFunction, self).__init__()
        self.bce = torch.nn.BCELoss()

    def forward(self, output, labels):
        labels = labels.float()
        loss0 = self.bce(output, labels)
        return loss0