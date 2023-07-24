import torch
from torch import nn
class DistilledFromTinyBert(nn.Module):

    def __init__(self, model):
        super(DistilledFromTinyBert, self).__init__()
        self.model = model
        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_ce = torch.nn.CrossEntropyLoss()
        self.a = 0.5

    def loss(self, x, tiny_bert_prob, real_label):
        output = self.model(x)
        return self.a * self.criterion_ce(output, real_label) + (1 - self.a) * self.criterion_mse(output, tiny_bert_prob)

