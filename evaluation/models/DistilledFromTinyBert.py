import torch
from torch import nn

class DistilledFromTinyBert(nn.Module):
    def __init__(self, model, a = 0.5):
        super(DistilledFromTinyBert, self).__init__()
        self.model = model
        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_ce = torch.nn.CrossEntropyLoss()
        self.a = a

    def loss(self, x, tiny_BERT_probabilities, true_labels):
        outputs = self.model(x)
        return self.a * self.criterion_ce(outputs, true_labels) + (1 - self.a) * self.criterion_mse(outputs, tiny_BERT_probabilities)
