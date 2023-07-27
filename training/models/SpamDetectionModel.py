from .TransformerEncoder import TransformerEncoder
import torch
from torch import nn
import torch.nn.functional as F

class SpamDetectionModel(nn.Module):
    def __init__(self, model, embed_dim, n_classes = 2):
        super().__init__()
        self.model = model
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = self.model(x)
        x = x[:, -1, :]
        outputs = self.fc(x)
        return outputs
