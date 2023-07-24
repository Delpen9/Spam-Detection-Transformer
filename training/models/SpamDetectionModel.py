from .TransformerEncoder import TransformerEncoder
import torch
from torch import nn
import torch.nn.functional as F
class SpamDetectionModel(nn.Module):

    def __init__(self, model, embed_dim, n_classes=2):

        super().__init__()
        self.model = model
        self.linear = nn.Linear(embed_dim, n_classes)

    def forward(self, x_in):

        # Feed into transformer encoder
        out = self.model(x_in)  # shape of out N*T*D

        # Gather the last relevant hidden state
        out = out[:, -1, :]  # N*D

        # FC layer
        y_pred = self.linear(out)
        y_pred = F.softmax(y_pred, dim=1)
        return y_pred
