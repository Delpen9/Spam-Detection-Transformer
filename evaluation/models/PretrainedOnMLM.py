import torch
import torch.nn as nn
from .TransformerEncoder import TransformerEncoder

class PretrainedOnMLM(nn.Module):
    def __init__(self, model, embed_dim, vocab_size):
        """
        :param vocab_size: total vocab size for masked_lm
        """
        super().__init__()
        self.model = model
        self.mask_lm = MaskedLanguageModel(embed_dim, vocab_size)

    def forward(self, x):
        x = self.model(x)
        x = self.mask_lm(x)
        return x

class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """
    def __init__(self, embed_dim, vocab_size):
        """
        :param hidden: output size of transformer encoder model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.fc(x)
        return x