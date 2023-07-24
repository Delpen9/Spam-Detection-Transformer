import torch
import torch.nn as nn
from .TransformerEncoder import TransformerEncoder

class PretrainedOnMLM(nn.Module):
    def __init__(self, model, vocab_size, embed_dim):
        """
        :param vocab_size: total vocab size for masked_lm
        """
        super().__init__()
        self.model = model
        self.mask_lm = MaskedLanguageModel(embed_dim, vocab_size)

    def forward(self, x, pos):
        x = self.model(x)
        x = self.mask_lm(x)
        x = x[torch.arange(x.size(0)).unsqueeze(1), pos]
        return x

class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """
    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of transformer encoder model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim = -1)

    def forward(self, x):
        return self.softmax(self.linear(x))
