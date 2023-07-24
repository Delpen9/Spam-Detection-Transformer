import torch
import torch.nn as nn
from .TransformerEncoder import TransformerEncoder

class PretrainedOnMLM(nn.Module):
    def __init__(self, model, vocab_size, embed_dim, seq_len, num_masked):
        """
        :param vocab_size: total vocab size for masked_lm
        """
        super().__init__()
        self.model = model
        self.mask_lm = MaskedLanguageModel(embed_dim, vocab_size, seq_len, num_masked)

    def forward(self, x, pos):
        x = self.model(x)
        x = self.mask_lm(x)
        return x

class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """
    def __init__(self, embed_dim, vocab_size, seq_len, num_masked):
        """
        :param hidden: output size of transformer encoder model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_masked = num_masked

        self.fc1 = nn.Linear(embed_dim, vocab_size)
        self.fc2 = nn.Linear(seq_len, num_masked)
        self.softmax = nn.LogSoftmax(dim = -1)

    def forward(self, x):
        x = self.fc1(x)
        x = x.permute(0, -1, -2)
        x = self.fc2(x)
        x = x.permute(0, -1, -2)
        return self.softmax(x)
