import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attended = self.attention(x, x, x, attn_mask=mask)[0]
        x = x + self.dropout(attended)
        x = self.norm1(x)
        x = x + self.dropout(self.ff(x))
        return self.norm2(x)

class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_blocks, dropout=0.1):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, mask=mask)
        return self.fc(x)
