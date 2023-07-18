import sys
sys.path.insert(0, '../models')
from transformer import Transformer

import torch
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss

def read_text_file(file_path):
    with open(file_path, "r") as f:
        lines = f.read().splitlines()
    return lines

# Assume that `sentences` is a list of sentences read from your text file
FILENAME = None
sentences = read_text_file(FILENAME)

# Parameters
vocab_size = 30522  # You would need to compute the actual vocab size from your data
max_length = 512  # Or any other value that fits your data
mask_prob = 0.15  # Masking probability for MLM
tokenizer = torch.nn.Embedding(vocab_size, max_length)

def mask_tokens(inputs, mask_prob):
    """ Prepare masked tokens inputs/labels for masked language modeling: """
    labels = inputs.clone()
    # Mask a fraction of tokens in each sequence for prediction.
    mask = torch.full(labels.shape, False)
    for i in range(inputs.shape[0]):
        mask[i, random.sample(range(inputs.shape[1]), round(mask_prob * inputs.shape[1]))] = True

    # Set labels to -1 for non-masked inputs
    labels[~mask] = -1
    # Set inputs to the MASK token for masked inputs
    inputs[mask] = tokenizer.encode('[MASK]')[0]  # Assuming the tokenizer uses '[MASK]' for the Mask token

    return inputs, labels

class MLMDataset(Dataset):
    def __init__(self, sentences, tokenizer):
        self.sentences = sentences
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        inputs = torch.tensor(self.tokenizer.encode(sentence, add_special_tokens=True))
        inputs = inputs[:max_length]  # Truncate to max_length
        inputs, labels = mask_tokens(inputs, mask_prob)
        return {"inputs": inputs, "labels": labels}

dataset = MLMDataset(sentences, tokenizer)
dataloader = DataLoader(dataset, batch_size = 32)

# Instantiate the model
model = Transformer(vocab_size, embed_dim = 256, num_heads = 8, ff_dim = 512, num_blocks = 6, dropout = 0.1)

device = 'cpu'
model.to(device)  # If using GPU
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
loss_fn = CrossEntropyLoss(ignore_index=-1)  # Ignore non-masked tokens

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    for batch in dataloader:
        inputs = batch["inputs"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs.view(-1, vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()
    print(f'Loss: {loss.item()}')
