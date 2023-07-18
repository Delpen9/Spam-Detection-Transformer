import torch
import random
from torch.utils.data import Dataset, DataLoader

# Assume that `sentences` is a list of sentences read from your text file
sentences = read_text_file("your_file.txt")

# Parameters
vocab_size = 30522  # You would need to compute the actual vocab size from your data
max_length = 64  # Or any other value that fits your data
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
dataloader = DataLoader(dataset, batch_size=32)
