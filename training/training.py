from transformer import Transformer
from transformers import BertTokenizerFast

import os

import torch
import numpy as np

if __name__ == '__main__':
    np.random.seed(1234)

    directory_path = '../data/masking/openwebtext/openwebtext'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    NUM_TOKENS = 30522

    # Load the model
    model = Transformer(vocab_size = 30522, embed_dim = 768, num_heads = 12, ff_dim = 3072, num_blocks = 12, dropout = 0.1)

    # Load the BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # Initialize a list to hold the sentences
    sentences = []
    masked_sentences = []
    mask_locations =  []

    # Read all .txt files in the directory
    for filename in os.listdir(directory_path)[:2]:
        if filename.endswith('.txt'):
            with open(os.path.join(directory_path, filename), 'r') as f:
                file_sentences = f.read().lower().strip().split('\n')
                sentences.extend([sentence.split() for sentence in file_sentences])
                masked_sentences.extend([masked_sentence.split() for masked_sentence in file_sentences])

    # Determine position of mask
    for masked_sentence in masked_sentences:
        index = np.random.choice(len(masked_sentence), 1)[0]
        mask_locations.append(index)
        masked_sentence[index] = '[MASK]'

    token_ids = tokenizer.__call__(
        masked_sentences[0],
        padding = 'max_length',
        truncation = True,
        max_length = 512,
        return_tensors = 'pt',
        is_split_into_words = True
    )['input_ids']
