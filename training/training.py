from transformer import Transformer
from transformers import BertTokenizerFast

import os

import torch
import numpy as np

def get_batch(
    batch_size : int = 32
) -> tuple[list, list, list]:
    '''
    '''
    # Initialize output values
    sentences = []
    masked_sentences = []
    mask_locations =  []

    # Get file index
    file_sample_index = np.random.choice(len(masked_sentence), 1)[0]

    for filename in os.listdir(directory_path)[file_sample_index]:
        if filename.endswith('.txt'):
            with open(os.path.join(directory_path, filename), 'r') as f:
                # Get all sentences in file
                file_sentences = f.read().lower().strip().split('\n')

                # Get batch
                batch_indices = np.random.choice(len(masked_sentence), batch_size, replace = False)
                batch_sentences = [file_sentences[batch_idx] for batch_idx in batch_indices]

                # Create tokens
                sentences.extend([sentence.split() for sentence in batch_sentences])
                masked_sentences.extend([sentence.split() for sentence in batch_sentences])

    # Perform masking
    for masked_sentence in masked_sentences:
        index = np.random.choice(len(masked_sentence), 1)[0]
        masked_sentence[index] = '[MASK]'

        # Maintain list of mask indices
        mask_locations.append(index)

    return (sentences, masked_sentences, mask_locations)

if __name__ == '__main__':
    np.random.seed(1234)

    # Set the directory and device
    directory_path = '../data/masking/openwebtext/openwebtext'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set model hyperparameters
    VOCAB_SIZE = 30522
    EMBED_DIM = 768
    NUM_HEADS = 12
    FF_DIM = 3072
    NUM_BLOCKS = 12
    DROPOUT = 0.1

    # Load the model
    model = Transformer(
        vocab_size = VOCAB_SIZE,
        embed_dim = EMBED_DIM,
        num_heads = NUM_HEADS,
        ff_dim = FF_DIM,
        num_blocks = NUM_BLOCKS,
        dropout = DROPOUT
    )

    # Load the BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # Perform training procedure
    NUM_EPOCHS = 10
    NUM_ITERATIONS = 1000
    BATCH_SIZE = 32
    for epoch in range(NUM_EPOCHS):
        for iteration in range(NUM_ITERATIONS):
            sentences, masked_sentences, mask_locations = get_batch(BATCH_SIZE)

            for sample_idx in range(BATCH_SIZE):
                token_ids = tokenizer.__call__(
                    masked_sentences[sample_idx],
                    padding = 'max_length',
                    truncation = True,
                    max_length = 512,
                    return_tensors = 'pt',
                    is_split_into_words = True
                )['input_ids']

                # TODO: Perform forward pass on single sample
                forward = None