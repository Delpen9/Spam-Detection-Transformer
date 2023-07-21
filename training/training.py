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

    directory_path = '../data/masking/openwebtext/openwebtext'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    NUM_TOKENS = 30522
    BATCH_SIZE = 32

    # Load the model
    model = Transformer(vocab_size = 30522, embed_dim = 768, num_heads = 12, ff_dim = 3072, num_blocks = 12, dropout = 0.1)

    # Load the BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # Get batch
    sentences, masked_sentences, mask_locations = get_batch(BATCH_SIZE)

    token_ids = tokenizer.__call__(
        masked_sentences[0],
        padding = 'max_length',
        truncation = True,
        max_length = 512,
        return_tensors = 'pt',
        is_split_into_words = True
    )['input_ids']
