from transformer import Transformer
from transformers import BertTokenizerFast

import os

import torch
import torch.nn as nn
import numpy as np

def get_batch(
    batch_size : int = 32
) -> list:
    '''
    '''
    # Initialize output value
    sentences = []

    # Get filename
    directory_list = os.listdir(directory_path)
    filename_idx = np.random.choice(len(directory_list), 1)[0]
    filename = directory_list[filename_idx]

    # Open the file
    with open(os.path.join(directory_path, filename), 'r') as f:
        # Get all sentences in file
        file_sentences = f.read().lower().strip().split('\n')

        # Get batch
        batch_indices = np.random.choice(len(file_sentences), batch_size, replace = False)
        batch_sentences = [file_sentences[batch_idx] for batch_idx in batch_indices]

        # Create tokens
        sentences.extend([sentence.split() for sentence in batch_sentences])

    return sentences

def train(
    device : str,
    model : any,
    optimizer : any,
    criterion : any,
    tokenizer : any,
    MASK_ID : int,
    MASK_RATIO : float,
    NUM_EPOCHS : int,
    NUM_ITERATIONS : int,
    BATCH_SIZE : int
) -> None:
    '''
    '''
    for epoch in range(NUM_EPOCHS):
        for iteration in range(NUM_ITERATIONS):
            sentences = get_batch(BATCH_SIZE)

            for sample_idx in range(BATCH_SIZE):
                # Perform tokenization on input
                input_ids = tokenizer.__call__(
                    sentences[sample_idx],
                    padding = 'max_length',
                    truncation = True,
                    max_length = 512,
                    return_tensors = 'pt',
                    is_split_into_words = True
                )['input_ids'][0]

                input = torch.tensor(input_ids, dtype = torch.long).to(device)
                target = input.clone()

                # Perform masking
                num_masks = int(len(sentences[sample_idx]) * MASK_RATIO)
                mask_idx = np.random.choice(len(sentences[sample_idx]), num_masks)

                # Make sure indices do not exceed input size
                mask_idx = [min(511, x) for x in mask_idx]
                input[mask_idx] = MASK_ID

                # Forward pass
                outputs = model(input)

                # Calculate loss
                loss = criterion(outputs.view(-1, outputs.size(-1)), target.view(-1))

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print loss after each batch
                print(fr'Epoch: {epoch + 1}, Iteration: {iteration + 1}, Batch: {sample_idx}, Loss: {loss.item()}')

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
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    criterion = nn.CrossEntropyLoss()

    # Load the BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # MASK ID for BERT Tokenizer
    MASK_ID = 103
    MASK_RATIO = 0.15

    # Perform training procedure
    NUM_EPOCHS = 10
    NUM_ITERATIONS = 1000
    BATCH_SIZE = 32

    train(device, model, optimizer, criterion, tokenizer, MASK_ID, MASK_RATIO, NUM_EPOCHS, NUM_ITERATIONS, BATCH_SIZE)
