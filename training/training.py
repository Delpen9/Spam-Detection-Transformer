# Modeling Version 1
from model1transformer import Model1Transformer
from transformers import BertTokenizerFast

# Modeling Version 2
from models.Transformer import Transformer
from models.TransformerEncoder import TransformerEncoder
from models.PretrainedOnMLM import PretrainedOnMLM
from models.SpamDetectionModel import SpamDetectionModel
from models.DistilledFromTinyBert import DistilledFromTinyBert

# Directory Library
import os

# Standard Data Science Libraries
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
    BATCH_SIZE : int,
    MAX_LENGTH : int,
    MODEL_VERSION : int = 1
) -> None:
    '''
    '''
    model = model.to(device)

    for epoch in range(NUM_EPOCHS):
        for iteration in range(NUM_ITERATIONS):
            sentences = get_batch(BATCH_SIZE)

            input_list = []
            mask_indices_list = []
            for sample_idx in range(BATCH_SIZE):
                # Perform tokenization on input
                input_ids = tokenizer(
                    sentences[sample_idx],
                    padding = 'max_length',
                    truncation = True,
                    max_length = MAX_LENGTH,
                    return_tensors = 'pt',
                    is_split_into_words = True
                )['input_ids'][0]

                input_list.append(input_ids)

            inputs = torch.stack(input_list).to(device)
            targets = inputs.clone()

            # Perform masking
            for i in range(BATCH_SIZE):
                unpadded_sentence_len = len(sentences[i])
                num_masks = int(unpadded_sentence_len * MASK_RATIO)
                mask_indices = torch.randperm(n = unpadded_sentence_len)[:num_masks]

                # Make sure indices do not exceed input size
                mask_indices = torch.min(mask_indices, torch.tensor(MAX_LENGTH - 1))
                mask_indices_list.append(mask_indices.tolist())
                inputs[i][mask_indices] = MASK_ID

            # Forward pass
            if MODEL_VERSION == 1:
                outputs = model(inputs)
            elif MODEL_VERSION == 2:
                outputs = model(inputs, mask_indices_list)

            # Calculate loss
            if MODEL_VERSION == 1:
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            elif MODEL_VERSION == 2:
                loss = criterion(torch.tensor([[10, 9],[10, 9]]), outputs)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss after each batch
            print(f'Epoch: {epoch + 1}, Iteration: {iteration + 1}, Loss: {loss.item()}')


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
    SEQ_LENGTH = 32
    NUM_LAYERS = 6 # Specific to model 2

    # Training Hyperparameters
    LEARNING_RATE = 1e-2

    # Select particular model to use
    MODEL_VERSION = 2

    # Load the model
    if MODEL_VERSION == 1:
        model = Model1Transformer(
            vocab_size = VOCAB_SIZE,
            embed_dim = EMBED_DIM,
            num_heads = NUM_HEADS,
            ff_dim = FF_DIM,
            num_blocks = NUM_BLOCKS,
            dropout = DROPOUT
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

    elif MODEL_VERSION == 2:
        transformer_encoder = TransformerEncoder(
            SEQ_LENGTH,
            VOCAB_SIZE,
            EMBED_DIM,
            NUM_LAYERS,
            expansion_factor = 4,
            n_heads = NUM_HEADS
        )
        model = PretrainedOnMLM(transformer_encoder, VOCAB_SIZE, EMBED_DIM)
        optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
        criterion = model.loss()

    # # Load the BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # MASK ID for BERT Tokenizer
    MASK_ID = 103
    MASK_RATIO = 0.15

    # Perform training procedure
    NUM_EPOCHS = 10

    # ================
    # NOTE: Based on a batch size of 32, a single epoch would be equal to 1,500,000 iterations
    # ================
    NUM_ITERATIONS = 1000
    BATCH_SIZE = 32

    train(
        device, model, optimizer, criterion,
        tokenizer, MASK_ID, MASK_RATIO,
        NUM_EPOCHS, NUM_ITERATIONS, BATCH_SIZE, SEQ_LENGTH,
        MODEL_VERSION
    )
