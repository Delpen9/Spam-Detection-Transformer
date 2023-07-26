# Tokenization
from transformers import BertTokenizerFast

# Modeling Version 1
from models.model1transformer import Model1Transformer

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
import math

class Trainer:
    def __init__(
        self,
        device, model, optimizer, criterion,
        tokenizer, MASK_ID, MASK_RATIO,
        NUM_EPOCHS, NUM_ITERATIONS, BATCH_SIZE, MAX_LENGTH,
        directory_path, VALIDATION_RATIO, VALIDATION_COUNT = None,
        SAVE_OUTPUT = False, SAVE_MODEL = False,
        TRAINING_OUTPUT_PATH = '', MODEL_OUTPUT_PATH = ''):
        '''
        '''
        super().__init__()
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.tokenizer = tokenizer
        self.MASK_ID = MASK_ID
        self.MASK_RATIO = MASK_RATIO
        self.NUM_EPOCHS = NUM_EPOCHS
        self.NUM_ITERATIONS = NUM_ITERATIONS
        self.BATCH_SIZE = BATCH_SIZE
        self.MAX_LENGTH = MAX_LENGTH
        self.directory_path = directory_path
        self.VALIDATION_RATIO = VALIDATION_RATIO
        self.VALIDATION_COUNT = VALIDATION_COUNT
        self.SAVE_OUTPUT = SAVE_OUTPUT
        self.SAVE_MODEL = SAVE_MODEL
        self.TRAINING_OUTPUT_PATH = TRAINING_OUTPUT_PATH
        self.MODEL_OUTPUT_PATH = MODEL_OUTPUT_PATH

    def get_training_batch(self):
        sentences = []
        directory_list = os.listdir(self.directory_path)

        num_files = len(directory_list)
        validation_count = int(num_files * self.VALIDATION_RATIO) if self.VALIDATION_COUNT == None else self.VALIDATION_COUNT
        training_count = num_files - validation_count
        file_probabilities = list(np.concatenate((
            np.zeros(validation_count),
            np.ones(training_count) / training_count
        )))

        filename_idx = np.random.choice(len(directory_list), 1, p = file_probabilities)[0]
        filename = directory_list[filename_idx]

        with open(os.path.join(self.directory_path, filename), 'r') as f:
            file_sentences = f.read().lower().strip().split('\n')
            batch_indices = np.random.choice(len(file_sentences), self.BATCH_SIZE, replace = False)
            batch_sentences = [file_sentences[batch_idx] for batch_idx in batch_indices]
            sentences.extend([sentence.split() for sentence in batch_sentences])

        return sentences

    def get_validation_samples(self):
        sentences = []
        directory_list = os.listdir(self.directory_path)

        num_files = len(directory_list)
        validation_count = int(num_files * self.VALIDATION_RATIO) if self.VALIDATION_COUNT == None else self.VALIDATION_COUNT
        filenames = directory_list[:validation_count]

        for filename in filenames:
            with open(os.path.join(self.directory_path, filename), 'r') as f:
                file_sentences = f.read().lower().strip().split('\n')
                sentences.extend([sentence.split() for sentence in file_sentences])

        return sentences
    
    def chunks(self, sentences):
        for i in range(0, len(sentences), self.BATCH_SIZE):
            chunk = sentences[i : i + self.BATCH_SIZE]
            if len(chunk) == self.BATCH_SIZE:
                yield chunk

    def calculate_validation_loss(self):
        self.model.eval()

        validation_loss = 0.0

        with torch.no_grad():
            sentences = self.get_validation_samples()
            for batch in self.chunks(sentences):
                inputs, targets = self.encode_sentences(batch)
                self.mask_inputs(inputs, batch)

                outputs = self.model(inputs)
                reshaped_outputs = outputs.view(-1, outputs.size(-1)).clone()
                desired_target = targets.view(-1).clone()
                loss = self.criterion(reshaped_outputs, desired_target)

                print(f'Validation loss for batch: {loss.item()}')

                validation_loss += loss.item()

        validation_loss /= float(math.floor(len(sentences) / self.BATCH_SIZE))

        return validation_loss

    def encode_sentences(self, sentences):
        input_list = []
        for sample_idx in range(self.BATCH_SIZE):
            input_ids = self.tokenizer(
                sentences[sample_idx],
                padding = 'max_length',
                truncation = True,
                max_length = self.MAX_LENGTH,
                return_tensors = 'pt',
                is_split_into_words = True
            )['input_ids'][0]

            input_list.append(input_ids)

        inputs = torch.stack(input_list).to(self.device)
        targets = inputs.clone()
        return (inputs, targets)

    def mask_inputs(self, inputs, sentences):
        for i in range(self.BATCH_SIZE):
            unpadded_sentence_len = len(sentences[i])
            num_masks = int(unpadded_sentence_len * self.MASK_RATIO)
            mask_indices = torch.randperm(n = unpadded_sentence_len)[:num_masks]
            mask_indices = torch.min(mask_indices, torch.tensor(self.MAX_LENGTH - 1))
            inputs[i][mask_indices] = self.MASK_ID

    def train(self):
        for epoch in range(self.NUM_EPOCHS):
            for iteration in range(self.NUM_ITERATIONS):
                sentences = self.get_training_batch()

                inputs, targets = self.encode_sentences(sentences)
                self.mask_inputs(inputs, sentences)

                outputs = self.model(inputs)
                reshaped_outputs = outputs.view(-1, outputs.size(-1)).clone()
                desired_target = targets.view(-1).clone()

                loss = self.criterion(reshaped_outputs, desired_target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if iteration % 50 == 0:
                    print('\n' + '#' * 25)
                    print('Calculate validation loss')
                    print('#' * 25)
                    validation_loss = self.calculate_validation_loss()
                    print('#' * 25)
                    print(f'Average validation loss for all batches: {validation_loss}')
                    print('#' * 25 + '\n')
                    print('#' * 25)
                    print('Training loss')
                    print('#' * 25)

                print(f'Epoch: {epoch + 1} of {self.NUM_EPOCHS}, Iteration: {iteration + 1} of {self.NUM_ITERATIONS}, Loss: {loss.item()}')

if __name__ == '__main__':
    np.random.seed(1234)
    DIRECTORY_PATH = '../data/masking/openwebtext/openwebtext'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    VOCAB_SIZE = 30522
    EMBED_DIM = 768
    NUM_HEADS = 12
    FF_DIM = 3072
    NUM_BLOCKS = 1 # TODO: Increase on GPU
    DROPOUT = 0.2
    SEQ_LENGTH = 16 # TODO: Increase on GPU
    MASK_RATIO = 0.15
    EXPANSION_FACTOR = 4
    LEARNING_RATE = 1e-2
    MODEL_VERSION = 2
    MASK_ID = 103
    NUM_EPOCHS = 10
    BATCH_SIZE = 256 # TODO: Increase on GPU
    VALIDATION_RATIO = 0.05 # Used if VALIDATION_COUNT = None
    VALIDATION_COUNT = 1 # Overrides validation ratio; represents number of files used for validation calculation
    NUM_ITERATIONS = int(1500000 * 32 / BATCH_SIZE * (1 - VALIDATION_RATIO)) if VALIDATION_COUNT == None \
                    else int(1500000 * 32 / BATCH_SIZE - VALIDATION_COUNT)

    SAVE_OUTPUT = True
    SAVE_MODEL = True
    TRAINING_OUTPUT_PATH = '../output'
    MODEL_OUTPUT_PATH = '../artifacts'

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
            seq_len = SEQ_LENGTH,
            vocab_size = VOCAB_SIZE,
            embed_dim = EMBED_DIM,
            num_layers = NUM_BLOCKS,
            expansion_factor = EXPANSION_FACTOR,
            n_heads = NUM_HEADS,
            dropout = DROPOUT
        ).to(device)
        model = PretrainedOnMLM(transformer_encoder, EMBED_DIM, VOCAB_SIZE).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    trainer = Trainer(
        device, model, optimizer, criterion,
        tokenizer, MASK_ID, MASK_RATIO,
        NUM_EPOCHS, NUM_ITERATIONS, BATCH_SIZE, SEQ_LENGTH,
        DIRECTORY_PATH, VALIDATION_RATIO, VALIDATION_COUNT
    )

    trainer.train()
