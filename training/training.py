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

class Trainer:
    def __init__(self, device, model, optimizer, criterion, tokenizer, MASK_ID, MASK_RATIO, NUM_EPOCHS, NUM_ITERATIONS, BATCH_SIZE, MAX_LENGTH, directory_path):
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

    def get_batch(self):
        sentences = []
        directory_list = os.listdir(self.directory_path)
        filename_idx = np.random.choice(len(directory_list), 1)[0]
        filename = directory_list[filename_idx]

        with open(os.path.join(self.directory_path, filename), 'r') as f:
            file_sentences = f.read().lower().strip().split('\n')
            batch_indices = np.random.choice(len(file_sentences), self.BATCH_SIZE, replace = False)
            batch_sentences = [file_sentences[batch_idx] for batch_idx in batch_indices]
            sentences.extend([sentence.split() for sentence in batch_sentences])

        return sentences

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
                sentences = self.get_batch()

                inputs, targets = self.encode_sentences(sentences)
                self.mask_inputs(inputs, sentences)

                outputs = self.model(inputs)
                reshaped_outputs = outputs.view(-1, outputs.size(-1)).clone()
                desired_target = targets.view(-1).clone()

                loss = self.criterion(reshaped_outputs, desired_target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print(f'Epoch: {epoch + 1} of {self.NUM_EPOCHS}, Iteration: {iteration + 1} of {self.NUM_ITERATIONS}, Loss: {loss.item()}')

if __name__ == '__main__':
    np.random.seed(1234)
    directory_path = '../data/masking/openwebtext/openwebtext'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    VOCAB_SIZE = 30522
    EMBED_DIM = 768
    NUM_HEADS = 12
    FF_DIM = 3072
    NUM_BLOCKS = 1
    DROPOUT = 0.2
    SEQ_LENGTH = 64
    MASK_RATIO = 0.15
    EXPANSION_FACTOR = 4
    LEARNING_RATE = 1e-2
    MODEL_VERSION = 2
    MASK_ID = 103
    NUM_EPOCHS = 10
    BATCH_SIZE = 64
    NUM_ITERATIONS = int(1500000 * 32 / BATCH_SIZE)

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

    trainer = Trainer(device, model, optimizer, criterion, tokenizer, MASK_ID, MASK_RATIO, NUM_EPOCHS, NUM_ITERATIONS, BATCH_SIZE, SEQ_LENGTH, directory_path)
    trainer.train()
