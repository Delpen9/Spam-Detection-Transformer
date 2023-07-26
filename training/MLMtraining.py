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

# Directory Libraries
import os
from datetime import datetime

# Standard Data Science Libraries
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math

# Serialize and De-Serialize Model
from joblib import dump

class MLMTrainer:
    def __init__(
        self,
        device, model, optimizer, criterion,
        tokenizer, MASK_ID, MASK_RATIO,
        NUM_EPOCHS, NUM_ITERATIONS, BATCH_SIZE, MAX_LENGTH,
        directory_path, VALIDATION_RATIO, VALIDATION_COUNT = None,
        SAVE_OUTPUT = False, SAVE_MODEL = False,
        TRAINING_OUTPUT_PATH = '', MODEL_OUTPUT_PATH = ''):
        '''
        Initialize the MLMTrainer instance with specified parameters.

        Parameters:
        device: Device to train the model on. 
        model: Instance of the model to train.
        optimizer: Optimizer to use for training.
        criterion: Criterion to calculate the loss.
        tokenizer: Tokenizer to tokenize the sentences.
        MASK_ID: ID for the mask token.
        MASK_RATIO: Ratio of tokens to mask in a sentence.
        NUM_EPOCHS: Number of epochs for training.
        NUM_ITERATIONS: Number of iterations for training.
        BATCH_SIZE: Size of the batch for training.
        MAX_LENGTH: Maximum length for the tokenized sentence.
        directory_path: Path of the directory containing data.
        VALIDATION_RATIO: Ratio of data to use for validation.
        VALIDATION_COUNT: Count of data to use for validation.
        SAVE_OUTPUT: Flag to decide whether to save training output or not.
        SAVE_MODEL: Flag to decide whether to save the model or not.
        TRAINING_OUTPUT_PATH: Path to save the training output.
        MODEL_OUTPUT_PATH: Path to save the trained model.
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

        self.training_output = pd.DataFrame([], columns = ['epoch', 'iteration', 'loss'])
        self.validation_output = pd.DataFrame([], columns = ['epoch', 'iteration', 'loss'])

    def get_training_batch(self):
        '''
        Generate a batch of sentences for training from the directory specified in directory_path.

        Returns:
        List of sentences for training.
        '''
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
        '''
        Get samples for validation from the directory specified in directory_path.

        Returns:
        List of sentences for validation.
        '''
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
        '''
        Generator to yield chunks of sentences of size BATCH_SIZE.

        Parameters:
        sentences: List of sentences to split into chunks.

        Returns:
        Generator yielding chunks of sentences of size BATCH_SIZE.
        '''
        for i in range(0, len(sentences), self.BATCH_SIZE):
            chunk = sentences[i : i + self.BATCH_SIZE]
            if len(chunk) == self.BATCH_SIZE:
                yield chunk

    def calculate_validation_loss(self):
        '''
        Calculate validation loss for the current state of the model.

        Returns:
        Validation loss.
        '''
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
        '''
        Encode sentences using the specified tokenizer.

        Parameters:
        sentences: List of sentences to encode.

        Returns:
        Tuple of inputs and targets tensors.
        '''
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
        '''
        Mask inputs according to the specified MASK_RATIO.

        Parameters:
        inputs: Tensor of input sentences.
        sentences: List of sentences.
        '''
        for i in range(self.BATCH_SIZE):
            unpadded_sentence_len = len(sentences[i])
            num_masks = int(unpadded_sentence_len * self.MASK_RATIO)
            mask_indices = torch.randperm(n = unpadded_sentence_len)[:num_masks]
            mask_indices = torch.min(mask_indices, torch.tensor(self.MAX_LENGTH - 1))
            inputs[i][mask_indices] = self.MASK_ID

    def train(self):
        '''
        Train the model using the specified optimizer and criterion. 
        Save the model and output if specified.
        '''
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
                    self.validation_output = pd.concat([
                        self.validation_output,
                        pd.DataFrame({
                            'epoch': [epoch + 1],
                            'iteration': [iteration + 1],
                            'loss': [validation_loss]
                        })], ignore_index = True
                    )

                print(f'Epoch: {epoch + 1} of {self.NUM_EPOCHS}, Iteration: {iteration + 1} of {self.NUM_ITERATIONS}, Loss: {loss.item()}')
                
                self.training_output = pd.concat([
                    self.training_output,
                    pd.DataFrame({
                        'epoch': [epoch + 1],
                        'iteration': [iteration + 1],
                        'loss': [loss.item()]
                    })], ignore_index = True
                )

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        if self.SAVE_OUTPUT == True:
            self.training_output.to_csv(f'{self.TRAINING_OUTPUT_PATH}/training_output_{timestamp}.csv', index = False)
            self.validation_output.to_csv(f'{self.TRAINING_OUTPUT_PATH}/validation_output_{timestamp}.csv', index = False)

        if self.SAVE_MODEL == True:
            dump(self.model, f'{MODEL_OUTPUT_PATH}/model_{timestamp}.joblib')

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
    # NUM_EPOCHS = 10
    NUM_EPOCHS = 1 # TODO: Delete
    BATCH_SIZE = 256 # TODO: Increase on GPU
    VALIDATION_RATIO = 0.05 # Used if VALIDATION_COUNT = None
    VALIDATION_COUNT = 1 # Overrides validation ratio; represents number of files used for validation calculation
    NUM_ITERATIONS = int(1500000 * 32 / BATCH_SIZE * (1 - VALIDATION_RATIO)) if VALIDATION_COUNT == None \
                    else int(1500000 * 32 / BATCH_SIZE - VALIDATION_COUNT)
    NUM_ITERATIONS = 50 # TODO: Delete

    SAVE_OUTPUT = True
    SAVE_MODEL = True
    TRAINING_OUTPUT_PATH = '../output'
    MODEL_OUTPUT_PATH = '../artifacts'

    LOAD_MODEL = False
    LOAD_MODEL_PATH = ''

    if LOAD_MODEL == False:
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
    else:
        model = load(f'{LOAD_MODEL_PATH}')
        optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    trainer = MLMTrainer(
        device, model, optimizer, criterion,
        tokenizer, MASK_ID, MASK_RATIO,
        NUM_EPOCHS, NUM_ITERATIONS, BATCH_SIZE, SEQ_LENGTH,
        DIRECTORY_PATH, VALIDATION_RATIO, VALIDATION_COUNT,
        SAVE_OUTPUT, SAVE_MODEL,
        TRAINING_OUTPUT_PATH, MODEL_OUTPUT_PATH
    )

    trainer.train()