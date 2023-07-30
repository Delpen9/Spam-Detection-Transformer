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

# Standard Data Science Libraries
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math

# Serialize and De-Serialize Model
from joblib import load, dump

# Graphing
import matplotlib.pyplot as plt
import seaborn as sns

# Logging
import logging

# Miscellaneous
from datetime import datetime
import time

class MLMTrainer:
    def __init__(
        self,
        device, model, optimizer, criterion,
        tokenizer, MASK_ID, MASK_RATIO,
        NUM_EPOCHS, NUM_ITERATIONS, BATCH_SIZE, MAX_LENGTH,
        directory_path, VALIDATION_RATIO, VALIDATION_COUNT = None, VALIDATION_EVALUATION_FREQUENCY = 50,
        SAVE_OUTPUT = False, SAVE_MODEL = False,
        TRAINING_OUTPUT_PATH = '', MODEL_OUTPUT_PATH = '', GRAPH_OUTPUT_PATH = '', LOGGING_OUTPUT_PATH = '../output/logging/output.log'
        ):
        '''
        Initialize the MLMTrainer instance with specified parameters.

        Parameters:
        device (str): Device to train the model on. 
        model (obj): Instance of the model to train.
        optimizer (obj): Optimizer to use for training.
        criterion (obj): Criterion to calculate the loss.
        tokenizer (obj): Tokenizer to tokenize the sentences.
        MASK_ID (int): ID for the mask token.
        MASK_RATIO (float): Ratio of tokens to mask in a sentence.
        NUM_EPOCHS (int): Number of epochs for training.
        NUM_ITERATIONS (int): Number of iterations for training.
        BATCH_SIZE (int): Size of the batch for training.
        MAX_LENGTH (int): Maximum length for the tokenized sentence.
        directory_path (str): Path of the directory containing data.
        VALIDATION_RATIO (float): Ratio of data to use for validation.
        VALIDATION_COUNT (int, optional): Count of data to use for validation. Defaults to None.
        VALIDATION_EVALUATION_FREQUENCY (int): Frequency at which to evaluate on the validation set during training.
        SAVE_OUTPUT (bool): Flag to decide whether to save training output or not. Defaults to False.
        SAVE_MODEL (bool): Flag to decide whether to save the model or not. Defaults to False.
        TRAINING_OUTPUT_PATH (str): Path to save the training output. Defaults to ''.
        MODEL_OUTPUT_PATH (str): Path to save the trained model. Defaults to ''.
        GRAPH_OUTPUT_PATH (str): Path to save the graph output. Defaults to ''.
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
        self.VALIDATION_EVALUATION_FREQUENCY = VALIDATION_EVALUATION_FREQUENCY

        self.SAVE_OUTPUT = SAVE_OUTPUT
        self.SAVE_MODEL = SAVE_MODEL
        self.TRAINING_OUTPUT_PATH = TRAINING_OUTPUT_PATH
        self.MODEL_OUTPUT_PATH = MODEL_OUTPUT_PATH
        self.GRAPH_OUTPUT_PATH = GRAPH_OUTPUT_PATH

        self.MODEL_VERSIONS = []

        self.training_output = pd.DataFrame([], columns = ['epoch', 'iteration', 'loss'])
        self.validation_output = pd.DataFrame([], columns = ['epoch', 'iteration', 'loss'])

        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.step = 0
        self.time = time.time()

        logging.basicConfig(filename = f'{LOGGING_OUTPUT_PATH}', filemode = 'w', level = logging.INFO, format = '%(message)s')

    def get_training_batch(self):
        '''
        Generate a batch of sentences for training from the directory specified in directory_path.

        Returns:
        List of sentences for training.
        '''
        try:
            file_row_count = 4160

            assert self.BATCH_SIZE < file_row_count, \
            'The batch size should not be greater than the file row count (4160)'

            sentences = []
            directory_list = os.listdir(self.directory_path)

            num_files = len(directory_list)
            validation_count = int(num_files * self.VALIDATION_RATIO) if self.VALIDATION_COUNT == None else self.VALIDATION_COUNT
            
            training_file_start_index = validation_count + math.floor((self.BATCH_SIZE * self.step) / file_row_count)

            batch_index_start = self.BATCH_SIZE * self.step - math.floor((self.BATCH_SIZE * self.step) / file_row_count) * file_row_count
            batch_index_end = batch_index_start + self.BATCH_SIZE

            # Handle file overflow
            if batch_index_end >= file_row_count:
                filenames = directory_list[training_file_start_index : training_file_start_index + 2]

                with open(os.path.join(self.directory_path, filenames[0]), 'r') as f:
                    file_sentences = f.read().lower().strip().split('\n')
                    batch_indices = np.arange(batch_index_start, file_row_count, 1)
                    batch_sentences = [file_sentences[batch_index] for batch_index in batch_indices]
                    sentences.extend([sentence.split() for sentence in batch_sentences])
                
                message = f'Batch file index: {training_file_start_index}; Row index start: {batch_index_start}; Row index end: {file_row_count}.'
                print(message)
                logging.info(message)

                batch_index_end %= file_row_count
                with open(os.path.join(self.directory_path, filenames[1]), 'r') as f:
                    file_sentences = f.read().lower().strip().split('\n')
                    batch_indices = np.arange(0, batch_index_end, 1)
                    batch_sentences = [file_sentences[batch_index] for batch_index in batch_indices]
                    sentences.extend([sentence.split() for sentence in batch_sentences])
                
                message = f'Batch file index: {training_file_start_index + 1}; Row index start: 0; Row index end: {batch_index_end}.\n'
                print(message)
                logging.info(message)

            # Base-case: No file overflow
            else:
                filenames = directory_list[training_file_start_index]

                with open(os.path.join(self.directory_path, filenames), 'r') as f:
                    file_sentences = f.read().lower().strip().split('\n')
                    batch_indices = np.arange(batch_index_start, batch_index_start + self.BATCH_SIZE, 1)
                    batch_sentences = [file_sentences[batch_index] for batch_index in batch_indices]
                    sentences.extend([sentence.split() for sentence in batch_sentences])

                message = f'Batch file index: {training_file_start_index}; Row index start: {batch_index_start}; Row index end: {batch_index_start + self.BATCH_SIZE}.\n'
                print(message)
                logging.info(message)
            
        except AssertionError as e:
            logging.info(e)

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

                message = f'Validation loss for batch: {loss.item()}'
                print(message)
                logging.info(message)

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

    def process_outputs(self):
        '''
        Processes the output dataframes for training and validation sets. 

        For the training output, it resets the index twice: the first reset is done with dropping the original 
        index, resulting in the default integer index. The second reset adds the default index as a new column 
        and resets the index again to the default integer index. The added index column is then renamed to 'step'.

        For the validation output, each row is duplicated 'VALIDATION_EVALUATION_FREQUENCY' number of times. Then,
        it resets the index in the same way as for the training output. After these steps, an 'iteration' column 
        is added to the validation output dataframe, which is derived from the 'iteration' column in the 
        training output dataframe.

        Note: The function directly modifies the 'training_output' and 'validation_output' attributes of the class.
        '''
        self.training_output = self.training_output.reset_index(drop = True).reset_index(drop = False) \
                                .rename(columns = {'index': 'step'})
        self.validation_output = self.validation_output.reindex(
            np.repeat(
                self.validation_output.index.values,
                self.VALIDATION_EVALUATION_FREQUENCY
            )
        )
        self.validation_output = self.validation_output.reset_index(drop = True).reset_index(drop = False) \
                                .rename(columns = {'index': 'step'})
        self.validation_output['iteration'] = self.training_output['iteration']

    def checkpoint(self):
        '''
        '''
        MODEL_NAME = f'{MODEL_OUTPUT_PATH}/version_{self.step}_model_{self.timestamp}.joblib'
        self.MODEL_VERSIONS.append(MODEL_NAME)

        if self.SAVE_MODEL == True:
            dump(self.model, MODEL_NAME)

        if len(self.MODEL_VERSIONS) > 5:
            os.remove(self.MODEL_VERSIONS[0])
            self.MODEL_VERSIONS = self.MODEL_VERSIONS[1:]

        if self.SAVE_OUTPUT == True:
            self.training_output.to_csv(f'{self.TRAINING_OUTPUT_PATH}/training_output_{self.timestamp}.csv', index = False)
            self.validation_output.to_csv(f'{self.TRAINING_OUTPUT_PATH}/validation_output_{self.timestamp}.csv', index = False)

    def train(self):
        '''
        Train the model using the specified optimizer and criterion. 
        Save the model and output if specified.
        '''
        try:
            assert self.VALIDATION_EVALUATION_FREQUENCY <= self.NUM_ITERATIONS, \
            'The VALIDATION_EVALUATION_FREQUENCY must be less than or equal to (<=) NUM_ITERATIONS'

            for epoch in range(self.NUM_EPOCHS):
                self.step = 0
                for iteration in range(self.NUM_ITERATIONS):
                    sentences = self.get_training_batch()

                    self.step += 1

                    inputs, targets = self.encode_sentences(sentences)
                    self.mask_inputs(inputs, sentences)

                    outputs = self.model(inputs)
                    reshaped_outputs = outputs.view(-1, outputs.size(-1)).clone()
                    desired_target = targets.view(-1).clone()
                    loss = self.criterion(reshaped_outputs, desired_target)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if self.step % 500 == 0:
                        self.checkpoint()

                    if iteration % self.VALIDATION_EVALUATION_FREQUENCY == 0:
                        message = '\n' + '#' * 25
                        print(message)
                        logging.info(message)

                        message = 'Calculate validation loss'
                        print(message)
                        logging.info(message)
                        
                        message = '#' * 25
                        print(message)
                        logging.info(message)

                        validation_loss = self.calculate_validation_loss()
                        
                        message = '#' * 25
                        print(message)
                        logging.info(message)

                        message = f'Average validation loss for all batches: {validation_loss}'
                        print(message)
                        logging.info(message)

                        message = '#' * 25
                        print(message)
                        logging.info(message)

                        message = '\n' + '#' * 25
                        print(message)
                        logging.info(message)

                        message = 'Training loss'
                        print(message)
                        logging.info(message)

                        message = '#' * 25
                        print(message)
                        logging.info(message)

                        self.validation_output = pd.concat([
                            self.validation_output,
                            pd.DataFrame({
                                'epoch': [epoch + 1],
                                'iteration': [iteration + 1],
                                'loss': [validation_loss]
                            })], ignore_index = True
                        )

                    message = f'Epoch: {epoch + 1} of {self.NUM_EPOCHS}, Iteration: {iteration + 1} of {self.NUM_ITERATIONS}, Loss: {loss.item()}'
                    print(message)
                    logging.info(message)

                    time_elapsed = time.time() - self.time
                    print(f'Time Elapsed (seconds): {time_elapsed}')

                    time_to_completion = self.NUM_EPOCHS * self.NUM_ITERATIONS / ((iteration + 1) * 3600) * time_elapsed
                    print(f'Estimated TTC (hours): {time_to_completion}')
                    
                    self.training_output = pd.concat([
                        self.training_output,
                        pd.DataFrame({
                            'epoch': [epoch + 1],
                            'iteration': [iteration + 1],
                            'loss': [loss.item()]
                        })], ignore_index = True
                    )

            self.process_outputs()

            self.checkpoint()

        except AssertionError as e:
            logging.info(e)
    
    def save_graphs(self, title = ''):
        '''
        Plots the training and validation loss as a function of steps, and saves the resulting figure.

        The method creates a line plot with 'step' on the x-axis and 'loss' on the y-axis for both the training
        and validation outputs. The training data is plotted in blue and labelled 'Train', and the validation data 
        is plotted in orange and labelled 'Validation'. 

        The resulting plot is saved as a PNG file in the path specified by the 'GRAPH_OUTPUT_PATH' attribute 
        of the class, with the filename 'training_validation_curves_{timestamp}.png', where '{timestamp}' is 
        replaced by the current value of the 'timestamp' attribute of the class.

        Note: This method directly uses the 'training_output' and 'validation_output' attributes of the class.
        '''
        plt.figure(figsize = (10, 6))

        sns.lineplot(data = self.training_output, x = 'step', y = 'loss', color = 'tab:blue', label = 'Train')
        sns.lineplot(data = self.validation_output, x = 'step', y = 'loss', color = 'tab:orange', label = 'Validation')

        plt.title(f'{title}')

        plt.xlabel(f'Step (Batch Size = {self.BATCH_SIZE})')
        plt.ylabel('Cross Entropy Loss')

        min_step = min(self.training_output['step'].min(), self.validation_output['step'].min())
        max_step = max(self.training_output['step'].max(), self.validation_output['step'].max())

        xticks = np.linspace(min_step, max_step, num = 10, dtype = int)
        plt.xticks(xticks)

        line_steps = np.arange(self.NUM_ITERATIONS, max_step + 1, self.NUM_ITERATIONS)
        for step in line_steps:
            plt.axvline(x = step, color = 'r', linestyle = 'dotted')

        plt.legend()

        plt.savefig(f'{self.GRAPH_OUTPUT_PATH}/linear_scale/training_validation_curves_{self.timestamp}.png')

        plt.yscale('log')
        plt.savefig(f'{self.GRAPH_OUTPUT_PATH}/log_scale/log_scale_training_validation_curves_{self.timestamp}.png')

if __name__ == '__main__':
    torch.cuda.empty_cache()

    np.random.seed(1234)
    DIRECTORY_PATH = '../data/masking/openwebtext/openwebtext'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    VOCAB_SIZE = 30522
    EMBED_DIM = 768
    NUM_HEADS = 12
    FF_DIM = 3072
    NUM_BLOCKS = 3 # TODO: Increase on GPU
    DROPOUT = 0.2
    SEQ_LENGTH = 64 # TODO: Increase on GPU
    MASK_RATIO = 0.15
    EXPANSION_FACTOR = 2
    LEARNING_RATE = 1e-2
    MODEL_VERSION = 2
    MASK_ID = 103 # NOTE: Specific to BERTTokenizerFast
    NUM_EPOCHS = 1
    BATCH_SIZE = 32 # TODO: Increase on GPU
    VALIDATION_RATIO = 0.05 # NOTE: Used if VALIDATION_COUNT = None
    VALIDATION_COUNT = 1 # NOTE: Overrides validation ratio; represents number of files used for validation calculation
    VALIDATION_EVALUATION_FREQUENCY = 1000 # NOTE: How often to evaluate the validation set in iterations
    NUM_ITERATIONS = int(1500000 * 32 / BATCH_SIZE * (1 - VALIDATION_RATIO)) if VALIDATION_COUNT == None \
                    else int(1500000 * 32 / BATCH_SIZE - VALIDATION_COUNT)

    NUM_ITERATIONS = math.floor(NUM_ITERATIONS / 128) # TODO: Remove line later

    SAVE_OUTPUT = True
    SAVE_MODEL = True
    TRAINING_OUTPUT_PATH = '../output'
    MODEL_OUTPUT_PATH = '../artifacts/MLM'
    GRAPH_OUTPUT_PATH = '../output/illustrations'

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
        DIRECTORY_PATH, VALIDATION_RATIO, VALIDATION_COUNT, VALIDATION_EVALUATION_FREQUENCY,
        SAVE_OUTPUT, SAVE_MODEL,
        TRAINING_OUTPUT_PATH, MODEL_OUTPUT_PATH, GRAPH_OUTPUT_PATH
    )

    trainer.train()

    trainer.save_graphs('Training/Validation Loss Curves for Masked Language Model')
