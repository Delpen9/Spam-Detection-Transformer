import warnings
warnings.filterwarnings('ignore', category = UserWarning)

# Tokenization
from transformers import BertTokenizerFast, AutoModelForSequenceClassification, pipeline

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
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math

# Serialize and De-Serialize Model
from joblib import load, dump

# Graphing
import matplotlib.pyplot as plt
import seaborn as sns

# Miscellaneous
from datetime import datetime
import copy

class DistillationTrainer:
    def __init__(
        self,
        device, model, optimizer,
        tokenizer,
        FREEZE_EARLY_LAYERS, VALIDATION_DATA_PERCENTAGE, VALIDATION_EVALUATION_FREQUENCY,
        NUM_EPOCHS, NUM_ITERATIONS, BATCH_SIZE, MAX_LENGTH,
        SAVE_OUTPUT, SAVE_MODEL,
        TRAINING_OUTPUT_PATH, MODEL_OUTPUT_PATH, GRAPH_OUTPUT_PATH,
        teacher_model = 'mrm8488/bert-tiny-finetuned-enron-spam-detection',
        classification_data_path = '../data/classification/Enron_spam',
    ):
        '''
        '''
        super().__init__()
        self.device = device
        self.model = model
        self.optimizer = optimizer

        self.tokenizer = tokenizer

        self.FREEZE_EARLY_LAYERS = FREEZE_EARLY_LAYERS
        self.VALIDATION_DATA_PERCENTAGE = VALIDATION_DATA_PERCENTAGE
        self.VALIDATION_EVALUATION_FREQUENCY = VALIDATION_EVALUATION_FREQUENCY

        self.NUM_EPOCHS = NUM_EPOCHS
        self.NUM_ITERATIONS = NUM_ITERATIONS
        self.BATCH_SIZE = BATCH_SIZE
        self.MAX_LENGTH = MAX_LENGTH

        self.SAVE_OUTPUT = SAVE_OUTPUT
        self.SAVE_MODEL = SAVE_MODEL
        self.TRAINING_OUTPUT_PATH = TRAINING_OUTPUT_PATH
        self.MODEL_OUTPUT_PATH = MODEL_OUTPUT_PATH
        self.GRAPH_OUTPUT_PATH = GRAPH_OUTPUT_PATH

        self.teacher_model = teacher_model
        self.classification_data_path = classification_data_path

        np.random.seed(1234)
        self.enron_data_df = self.prepare_enron_data()

        self.training_output = pd.DataFrame([], columns = ['epoch', 'iteration', 'loss'])
        self.validation_output = pd.DataFrame([], columns = ['epoch', 'iteration', 'loss'])

        self.step = 0

    def prepare_enron_data(self):
        enron_groups = [f'enron{i}' for i in range(1, 7)]

        ham_enron_directories = np.array([
            f'{self.classification_data_path}/{group}/ham'
            for group in enron_groups
        ])

        spam_enron_directories = np.array([
            f'{self.classification_data_path}/{group}/spam'
            for group in enron_groups
        ])

        ham_enron_files = []
        for ham_enron_directory in ham_enron_directories:
            new_directories = [fr"{ham_enron_directory}/{filepath.decode('utf-8')}" for filepath in os.listdir(ham_enron_directory)]
            ham_enron_files.extend(new_directories)
        ham_enron_files = [str(filename) for filename in ham_enron_files]
        ham_enron_files = np.array(ham_enron_files)

        spam_enron_files = []
        for spam_enron_directory in spam_enron_directories:
            new_directories = [fr"{spam_enron_directory}/{filepath.decode('utf-8')}" for filepath in os.listdir(spam_enron_directory)]
            spam_enron_files.extend(new_directories)
        spam_enron_files = [str(filename) for filename in spam_enron_files]
        spam_enron_files = np.array(spam_enron_files)

        ham_enron_np = np.vstack((ham_enron_files, np.zeros(len(ham_enron_files)))).T
        ham_enron_df = pd.DataFrame(ham_enron_np, columns = ['Path', 'Target'])

        spam_enron_np = np.vstack((spam_enron_files, np.ones(len(spam_enron_files)))).T
        spam_enron_df = pd.DataFrame(spam_enron_np, columns = ['Path', 'Target'])

        endron_data_df = pd.concat((ham_enron_df, spam_enron_df), axis = 0).sample(frac = 1).astype({
            'Path': str,
            'Target': float
        })
        return endron_data_df

    def get_enron_training_data(self):
        '''
        '''
        validation_end_index = math.floor(33716 * self.VALIDATION_DATA_PERCENTAGE)
        start_index = validation_end_index + self.step * self.BATCH_SIZE
        end_index = validation_end_index + (self.step + 1) * self.BATCH_SIZE

        sample_rows_df = self.enron_data_df.iloc[start_index : end_index]

        contents = []
        for _, row in sample_rows_df.iterrows():
            filename = row['Path']
            with open(filename, 'r', errors = 'replace') as f:
                content = f.read()
                contents.append(content)

        targets = torch.tensor(sample_rows_df['Target'].to_numpy().flatten()).long()

        contents = [str(content) for content in contents]

        return (contents, targets, len(targets))

    def get_enron_validation_data(self):
        '''
        '''
        start_index = 0
        end_index = math.floor(33716 * self.VALIDATION_DATA_PERCENTAGE)

        sample_rows_df = self.enron_data_df.iloc[start_index : end_index]

        contents = []
        for _, row in sample_rows_df.iterrows():
            filename = row['Path']
            with open(filename, 'r', errors = 'replace') as f:
                content = f.read()
                contents.append(content)

        targets = torch.tensor(sample_rows_df['Target'].to_numpy().flatten()).long()

        contents = [str(content) for content in contents]

        return (contents, targets)

    def chunks(self, contents, targets):
        '''
        Generator to yield chunks of sentences of size BATCH_SIZE.

        Parameters:
        sentences: List of sentences to split into chunks.

        Returns:
        Generator yielding chunks of sentences of size BATCH_SIZE.
        '''
        for i in range(0, len(contents), self.BATCH_SIZE):
            chunk = (contents[i : i + self.BATCH_SIZE], targets[i : i + self.BATCH_SIZE])
            if len(chunk[0]) == self.BATCH_SIZE:
                yield chunk

    def calculate_validation_loss(self, teacher_classifier):
        '''
        Calculate validation loss for the current state of the model.

        Returns:
        Validation loss.
        '''
        self.model.eval()

        validation_loss = 0.0

        with torch.no_grad():
            contents, targets = self.get_enron_validation_data()

            for batch in self.chunks(contents, targets):
                batch_contents = batch[0]
                batch_targets = batch[1]
                loss = self.calculate_loss(batch_contents, batch_targets, self.BATCH_SIZE, teacher_classifier)

                message = f'Validation loss for batch: {loss.item()}'
                print(message)

                validation_loss += loss.item()

        validation_loss /= float(math.floor(len(contents) / self.BATCH_SIZE))

        return validation_loss

    def freeze_layers(self):
        '''
        '''
        for param in self.model.model.model.parameters():
            param.requires_grad = False

        self.model.model.fc.requires_grad = True
    
    def unfreeze_layers(self):
        '''
        '''
        for param in self.model.model.model.parameters():
            param.requires_grad = True

    def calculate_loss(self, contents, targets, batch_size, teacher_classifier):
        contents = [content.split() for content in contents]

        input_list = []
        for sample_idx in range(batch_size):
            try:
                input_ids = tokenizer(
                    contents[sample_idx],
                    padding = 'max_length',
                    truncation = True,
                    max_length = self.MAX_LENGTH,
                    return_tensors = 'pt',
                    is_split_into_words = True
                )['input_ids'][0]

                input_list.append(input_ids)
            except:
                pass

        inputs = torch.stack(input_list).to(self.device)

        teacher_probabilities = F.softmax(teacher_classifier(inputs)['logits'])

        loss = self.model.loss(inputs, teacher_probabilities, targets)

        return loss

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

    def train(self):
        '''
        '''
        teacher_classifier = AutoModelForSequenceClassification.from_pretrained(self.teacher_model)

        if self.FREEZE_EARLY_LAYERS == True:
            self.freeze_layers()

        for epoch in range(self.NUM_EPOCHS):
            for iteration in range(self.NUM_ITERATIONS):
                # batch_size is explicitly mentioned here to handle end of dataframe
                contents, targets, batch_size = self.get_enron_training_data()

                loss = self.calculate_loss(contents, targets, batch_size, teacher_classifier)

                message = f'Epoch: {epoch + 1} of {self.NUM_EPOCHS}, Iteration: {iteration + 1} of {self.NUM_ITERATIONS}, Loss: {loss.item()}'
                print(message)

                self.training_output = pd.concat([
                    self.training_output,
                    pd.DataFrame({
                        'epoch': [epoch + 1],
                        'iteration': [iteration + 1],
                        'loss': [loss.item()]
                    })], ignore_index = True
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.step += 1
                if self.step % self.VALIDATION_EVALUATION_FREQUENCY == 0:
                    validation_loss = self.calculate_validation_loss(teacher_classifier)

                    self.validation_output = pd.concat([
                        self.validation_output,
                        pd.DataFrame({
                            'epoch': [epoch + 1],
                            'iteration': [iteration + 1],
                            'loss': [validation_loss]
                        })], ignore_index = True
                    )
        
        if self.FREEZE_EARLY_LAYERS == True:
            self.unfreeze_layers()

        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        self.process_outputs()

        if self.SAVE_OUTPUT == True:
            self.training_output.to_csv(f'{self.TRAINING_OUTPUT_PATH}/distillation_training_output_{self.timestamp}.csv', index = False)
            self.validation_output.to_csv(f'{self.TRAINING_OUTPUT_PATH}/distillation_validation_output_{self.timestamp}.csv', index = False)

        if self.SAVE_MODEL == True:
            dump(self.model, f'{MODEL_OUTPUT_PATH}/distilled_model_{self.timestamp}.joblib')

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
        plt.ylabel('Distillation Loss')

        min_step = min(self.training_output['step'].min(), self.validation_output['step'].min())
        max_step = max(self.training_output['step'].max(), self.validation_output['step'].max())

        xticks = np.linspace(min_step, max_step, num = 10, dtype = int)
        plt.xticks(xticks)

        line_steps = np.arange(self.NUM_ITERATIONS, max_step + 1, self.NUM_ITERATIONS)
        for step in line_steps:
            plt.axvline(x = step, color = 'r', linestyle = 'dotted')

        plt.legend()

        plt.savefig(f'{self.GRAPH_OUTPUT_PATH}/linear_scale/distillation_training_validation_curves_{self.timestamp}.png')

        plt.yscale('log')
        plt.savefig(f'{self.GRAPH_OUTPUT_PATH}/log_scale/distillation_log_scale_training_validation_curves_{self.timestamp}.png')

if __name__ == '__main__':
    np.random.seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MODEL_PATH = '../artifacts/MLM/model_2023-07-28_17-59-18.joblib'
    model = load(MODEL_PATH)
    
    LEARNING_RATE = 1e-2
    EMBED_DIM = 768
    SEQ_LENGTH = 16

    VALIDATION_DATA_PERCENTAGE = 0.1
    VALIDATION_EVALUATION_FREQUENCY = 20
    BATCH_SIZE = 256
    NUM_EPOCHS = 1
    NUM_ITERATIONS = math.floor(33716 * (1 - VALIDATION_DATA_PERCENTAGE) / BATCH_SIZE)

    FREEZE_EARLY_LAYERS = True

    SAVE_OUTPUT = True
    SAVE_MODEL = True
    TRAINING_OUTPUT_PATH = '../output'
    MODEL_OUTPUT_PATH = '../artifacts/Fine-tuned'
    GRAPH_OUTPUT_PATH = '../output/illustrations'

    MODEL_VERSION = 2

    if MODEL_VERSION == 1:
        transformerEncoder = copy.deepcopy(model)
    elif MODEL_VERSION == 2:
        transformerEncoder = copy.deepcopy(model.model)

    spamDetectionModel = SpamDetectionModel(transformerEncoder, EMBED_DIM, n_classes = 2)
    model = DistilledFromTinyBert(spamDetectionModel)
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    trainer = DistillationTrainer(
        device, model, optimizer,
        tokenizer,
        FREEZE_EARLY_LAYERS, VALIDATION_DATA_PERCENTAGE, VALIDATION_EVALUATION_FREQUENCY,
        NUM_EPOCHS, NUM_ITERATIONS, BATCH_SIZE, SEQ_LENGTH,
        SAVE_OUTPUT, SAVE_MODEL,
        TRAINING_OUTPUT_PATH, MODEL_OUTPUT_PATH, GRAPH_OUTPUT_PATH
    )

    trainer.train()

    trainer.save_graphs('Training/Validation Loss Curves for Distilled Model')
