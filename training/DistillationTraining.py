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
        FREEZE_EARLY_LAYERS,
        NUM_EPOCHS, NUM_ITERATIONS, BATCH_SIZE, MAX_LENGTH,
        teacher_model = 'mrm8488/bert-tiny-finetuned-enron-spam-detection',
        classification_data_path = '../data/classification/Enron_spam'
    ):
        '''
        '''
        super().__init__()
        self.device = device
        self.model = model
        self.optimizer = optimizer

        self.tokenizer = tokenizer

        self.FREEZE_EARLY_LAYERS = FREEZE_EARLY_LAYERS

        self.NUM_EPOCHS = NUM_EPOCHS
        self.NUM_ITERATIONS = NUM_ITERATIONS
        self.BATCH_SIZE = BATCH_SIZE
        self.MAX_LENGTH = MAX_LENGTH

        self.teacher_model = teacher_model
        self.classification_data_path = classification_data_path

        np.random.seed(1234)
        self.enron_data_df = self.prepare_enron_data()

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

    def get_enron_data(self):
        '''
        '''
        start_index = self.step * self.BATCH_SIZE
        end_index = (self.step + 1) * self.BATCH_SIZE

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

    def train(self):
        '''
        '''
        teacher_classifier = AutoModelForSequenceClassification.from_pretrained(self.teacher_model)

        if self.FREEZE_EARLY_LAYERS == True:
            self.freeze_layers()

        for epoch in range(self.NUM_EPOCHS):
            for iteration in range(self.NUM_ITERATIONS):
                # batch_size is explicitly mentioned here to handle end of dataframe
                contents, targets, batch_size = self.get_enron_data()

                self.step += 1

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

                message = f'Epoch: {epoch + 1} of {self.NUM_EPOCHS}, Iteration: {iteration + 1} of {self.NUM_ITERATIONS}, Loss: {loss.item()}'
                print(message)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        if self.FREEZE_EARLY_LAYERS == True:
            self.unfreeze_layers()

if __name__ == '__main__':
    np.random.seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MODEL_PATH = '../artifacts/MLM/model_2023-07-28_17-59-18.joblib'
    model = load(MODEL_PATH)
    
    BATCH_SIZE = 256
    NUM_EPOCHS = 10
    NUM_ITERATIONS = math.floor(33716 / BATCH_SIZE)
    LEARNING_RATE = 1e-2
    EMBED_DIM = 768
    SEQ_LENGTH = 16

    FREEZE_EARLY_LAYERS = True

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
        FREEZE_EARLY_LAYERS,
        NUM_EPOCHS, NUM_ITERATIONS, BATCH_SIZE, SEQ_LENGTH
    )

    trainer.train()
