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
        NUM_EPOCHS, NUM_ITERATIONS, BATCH_SIZE, MAX_LENGTH,
        teacher_model = 'mrm8488/bert-tiny-finetuned-enron-spam-detection'
    ):
    '''
    '''
    super().__init__()
    self.device = device
    self.model = model
    self.optimizer = optimizer

    self.tokenizer = tokenizer

    self.NUM_EPOCHS = NUM_EPOCHS
    self.NUM_ITERATIONS = NUM_ITERATIONS
    self.BATCH_SIZE = BATCH_SIZE
    self.MAX_LENGTH = MAX_LENGTH

    self.teacher_model = teacher_model

    def get_enron_data(self):
        '''
        '''
        sentences = None
        targets = None
        # sentences = [
        #     'We are very happy that we bought this car.',
        #     'aquila dave marks just got a call from someone at aquila saying they disliked trading on enrononline anymore . - r'
        # ]
        return (sentences, targets)

    def train(self):
        '''
        '''
        teacher = AutoModelForSequenceClassification.from_pretrained(self.teacher_model)
        teacher_classifier = pipeline('text-classification', model = teacher, tokenizer = self.tokenizer)

        for epoch in range(self.NUM_EPOCHS):
            for iteration in range(self.NUM_ITERATIONS):
                sentences, targets = get_enron_data()

                teacher_outputs = classifier(sentences)

                teacher_probabilities = []
                for result in results:
                    if result['label'] == 'LABEL_0':
                        prob = [result['score'], 1 - result['score']]
                    elif result['label'] == 'LABEL_1':
                        prob = [1 - result['score'], result['score']]

                    teacher_probabilities.append(prob)
                
                teacher_probabilities = torch.tensor(teacher_probabilities)

                sentences = [sentence.split() for sentence in sentences]

                input_list = []
                for sample_idx in range(self.BATCH_SIZE):
                    input_ids = tokenizer(
                        sentences[sample_idx],
                        padding = 'max_length',
                        truncation = True,
                        max_length = self.MAX_LENGTH,
                        return_tensors = 'pt',
                        is_split_into_words = True
                    )['input_ids'][0]

                    input_list.append(input_ids)

                inputs = torch.stack(input_list).to(self.device)

                loss = self.model.loss(inputs, teacher_probabilities, targets)

                message = f'Epoch: {epoch + 1} of {self.NUM_EPOCHS}, Iteration: {iteration + 1} of {self.NUM_ITERATIONS}, Loss: {loss.item()}'
                print(message)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

if __name__ == '__main__':
    np.random.seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MODEL_PATH = '../artifacts/MLM/model_2023-07-28_17-59-18.joblib'
    model = load(MODEL_PATH)
    
    NUM_EPOCHS = 10
    NUM_ITERATIONS = 10
    BATCH_SIZE = 256
    SEQ_LENGTH = 16

    DATA_PATH = '../data/classification/Enron_spam'

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
        NUM_EPOCHS, NUM_ITERATIONS, BATCH_SIZE, SEQ_LENGTH
    )
