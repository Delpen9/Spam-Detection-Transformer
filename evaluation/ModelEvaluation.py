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
from torch.utils.data import DataLoader, Dataset
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

class SpamDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.data = self.generate_data()

    def generate_data(self):
        data = []
        for idx in range(len(self.X)):
            data.append((self.X[idx], self.y[idx]))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == "__main__":
    np.random.seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    DATA_DIRECTORY = '../data/classification/LingSpam/messages.csv'
    test_df = pd.read_csv(DATA_DIRECTORY)

    test_df['subject'] = test_df['subject'].fillna('-')
    test_df['message'] = test_df['message'].fillna('-')
    test_df['text'] = test_df['subject'] + " " + test_df['message']

    dataset_size = test_df.shape[0]

    MODEL_PATH = '../artifacts/Fine-tuned/distilled_model_2023-07-30_18-50-34.joblib'
    model = load(MODEL_PATH).model

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    dataset = SpamDataset(test_df['text'], test_df['label'])
    test_loader = DataLoader(dataset, batch_size = 32)

    model.eval()

    with torch.no_grad():
        y_true = torch.Tensor()
        y_pred = torch.Tensor()

        for batch_idx, test_data in enumerate(test_loader):
            inputs = test_data[0]
            inputs = tokenizer(
                inputs,
                truncation = True,
                padding = True,
                return_tensors = 'pt'
            )['input_ids']

            targets = test_data[1]

            output = model(inputs..long())
            output = F.softmax(output, dim = -1)

            y_true = torch.cat((y_true, targets), 0)
            y_pred = torch.cat((y_pred, torch.argmax(output, dim = -1)), 0)
                
        test_acc = (y_true == y_pred).float().sum()
        test_acc /= dataset_size
        test_acc *= 100
        print(f'Test Accuracy: {test_acc}%')