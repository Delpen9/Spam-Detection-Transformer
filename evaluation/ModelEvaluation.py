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
from sklearn.metrics import roc_curve, auc

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
    model = load(MODEL_PATH).model.to(device)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    dataset = SpamDataset(test_df['text'], test_df['label'])
    test_loader = DataLoader(dataset, batch_size = 32)

    SEQ_LENGTH = 64

    model.eval()

    with torch.no_grad():
        y_true = torch.Tensor().to(device)
        y_pred = torch.Tensor().to(device)

        for batch_idx, test_data in enumerate(test_loader):
            inputs = test_data[0]
            inputs = tokenizer(
                inputs,
                padding = 'max_length',
                truncation = True,
                max_length = SEQ_LENGTH,
                return_tensors = 'pt'
            )['input_ids'].long().to(device)
            
            targets = test_data[1].to(device)

            output = model(inputs)
            y_prob = F.softmax(output, dim = -1)

            y_true = torch.cat((y_true, targets), 0)
            y_pred = torch.cat((y_pred, torch.argmax(y_prob, dim = -1)), 0)
                
        test_acc = (y_true == y_pred).float().sum()
        test_acc /= dataset_size
        test_acc *= 100
        print(f'Test Accuracy: {test_acc}%')

    y_true_np = y_true.cpu().numpy()
    y_prob_np = y_prob[:, 1].cpu().numpy()

    # compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true_np, y_prob_np)
    roc_auc = auc(fpr, tpr)

    # plot settings
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 8))

    # plot the ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    # show the plot
    plt.savefig('roc_chart.png')