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

if __name__ == '__main__':
    np.random.seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    EMBED_DIM = 768
    BATCH_SIZE = 256

    MODEL_PATH = '../artifacts/MLM/model_2023-07-27_20-32-20.joblib'
    model = load(MODEL_PATH)

    MODEL_VERSION = 2

    if MODEL_VERSION == 1:
        transformerEncoder = copy.deepcopy(model)
    elif MODEL_VERSION == 2:
        transformerEncoder = copy.deepcopy(model.model)

    spamDetectionModel = SpamDetectionModel(transformerEncoder, EMBED_DIM, n_classes = 2)

    distillationModel = DistilledFromTinyBert(spamDetectionModel)

    model_name = 'mrm8488/bert-tiny-finetuned-enron-spam-detection'
    tinyBERTModel = AutoModelForSequenceClassification.from_pretrained(model_name)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    classifier = pipeline('text-classification', model = tinyBERTModel, tokenizer = tokenizer)
    results = classifier(['We are very happy that we bought this car.', 'aquila dave marks just got a call from someone at aquila saying they disliked trading on enrononline anymore . - r'])
    
    probabilities = []
    for result in results:
        if result['label'] == 'LABEL_0':
            prob = [result['score'], 1 - result['score']]
        elif result['label'] == 'LABEL_1':
            prob = [1 - result['score'], result['score']]

        probabilities.append(prob)

    print(probabilities)


