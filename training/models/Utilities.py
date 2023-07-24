import pandas as pd
import os
import re
from math import ceil
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords


snow_stemmer = SnowballStemmer(language='english')
stop_words = set(stopwords.words('english'))

def load_sample_data():

    df = pd.DataFrame(columns=['text', 'sentiment', 'truthful'])
    paths = [
        '../data/op_spam_v1.4/positive_polarity/truthful_from_TripAdvisor/fold1/t_hilton_1.txt',
        '../data/op_spam_v1.4/positive_polarity/truthful_from_TripAdvisor/fold1/t_hilton_2.txt',
        '../data/op_spam_v1.4/positive_polarity/truthful_from_TripAdvisor/fold1/t_hilton_3.txt',
        '../data/op_spam_v1.4/positive_polarity/truthful_from_TripAdvisor/fold1/t_hilton_4.txt',
        '../data/op_spam_v1.4/positive_polarity/truthful_from_TripAdvisor/fold1/t_hilton_5.txt'
    ]
    for file in paths:
        f = open(file, 'r')
        data = f.read()
        data = {'text': data, 'sentiment': 'positive', 'truthful': True}
        df = df.append(data, ignore_index=True)

    return df


def load_data():
    """
    Load as dataframe: col1: txt, col2: label
    Combine all folds together
    Output: return df
    """

    df = pd.DataFrame(columns=['text', 'sentiment', 'truthful'])
    sentiment = ['positive_polarity', 'negative_polarity']
    truthful = ['truthful_from_Web', 'deceptive_from_MTurk', 'truthful_from_TripAdvisor']

    path = '../data/op_spam_v1.4/'
    dir_list = os.listdir(path)
    for dir1 in dir_list:
        if dir1 in sentiment:
            if dir1 == 'positive_polarity':
                sent = 'positive'
            else:
                sent = 'negative'
            dir1_list = os.listdir(f'{path}/{dir1}')
            for dir2 in dir1_list:
                if dir2 in truthful:
                    if dir2 == 'truthful_from_Web' or dir2 == 'truthful_from_TripAdvisor':
                        truth = 1
                    else:
                        truth = 0
                    dir2_list = os.listdir(f'{path}/{dir1}/{dir2}')
                    for dir3 in dir2_list:
                        if 'fold' in dir3:
                            file_list = os.listdir(f'{path}/{dir1}/{dir2}/{dir3}')
                            for file in file_list:
                                new_path = f'{path}/{dir1}/{dir2}/{dir3}/{file}'
                                # data = pd.read_csv(new_path)
                                f = open(new_path, 'r')
                                data = f.read()
                                data = {'text': data, 'sentiment': sent, 'truthful': truth}
                                df.loc[len(df.index)] = data
    return df


def process_text(corpus, mask=False, mask_rate=0.15):
    
    MASK_TOKEN = "<mask>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    NUMBER_TOKEN = "<num>"

    processed_corpus = []

    for x in corpus:
        x = x.lower()
        x = re.sub(' +', ' ', x)
        x = re.sub('[^a-zA-Z]', ' ', x)
        x = ' '.join([snow_stemmer.stem(word) for word in x.split() if word not in stop_words])
        x = re.sub('\d', NUMBER_TOKEN, x)
        x = x.split()
        if mask:
            mask_idx = np.random.permutation(len(x))[:ceil(len(x)*mask_rate)]
            for idx in mask_idx:
                x[idx] = MASK_TOKEN

        x.insert(0, SOS_TOKEN)
        x.append(EOS_TOKEN)
        x = ' '.join(x)
        processed_corpus.append(x)
    
    return np.array(processed_corpus)


def shuffleData():
    pass


def splitData():
    pass