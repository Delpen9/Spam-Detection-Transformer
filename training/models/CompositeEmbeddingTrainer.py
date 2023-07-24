import os
import re
import pandas as pd
import CompositeEmbeddings
import Utilities


class CompositeTrainer(object):

    def __init__(self, corpus, targets, embedding_dim, topics, num_epochs, learning_rate, fit=True):
        self.corpus = corpus
        self.targets = targets
        self.embedding_dim = embedding_dim
        self.topics = topics
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.fit = fit

    def train_embeddings(self):

        embeddings = CompositeEmbeddings.train_composite(self.corpus, self.targets, self.embedding_dim, self.topics,
                                                          self.num_epochs, self.learning_rate, self.fit)
        return embeddings


if __name__ == "__main__":
 
    print('Loading Data...')
    data = Utilities.load_data()

    # data.to_excel('data/op_spam_data.xlsx', index=False)

    processed_text = Utilities.process_text(data['text'].values)
    masked_text = Utilities.process_text(data['text'].values, mask=True)
    targets = data['truthful'].values

    embedding_dim = 256
    num_epochs = 100
    learning_rate = 0.01
    topics = 16
    fit = True

    trainer = CompositeTrainer(masked_text, targets, embedding_dim, topics, num_epochs, learning_rate, fit)
    embeddings = trainer.train_embeddings()

    print(embeddings)
