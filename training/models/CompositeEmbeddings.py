import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from collections import Counter
from torchtext.vocab import vocab
from torch.nn.utils.rnn import pad_sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np

import yaml
import pickle


class CompositeDataset(Dataset):
    

    def __init__(self, corpus, embedding_size=128, topics=10, fit=True):
        
        self.corpus = corpus
        self.embedding_size = embedding_size
        self.topics = topics
        self.fit = fit
        self.data = self.generate_data()


    def apply_tfidf(self, corpus=None):
        
        if self.fit:
            vectorizer = TfidfVectorizer()
            vectorizer.fit(corpus)
            # with open('models/tfidf_model.pickle', 'wb') as f:
            #     pickle.dump(vectorizer, f)
        else:
            with open('models/tfidf_model.pickle', 'rb') as f:
                vectorizer = pickle.load(f)
        tfidf = vectorizer.transform(corpus)
        
        return tfidf


    def apply_tsvd(self, vectors=None, magnitude=1):
        
        if self.fit:
            decomposer = TruncatedSVD(n_components=self.embedding_size)
            decomposer.fit(vectors)
            # with open('models/tsvd_model.pickle', 'wb') as f:
            #     pickle.dump(decomposer, f)
        else:
            with open('models/tsvd_model.pickle', 'rb') as f:
                decomposer = pickle.load(f)
        tsvd = decomposer.transform(vectors)
        tsvd = tsvd + 1
        tsvd = tsvd * 10**magnitude
        tsvd = tsvd.astype(int)
        
        return tsvd
    

    def apply_docvec(self):
        
        if self.fit:
            documents = [TaggedDocument(self.corpus[i], [i]) for i in range(len(self.corpus))]
            doc2vec_model = Doc2Vec(documents, vector_size=256, min_count=2)
            # with open('models/docvec_model.pickle', 'wb') as f:
            #     pickle.dump(doc2vec_model, f)
        else:
            with open('models/docvec_model.pickle', 'rb') as f:
                doc2vec_model = pickle.load(f)
        doc2vecs = []
        for sentence in self.corpus:
            doc2vecs.append(doc2vec_model.infer_vector(sentence.split()))
        doc2vecs = np.absolute(np.array(doc2vecs))
        
        return doc2vecs


    def apply_lda(self, doc2vecs=None):
        
        if self.fit:
            lda_decomposer = LatentDirichletAllocation(n_components=self.topics)
            lda_decomposer.fit(doc2vecs)
            # with open('models/lda_model.pickle', 'wb') as f:
            #     pickle.dump(lda_decomposer, f)
        else:
            with open('models/lda_model.pickle', 'rb') as f:
                lda_decomposer = pickle.load(f)
        lda = lda_decomposer.transform(doc2vecs)
        lda = np.argmax(lda, axis=1)
        
        return lda


    def build_vocab(self):
        
        counter = Counter()
        f = self.corpus
        for string_ in f:
            counter.update(string_.split())
        
        return vocab(counter, specials=['<pad>', '<sos>', '<eos>', '<mask>'], min_freq=1)


    def generate_data(self):
        
        print('Tfidf...')
        tfidf = self.apply_tfidf(self.corpus)
        print('Tsvd...')
        tsvd = self.apply_tsvd(vectors=tfidf)
        tsvd_tensors = torch.from_numpy(tsvd)
        print('Doc2Vec...')
        doc2vecs = self.apply_docvec()
        print('LDA...')
        lda = self.apply_lda(doc2vecs=doc2vecs)
        _vocab = self.build_vocab()
        model_data = []
        MAX_LEN = len(max(self.corpus, key=len))
        fix = torch.ones(MAX_LEN)
        print('Generating Data...')
        for idx in range(len(self.corpus)):
            sentence_tensor = torch.tensor([_vocab[token] for token in self.corpus[idx].split()], dtype=torch.long)
            tsvd_tensor = tsvd_tensors[idx]
            lda_tensor = lda[idx]
            two = pad_sequence([sentence_tensor, fix], padding_value=_vocab['<pad>'])
            sentence_tensor = two[:, 0, ]
            two = pad_sequence([tsvd_tensor, fix], padding_value=torch.max(tsvd_tensor))
            tsvd_tensor = two[:, 0, ]
            lda_tensor = np.array([lda_tensor for i in range(MAX_LEN)])
            lda_tensor = torch.from_numpy(lda_tensor)
            model_data.append((sentence_tensor, tsvd_tensor, lda_tensor))

        config = {}
        config['composite_emb_vocab_size'] = _vocab.__len__()
        config['composite_emb_max_len'] = MAX_LEN
        with open('../config/composite_config.yaml', 'w') as f:
            yaml.dump(config, f)
        
        return model_data


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.data[idx]


class CompositeEmbedding(nn.Module):
    
    def __init__(self, input_size, embedding_dim, max_length, device):
        super(CompositeEmbedding, self).__init__()

        self.device = device

        self.max_length = max_length
        self.sentence_embedding = nn.Embedding(input_size, embedding_dim, device=self.device)
        self.tsvd_embedding = nn.Embedding(input_size, embedding_dim, device=self.device)
        self.lda_embedding = nn.Embedding(input_size, embedding_dim, device=self.device)
        self.position_embedding = nn.Embedding(max_length, embedding_dim, device=self.device)

        self.linear = nn.Linear(in_features=embedding_dim, out_features=2, device=self.device)
        

    def forward(self, inputs):
        
        sentence_embedding = self.sentence_embedding(inputs[0])
        tsvd_embedding = self.tsvd_embedding(inputs[1])
        lda_embedding = self.lda_embedding(inputs[2])
        position = torch.arange(self.max_length, device=self.device)
        position = self.position_embedding(position)
        for i in range(len(inputs)):
            sentence_embedding[i] = sentence_embedding[i] #+ position + tsvd_embedding[i] + lda_embedding[i]
        
        output = self.linear(sentence_embedding)
        output = F.softmax(output)
        
        return output
    

def batch_target(batch_targets):
    batch_targets = np.array(batch_targets)
    batch_targets = torch.from_numpy(batch_targets)
    batch_targets = F.one_hot(batch_targets, num_classes=2)
    return batch_targets


def train_composite(corpus, targets, embedding_dim, topics, num_epochs, learning_rate, fit=True):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 8

    dataset = CompositeDataset(corpus, embedding_dim, topics, fit)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=lambda x: tuple(_x.to(device) for _x in default_collate(x)))

    with open('../config/composite_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model = CompositeEmbedding(input_size=config['composite_emb_vocab_size'],
                               embedding_dim=embedding_dim, max_length=config['composite_emb_max_len'],
                               device=device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    model.to(device)

    for epoch in range(num_epochs):
        total_loss = 0
        i = 0
        for batch_idx, model_input in enumerate(dataloader):
            
            batch_targets = targets[i:i+BATCH_SIZE]
            i += BATCH_SIZE
            batch_targets = batch_target(batch_targets)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            output = model(model_input)
            # print(output.device, batch_targets.device)
            loss = F.cross_entropy(output, batch_targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}')

    torch.save('../data/embeddings/composite_embeddings.h5')

    return model.sentence_embedding.weight.data.cpu().numpy()