# reference: http://linanqiu.github.io/2015/10/07/word2vec-sentiment/
from data_handling import Data
import multiprocessing
import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.naive_bayes import GaussianNB

data = Data()
train_x, train_y = data.getTrainData()
dev_x, dev_y = data.getDevData()

# Preprocessing
print("Preprocessing Started")
tokenizer = RegexpTokenizer('\w+|\d+')
train_x_clean = train_x.apply(lambda row: tokenizer.tokenize(row['review'].lower()), axis=1)
dev_x_clean = dev_x.apply(lambda row: tokenizer.tokenize(row['review'].lower()), axis=1)
print("Preprocessing Completed, Training Started")
# print(train_x_clean.head(3))
cores = multiprocessing.cpu_count()
train_size = len(train_x)
# label = [i for i in range(train_size)]
# documents = [TaggedDocument(doc, [i]) for i, doc in zip(label, train_x_clean)]
# modelD2V = Doc2Vec(min_count=1, workers=cores)
# modelD2V.build_vocab(documents)
# modelD2V.train(documents, epochs=100, total_examples=train_size)
# modelD2V.save('./temp.d2v')
# print("doc2vec Training Completed")
modelD2V = Doc2Vec.load('./trained.d2v')
train_embedded = modelD2V[range(train_size)]
modelNB = GaussianNB()
modelNB.fit(train_embedded, train_y.values.reshape(-1))
print("GaussianNB Training Completed, Predicting Started")
dev_embedded = [modelD2V.infer_vector(comment) for comment in dev_x_clean]
print("Embedding Completed")
dev_predict = modelNB.predict(dev_embedded)
scoreNB = modelNB.score(dev_embedded, dev_y.values.reshape(-1))
print("The score of the Naive Bayes model is:", scoreNB) # 50.53%
