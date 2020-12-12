# reference: http://linanqiu.github.io/2015/10/07/word2vec-sentiment/
from helpers import Data, format_time
import multiprocessing
import os, pickle
import time
import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.naive_bayes import GaussianNB

# Loading Data
# Data class is defined in helpers.py
data = Data('NB')
train_x, train_y = data.getTrainData()
dev_x, dev_y = data.getDevData()

# Tokenizing
print("Tokenization Started")
tokenizer = RegexpTokenizer('\w+|\d+')
train_x_clean = train_x.apply(lambda row: tokenizer.tokenize(row['review'].lower()), axis=1)
dev_x_clean = dev_x.apply(lambda row: tokenizer.tokenize(row['review'].lower()), axis=1)

# Training Word Embedder
print("Tokenization Completed, Training Started")
train_size = len(train_x)

# Preparing for time recording
elapsed = []
interval_name = []
t0 = time.time()

# Generating Tagged Documents
documents = [TaggedDocument(doc, [i]) for doc, i in \
                    zip(train_x_clean, range(train_size))]

# Initializing the Doc2Vec Model
modelD2V = Doc2Vec(min_count=1, workers=multiprocessing.cpu_count())

# Building the Vocabulary Dict
modelD2V.build_vocab(documents)

# Training the Model
# Epoch number is set to 100 to generage a better prediction
modelD2V.train(documents, epochs=100, total_examples=train_size)

# Calculate the elapsed time
# format_time is defined in helpers.py
elapsed.append(format_time(time.time() - t0))
interval_name.append("Training Doc2Vec")

# Saving the Word Embedder
print("doc2vec Training Completed")
folderName = './NB_model'
modelD2V_filename = 'trained.d2v'
modelD2V.save(os.path.join(folderName, modelD2V_filename))

# Getting Embedded Vectors for Training Set
train_embedded = modelD2V[range(train_size)]

# Generating Embedded Vectors for Development Set
# It is time-expensive because infer_vector() does not accept vectorized input,
# and it is set to run 100 epochs to generage a better prediction
print("Generating Embedded Vectors for Development Set")
t0 = time.time()
dev_embedded = [modelD2V.infer_vector(comment) for comment in dev_x_clean]

# Calculate the elapsed time
elapsed.append(format_time(time.time() - t0))
interval_name.append("Generating Embedded Vectors for Development Set")

print("Embedding Completed, Naive Bayes Classification Started")

# Training Naive Bayes Classifier
modelNB = GaussianNB()

# Cross-validation could be performed to fine-tune this model. Methods such as
# GridSearchCV could be used in such a process. However, since it is merely a
# baseline model, fine-tuning it is arguably redundant
t0 = time.time()
modelNB.fit(train_embedded, train_y)

# Calculate the elapsed time
elapsed.append(format_time(time.time() - t0))
interval_name.append("Training GaussianNB")

print("GaussianNB Training Completed")
scoreNB = modelNB.score(dev_embedded, dev_y)
print("The accuracy is: {:2.2%}".format(scoreNB)) # 51.45%

# Saving the classifier
modelNB_filename = 'GaussianNB.pkl'
with open(os.path.join(folderName, modelNB_filename), 'wb') as file:
    pickle.dump(modelNB, file)

log = pd.DataFrame({"Interval": interval_name, "Time": elapsed})
log.to_csv('baseline_log.csv')
