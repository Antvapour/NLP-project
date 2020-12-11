# reference:
# http://linanqiu.github.io/2015/10/07/word2vec-sentiment/
from data_handling import Data
import multiprocessing
import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

# Loading Data
data = Data('NB')
train_x, train_y = data.getTrainData()
dev_x, dev_y = data.getDevData()
test_x = data.getTestData()

# Preprocessing
print("Preprocessing Started")
tokenizer = RegexpTokenizer('\w+|\d+')
train_x_clean = train_x.apply(lambda row: tokenizer.tokenize(row['review'].lower()), axis=1)
dev_x_clean = dev_x.apply(lambda row: tokenizer.tokenize(row['review'].lower()), axis=1)
test_x_clean = test_x.apply(lambda row: tokenizer.tokenize(row['review'].lower()), axis=1)

# Training Word Embedder
print("Preprocessing Completed, Training Started")
cores = multiprocessing.cpu_count()
train_size = len(train_x)

#===============================================================================
# Word Embedder Training Code is Commented out to Save Time
#-------------------------------------------------------------------------------
# label = [i for i in range(train_size)]
# documents = [TaggedDocument(doc, [i]) for i, doc in zip(label, train_x_clean)]
# modelD2V = Doc2Vec(min_count=1, workers=cores)
# modelD2V.build_vocab(documents)
# modelD2V.train(documents, epochs=100, total_examples=train_size)
# modelD2V.save('./NB_model/trained.d2v')
# print("doc2vec Training Completed")
#===============================================================================
modelD2V = Doc2Vec.load('./NB_model/trained.d2v')

# Getting Embedded Vectors for Training Set
train_embedded = modelD2V[range(train_size)]

# Training Naive Bayes Classifier
modelNB = GaussianNB()
modelNB.fit(train_embedded, train_y)
print("GaussianNB Training Completed, Predicting Started")

# Generating Embedded Vectors for Dev Set and Test Set
# It is time-expensive because infer_vector() does not accept vectorized input,
# and it is set to run 100 epochs to generage a better prediction
print("Generating Embedded Vectors for Development Set")
dev_embedded = [modelD2V.infer_vector(comment) for comment in dev_x_clean]
print("Generating Embedded Vectors for Test Set")
test_embedded = [modelD2V.infer_vector(comment) for comment in test_x_clean]
print("Embedding Completed, Naive Bayes Classification Started")

# Generating Results
dev_predict = modelNB.predict(dev_embedded)
df_dev = pd.DataFrame({"prediction": dev_predict, "actual": dev_y})
df_dev.index.name = "id"
df_dev.to_csv("NB_dev_prediction.csv")
test_predict = modelNB.predict(test_embedded)
df_test = pd.DataFrame(data=test_predict, columns=["prediction"])
df_test.index.name = "id"
df_test.to_csv("NB_test_prediction.csv")
scoreNB = modelNB.score(dev_embedded, dev_y)
print("The score of the Naive Bayes model on Development set is: {:2.2%}".format(scoreNB)) # 50.82%
