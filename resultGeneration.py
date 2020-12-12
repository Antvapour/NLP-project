from helpers import Data, generate_dataset, evaluate
import os, pickle
import pandas as pd
import torch
from nltk.tokenize import RegexpTokenizer
from gensim.models.doc2vec import Doc2Vec
from sklearn.naive_bayes import GaussianNB
from torch.utils.data import TensorDataset, DataLoader

try:
    import transformers
except :
    os.system('pip install transformers')
from transformers import BertForSequenceClassification

# Generate results for the baseline model
print("Generating results for the baseline model")
print("")

# Loading Data
data = Data('NB')
dev_x, dev_y = data.getDevData()
test_x = data.getTestData()

# Tokenizing
print("Tokenization Started")
tokenizer = RegexpTokenizer('\w+|\d+')
dev_x_clean = dev_x.apply(lambda row: tokenizer.tokenize(row['review'].lower()), axis=1)
test_x_clean = test_x.apply(lambda row: tokenizer.tokenize(row['review'].lower()), axis=1)
print("Tokenization Completed, Loading Models")

# Loading Model
folderName = './NB_model'
modelD2V_filename = 'trained.d2v'
modelNB_filename = 'GaussianNB.pkl'
modelD2V = Doc2Vec.load(os.path.join(folderName, modelD2V_filename))
with open(os.path.join(folderName, modelNB_filename), 'rb') as file:
    modelNB = pickle.load(file)
print("Loading Models Completed, Start Embedding")

# Generating Embedded Vectors for Development Set and Test Set
# It is time-expensive because infer_vector() does not accept vectorized input,
# and it is set to run 100 epochs to generage a better prediction
print("Generating Embedded Vectors for Development Set")
dev_embedded = [modelD2V.infer_vector(comment) for comment in dev_x_clean]
print("Generating Embedded Vectors for Test Set")
test_embedded = [modelD2V.infer_vector(comment) for comment in test_x_clean]
print("Embedding Completed, Naive Bayes Classification Started")

# Predicting and Generating Results
# Development Set
dev_predict = modelNB.predict(dev_embedded)
scoreNB = modelNB.score(dev_embedded, dev_y)
df_dev = pd.DataFrame({"prediction": dev_predict, "actual": dev_y})
df_dev.index.name = "id"
folderName = './results'
filename = "NB_dev_prediction_acc_" + "{:2.2%}".format(scoreNB) + ".csv"
df_dev.to_csv(os.path.join(folderName, filename))

# Test Set
test_predict = modelNB.predict(test_embedded)
df_test = pd.DataFrame({"prediction": test_predict})
df_test.index.name = "id"
filename = "NB_test_prediction.csv"
df_test.to_csv(os.path.join(folderName, filename))
print("Results for the baseline model has been generated")
print("")

# Generate results for the SOTA model
print("Generating results for the SOTA(BERT) model")
print("")

# Loading Data
data = Data('BERT')
dev_x, dev_y = data.getDevData()
test_x = data.getTestData()

# Tokenizing
print("Tokenization Started")
dev_set = generate_dataset(dev_x, ratings = dev_y)
test_set = generate_dataset(dev_x, is_test = True)

# Create DataLoaders for Training and Development Sets.
# From the referenced article, a batch size of 16 or 32 is recommended
# My GPU is not large enough to handle batches with size of 32
dev_loader = DataLoader(dev_set, batch_size = 16)
test_loader = DataLoader(test_set, batch_size = 16)
print("Tokenization Completed, Loading Models")

# Loading Model
folderName = './BERT_model/'
model = BertForSequenceClassification.from_pretrained(folderName, num_labels = 6)

if torch.cuda.is_available():
    device = torch.device("cuda")
    model.cuda()
else:
    device = torch.device("cpu")

print("Loading Completed, BERT Classification Started")
dev_predict, accuracy, _ = evaluate(model, dev_loader, device)
df_dev = pd.DataFrame({"prediction": dev_predict, "actual": dev_y})
df_dev.index.name = "id"
folderName = './results'
filename = "BERT_dev_prediction_acc_" + "{:2.2%}".format(accuracy) + ".csv"
df_dev.to_csv(os.path.join(folderName, filename))

test_predict = evaluate(model, test_loader, device, is_test = True)
df_test = pd.DataFrame({"prediction": test_predict})
df_test.index.name = "id"
filename = "BERT_test_prediction.csv"
df_test.to_csv(os.path.join(folderName, filename))
print("Results for the SOTA model has been generated")
