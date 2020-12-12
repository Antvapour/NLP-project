# Functions and classes used in this project are presented here
import datetime
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset

try:
    from transformers import BertTokenizer, BertForSequenceClassification
except :
    os.system('pip install transformers')
    from transformers import BertTokenizer, BertForSequenceClassification

# Common Classes and functions
# Data Handling Class
class Data(object):
    """Data Handler"""

    def __init__(self, model):
        super(Data, self).__init__()
        train_df = pd.read_csv('data/sentiment_dataset_train.csv')
        # print(list(train_df.columns)) # ['id', 'review', 'rating']
        # print("Null value check")
        # print(train_df.isnull().sum()) # no null values
        # During later processes, it was found that id 30944 contains invalid
        # information. Thus I decide to delete this row.
        train_df = train_df.drop(30944)

        dev_df = pd.read_csv('data/sentiment_dataset_dev.csv')
        # print(list(train_df.columns)) # ['id', 'review', 'rating']
        # print("Null value check")
        # print(train_df.isnull().sum()) # no null values

        test_df = pd.read_csv('data/sentiment_dataset_test.csv')
        # print(list(test_df.columns)) # ['id', 'review']
        # print("Null value check")
        # print(test_df.isnull().sum()) # no null values

        def str2int(str):
            return int(float(str))

        if model == 'NB' :
            self.train_x, self.dev_x = train_df[['review']], dev_df[['review']]
            self.train_y = train_df[['rating']].applymap(str).values.reshape(-1)
            self.dev_y   = dev_df[['rating']].applymap(str).values.reshape(-1)
            self.test_x  = test_df[['review']]
        elif model == 'BERT' :
            self.train_x = train_df[['review']].values.reshape(-1)
            self.dev_x   = dev_df[['review']].values.reshape(-1)
            self.train_y = train_df[['rating']].applymap(str2int).values.reshape(-1)
            self.dev_y   = dev_df[['rating']].applymap(str2int).values.reshape(-1)
            self.test_x  = test_df[['review']].values.reshape(-1)

    def getTrainData(self):
        return [self.train_x, self.train_y]

    def getDevData(self):
        return [self.dev_x, self.dev_y]

    def getTestData(self):
        return self.test_x

# Function to format elapsed time
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# Functions for SOTA model
# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Generating the dataset
def generate_dataset(reviews, ratings = None, is_test = False):
    # Tokenize all of the sentences and map the tokens to thier word IDs
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_ids = []
    attention_masks = []

    for review in reviews:
        # My GPU is not large enough to handle 512 tokens
        # For each review I only took the first 256 tokens
        # https://stackoverflow.com/questions/58636587/how-to-use-bert-for-long-text-classification
        # According to the post above, cutting out the middle of the text has
        # The best performance. But if the default truncation works just fine,
        # cutting out the middle of the text might not improve the performance
        # significantly. After getting an 80% accuracy using default truncation,
        # I decide to stay with the default method
        encoded_dict = tokenizer.encode_plus(review, max_length = 256,
                padding = 'max_length', truncation = True,
                return_attention_mask = True, return_tensors = 'pt')

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    # If it is for the test set, there would be no labels
    if is_test :
        dataset = TensorDataset(input_ids, attention_masks)
    else:
        labels = torch.tensor(ratings)
        dataset = TensorDataset(input_ids, attention_masks, labels)
    return dataset

# Define the evaluation function
def evaluate(model, loader, device, is_test=False):
    # Put the model in evaluation mode
    model.eval()

    # Tracking variables if it is for validation
    total_eval_accuracy = 0
    total_eval_loss = 0

    # Initiate the prediction list
    predictions = []

    # Evaluate data for one epoch
    for batch in loader:

        # Unpack this development batch from dataloader and copy each tensor to GPU
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        if is_test :
            b_labels = None
        else:
            b_labels = batch[2].to(device)


        # There is no need to calculate the gradients for validation
        with torch.no_grad():

            # Perform a forward pass
            outputs = model(b_input_ids,attention_mask=b_input_mask,labels=b_labels)
            logits = outputs.logits

            # Move logits to CPU
            logits = logits.detach().cpu().numpy()

            # Store the predictions
            
            predictions += np.argmax(logits, axis=1).flatten().tolist()

            if not is_test :
                # Get the validation loss for this batch
                loss = outputs.loss

                # Accumulate the validation loss
                total_eval_loss += loss.item()

                # Move labels to CPU
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch and accumulate it over all batches
                total_eval_accuracy += flat_accuracy(logits, label_ids)


    if not is_test :
        # Report the final accuracy for this validation run
        avg_val_accuracy = total_eval_accuracy / len(loader)

        # Calculate the average loss over all of the batches
        avg_val_loss = total_eval_loss / len(loader)

    return predictions if is_test else [predictions, avg_val_accuracy, avg_val_loss]
