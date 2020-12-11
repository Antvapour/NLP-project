# reference: https://mccormickml.com/2019/07/22/BERT-fine-tuning/
from data_handling import Data
import os
import time, datetime
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

try:
    import transformers
except :
    os.system('pip install transformers')
from transformers import BertTokenizer, BertForSequenceClassification, \
                        AdamW, BertConfig, get_linear_schedule_with_warmup
# Loading Data
data = Data('BERT')
train_x, train_y = data.getTrainData()
dev_x, dev_y = data.getDevData()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Preprocessing
print("Preprocessing Started")

# This part of code was to identify the maximum lenth after tokenization.
# It appeares that some reviews generate tokens more than the specified maximum
# sequence length for BERT (512).
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# max_len = 0
# for review in train_x.values.reshape(-1):
#     max_len = max(max_len, len(tokenizer.encode(review)))
# print('Max sentence length: ', max_len)

def generate_dataset(reviews, ratings):
    # Tokenize all of the sentences and map the tokens to thier word IDs
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_ids = []
    attention_masks = []

    for review in reviews:
        # My GPU is not large enough to handle 512 tokens
        # For each review I only took the first 256 tokens
        encoded_dict = tokenizer.encode_plus(review, max_length = 256,
                padding = 'max_length', truncation = True,
                return_attention_mask = True, return_tensors = 'pt')

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(ratings)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    return dataset

# Generate Datasets
train_set = generate_dataset(train_x, train_y)
print("Training Set Tokenization Completed")
dev_set = generate_dataset(dev_x, dev_y)
print("Development Set Tokenization Completed")

# Create DataLoaders for Training and Development Sets.
# From the referenced article, a batch size of 16 or 32 is recommended
# My GPU is not large enough to handle batches with size of 32
train_loader = DataLoader(train_set, shuffle = True, batch_size = 16)
dev_loader = DataLoader(dev_set, batch_size = 16)

# Use BertForSequenceClassification Pretrained model for Classification
model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels = 6)

# Run the model on cuda if available
if torch.cuda.is_available():
    model.cuda()

# Use Adam Weight Decay as the optimizer
optimizer = AdamW(model.parameters(), lr = 2e-5)

# The BERT authors recommend the number of epochs to be 2~4. Four epochs
# were implemented, while The 4th epoch would not increase the validation score.
# Therefore only 3 epochs were kept
epochs = 3

# Create the learning rate scheduler
# Total number of training steps is [number of batches] x [number of epochs]
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0,
                num_training_steps = len(train_loader) * epochs)

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Function to format elapsed time
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# ===================================
# Start Training
# ===================================

# Measure the total training time for the whole run
total_t0 = time.time()

for epoch in range(epochs):
    print("")
    print("Training...")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))

    # Measure how long the training epoch takes
    t0 = time.time()

    # Reset the total loss for this epoch
    total_train_loss = 0

    # Put the model into training mode
    model.train()

    for step, batch in enumerate(train_loader):

        # Progress update every 50 batches
        if step % 50 == 0 and not step == 0:
            # Calculate elapsed time in minutes
            elapsed = format_time(time.time() - t0)

            # Report progress
            print('  Batch {:>5,}  of  {:>5d,}.    Elapsed: {:}.'.format(step,
                    len(train_loader), elapsed))

        # Unpack this training batch from dataloader and copy each tensor to GPU
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Clear previously calculated gradients
        model.zero_grad()

        # Perform a forward pass
        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss, logits = outputs.loss, outputs.logits

        # Accumulate the training loss over all of the batches
        total_train_loss += loss.item()

        # Perform a backward pass
        loss.backward()

        # Clip the norm of the gradients to 1.0
        # This is to help prevent the "exploding gradients" problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient
        optimizer.step()

        # Update the learning rate
        scheduler.step()

    # Calculate the average loss over all of the batches
    avg_train_loss = total_train_loss / len(train_loader)

    # Measure how long this epoch took
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    # ==============================================
    # Measure the Performance on the Development Set
    # ==============================================

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode
    model.eval()

    # Tracking variables
    total_eval_accuracy = 0
    total_eval_loss = 0

    # Evaluate data for one epoch
    for batch in dev_loader:

        # Unpack this development batch from dataloader and copy each tensor to GPU
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # There is no need to calculate the gradients for validation
        with torch.no_grad():

            # Perform a forward pass
            outputs = model(b_input_ids,attention_mask=b_input_mask,labels=b_labels)
            loss, logits = outputs.loss, outputs.logits

        # Accumulate the validation loss
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch and accumulate it over all batches
        total_eval_accuracy += flat_accuracy(logits, label_ids)


    # Report the final accuracy for this validation run
    avg_val_accuracy = total_eval_accuracy / len(dev_loader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches
    avg_val_loss = total_eval_loss / len(dev_loader)

    # Measure how long the validation run took
    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
output_dir = './BERT_model'
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
