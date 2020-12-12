# reference: https://mccormickml.com/2019/07/22/BERT-fine-tuning/
from helpers import *
import os
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

try:
    from transformers import BertForSequenceClassification, \
                        AdamW, get_linear_schedule_with_warmup
except :
    os.system('pip install transformers')
from transformers import BertForSequenceClassification, \
                        AdamW, get_linear_schedule_with_warmup
# Loading Data
# Data class is defined in helpers.py
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

# Generate Datasets
# generate_dataset() is defined in helpers.py
train_set = generate_dataset(train_x, ratings = train_y)
print("Training Set Tokenization Completed")
dev_set = generate_dataset(dev_x, ratings = dev_y)
print("Development Set Tokenization Completed")

# Create DataLoaders for Training and Development Sets.
# From the referenced article, a batch size of 16 or 32 is recommended
# My GPU is not large enough to handle batches with size of 32
train_loader = DataLoader(train_set, shuffle = True, batch_size = 16)
dev_loader = DataLoader(dev_set, batch_size = 16)

# Use BertForSequenceClassification Pretrained model for Classification
# BERT requires the lables to be in the range (0, num_labels-1). Since
# no prediction of label 0 will be made by the model if no training sample
# has label 0, I simply set num_labels to six to avoid further conversions
model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels = 6)

# Run the model on cuda if available
if torch.cuda.is_available():
    model.cuda()

# Use Adam Weight Decay as the optimizer
optimizer = AdamW(model.parameters(), lr = 2e-5)  # lr = [1e-3, 1e-4]

# The BERT authors recommend the number of epochs to be 2~4. Four epochs
# were implemented, while The 4th epoch would not increase the validation score.
# Therefore only 3 epochs were kept
epoch_max = 3

# Create the learning rate scheduler
# Total number of training steps is [number of batches] x [number of epochs]
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0,
                num_training_steps = len(train_loader) * epoch_max)

# Preparing for time and loss recording
log_contents = {
    "epochs" : [1, 2, 3],
    "training_time" : [],
    "train_loss" : [],
    "val_loss" : [],
}

# ===================================
# Start Training
# ===================================

# Measure the total training time for the whole run
total_t0 = time.time()

for epoch in range(epoch_max):
    print("")
    print("Training...")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epoch_max))

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
            # format_time() is defined in helpers.py
            elapsed = format_time(time.time() - t0)

            # Report progress
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step,
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

    # Update the log contents
    log_contents["training_time"].append(training_time)
    log_contents["train_loss"].append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    # ==============================================
    # Measure the Performance on the Development Set
    # ==============================================

    print("")
    print("Running Validation...")

    # evaluate() is defined in helpers.py
    _, avg_val_accuracy, avg_val_loss = evaluate(model, dev_loader, device)

    # Update the log contents
    log_contents["val_loss"].append(avg_val_loss)

    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

# Saving the model
output_dir = './BERT_model'
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)

log = pd.DataFrame(log_contents)
log.to_csv('sota_log.csv')
