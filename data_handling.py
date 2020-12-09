import os
import pandas as pd
import numpy as np
class Data(object):
    """docstring for Data."""

    def __init__(self):
        super(Data, self).__init__()
        train_df = pd.read_csv('data/sentiment_dataset_train.csv')
        # print(list(train_df.columns)) # ['id', 'review', 'rating']
        # print("Null value check")
        # print(train_df.isnull().sum()) # no null values

        dev_df = pd.read_csv('data/sentiment_dataset_dev.csv')
        # print(list(train_df.columns)) # ['id', 'review', 'rating']
        # print("Null value check")
        # print(train_df.isnull().sum()) # no null values

        labels = np.loadtxt('data/sentiment_ratings_labels.txt',dtype=str).reshape(-1)

        self.train_x, self.train_y = train_df[['review']], train_df[['rating']].applymap(str)
        self.dev_x, self.dev_y = dev_df[['review']], dev_df[['rating']].applymap(str)

    def getTrainData(self):
        return [self.train_x, self.train_y]

    def getDevData(self):
        return [self.dev_x, self.dev_y]
