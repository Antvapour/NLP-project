import os
import pandas as pd
import numpy as np
class Data(object):
    """docstring for Data."""

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

        labels = np.loadtxt('data/sentiment_ratings_labels.txt',dtype=str).reshape(-1)

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
