# reference: https://mccormickml.com/2019/07/22/BERT-fine-tuning/
import os
import pandas as pd
import numpy as np
from data_handling import Data

try:
    from transformers import BertTokenizer
except :
    os.system('pip install transformers')
    from transformers import BertTokenizer


data = Data()
train_x, train_y = data.getTrainData()
dev_x, dev_y = data.getDevData()
