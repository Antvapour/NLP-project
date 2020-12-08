# reference: https://www.kaggle.com/residentmario/notes-on-word-embedding-algorithms
# reference: https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
import os
try:
    import transformers
except :
    os.system('pip install transformers')
    import transformers
