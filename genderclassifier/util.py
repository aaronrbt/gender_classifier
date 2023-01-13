from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
import re

# creating a vocab and dict map                              
vocab = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
vocab.append("END")
len_vocab = len(vocab)
char_index = {c:i for i, c in enumerate(vocab)}   # creating a dictionary
code_to_label = {0:'F',1:'M'}

# text preprocessing (mainly cleaning)
def preprocessing(name_series):
    # 1. keep alphabets only and 2. (standardize) lowercase names  
    name_series = name_series.str.replace('[^a-zA-Z]', '').str.lower()
    return name_series

# Truncate names and create the matrix
def get_encod_names(X,maxlen):
    vec_names = []
    trunc_names = [str(i)[0:maxlen] for i in X]  # consider only the first 20 characters
    for name in trunc_names:
        tmp = [set_flag(char_index[i]) for i in str(name)]
        for k in range(0,maxlen - len(str(name))):
            tmp.append(set_flag(char_index["END"]))
        vec_names.append(tmp)
    return vec_names

def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

# Builds an empty line with a 1 at the index of character
def set_flag(i):
    aux = np.zeros(len_vocab)
    aux[i] = 1
    return list(aux)
    
def process_unit_ipt(ipt:str, maxlen):
    processed_name = re.sub('[^a-zA-Z]','',ipt).lower()[0:maxlen]
    if processed_name == '':
        return -1     
    encoded_name = [set_flag(char_index[i]) for i in processed_name]
    for k in range(0,maxlen - len(str(processed_name))):
        encoded_name.append(set_flag(char_index["END"]))
    return np.asarray([encoded_name])

def data_to_df(x, maxlen, y=None):
    df_x = pd.DataFrame(data=x.reshape((x.shape[0],-1)))
    y_column = []
    if y is not None and isinstance(y,np.ndarray):
        df_y = pd.DataFrame(data=y.reshape((y.shape[0],-1)))
        df = pd.concat([df_x, df_y], axis=1)
        y_column.append('gender')
    else:
        df = df_x
    df.columns = [str(pos+1)+char for pos in range(maxlen) for char in vocab] + y_column
    df = df.astype('int8')
    return df
