 
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from util import Data_Proprocesing

file_path = "../raw_data/"

debug = True

# Load datasets
X_train = np.load(os.path.join(file_path,"train_data.npy"))

y_train = np.load(os.path.join(file_path,"train_label.npy"))

X_test = np.load(os.path.join(file_path,"test_data.npy"))

y_test = np.load(os.path.join(file_path,"test_label.npy"))

if debug:
    print('X_train: ', X_train.shape)
    print('y_train: ', y_train.shape)
    print('X_test: ', X_test.shape)
    print('y_test: ', y_test.shape)
    print(y_train[:10,:])


# Preprocess data
X_train = Data_Proprocesing.standardize(X_train)
X_test = Data_Proprocesing.standardize(X_test)
y_test = Data_Proprocesing.one_encoding(y_test)
y_train = Data_Proprocesing.one_encoding(y_train)

if debug:
    print("--"*30)
    print(y_train[:10,:])
    print(Data_Proprocesing.decode_one_encoding(y_train[:10]))
    X_shuffle, y_shuffle = Data_Proprocesing.shuffle_randomly(X_train[:10,:], y_train[:10,:])
    print(y_shuffle)
    print(Data_Proprocesing.decode_one_encoding(y_shuffle))
 