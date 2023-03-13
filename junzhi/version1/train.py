 
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from util import Data_Proprocesing
from mlp import MLP

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
    
    
# Hyperparameters
LAYER_NEURONS = [128, 100, 10]
LAYER_ACTIVATION_FUNCS = [None, 'relu', 'relu']
LEARNING_RATE = 0.0005
EPOCHS = 20
DROPOUT_PROB =1
BATCH_SIZE = 1
WEIGHT_DECAY = 0.98
# Instantiate the multi-layer neural network
nn = MLP(X_test[:5], y_test[:5], layers=LAYER_NEURONS, activation=LAYER_ACTIVATION_FUNCS,
         dropoutRate=DROPOUT_PROB, weight_decay=WEIGHT_DECAY, loss='MSE', batch_size=BATCH_SIZE)

# Perform fitting using the training dataset
t0 = time.time()
trial1_logger = nn.fit(X_train[:5], y_train[:5], learning_rate=LEARNING_RATE, epochs=EPOCHS)
t1 = time.time()
print(f"============= Model Build Done =============")
print(f"Time taken to build model: {round(t1 - t0, 4)} seconds.")