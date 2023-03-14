 
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from util import Data_Proprocesing
from mlp import MLP

file_path = "../../raw_data/"

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

if debug:
    print("--"*30)
    print(y_train[:10,:])
    print(Data_Proprocesing.decode_one_encoding(y_train[:10]))
    X_shuffle, y_shuffle = Data_Proprocesing.shuffle_randomly(X_train[:10,:], y_train[:10,:])
    print(y_shuffle)
    print(Data_Proprocesing.decode_one_encoding(y_shuffle))
    print("--"*30)
    
    
# Hyperparameters
LAYER_NEURONS = [128, 100, 80, 50,10]
LAYER_ACTIVATION_FUNCS = [None, 'tanh', 'tanh', 'relu', 'relu']

assert len(LAYER_NEURONS) == len(LAYER_ACTIVATION_FUNCS)

LEARNING_RATE = 0.05
EPOCHS = 50
DROPOUT_PROB = 1
BATCH_SIZE = 5000

WEIGHT_DECAY = 0.01  # if WEIGHT_DECAY is None, then no weight decay is applied

size = 50000
loss = 'MSE'
optimizer = 'momentum'

# Instantiate the multi-layer neural network
nn = MLP(X_test[:int(size*0.2)], y_test[:int(size*0.2)], layers=LAYER_NEURONS, activation=LAYER_ACTIVATION_FUNCS,
         dropoutRate=DROPOUT_PROB, weight_decay=WEIGHT_DECAY, loss=loss, batch_size=BATCH_SIZE)

# Perform fitting using the training dataset
t0 = time.time()
trial1_logger = nn.fit(X_train[:size], y_train[:size], learning_rate=LEARNING_RATE, epochs=EPOCHS, opt = optimizer)
t1 = time.time()
print(f"============= Model Build Done =============")
print(f"Time taken to build model: {round(t1 - t0, 4)} seconds with {EPOCHS} Epochs of training.")