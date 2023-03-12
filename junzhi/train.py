 
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt

file_path = "../raw_data/"

debug = True

# Load datasets
X_train = np.load(os.path.join(file_path,"train_data.npy"))
print('X_train: ', X_train.shape)
y_train = np.load(os.path.join(file_path,"train_label.npy"))
print('y_train: ', y_train.shape)
X_test = np.load(os.path.join(file_path,"test_data.npy"))
print('X_test: ', X_test.shape)
y_test = np.load(os.path.join(file_path,"test_label.npy"))
print('y_test: ', y_test.shape)
