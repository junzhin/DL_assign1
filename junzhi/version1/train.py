 
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from util import Data_Proprocesing
from mlp import MLP
import argparse
import yaml

file_location = "../../raw_data/"
debug = False
# ----------------------------------------------------------------------------------   
# Load datasets
X_train = np.load(os.path.join(file_location,"train_data.npy"))
y_train = np.load(os.path.join(file_location,"train_label.npy"))
X_test = np.load(os.path.join(file_location,"test_data.npy"))
y_test = np.load(os.path.join(file_location,"test_label.npy"))
if debug:
    print('X_train: ', X_train.shape)
    print('y_train: ', y_train.shape)
    print('X_test: ', X_test.shape)
    print('y_test: ', y_test.shape)
    print(y_train[:10,:])

# ----------------------------------------------------------------------------------   
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
 # ----------------------------------------------------------------------------------
#Set default hyperparameters
# default_layer_neurons = [128, 90, 80, 50, 10]

default_layer_neurons = [128, 100, 110, 100, 10]
default_layer_activation_funcs = [None, 'leakyrelu', 'leakyrelu', 'leakyrelu', 'softmax']
default_learning_rate = 0.0005
default_epochs = 1
default_dropout_prob = 0.8
assert 0 <= default_dropout_prob <= 1
default_batch_size = 2 # if batch_size is None, then no batch is used
default_weight_decay = 0 # if weight_decay is None, then no weight decay is applied
default_beta = [0.9, 0.999]
default_size = 100 # Size of training dataset, 50000 is the full dataset
default_batchnorm = True
default_loss = 'CE' # 'CE' or 'MSE'
default_optimizer = 'adam' # 'sgd' or 'adam', 'sgd_momentum'
 
parser = argparse.ArgumentParser(
    description='Multi-layer neural network arguments')

parser.add_argument('--layer_neurons', nargs='+', type=int, default=default_layer_neurons,
                    help='List of integers specifying number of neurons in each layer')
parser.add_argument('--activation_funcs', nargs='+', type=str, default=default_layer_activation_funcs,
                    help='List of activation functions for each layer')
parser.add_argument('--learning_rate', type=float, default=default_learning_rate,
                    help='Learning rate for the optimizer')
parser.add_argument('--epochs', type=int, default=default_epochs,
                    help='Number of training epochs')
parser.add_argument('--dropout_prob', type=float, default=default_dropout_prob,
                    help='Dropout probability (between 0 and 1)')
parser.add_argument('--batch_size', type=int, default=default_batch_size,
                    help='Batch size for training. If None, then no batch is used')
parser.add_argument('--weight_decay', type=float, default=default_weight_decay,
                    help='Weight decay for the optimizer. If None, then no weight decay is applied')
parser.add_argument('--beta', nargs='+', type=float, default=default_beta,
                    help='List of beta values for the optimizer')
parser.add_argument('--size', type=int, default=default_size,
                    help='Size of the training dataset. 50000 is the full dataset')
parser.add_argument('--batch_norm', type=bool, default=default_batchnorm,
                    help='Whether to use batch normalization or not')
parser.add_argument('--loss', type=str, default=default_loss,
                    help='Loss function for the optimizer (CE or MSE)')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='Optimizer to use (sgd, adam, or sgd_momentum)')
parser.add_argument('--save_path', type=str, default='./results/')

args = parser.parse_args()
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    
# Convert the args object to a dictionary
args_dict = vars(args)
print("-----------------------------------")
print("setting of the training!")
# Loop through the dictionary and print each key-value pair
for arg, val in args_dict.items():
    print(f'{arg}: {val}')
    
    import xml.etree.ElementTree as ET


# Convert args object to a dictionary
args_dict = vars(args)

# Save the dictionary to a YAML file
with open(os.path.join(args.save_path,'hyperparameters.yaml'), 'w') as f:
    yaml.dump(args_dict, f)

 

print("-----------------------------------")
# ----------------------------------------------------------------------------------   
# Instantiate the multi-layer neural network
assert len(args.layer_neurons) == len(args.activation_funcs)
nn = MLP(X_test[:int(args.size*0.2)], y_test[:int(args.size*0.2)],
         layers=args.layer_neurons, activation=args.activation_funcs,
      
         dropoutRate=args.dropout_prob, weight_decay=args.weight_decay,
         loss=args.loss, batch_size=args.batch_size, beta=args.beta,
         batch_norm=args.batch_norm)

# Perform fitting using the training dataset
t0 = time.time()
print(f"============= Model Starts Building =============")
trial1_logger = nn.fit(X_train[:args.size], y_train[:args.size],
                        learning_rate=args.learning_rate, epochs=args.epochs,
                        opt=args.optimizer)

t1 = time.time()
print(f"============= Model Build Done =============")
print(f"Time taken to build model: {round(t1 - t0, 4)} seconds with {args.epochs} Epochs of training.")