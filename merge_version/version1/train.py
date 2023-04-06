
 # Import libraries
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from util import Data_Proprocesing
from mlp import MLP
import argparse
import yaml
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
from util import Data_Proprocesing
from sklearn.metrics import accuracy_score

debug = False

#----------------------------------------------------------------------------------
#Set default hyperparameters
default_layer_neurons = [128, 150, 100, 10]  # specify the number of layers and neurons in each layer
default_layer_activation_funcs = ['None', 'relu', 'relu', 'softmax']  # optins: None, linear,leakyrelu,relu,softmax,logistic
default_learning_rate = 0.005 # learning rate for the optimizer
default_epochs = 150 # number of training epochs
default_dropout_prob = 0.9  # dropout probability that perserve the neuron
assert 0 <= default_dropout_prob <= 1 # dropout probability must be between 0 and 1
default_batch_size = 1024 # if batch_size is None, then no batch is used
default_weight_decay = 0.002 # if weight_decay is None, then no weight decay is applied
default_beta = [0.9, 0.999] # beta values for the adam optimizer
default_size = 50000 # Size of training dataset, 50000 is the full dataset
default_batchnorm = False # True or False for batch normalization
default_loss = 'CE' # 'CE' or 'MSE'
default_optimizer = 'adam'  # 'sgd' or 'adam', 'sgd_momentum' 'rmsprop'
# path to save the results
default_save_path = './results/Basic_Model_bn_weight_decay_adam_early_stoppingv3'
default_file_location= "../../raw_data/" # path to the raw data
default_early_stopping = True # True or False for early stopping

# ----------------------------------------------------------------------------------
# Parse arguments

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
parser.add_argument('--batch_norm', type=str, default=default_batchnorm, choices=["True", "False"],
                    help='Whether to use batch normalization or not')
parser.add_argument('--loss', type=str, default=default_loss,
                    help='Loss function for the optimizer (CE or MSE)')
parser.add_argument('--optimizer', type=str, default=default_optimizer,
                    help='Optimizer to use (sgd, adam, or sgd_momentum)')
parser.add_argument('--save_path', type=str, default=default_save_path)
parser.add_argument('--file_location', type=str, default=default_file_location)
parser.add_argument('--early_stopping', type=bool, default=default_early_stopping)

args = parser.parse_args()


# This means that the default batch normalization is not used, there is a argument being passed to this args.batch_norm
if isinstance(args.batch_norm, str):
    if args.batch_norm == "False":
        args.batch_norm = False
    else:
        args.batch_norm = True

# ----------------------------------------------------------------------------------   
# Load datasets
X_train = np.load(os.path.join(args.file_location,"train_data.npy"))
y_train = np.load(os.path.join(args.file_location,"train_label.npy"))
X_test = np.load(os.path.join(args.file_location,"test_data.npy"))
y_test = np.load(os.path.join(args.file_location, "test_label.npy"))
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
# Create the directory to save the results
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

# Save the dictionary to a yaml file
for index, each in enumerate(args.activation_funcs):
    if each == "None":
        args.activation_funcs[index] = None 

print("-----------------------------------")
# ----------------------------------------------------------------------------------   
# Instantiate the multi-layer neural network
assert len(args.layer_neurons) == len(args.activation_funcs)
nn = MLP(X_test, y_test,
         layers=args.layer_neurons, activation=args.activation_funcs,
      
         dropoutRate=args.dropout_prob, weight_decay=args.weight_decay,
         loss=args.loss, batch_size=args.batch_size, beta=args.beta,
         batch_norm=args.batch_norm)

# Perform fitting using the training dataset
t0 = time.time()
print(f"============= Model Starts Building =============")
trial1_logger = nn.fit(X_train[:args.size], y_train[:args.size],
                        learning_rate=args.learning_rate, epochs=args.epochs,
                        opt=args.optimizer,early_stop=args.early_stopping)

t1 = time.time() # end time
print(f"============= Model Build Done =============")
print(f"Time taken to build model: {round(t1 - t0, 4)} seconds with {args.epochs} Epochs of training.")


# ----------------------------------------------------------------------------------
# Plot the results
print(f"============= Results plotting =============")
# save a csv file
df_stats = pd.DataFrame.from_dict(trial1_logger)
# Save the DataFrame as a CSV file using args.save_path location
df_stats.to_csv(args.save_path + '/stats.csv', index=False)


sns.set(style='whitegrid', font_scale=1.2) # set the style of the plots

#  plot training and validation loss
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=trial1_logger['train_loss_per_epochs'], label='Training Loss')
sns.lineplot(
    data=trial1_logger['val_loss_per_epochs'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig(os.path.join(args.save_path, 'loss.png'), dpi=300)

#  plot training and validation accuracy
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=trial1_logger['train_acc_per_epochs'], label='Training Accuracy')
sns.lineplot(
    data=trial1_logger['val_acc_per_epochs'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig(os.path.join(args.save_path, 'accuracy.png'), dpi=300)

#   plot training and validation F1 score
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=trial1_logger['train_f1_per_epochs'], label='Training F1 Score')
sns.lineplot(data=trial1_logger['val_f1_per_epochs'],
             label='Validation F1 Score')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('Training and Validation F1 Score')
plt.legend()
plt.savefig(os.path.join(args.save_path, 'f1_score.png'), dpi=300)

 
# heatmap of confusion matrixss
if args.early_stopping:
    y_pred = nn.predict_early_stop(X_test)
else:
    y_pred = nn.predict(X_test)   
y_pred_decoded = Data_Proprocesing.decode_one_encoding(y_pred) 

cm = confusion_matrix(y_test, y_pred_decoded)  
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
val_acc = accuracy_score(y_test,  np.expand_dims(np.argmax(y_pred, axis=1),axis=1))
print('Final_evaluate_val_acc: ', val_acc)
recall, precision = Data_Proprocesing.recall_precision_from_confusion_matrix(cm)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_norm, annot=True, cmap='YlGnBu', fmt='.2%', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(args.save_path, 'confusion_matrix.png'), dpi=300)




# Compute recall and precision for each class
class_labels = [f'Class {i}' for i in range(cm.shape[0])]

# Define color map for recall and precision
colors = ['tab:red', 'tab:blue']

# Create subplots
fig, ax = plt.subplots(figsize=(10, 6))

# Plot recall and precision as comparative bar chart
bar_width = 0.4
x_pos = np.arange(len(class_labels))
ax.bar(x_pos - bar_width/2, recall, width=bar_width, color=colors[0], label='Recall')
ax.bar(x_pos + bar_width/2, precision, width=bar_width, color=colors[1], label='Precision')

# Set axis labels and title
ax.set_xticks(x_pos)
ax.set_xticklabels(class_labels, rotation=45)
ax.set_xlabel('Class')
ax.set_ylabel('Score')
ax.set_title('Recall and Precision by Class')

# Add legend
ax.legend()

# Adjust layout and save figure
fig.tight_layout()
plt.savefig(os.path.join(args.save_path, 'recall_precision.png'), dpi=300)
print(f"============= Results plotting finished =============")

 
