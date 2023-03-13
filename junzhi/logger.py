import numpy as np
import matplotlib.pyplot as plt
from typing import *

class MetricLogger:
    def __init__(self, length: int, mode = "iter",batch_size = 1):
        self.train_loss: List[float] = []
        self.train_acc: List[float] = []
        self.val_loss: List[float] = []
        self.val_acc: List[float] = []
        self.train_epochs: List[int] = []
        self.val_epochs: List[int] = []
        self.train_iterations: int = 0
        self.val_iterations: int = 0
        self.mode = mode
        self.batch_size = batch_size
        self.data_length = length   
        

    def log_train(self, loss: float, acc: float, current_epoch: int):
        self.train_loss.append(loss)
        self.train_acc.append(acc)
        self.train_epochs(current_epoch)
        self.train_iterations += 1

    def log_val(self, loss: float, acc: float,  current_epoch: int):
        self.val_loss.append(loss)
        self.val_acc.append(acc)
        self.val_epochs(current_epoch)
        self.val_iterations += 1

    def print_last(self, current_epoch: int, current_iteration: int):
        print(f'Epoch {current_epoch} | Iteration {current_iteration} | ' \
              f'Train loss: {np.mean(self.train_loss[self.train_epochs == current_epoch])/self.batch_size:.4f} | Train accuracy: {np.mean(self.train_acc[self.train_epochs == current_epoch])/self.batch_size:.4f} | '
          f'Validation loss: {np.mean(self.val_loss[self.val_epochs == current_epoch])/self.batch_size:.4f} | Validation accuracy: {np.mean(self.val_acc[self.val_epochs == current_epoch])/self.batch_size:.4f}')

    def plot_loss(self) -> None:
        mode = self.mode
        if mode == "iter":
            plt.plot(self.train_loss, label='Train')
            plt.plot(self.val_loss, label='Validation')
            plt.title('Loss')
            plt.xlabel('itertions')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
        elif mode == "epoch":
            plt.plot(self.compute_averages(self.train_loss,self.train_epochs)/self.data_length, label='Train')
            plt.plot(self.compute_averages(self.val_loss,
                     self.val_epochs)/self.data_length, label='Validation')
            plt.title('Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

    def plot_accuracy(self) -> None:
        mode = self.mode
        if mode == "iter":
            plt.plot(self.train_acc, label='Train')
            plt.plot(self.val_acc, label='Validation')
            plt.title('Accuracy')
            plt.xlabel('iter')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()
        elif mode == "epoch":
            plt.plot(self.compute_averages(
                self.train_acc, self.train_epochs), label='Train')/self.data_length
            plt.plot(self.compute_averages(self.val_acc,self.val_epochs)/self.data_length, label='Validation')
            plt.title('Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()
            
            
    def compute_averages(self, arra: np.ndarray, indicator: str) -> np.ndarray:
        counts = np.bincount(indicator)
        sums = np.bincount(indicator, weights=arra) 
        averages = sums / counts
 
        return averages