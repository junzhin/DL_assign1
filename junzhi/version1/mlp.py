import numpy as np
from components import * 
from typing import *
from logger import MetricLogger
from util import *


class MLP:
    # for initiallization, the code will create all layers automatically based on the provided parameters.     
    def __init__(self, X_test, y_test,layers: List[int], activation: List[Optional[str]], weight_decay = 0.99, loss = "MSE", batch_size = 1, dropoutRate = 0.5):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """        
        ### initialize layers
         
        self.layers: List[HiddenLayer]=[]
        self.params=[]
        self.masks=[]
        self.weight_decay = weight_decay
        self.loss = loss
        self.batch_size = batch_size
        self.X_test = X_test
        self.y_test = y_test
        self.y_test_label = Data_Proprocesing.decode_one_encoding(self.y_test)
        self.dropoutRate = dropoutRate
        self.activation=activation
        
        output_layer = False
        for i in range(len(layers)-1):      
            if i == len(layers) - 2:
                output_layer = True
            self.layers.append(HiddenLayer(layers[i],layers[i+1],activation[i],activation[i+1], output_layer = output_layer,dropout=self.dropoutRate))


    # define the objection/loss function, we use mean sqaure error (MSE) as the loss
    # you can try other loss, such as cross entropy.
    # when you try to change the loss, you should also consider the backward formula for the new loss as well!
 

    
    def criterion_CE(self, y, y_hat, isTraining = True):
        y = Data_Proprocesing.one_encoding(y)
        assert y.shape == y_hat.shape
        number_of_sample = y.shape[0]
        loss = - np.nansum(y * np.log(y_hat + 1e-30))
        loss = loss / number_of_sample
        loss = loss*self.weight_decay
        
        if isTraining == False:
            return loss, None
        delta = (y_hat - y) / (self.batch_size)

        return loss, delta
    
    
    # forward progress: pass the information through the layers and out the results of final output layer
    def forward(self,input, isTraining = True):
        # reset self.masks to empty list            
        self.masks=[]
        for layer in self.layers:
            output=layer.forward(input, isTraining=isTraining)
            if isTraining == True:
                self.masks.append(layer.obtain_mask())
            input=output
        return output

    # backward progress  
    def backward(self,delta):
        for layerIndex in reversed(range(len(self.layers))):
            if layerIndex == 0:
                delta = self.layers[layerIndex].backward(delta, None)
            else:
                 delta = self.layers[layerIndex].backward(delta, self.masks[layerIndex-1])
                
         

    # update the network weights after backward.
    # make sure you run the backward function before the update function!    
    def update(self,lr, method: str = "sgd"):
        
        if method == "sgd":
            for layer in self.layers:
                layer.W -= lr * layer.grad_W
                layer.b -= lr * layer.grad_b
        elif method == "momentum":
            for layer in self.layers:
                layer.v_W = (0.9* layer.v_W) + (lr * layer.grad_W)
                layer.v_b = (0.9 * layer.v_b) + (lr * layer.grad_b)
                layer.W = layer.W - layer.v_W
                layer.b = layer.b - layer.v_b
        elif method == "adam":
            pass
            
            

    # define the training function
    # it will return all losses within the whole training process.
    def fit(self,X,y,learning_rate=0.1, epochs=100, opt = 'sgd'):
        """
        Online learning.
        :param X: Input data or features
        :param y: Input targets
        :param learning_rate: parameters defining the speed of learning
        :param epochs: number of times the dataset is presented to the network for learning
        """ 
  
        X=np.array(X)
        y=np.array(y)
        
        print("X shape",X.shape)
        print("y shape",y.shape)
        
        train_loss_per_epochs = []
        val_loss_per_epochs = []
        train_acc_per_epochs = []
        val_acc_per_epochs = []     
 
        
        num_batches = int(np.ceil(X.shape[0] / self.batch_size))

        for k in range(epochs):
            loss=np.zeros(X.shape[0])
            
            index = 0
            current_batch_size = self.batch_size
            X,y = Data_Proprocesing.shuffle_randomly(X,y)
     
            
            current_train_loss = []
            for _ in range(num_batches):
                
                # forward pass
                y_hat = self.forward(X[index: index + current_batch_size,])
                # print('y_hat: ', y_hat.shape)
                # print('y[index:index + current_batch_size, :].shape',
                #       y[index:index + current_batch_size, :].shape)
        
                # backward pass
                loss, delta = self.criterion_CE(
                    y[index:index + current_batch_size, :], y_hat, isTraining=True)
                
                current_train_loss.append(loss)
    
                self.backward(delta) 
                
                # update the model parameters
                self.update(learning_rate,method = opt)   
                
           
                if index + self.batch_size > X.shape[0]:
                    current_batch_size = X.shape[0] - index
                else:
                    index += current_batch_size
                
                # print(current_train_loss)
            # train_loss_per_epochs.append(np.mean(current_train_loss))
            y_train_pred = self.predict(X)
            train_loss,_ = self.criterion_CE(
                y, y_train_pred, isTraining=False)
            train_loss_per_epochs.append(train_loss)
            
            train_acc_per_epochs.append(
                Data_Proprocesing.accuarcy(y, y_train_pred))
            
            y_test_pred = self.predict(self.X_test)
            val_loss,_ = self.criterion_CE(
                self.y_test, y_test_pred, isTraining=False)
            val_loss_per_epochs.append(val_loss)
            val_acc_per_epochs.append(
                Data_Proprocesing.accuarcy(self.y_test, y_test_pred))
            
           
            print(
                f'Epoch {k} |' f'train_loss_per_epochs : {train_loss_per_epochs[-1]:.4f} |' f' train_acc_per_epochs : {train_acc_per_epochs[-1]:.4f} |' f'val_loss_per_epochs : {val_loss_per_epochs[-1]:.4f} |' f' val_acc_per_epochs : {val_acc_per_epochs[-1]:.4f} |')
     
            
        return  

    # define the prediction function
    # we can use predict function to predict the results of new data, by using the well-trained network.
    def predict(self, x):
        x = np.array(x)
        output = []
        for i in np.arange(x.shape[0]):
            output.append(self.forward(x[i,:], isTraining=False))
        return np.array(output).squeeze(axis=1)