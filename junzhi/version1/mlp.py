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
        print("self.X_test.shape", self.X_test.shape)
        print("self.y_test", self.y_test.shape)
        self.y_test_label = Data_Proprocesing.decode_one_encoding(self.y_test)
        self.dropoutRate = dropoutRate
        self.activation=activation
        
        output_layer = False
        for i in range(len(layers)-1):      
            if i == len(layers) - 2:
                output_layer = True
            self.layers.append(HiddenLayer(layers[i],layers[i+1],activation[i],activation[i+1], output_layer = output_layer,dropout=self.dropoutRate))

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

    # define the objection/loss function, we use mean sqaure error (MSE) as the loss
    # you can try other loss, such as cross entropy.
    # when you try to change the loss, you should also consider the backward formula for the new loss as well!
    
    def criteron(self, y, y_hat,require_grad = True):
        if self.loss == "MSE":
            return self.criterion_MSE(y, y_hat, require_grad = require_grad)
        elif self.loss == "CE":
            return self.criterion_CE(y, y_hat, require_grad=require_grad)
        else:
            raise ValueError("Unknown method: {}".format(self.loss))

                             
    def criterion_MSE(self,y,y_hat, require_grad = True):
        activation_deriv=Activation(self.activation[-1]).f_deriv
        # MSE
        error = y-y_hat
        loss=error**2
        # calculate the MSE's delta of the output layer
        delta=-error*activation_deriv(y_hat)    
        # return loss and delta
        return loss,delta
    
    def criterion_CE(self,y,y_hat, require_grad = True):
        activation_deriv=Activation(self.activation[-1]).f_deriv
        
        assert y.shape == y_hat.shape
        
        loss = -np.sum(y * np.log(y_hat + 1e-8))
        loss *= self.weight_decay
        
        print("y shape: ", y.shape)
        print("y_hat shape: ", y_hat.shape)
        
        if require_grad == False:
            return loss, None
        delta = (y - y_hat) * activation_deriv(y_hat)
        return loss, delta
        
        return loss,delta
    # backward progress  
    def backward(self,delta):
        delta=self.layers[-1].backward(delta,self.masks[-1])
        for layerIndex in reversed(range(len(self.layers[:-1]))):
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
            pass
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
        
        print(X.shape)
        print(y.shape)
        
        self.logger = MetricLogger(length = X.shape[0],mode='iter',batch_size=self.batch_size)
        num_batches = int(np.ceil(X.shape[0] / self.batch_size))

        for k in range(epochs):
            loss=np.zeros(X.shape[0])
            
            index = 0
            current_batch_size = self.batch_size
            X,y = Data_Proprocesing.shuffle_randomly(X,y)
            y_label = Data_Proprocesing.decode_one_encoding(y)
            
            for _ in range(num_batches):
                
                # forward pass
                y_hat = self.forward(X[index: index + current_batch_size,])
                
                print('y_hat: ', y_hat)
                print('y', y[index:index + current_batch_size,:])
                
                
                # backward pass
                loss, delta=self.criteron(y[index:index + current_batch_size,:],y_hat)
                
           
 
                self.backward(delta) 
                
                # update the model parameters
                self.update(learning_rate,method = opt)   
                
                # keep track of the training loss and accuracy
            
                y_predict_label = Data_Proprocesing.decode_one_encoding(y_hat)
                print('y_predict_label: ', y_predict_label)
                print("y_label[index:index + current_batch_size]",
                      y_label[index:index + current_batch_size])
                
               
                self.logger.log_train(loss,Data_Proprocesing.accuarcy(y_label[index:index + current_batch_size],y_predict_label),k)
    
                # update of the current batch size if the batch size is not divisible by the number of batch size
                if index + self.batch_size > X.shape[0]:
                    current_batch_size = X.shape[0] - index
                else:
                    index += current_batch_size
                    
            # keep track of the validation loss and accuracy
            y_test_predict = self.predict(self.X_test)
            y_test_predict_label = Data_Proprocesing.decode_one_encoding(y_test_predict)
            print("self.y_test", self.y_test.shape)
            print("y_test_predict.shape",y_test_predict.squeeze(axis=1).shape)
            val_loss= self.criteron(self.y_test, y_test_predict.squeeze(axis=1),require_grad = False)[0]
            self.logger.log_val(val_loss, Data_Proprocesing.accuarcy(self.y_test_label,y_test_predict_label),k)
            
            self.logger.print_last(k)
    
        return self.logger

    # define the prediction function
    # we can use predict function to predict the results of new data, by using the well-trained network.
    def predict(self, x):
        x = np.array(x)
        print("++++")
        print("x.shape", x.shape)
        print("+++++")
        output = []
        for i in np.arange(x.shape[0]):
            output.append(self.forward(x[i,:], isTraining=False))
        return np.array(output)