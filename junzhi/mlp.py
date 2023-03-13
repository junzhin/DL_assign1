import numpy as np
from components import * 
from typing import *
from logger import MetricLogger


class MLP:
    # for initiallization, the code will create all layers automatically based on the provided parameters.     
    def __init__(self, layers: List[int], activation: List[Optional[str]], weigth_decay = 0.99, loss = "MSE", batch_size = 1):
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
        self.weight_decay = weigth_decay
        self.loss = loss
        self.bs = batch_size
       
    
        
        self.activation=activation
        for i in range(len(layers)-1):      
            if i == len(layers) - 2:
                output_layer = True
            self.layers.append(HiddenLayer(layers[i],layers[i+1],activation[i],activation[i+1], output_layer = output_layer))

    # forward progress: pass the information through the layers and out the results of final output layer
    def forward(self,input):
        # reset self.masks to empty list            
        self.masks=[]
        for layer in self.layers:
            output=layer.forward(input)
            self.masks.append(layer.obtain_mask())
            input=output
        return output

    # define the objection/loss function, we use mean sqaure error (MSE) as the loss
    # you can try other loss, such as cross entropy.
    # when you try to change the loss, you should also consider the backward formula for the new loss as well!
    
    def criteron(self, y, y_hat):
        if self.loss == "MSE":
            return self.criterion_MSE(y, y_hat)
        elif self.loss == "CE":
            return self.criterion_CE(y, y_hat)
        else:
            raise ValueError("Unknown method: {}".format(self.loss))

                             
    def criterion_MSE(self,y,y_hat):
        activation_deriv=Activation(self.activation[-1]).f_deriv
        # MSE
        error = y-y_hat
        loss=error**2
        # calculate the MSE's delta of the output layer
        delta=-error*activation_deriv(y_hat)    
        # return loss and delta
        return loss,delta
    
    def criterion_CE(self,y,y_hat):
        activation_deriv=Activation(self.activation[-1]).f_deriv
        
        assert y.shape == y_hat.shape
        
        loss = -np.sum(y * np.log(y_hat + 1e-8))
        loss *= self.weight_decay
        
        delta = y/(y_hat + 1e-8) * activation_deriv(y_hat)
        
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
    def fit(self,X,y,learning_rate=0.1, epochs=100):
        """
        Online learning.
        :param X: Input data or features
        :param y: Input targets
        :param learning_rate: parameters defining the speed of learning
        :param epochs: number of times the dataset is presented to the network for learning
        """ 
  
        X=np.array(X)
        y=np.array(y)
        
        self.logger = MetricLogger(length = X.shape[0],mode='iter',batch_size=self.batch_size)
        num_batches = int(np.ceil(X.shape[0] / self.batch_size))
        
        
        
        for k in range(epochs):
            loss=np.zeros(X.shape[0])
            for it in range(X.shape[0]):
                i=np.random.randint(X.shape[0])
                
                # forward pass
                y_hat = self.forward(X[i])
                
                # backward pass
                loss,delta=self.criteron(y[i],y_hat)
                self.backward(delta)    
                
                # keep track of the loss
                self.logger.log(loss,)
    
                # update
                self.update(learning_rate)
            to_return[k] = np.mean(loss)
        return to_return

    # define the prediction function
    # we can use predict function to predict the results of new data, by using the well-trained network.
    def predict(self, x):
        x = np.array(x)
        output = np.zeros(x.shape[0])
        for i in np.arange(x.shape[0]):
            output[i] = self.forward(x[i,:])
        return output