import numpy as np
from elements import Activation, HiddenLayer
class MLP:
    """
    """ 

    # for initiallization, the code will create all layers automatically based on the provided parameters.     
    def __init__(self, layers, activation=[None,'tanh','tanh']):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """        
        ### initialize layers
        self.layers=[]
        self.params=[]
        self.masks=[]
        
        self.activation=activation
        for i in range(len(layers)-1):
            self.layers.append(HiddenLayer(layers[i],layers[i+1],activation[i],activation[i+1]))


    # define the objection/loss function, we use mean sqaure error (MSE) as the loss
    # you can try other loss, such as cross entropy.
    # when you try to change the loss, you should also consider the backward formula for the new loss as well!
    
    def criterion_MSE(self,y,y_hat):
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
    
    # forward progress: pass the information through the layers and out the results of final output layer
    def forward(self,input, isTraining = True):
        for layer in self.layers:
            output=layer.forward(input)
            if isTraining:
                self.masks.append(layer.mask)
            input=output
        return output

    # backward progress  
    def backward(self,delta):

        for index in reversed(range(len(self.layers))):
            
            delta=self.layers[index].backward(delta,output_layer=True)
            delta=layer.backward(delta)

    # update the network weights after backward.
    # make sure you run the backward function before the update function!    
    def update(self,lr):
        for layer in self.layers:
            layer.W -= lr * layer.grad_W
            layer.b -= lr * layer.grad_b

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
        return_info = np.zeros(epochs)
        
        for k in range(epochs):
            loss=np.zeros(X.shape[0])
            for it in range(X.shape[0]):
                i=np.random.randint(X.shape[0])
                
                # forward pass
                y_hat = self.forward(X[i])
                
                # backward pass
                loss[it],delta=self.criterion_MSE(y[i],y_hat)
                self.backward(delta)
                y
                # update
                self.update(learning_rate)
            return_info[k] = np.mean(loss)
        return return_info

    # define the prediction function
    # we can use predict function to predict the results of new data, by using the well-trained network.
    def predict(self, x):
        x = np.array(x)
        output = []
        for i in np.arange(x.shape[0]):
            output[i] = self.forward(x[i,:])
        return np.array(output)