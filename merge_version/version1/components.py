import numpy as np
from typing import *

class Activation(object):
    def __tanh(self, x: np.ndarray) -> float:
        return np.tanh(x)

    def __tanh_deriv(self, a: np.ndarray) -> np.ndarray:
        # a = np.tanh(x)
        return 1.0 - a**2

    def __logistic(self, x: np.ndarray) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def __logistic_deriv(self, a: np.ndarray) -> float:
    
        return a * (1 - a)
    
    def __relu(self,a: np.ndarray) -> np.ndarray:
        return np.maximum(0,a)
  
    def __relu_deriv(self,a: np.ndarray) -> np.ndarray:
        return np.where(a <= 0, 0, 1)
 
        
    def __leakyrelu(self,a: np.ndarray) -> float:
        return np.maximum(self.delta * a, a)
    
    def __leakyrelu_deriv(self, a: np.ndarray) -> np.ndarray:
        return np.where(a <= 0, self.delta, 1)
    
    def __softmax(self, a: np.ndarray) -> np.ndarray:
        shift = a - np.max(a, axis=1, keepdims=True)
        return np.exp(shift) / np.sum(np.exp(shift), axis=1, keepdims=True)
    
    # 这里不一定用的到， 其中对于softmax 的导数， 一般用的是交叉熵的导数 结合使用
    def __softmax_deriv(self, a: np.ndarray) -> np.ndarray:
        a = a.reshape((-1,1))
        jac = np.diagflat(a) - np.dot(a, a.T)
        return jac
        
 
    def __init__(self, activation: str='tanh', delta: float = 0.01):
        self.indicator = False
        if activation == 'logistic':
            self.f = self.__logistic
            self.f_deriv = self.__logistic_deriv
        elif activation == 'tanh':
            self.f = self.__tanh
            self.f_deriv = self.__tanh_deriv
        elif activation == 'relu':
            self.f = self.__relu
            self.f_deriv = self.__relu_deriv
        elif activation == "leakyrelu":
            self.delta = delta
            self.f = self.__leakyrelu
            self.f_deriv = self.__leakyrelu_deriv
        elif activation == "softmax":
      
            self.f = self.__softmax
            self.f_deriv = self.__softmax_deriv
                 
class HiddenLayer(object):
    def __init__(self, n_in: int, n_out: int,
                 activation_last_layer='tanh', activation='tanh', W=None, b=None, output_layer = False, dropout = 1.0, weight_decay = None, batch_norm = False):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: string
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.mask = None
        self.input = None
        self.activation = Activation(activation).f
        self.dropoutrate=dropout
        self.output_layer = output_layer
        self.weight_decay = weight_decay
        self.dropout = dropout
        
        
        self.batch_norm = batch_norm
        self.batch_mean = np.zeros((1, n_in))
        self.batch_var = np.zeros((1, n_in))
        self.gamma =  np.ones((1, n_in)) 
        self.beta =  np.zeros((1, n_in)) 
        # activation deriv of last layer
        self.activation_deriv = None
        if activation_last_layer:
            self.activation_deriv = Activation(activation_last_layer).f_deriv

        # we randomly assign small values for the weights as the initiallization
        
        if self.activation == 'relu' or self.activation == 'leakyrelu':
            self.W = np.random.uniform(low=-np.sqrt(6. / n_in), high=np.sqrt(6. / n_in), size=(n_in, n_out))
        else:            
            self.W = np.random.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)
            )
        

        # we set the size of bias as the size of output dimension
        self.b = np.zeros((1,n_out),)
        
        # we set he size of weight gradation as the size of weight
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)
        self.v_W = np.zeros_like(self.grad_W)
        self.v_b = np.zeros_like(self.grad_b)


    
    def forward(self, input: np.ndarray, isTraining: bool = True,dropout_predict = False) -> np.ndarray:
        '''
        :type input: numpy.array
        :param input: a symbolic tensor of shape (n_in,)
        '''
        # https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwimv42a_tv9AhUsmlYBHSO9BYQQFnoECAwQAQ&url=https%3A%2F%2Fgithub.com%2Frenan-cunha%2FBatchNormalization&usg=AOvVaw28oNAzfY7iGhQg3qVBktzV
        # https://github.com/renan-cunha/BatchNormalization
        
        if isTraining and not self.output_layer:
            input = self.dropout_forward(input)
        elif self.output_layer:
            self.mask = np.ones(input.shape)
        
        if self.batch_norm and isTraining is True:
            mean = input.mean(axis=0, keepdims=True)
            var = input.var(axis=0, keepdims=True)
            self.input_normalized = (input - mean) / np.sqrt(var + 1e-18)
            input = self.gamma * self.input_normalized + self.beta
            
            # we implement the batch normalization in the forward progress with a momentum to keep the mean and var stable, instead of using the numpy arrays to keep all means and var for each iteration during trianing and compute the means for batch mean and standard deviation during inference, we found that this approach is more efficient and speed up the training process significantly.
            self.batch_mean = self.batch_mean * 0.9 + mean* 0.1
            self.batch_var = self.batch_var * 0.9 + var * 0.1  
        elif self.batch_norm and isTraining is False:
            input = (input - self.batch_mean) / np.sqrt(self.batch_var + 1e-18)
            input = input * self.gamma + self.beta
        
        scale_factor = 1.0
        if isTraining is False and self.dropoutrate < 1.0:
            scale_factor = self.dropoutrate
        else:
           scale_factor = 1.0
                           
        lin_output = np.dot(input, self.W * scale_factor) + self.b * scale_factor
        self.output = (
          
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
        self.input = input
       
    
        return self.output
    
    
    def backward(self, delta: np.ndarray) -> None:
        self.grad_W = np.atleast_2d(self.input).T.dot(
            np.atleast_2d(delta))
        self.grad_b = np.average(delta, axis=0) 
        
        if self.weight_decay is not None:
            self.grad_W += self.weight_decay * self.W  
            
        if self.activation_deriv:
            delta = delta.dot(self.W.T) * self.activation_deriv(self.input)
            delta = self.dropout_backward(delta)
            
            # retreved from https://towardsdatascience.com/implementing-batch-normalization-in-python-a044b0369567
            if self.batch_norm:
            
                self.grad_gamma = np.sum(
                    delta  * self.input_normalized, axis=0)
                self.grad_beta = np.sum(delta, axis=0)
                
        return delta
    
    def dropout_forward(self, input: np.ndarray) -> np.ndarray:
        self.mask = np.random.binomial(1, self.dropoutrate, size=input.shape) 
        # self.mask = np.random.choice([0, 1], size=input.shape, p=[1-self.dropoutrate, self.dropoutrate])
        input = input * self.mask
        
        #inverted dropout
        # input = input * self.mask/self.dropoutrate
        return input
    
    def dropout_backward(self, delta: np.ndarray) -> np.ndarray:
        assert self.mask.shape == delta.shape
        return delta * self.mask 
    
    def obtain_mask(self):
        return self.mask