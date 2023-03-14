import numpy as np
from typing import *

class Activation(object):
    def __tanh(self, x):
        return np.tanh(x)

    def __tanh_deriv(self, a):
        # a = np.tanh(x)
        return 1.0 - a**2

    def __logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __logistic_deriv(self, a):
    
        return a * (1 - a)
    
    def __relu(self,a):
        return np.maximum(0,a)
  
    def __relu_deriv(self,a):
        return np.where(a <= 0, 0, 1)
 
        
    def __leakyrelu(self,a):
        return np.maximum(self.delta * a, a)
    
    def __leakyrelu_deriv(self, a):
        return np.where(a <= 0, self.delta, 1)
    
    def __softmax(self, a):
        shift = a - np.max(a, axis=1, keepdims=True)

        # if self.indicator  is False:
        #     print("a", a.shape)
        #     print('a : ', a )
        #     print('np.max(a, axis=1, keepdims=True): ', np.max(a, axis=1, keepdims=True))
        #     print('shift: ', shift.shape)
        #     print("shift")
        #     print(shift)
        #     print(np.exp(shift) / np.sum(np.exp(shift)))
        #     print(' np.sum(np.exp(shift)): ',  np.sum(np.exp(shift), axis=1, keepdims=True))
        #     self.indicator = True
      
        return np.exp(shift) / np.sum(np.exp(shift), axis=1, keepdims=True)
    
    # 这里不一定用的到， 其中对于softmax 的导数， 一般用的是交叉熵的导数 结合使用
    def __softmax_deriv(self,a):
        a = a.reshape((-1,1))
        print("a",a.shape)
        jac = np.diagflat(a) - np.dot(a, a.T)
        return jac
        
 
    def __init__(self, activation='tanh', delta  = 0.01):
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
    def __init__(self, n_in, n_out,
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
        self.batch_mean = []
        self.batch_var = []
        self.gamma =  np.ones((1, n_in)) 
        self.beta =  np.zeros((1, n_in)) 
        # activation deriv of last layer
        self.activation_deriv = None
        if activation_last_layer:
            self.activation_deriv = Activation(activation_last_layer).f_deriv

        # we randomly assign small values for the weights as the initiallization
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

    # the forward and backward progress (in the hidden layer level) for each training epoch
    # please learn the week2 lec contents carefully to understand these codes.
    
    def forward(self, input: np.ndarray, isTraining: bool = True) -> np.ndarray:
        '''
        :type input: numpy.array
        :param input: a symbolic tensor of shape (n_in,)
        '''
        # https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwimv42a_tv9AhUsmlYBHSO9BYQQFnoECAwQAQ&url=https%3A%2F%2Fgithub.com%2Frenan-cunha%2FBatchNormalization&usg=AOvVaw28oNAzfY7iGhQg3qVBktzV
        if self.batch_norm and isTraining is True:
            mean = input.mean(axis=0, keepdims=True)
            var = input.var(axis=0, keepdims=True)
            self.input_normalized = (input - mean) / np.sqrt(var + 1e-18)
            input = self.gamma * self.input_normalized + self.beta
            self.batch_mean.append(mean)
            self.batch_var.append(var)  
        elif self.batch_norm and isTraining is False:
            input_mean = np.mean(self.batch_mean)
            input_var = np.mean(self.batch_var)
            input = (input - input_mean) / np.sqrt(input_var + 1e-18)
            input = input * self.gamma + self.beta
            
                      
        lin_output = np.dot(input, self.W) + self.b
        self.output = (
          
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
        self.input = input
        if isTraining and not self.output_layer:
            self.output = self.dropout_forward(self.output)
        else:
            self.mask = np.ones(input.shape)
        return self.output
    
    def backward(self, delta, mask = None):
        self.grad_W = np.atleast_2d(self.input).T.dot(
            np.atleast_2d(delta))
        self.grad_b = np.average(delta, axis=0) 
        
       
        
        if self.weight_decay is not None:
            self.grad_W += self.weight_decay * self.W  
            
        if self.activation_deriv:
            delta = delta.dot(self.W.T) * self.activation_deriv(self.input)
            delta = self.dropout_backward(delta, mask) if mask is not None else delta
            
            if self.batch_norm:
                self.grad_gamma_BN = np.mean(delta, axis=0, keepdims=True) *self.input_normalized
                self.grad_beta_BN = np.mean(delta)
                # print("self.grad_gamma_BN.shape", self.grad_gamma_BN.shape)
                # print("self.grad_gamma_BN.shape", self.grad_beta_BN.shape)
                
        return delta
    
    def dropout_forward(self, input):
        self.mask = np.random.binomial(1, 1 - self.dropoutrate, size=input.shape) 
        # self.mask = np.random.choice([0, 1], size=input.shape, p=[1-self.dropoutrate, self.dropoutrate])
        input *= self.mask
        return input
    
    def dropout_backward(self, delta, previous_masking):
        return delta *  previous_masking
    
    def obtain_mask(self):
        return self.mask