import numpy as np


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
        shift = a - np.max(a)
        return np.exp(shift) / np.sum(np.exp(shift))
    
    def __softmax_deriv(self,s):
        # SM = s.reshape((-1, 1))
        # print(SM)
        jac = np.diagflat(s) - np.dot(s, s.T)
        return jac
        
 
    def __init__(self, activation='tanh', delta  = 0.01):
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
                 activation_last_layer='tanh', activation='tanh', W=None, b=None, output_layer = False, dropout = 1.0):
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
        # if activation == 'logistic':
        #     self.W *= 4

        # we set the size of bias as the size of output dimension
        self.b = np.zeros((1,n_out))

        # we set he size of weight gradation as the size of weight
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

    # the forward and backward progress (in the hidden layer level) for each training epoch
    # please learn the week2 lec contents carefully to understand these codes.
    
    def forward(self, input, isTraining = True):
        '''
        :type input: numpy.array
        :param input: a symbolic tensor of shape (n_in,)
        '''
        print("input size", input.shape)
        lin_output = np.dot(input, self.W) + self.b
        self.output = (
          
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
        self.input = input
        print('output size: ', self.output.shape)
        if isTraining and not self.output_layer:
            self.output = self.dropout_forward(self.output)
        else:
            self.mask = np.ones(input.shape)
        print("----"*50)
        return self.output
    
    def backward(self, delta, mask = None):
        
        self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
        self.grad_b = delta
        print("self.grad_W",self.grad_W.shape)
        print("self.grad_b",self.grad_b.shape)
        if self.activation_deriv:
            delta = delta.dot(self.W.T) * self.activation_deriv(self.input)
            delta = self.dropout_backward(delta, mask) if mask is not None else delta
        return delta
    
    def dropout_forward(self, input):
        self.mask = np.random.choice([0, 1], size=input.shape, p=[1-self.dropoutrate, self.dropoutrate])
        input *= self.mask
        return input
    
    def dropout_backward(self, delta, previous_masking):
        return delta *  previous_masking
    
    def obtain_mask(self):
        return self.mask