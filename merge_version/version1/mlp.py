import numpy as np
from components import * 
from typing import *
from util import *
from optimizer import *
from sklearn.metrics import *
from tqdm import tqdm
class MLP:
    # for initiallization, the code will create all layers automatically based on the provided parameters.     
    def __init__(self, X_test: np.ndarray, y_test: np.ndarray, layers: List[int], activation: List[Optional[str]], weight_decay: float = 0.01, loss: str = "MSE", batch_size: int = 1, dropoutRate: float = 0.5, beta: List[float] = [0.9,0.999], batch_norm: bool = False):
         
        # initialize layers
        self.layers: List[HiddenLayer]=[]
        self.params=[]
        self.masks=[]
        self.weight_decay = weight_decay
        self.loss = loss
        self.batch_size = batch_size
        self.X_test = X_test
        self.y_test = y_test
        self.dropoutRate = dropoutRate
        self.activation=activation
        self.step_count = 0
        self.batch_norm = batch_norm
        
        self.beta1 = beta[0]
        self.beta2 = beta[1]
        self.breaking_point = 5
 
        
        
        first_layer = True
        for i in range(len(layers)-1):      
            if i > 0 :
                first_layer = False
            self.layers.append(HiddenLayer(layers[i],layers[i+1],activation[i],activation[i+1], output_layer = first_layer,dropout=self.dropoutRate, weight_decay=self.weight_decay, batch_norm=self.batch_norm if i != 0 else False))  

 
    def criterion(self, y: np.ndarray, y_hat: np.ndarray, isTraining: bool = True):
        """Compute the loss of the network's prediction with respect to the ground truth.

        Args:
            y (np.ndarray): The ground truth.
            y_hat (np.ndarray): The network's prediction.
            isTraining (bool, optional): Whether the network is in training mode. Defaults to True.

        Returns:
            float: The loss value.
        """
        if self.loss == "MSE":
            return self.criterion_MSE(y,y_hat, isTraining)
        elif self.loss == "CE":
            return self.criterion_CE(y,y_hat, isTraining)
        
        
           
    def criterion_MSE(self, y: np.ndarray, y_hat: np.ndarray, isTraining: bool = True):
        """Computes the Mean Squared Error between the true and predicted values of the data.
    
        Args:
            y: The true values of the data.
            y_hat: The predicted values of the data.
            isTraining: Whether this is being used for training or testing.
    
        Returns:
            The Mean Squared Error between the true and predicted values of the data.
        """

 
        activation_deriv=Activation(self.activation[-1]).f_deriv
        # MSE
        y = Data_Proprocesing.one_encoding(y)
        
        error = y - y_hat
        loss = error**2
        
        if isTraining is False:
            
            return np.sum(loss)/y.shape[0], None
        
        delta = -error * activation_deriv(y_hat) / y.shape[0]
        return np.sum(loss)/y.shape[0], delta
    
    
    def criterion_CE(self, y:  np.ndarray, y_hat:  np.ndarray, isTraining: bool = True):
        """Computes the cross-entropy loss between the true labels, y, and the predictions, y_hat.
        Args:
            y: The true labels, of shape (batch_size, num_classes).
            y_hat: The model predictions, of shape (batch_size, num_classes).
            isTraining: A boolean indicating whether this is a training step or not. This is used to determine whether to update the running mean and standard deviation.
        Returns:
            The loss, as a scalar.
        """
        
        y = Data_Proprocesing.one_encoding(y)
        # print("y: ",y.shape)
        # print("y_hat", y_hat.shape)
        # print("After scaling loss", loss)
       

        assert y.shape == y_hat.shape

        number_of_sample = y.shape[0]
        loss = - np.nansum(y * np.log(y_hat + 1e-30))
        loss = loss / number_of_sample
        # print("Original loss", loss)
        if isTraining == False:
            return loss, None
        
        # see https://levelup.gitconnected.com/killer-combo-softmax-and-cross-entropy-5907442f60ba#:~:text=Putting%20It%20All%20Together
        
        delta = -(y - y_hat) / number_of_sample
        return loss, delta

    # forward progress: pass the information through the layers and out the results of final output layer
    def forward(self, input:  np.ndarray, isTraining: bool = True, dropout_predict: bool = False, early_stop: bool = False):
        # This is the forward propagation function that is used to predict the output
        # for a given input. It is called when the predict() function is called.
 
        # reset self.masks to empty list            
        for layer in self.layers:
            output=layer.forward(input, isTraining=isTraining, dropout_predict=dropout_predict, early_stopping=early_stop)
            input=output
        return output

    # backward progress  
    def backward(self, delta: np.ndarray):
        # Computes the backward pass of the ReLU activation function
        # delta is the gradient of the loss with respect to the output of the activation function
        # Returns the gradient of the loss with respect to the input of the activation function
 
        for layerIndex in reversed(range(len(self.layers))):
            # print("layer: ", layerIndex)
            delta = self.layers[layerIndex].backward(delta)
                
         

    # update the network weights after backward.
    # make sure you run the backward function before the update function! 
    def optimizer_init(self, method: str) -> None:
        """
        Initialize the optimizer with the given method.
        """
 
        
        if method == "sgd":
            self.opt = sgd()
        elif method == "sgd_momentum":
            self.opt= sgd_momentum()  
        elif method == "adam":
            self.opt= adam(self.beta1, self.beta2)  
        elif method == "rmsprop":
            self.opt = RMSprop(self.beta2)
        
    def update(self,lr: float, step_count: int)->None:         
        if step_count == 1:
            self.opt.reset()
            
        for index, layer in enumerate(self.layers):
            # print("layer: ", index)
            if step_count == 1:
                self.opt.init_first_step(layer.grad_W, layer.grad_b)
                
            layer.W, layer.b = self.opt.update_parameter(
                index, step_count,layer,lr, layer.W, layer.b, layer.grad_W, layer.grad_b)
            
            if layer.batch_norm == True:
                # print("layer.gamma", layer.gamma.shape)
                # print("layer.grad_gamma", layer.grad_gamma.shape)
                layer.gamma -= lr * layer.grad_gamma
                layer.beta -= lr * layer.grad_beta
       

    # define the training function
    # it will return all losses within the whole training process.
    def fit(self,X:np.ndarray,y:np.ndarray,learning_rate:float=0.1, epochs:int=100, opt: str ='sgd', early_stop: bool = False):
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
        self.early_stop_mode = early_stop
        
        early_stop_torlerance_runs = 0
        train_loss_per_epochs = []
        val_loss_per_epochs = []
        train_acc_per_epochs = []
        val_acc_per_epochs = []
        train_f1_per_epochs = []
        val_f1_per_epochs = []
        
        best_metric_score = np.inf
        self.optimizer_init(opt)
        num_batches = int(np.ceil(X.shape[0] / self.batch_size))
        
        self.step_count = 1

        for k in range(epochs):
            
            
            index = 0
            current_batch_size = self.batch_size
            X,y = Data_Proprocesing.shuffle_randomly(X,y)


            for batch_indx in tqdm(range(num_batches)):
            
                # forward pass
                y_hat = self.forward(X[index: index + current_batch_size,])
                
                
                # backward pass
                loss, delta = self.criterion(
                    y[index:index + current_batch_size, :], y_hat, isTraining=True)
                
           
                self.backward(delta) 
                
                # update the model parameters
                self.update(learning_rate,step_count=self.step_count)   
                
                
                # correct the batch size if it is the last batch is not full
                index += current_batch_size
                if index + self.batch_size > X.shape[0]:
                    current_batch_size = X.shape[0] - index
                                
                
                
                self.step_count += 1
                
            # keep track of experiment results
            y_train_pred = self.predict(X)
            train_loss, _ = self.criterion(
                y, y_train_pred, isTraining=False)
            train_loss_per_epochs.append(train_loss)
            train_acc_per_epochs.append(accuracy_score(y,  np.expand_dims(np.argmax(y_train_pred, axis=1),axis=1)))
            train_f1_per_epochs.append(f1_score(y,  np.expand_dims(np.argmax(y_train_pred, axis=1),axis=1), average='macro'))
            
            y_test_pred = self.predict(self.X_test)
            val_loss, _ = self.criterion(
                self.y_test, y_test_pred, isTraining=False)
            val_loss_per_epochs.append(val_loss)
            val_acc_per_epochs.append(accuracy_score(
                self.y_test, np.expand_dims(np.argmax(y_test_pred, axis=1), axis=1)))
            val_f1_per_epochs.append(f1_score(self.y_test,  np.expand_dims(np.argmax(y_test_pred, axis=1),axis=1), average='macro')) 
            
            # early stopping implementation
            if self.early_stop_mode:
                if val_loss < best_metric_score:
                    best_metric_score = val_loss
                    early_stop_torlerance_runs = 0
                    self.early_stop_save()
                    print(".....saving model!")
                    
                else:
                    early_stop_torlerance_runs += 1
                    
                print('val_loss: ', val_loss)
                print('best_val_loss: ', best_metric_score)
                print('early_stop_torlerance_runs: ', early_stop_torlerance_runs)
                
                if early_stop_torlerance_runs >= self.breaking_point:
                    break
                   
        
            print(
                f'Epoch: {k:3} | ' f'itrs: {self.step_count:5} |'
                f' train_loss_per_epochs : {train_loss_per_epochs[-1]:.4f} | '
                f' train_acc_per_epochs : {train_acc_per_epochs[-1]:.4f} | '
                f'val_loss_per_epochs : {val_loss_per_epochs[-1]:.4f} |'
                f' val_acc_per_epochs : {val_acc_per_epochs[-1]:.4f} |'
                f' train_f1_per_epochs : {train_f1_per_epochs[-1]:.4f} |'
                f' val_f1_per_epochs : {val_f1_per_epochs[-1]:.4f} |'
            )

        statistic = dict()
        statistic['train_loss_per_epochs'] = train_loss_per_epochs
        statistic['val_loss_per_epochs'] = val_loss_per_epochs
        statistic['train_acc_per_epochs'] = train_acc_per_epochs
        statistic['val_acc_per_epochs'] = val_acc_per_epochs  
        statistic['train_f1_per_epochs'] = train_f1_per_epochs
        statistic['val_f1_per_epochs'] = val_f1_per_epochs

        
        return  statistic

    
    def early_stop_save(self):
        """Save the model if the validation loss has decreased"""

        for layerIndex in reversed(range(len(self.layers))):
            self.layers[layerIndex].early_stopping_update()
       
       
    def predict_early_stop(self, x: np.ndarray):
        """
        Predicts the output of the model given a set of input data points.
        If the model has not implemented early stopping in a fit method, this function will raise an
        exception.
        
        :param x: the input data points
        :return: the output of the model
        """
    
        assert self.early_stop_mode == True, "early_stop_mode is not activated"
        x = np.array(x)
        output = []
        for i in np.arange(x.shape[0]):
            output.append(self.forward(x[i,:], isTraining=False, early_stop=True))
        return np.array(output).squeeze(axis=1) 
        
            
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Given a set of input features, predicts the output values for each
        input feature. This function is called in the predict function in
        the Model class.
        """ 
        x = np.array(x)
        output = []
        for i in np.arange(x.shape[0]):
            output.append(self.forward(x[i,:], isTraining=False))
    
        return np.array(output).squeeze(axis=1) 
    
    
    
    
