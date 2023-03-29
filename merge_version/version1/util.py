import numpy as np

class Data_Proprocesing:
    # This is a class for data preprocessing. It contains some useful functions for data preprocessing.
    def __init__(self) -> None:
        pass
    
    # This function is used to standardize the data.
    def standardize(data:np.ndarray):
        return (data - np.mean(data)) / np.std(data)
    
    # This function is used to one hot encoding the data.
    def one_encoding(X: np.ndarray) -> np.ndarray:
        
        numOfClasses = 10
        
        one_encoding_of_X = []
        
        for each_x in X:
            one_encoding_of_each_x = np.zeros(numOfClasses)
            one_encoding_of_each_x[int(each_x)] = 1
            one_encoding_of_X.append(one_encoding_of_each_x)
                    
        return np.array(one_encoding_of_X)
    
    # This function is used to decode the one hot encoding data.
    def decode_one_encoding(one_encoding: np.ndarray):
        return np.expand_dims(np.argmax(one_encoding, axis=1), axis=1)
        
   # This function is used to shuffle the data into random training set and validation set      
    def shuffle_randomly(X, y):  
        randomize = np.arange(X.shape[0])
        np.random.shuffle(randomize)
        return X[randomize], y[randomize]
    
    # This function is used to compute the accuarcy of the model.
    def accuarcy( y_true: np.ndarray, y_pred:np.ndarray):
        y_pred = np.expand_dims(np.argmax(y_pred, axis=1),axis=1)
        return np.sum(y_true == y_pred) / len(y_true)
    
    # This function is used to compute the confusion matrix of the model.
    def recall_precision_from_confusion_matrix( confusion_matrix: np.ndarray):
        tp = np.diag(confusion_matrix)
        fp = np.sum(confusion_matrix, axis=0) - tp
        fn = np.sum(confusion_matrix, axis=1) - tp
        
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        
        return recall, precision
