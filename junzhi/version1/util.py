import numpy as np

class Data_Proprocesing:
    def __init__(self) -> None:
        pass
    
    def standardize(data:np.ndarray):
        return (data - np.mean(data)) / np.std(data)
    
    def one_encoding(X: np.ndarray) -> np.ndarray:
        
        numOfClasses = 10
        
        one_encoding_of_X = []
        
        for each_x in X:
            one_encoding_of_each_x = np.zeros(numOfClasses)
            one_encoding_of_each_x[int(each_x)] = 1
            one_encoding_of_X.append(one_encoding_of_each_x)
                    
        return np.array(one_encoding_of_X)
        
    def decode_one_encoding(one_encoding: np.ndarray):
        return np.expand_dims(np.argmax(one_encoding, axis=1), axis=1)
        
        
    def shuffle_randomly(X, y):  
        randomize = np.arange(X.shape[0])
        np.random.shuffle(randomize)
        return X[randomize], y[randomize]
    
    def accuarcy( y_true: np.ndarray, y_pred:np.ndarray):
        y_pred = np.expand_dims(np.argmax(y_pred, axis=1),axis=1)
        return np.sum(y_true == y_pred) / len(y_true)
