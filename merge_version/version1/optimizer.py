import numpy as np

# This is a base class for all optimizers. All optimizers should inherit from this.
class base_optimizer():
    def __init__(self):
        pass
    
    def reset(self):
        pass
    
    def init_first_step(self, dw, db):
        pass
    
    def update_parameter(self, idx, step_count,layer, lr, w, b, dw, db):
        pass
    
    
# This is a class for the stochastic gradient descent optimizer. It inherits the base_optimizer class. 
# The update_parameter function does a simple weight update using the SGD algorithm. 
class sgd(base_optimizer):
    def __init__(self):
        base_optimizer.__init__(self)
        pass

    def update_parameter(self, idx, step_count,layer, lr, w, b, dw, db):
        w -= lr * dw
        b -= lr * db
        return w, b
         
class sgd_momentum(base_optimizer):
    def __init__(self) -> None:
        base_optimizer.__init__(self)
        pass  
    
    def update_parameter(self, idx, step_count, layer, lr, w, b, dw, db):
        layer.v_W = (0.9* layer.v_W) + (lr * dw)
        layer.v_b = (0.9 * layer.v_b) + (lr * db)
        w = w -  layer.v_W
        b = b - layer.v_b
        
        return w, b


 
# This is a class for adam optimizer. It inherits the base_optimizer class. 
# The update_parameter function does a simple weight update using the adam algorithm.        
class adam(base_optimizer):
    def __init__(self, beta1, beta2):
        base_optimizer.__init__(self)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 1e-16
        self.m_dws = []
        self.v_dws = []
        self.m_dbs = []
        self.v_dbs = []

    def reset(self):
        self.m_dws = []
        self.v_dws = []
        self.m_dbs = []
        self.v_dbs = []

    def init_first_step(self, dw, db):
        self.m_dws.append(np.zeros(dw.shape))
        self.v_dws.append(np.zeros(dw.shape))
        self.m_dbs.append(np.zeros(db.shape))
        self.v_dbs.append(np.zeros(db.shape))

    def update_parameter(self, idx, step_count,layer,lr, w, b, dw, db):
        
        # beta 1
        self.m_dws[idx] = self.beta1 * self.m_dws[idx] + (1 - self.beta1) * dw
        self.m_dbs[idx] = self.beta1 * self.m_dbs[idx] + (1 - self.beta1) * db

        # beta 2
        self.v_dws[idx] = self.beta2 * \
            self.v_dws[idx] + (1 - self.beta2) * (dw ** 2)
        self.v_dbs[idx] = self.beta2 * \
            self.v_dbs[idx] + (1 - self.beta2) * (db ** 2)

        # correct bias
        m_dw_corrected = self.m_dws[idx] / (1 - self.beta1 ** (step_count))
        m_db_corrected = self.m_dbs[idx] / (1 - self.beta1 ** (step_count))
        v_dw_corrected = self.v_dws[idx] / (1 - self.beta2 ** (step_count))
        v_db_corrected = self.v_dbs[idx] / (1 - self.beta2 ** (step_count))

        # update parameters with adjusted bias and learning rate
        w = w - lr * m_dw_corrected / (np.sqrt(v_dw_corrected) + self.epsilon)
        b = b - lr * m_db_corrected / (np.sqrt(v_db_corrected) + self.epsilon)

        return w, b
    
# This is a class for RMSprop optimizer. It inherits the base_optimizer class. 
# The update_parameter function does a simple weight update using the RMSprop algorithm.
class RMSprop(base_optimizer):
    def __init__(self,  beta2):
        base_optimizer.__init__(self)
        self.beta2 = beta2
        self.epsilon = 1e-16
        self.v_dws = []
        self.v_dbs = []

    def reset(self):
        self.v_dws = []
        self.v_dbs = []

    def init_first_step(self, dw, db):
        self.v_dws.append(np.zeros(dw.shape))
        self.v_dbs.append(np.zeros(db.shape))

    def update_parameter(self, idx, step_count, layer, lr, w, b, dw, db):
        # beta 2
        self.v_dws[idx] = self.beta2 * \
                          self.v_dws[idx] + (1 - self.beta2) * (dw ** 2)
        self.v_dbs[idx] = self.beta2 * \
                          self.v_dbs[idx] + (1 - self.beta2) * (db ** 2)

        # update parameters with adjusted bias and learning rate
        w = w - lr * dw/ (np.sqrt(self.v_dws[idx]) + self.epsilon)
        b = b - lr * db / (np.sqrt(self.v_dbs[idx]) + self.epsilon)

        return w, b