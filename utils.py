import numpy as np


class Scaler:
    def __init__(self, method):
        self.method = method 
        if 'static' in method:
            self.scalers_ = np.array([1e-2, 1e-5, 1]).reshape(1, -1)
        elif 'dynamic' in method:
            self.scalers_ = np.array([1e-2, 1e-2, 1e-2]).reshape(1, -1)
        elif 'wdfp' in method:
            self.scalers_ = np.array([1e-1]).reshape(1, -1)
        else:
            raise AttributeError('Scaler method not recognized...')
    
    def fit(self, X):
        assert X.ndim == 2
        return self
    
    def transform(self, X):
        assert X.ndim == 2
        return X * self.scalers_
    
    def inverse_transform(self, X):
        assert X.ndim == 2
        return X / self.scalers_