import numpy as np

class Scaler:
    def __init__(self, X: np.ndarray = None): # n * F matrix
        self.is_init = False
        if not X is None:
            self.init_transform(X)
            self.is_init = True

    def init_transform(self, X: np.ndarray):
        raise NotImplementedError

    def transform(self, X: np.ndarray):
        raise NotImplementedError
    
    def transform_rev(self, X: np.ndarray):
        raise NotImplementedError


class MaxMinScaler(Scaler):
    def init_transform(self, X: np.ndarray):
        self.xmax = np.max(X, axis=0)
        self.xmin = np.min(X, axis=0)

    def transform(self, X: np.ndarray):
        if not self.is_init:
            self.init_transform(X)
            self.is_init = True
        return (X - self.xmin) / (self.xmax - self.xmin)
    
    def transform_rev(self, X: np.ndarray):
        if not self.is_init:
            raise ValueError('Scaler is not initialized')
        return X * (self.xmax - self.xmin) + self.xmin
    

class SignMaxMinScaler(Scaler):
    def init_transform(self, X: np.ndarray):
        xmax = np.max(X, axis=0)
        xmin = np.min(X, axis=0)
        self.bounds = np.stack((xmax, xmin), axis=0)
        self.map_bounds = np.ones_like(self.bounds[0, :])
        for i in range(self.bounds.shape[1]):
            if self.bounds[0, i] <= 0:
                self.map_bounds[i] = -1
            elif self.bounds[1, i] >= 0:
                self.map_bounds[i] = 1
            else:
                self.map_bounds[i] = 0

    def transform(self, X: np.ndarray):
        X = X.copy()
        if not self.is_init:
            self.init_transform(X)
            self.is_init = True
        for i in range(X.shape[1]):
            if self.map_bounds[i] == 1:
                X[:, i] = (X[:, i] - self.bounds[1, i]) / (self.bounds[0, i] - self.bounds[1, i])
            elif self.map_bounds[i] == -1:
                X[:, i] = (X[:, i] - self.bounds[0, i]) / (self.bounds[1, i] - self.bounds[0, i])
            else:
                X[:, i] = 2*(X[:, i] - self.bounds[1, i]) / (self.bounds[0, i] - self.bounds[1, i]) - 1
        return X

    def transform_rev(self, X: np.ndarray):
        X = X.copy()
        if not self.is_init:
            raise ValueError('Scaler is not initialized')
        for i in range(X.shape[1]):
            if self.map_bounds[i] == 1:
                X[:, i] = X[:, i] * (self.bounds[0, i] - self.bounds[1, i]) + self.bounds[1, i]
            elif self.map_bounds[i] == -1:
                X[:, i] = X[:, i] * (self.bounds[1, i] - self.bounds[0, i]) + self.bounds[0, i]
            else:
                X[:, i] = (X[:, i] + 1) * (self.bounds[0, i] - self.bounds[1, i]) / 2 + self.bounds[1, i]
        return X
        