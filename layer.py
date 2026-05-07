from abc import ABC, abstractmethod
from activation import Activation
import numpy as np

class Layer(ABC):

    W:np.ndarray
    b:np.ndarray
    f:Activation
    x:np.ndarray
    z:np.ndarray
    a:np.ndarray

    @abstractmethod
    def forward(self, x:np.ndarray) -> np.ndarray:
        pass

    def __getstate__(self):
        return {'W': self.W, 'b': self.b, 'f': self.f, 'x': None, 'z': None, 'a': None}

class Dense(Layer):

    def __init__(self, m:int, n:int, f:Activation):
        self.W = np.random.uniform(-1, 1, (m, n))
        self.b = np.zeros(n)
        self.f = f
        self.x = None
        self.z = None
        self.a = None
    
    def forward(self, x:np.ndarray) -> np.ndarray:
        self.x = x
        self.z = np.dot(x, self.W) + self.b
        self.a = self.f.function(self.z)
        return self.a