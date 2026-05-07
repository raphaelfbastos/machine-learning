from abc import ABC, abstractmethod
import numpy as np

class Activation(ABC):

    @staticmethod
    @abstractmethod
    def function(z:np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def derivative(a:np.ndarray) -> np.ndarray:
        pass

class Linear(Activation):

    @staticmethod
    def function(z:np.ndarray) -> np.ndarray:
        return z
    
    @staticmethod
    def derivative(a:np.ndarray) -> np.ndarray:
        return np.ones(a.shape)

class ReLU(Activation):

    @staticmethod
    def function(z:np.ndarray) -> np.ndarray:
        return np.maximum(0, z)
    
    @staticmethod
    def derivative(a:np.ndarray) -> np.ndarray:
        return np.where(a > 0, 1, 0)
    
class Sigmoid(Activation):

    @staticmethod
    def function(z:np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def derivative(a:np.ndarray) -> np.ndarray:
        return a * (1 - a)
    
class Tanh(Activation):

    @staticmethod
    def function(z:np.ndarray) -> np.ndarray:
        return np.tanh(z)
    
    @staticmethod
    def derivative(a:np.ndarray) -> np.ndarray:
        return 1 - a * a
