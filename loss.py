from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):

    @staticmethod
    @abstractmethod
    def function(a:np.ndarray, y:np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def derivative(a:np.ndarray, y:np.ndarray) -> np.ndarray:
        pass

class MeanSquaredError(Loss):

    @staticmethod
    def function(a:np.ndarray, y:np.ndarray) -> np.ndarray:
        return np.mean((y - a) ** 2, axis=1)
    
    @staticmethod
    def derivative(a:np.ndarray, y:np.ndarray) -> np.ndarray:
        return 2 * (a - y)