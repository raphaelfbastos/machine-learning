from __future__ import annotations
from datetime import datetime
from pickle import dumps, loads
from layer import Layer
from loss import Loss
import numpy as np

class Model:

    def __init__(self, *layers:Layer):
        self.layers = layers

    def size(self) -> int:
        return sum(layer.W.size + layer.b.size for layer in self.layers)

    def forward(self, x:np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, a:np.ndarray, y:np.ndarray, loss:Loss, learning_rate:float) -> None:
        grad = loss.derivative(a, y)
        for layer in self.layers[::-1]:
            grad = np.sum(grad, axis=1, keepdims=True) * layer.f.derivative(layer.a)
            layer.W -= learning_rate * np.mean(np.einsum('ij,ik->ijk', grad, layer.x), axis=0).T
            layer.b -= learning_rate * np.sum(grad, axis=0)

    def export(self, directory:str) -> str:
        with open(f"{directory}/export{datetime.now().strftime('%Y%m%d%H%M%S%f')}.model", 'wb') as file:
            file.write(dumps(self))
            return file.name

    @staticmethod
    def load(path:str) -> Model:
        with open(path, 'rb') as file:
            return loads(file.read())
