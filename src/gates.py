import numpy as np
from abc import ABC, abstractmethod

class Node(ABC):

    @property
    @abstractmethod
    def requires_grad(self) -> bool:
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def value(self) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def _backward(self, dy: np.ndarray) -> None:
        raise NotImplementedError()
    
    def backward(self) -> None:
        self._backward(np.eye(np.array(self.shape).prod()).reshape(self.shape + self.shape))
