from abc import ABC
from typing import Protocol, Optional, Callable, Self

import numpy as np


class Node(Protocol):

    @property
    def requires_grad(self) -> bool:
        raise NotImplementedError()

    @property
    def ndim(self) -> int:
        raise NotImplementedError()

    @property
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError()

    @property
    def value(self) -> np.ndarray:
        raise NotImplementedError()

    def _backward(self, dy: np.ndarray) -> None:
        raise NotImplementedError()

    def backward(self, dy: Optional[np.ndarray] = None) -> None:
        raise NotImplementedError()

    def __neg__(self) -> Self:
        raise NotImplementedError()

    def __add__(self, other: Self) -> Self:
        raise NotImplementedError()

    def __sub__(self, other: Self) -> Self:
        raise NotImplementedError()

    def __mul__(self, other: Self) -> Self:
        raise NotImplementedError()

    def __truediv__(self, other: Self) -> Self:
        raise NotImplementedError()

    def __matmul__(self, other: Self) -> Self:
        raise NotImplementedError()

class AbstractNode(ABC, Node):

    @property
    def ndim(self) -> int:
        return self.value.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape
    
    def backward(self, dy: Optional[np.ndarray] = None) -> None:
        if dy is None:
            dy = np.eye(np.array(self.shape).prod()).reshape(self.shape + self.shape)
        self._backward(dy)

    def __neg__(self):
        return Func(self, lambda x: -x, lambda x: -np.ones_like(x))

    def __add__(self, other):
        return Add(self, other)

    def __sub__(self, other):
        return Add(self, -other)

    def __mul__(self, other):
        return Mult(self, other)

    def __truediv__(self, other):
        return Mult(self, Func(self, lambda x: 1/x, lambda x: -1/x**2))

    def __matmul__(self, other):
        return Dot(self, other)

class Const(AbstractNode):

    value: np.ndarray

    def __init__(self, value: np.ndarray):
        self.value = value

    @property
    def requires_grad(self) -> bool:
        return False

    def _backward(self, dy: np.ndarray) -> None:
        pass

class WeightNode(AbstractNode):

    value: np.ndarray
    _grad: Optional[np.ndarray] = None

    def __init__(self, value: np.ndarray):
        self.value = value

    @property
    def requires_grad(self) -> bool:
        return True

    def _backward(self, dy: np.ndarray) -> None:
        if self._grad is None:
            self._grad = dy
        else:
            self._grad += dy

    @property
    def grad(self) -> Optional[np.ndarray]:
        return self._grad

class Proj(AbstractNode):

    base: Node
    mask: tuple[int | slice | np.ndarray, ...]
    requires_grad: bool
    value: np.ndarray

    def __init__(self, base: Node, masks: tuple[tuple[int, int | slice | np.ndarray], ...]):
        self.base = base
        mask: list[int | slice | np.ndarray] = [slice(None)] * base.ndim
        for axis, entry in masks:
            mask[axis] = entry
        self.mask = tuple(mask)
        self.requires_grad = base.requires_grad
        self.value = base.value[self.mask]

    def _backward(self, dy: np.ndarray) -> None:
        if self.base.requires_grad:
            y_shape = dy.shape[:-self.ndim]
            grad = np.zeros(y_shape + self.base.shape)
            np.add.at(grad, (slice(None),) * len(y_shape) + self.mask, dy)
            self.base._backward(grad)

class Tr(AbstractNode):

    base: Node
    perm: tuple[int, ...]
    requires_grad: bool
    value: np.ndarray

    def __init__(self, base: Node, perm: tuple[int, ...]):
        self.base = base
        self.perm = perm
        self.requires_grad = base.requires_grad
        self.value = base.value.transpose(perm)

    def _backward(self, dy: np.ndarray) -> None:
        if self.base.requires_grad:
            y_dim = dy.ndim - self.ndim
            self.base._backward(dy.transpose(np.concatenate((np.arange(y_dim), np.argsort(self.perm) + y_dim))))

class Reshape(AbstractNode):

    base: Node
    requires_grad: bool
    value: np.ndarray

    def __init__(self, base: Node, shape: tuple[int, ...]):
        self.base = base
        self.requires_grad = base.requires_grad
        self.value = base.value.reshape(shape)

    def _backward(self, dy: np.ndarray) -> None:
        if self.base.requires_grad:
            self.base._backward(dy.reshape(dy.shape[:-self.ndim] + self.base.shape))

class BroadcastHelper:

    shape_left: tuple[int, ...]
    shape_right: tuple[int, ...]
    ndim: int
    stretch_left: tuple[int, ...]
    stretch_right: tuple[int, ...]

    def __init__(self, shape_left: tuple[int, ...], shape_right: tuple[int, ...]):
        self.ndim = max(len(shape_left), len(shape_right))
        shape_left_padded = (1,) * (self.ndim - len(shape_left)) + shape_left
        shape_right_padded = (1,) * (self.ndim - len(shape_right)) + shape_right
        stretch_left: list[int] = []
        stretch_right: list[int] = []
        for i, (l, r) in enumerate(zip(shape_left_padded, shape_right_padded)):
            if l != r and min(l, r) != 1:
                raise ValueError(f"Invalid shape for broadcast: {shape_left} and {shape_right}")
            if l < r:
                stretch_left.append(l)
            elif r < l:
                stretch_right.append(r)
        self.shape_left = shape_left
        self.shape_right = shape_right
        self.stretch_left = tuple(stretch_left)
        self.stretch_right = tuple(stretch_right)

    def unbroadcast_left_grad(self, dy: np.ndarray) -> np.ndarray:
        y_shape = dy.shape[:-self.ndim]
        return dy.sum(axis=tuple(axis + len(y_shape) for axis in self.stretch_left), keepdims=True).reshape(y_shape + self.shape_left)

    def unbroadcast_right_grad(self, dy: np.ndarray) -> np.ndarray:
        y_shape = dy.shape[:-self.ndim]
        return dy.sum(axis=tuple(axis + len(y_shape) for axis in self.stretch_right), keepdims=True).reshape(y_shape + self.shape_right)

class Add(AbstractNode):

    left: Node
    right: Node
    broadcast: BroadcastHelper
    requires_grad: bool
    value: np.ndarray

    def __init__(self, left: Node, right: Node):
        self.left = left
        self.right = right
        self.broadcast = BroadcastHelper(self.left.shape, self.right.shape)
        self.requires_grad = left.requires_grad or right.requires_grad
        self.value = left.value + right.value

    def _backward(self, dy: np.ndarray) -> None:
        if self.left.requires_grad:
            self.left._backward(self.broadcast.unbroadcast_left_grad(dy))
        if self.right.requires_grad:
            self.right._backward(self.broadcast.unbroadcast_right_grad(dy))

class Mult(AbstractNode):

    left: Node
    right: Node
    broadcast: BroadcastHelper
    requires_grad: bool
    value: np.ndarray

    def __init__(self, left: Node, right: Node):
        self.left = left
        self.right = right
        self.broadcast = BroadcastHelper(self.left.shape, self.right.shape)
        self.requires_grad = left.requires_grad or right.requires_grad
        self.value = left.value * right.value

    def _backward(self, dy: np.ndarray) -> None:
        if self.left.requires_grad:
            self.left._backward(self.broadcast.unbroadcast_left_grad(dy * self.right.value))
        if self.right.requires_grad:
            self.right._backward(self.broadcast.unbroadcast_right_grad(dy * self.left.value))

class Dot(AbstractNode):

    left: Node
    right: Node
    axes: int
    requires_grad: bool
    value: np.ndarray

    def __init__(self, left: Node, right: Node, axes: int = 1):
        self.left = left
        self.right = right
        self.axes = axes
        self.requires_grad = left.requires_grad or right.requires_grad
        self.value = np.tensordot(left.value, right.value, axes=axes)

    def _backward(self, dy: np.ndarray) -> None:
        if self.left.requires_grad:
            free_axes = self.right.ndim - self.axes
            self.left._backward(np.tensordot(dy, self.right.value, axes=(range(-free_axes, 0), range(-free_axes, 0))))
        if self.right.requires_grad:
            free_axes = self.left.ndim - self.axes
            y_dim = dy.ndim - self.ndim
            self.right._backward(
                np.tensordot(dy, self.left.value, axes=(range(-self.ndim, 0)[:free_axes], range(free_axes)))
                    .transpose(np.concatenate((np.arange(y_dim), (np.arange(self.right.ndim) - self.axes) % self.right.ndim + y_dim)))
            )

class Func(AbstractNode):

    base: Node
    df: Callable[[np.ndarray], np.ndarray]
    requires_grad: bool
    value: np.ndarray

    def __init__(self, base: Node, f: Callable[[np.ndarray], np.ndarray], df: Callable[[np.ndarray], np.ndarray]):
        self.base = base
        self.df = df
        self.requires_grad = base.requires_grad
        self.value = f(base.value)

    def _backward(self, dy: np.ndarray) -> None:
        if self.base.requires_grad:
            self.base._backward(dy * self.df(self.base.value))