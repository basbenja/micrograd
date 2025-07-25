import math

from typing import Union

class Value:
    def __init__(self, data: float, _children: tuple = (), _op: str = '', label: str = '') -> "Value":
        """
        Parameters:
            data (float): the actual value wrapped around Value.
            _children (float): placeholder for the previous values that generated this Value.
            _op (str): the operation applied between the children that gave as a result this Value.
        """
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label

        # Maintains the derivative of the loss function with respect to this variable
        # 0 means no effect. This means that at initialization, we are assuming that
        # no variable affects the output
        self.grad = 0

        # It is a function that is going to apply the chain rule for a certain value
        # We are going to store how we are going to change the outputs gradients
        # with the inputs gradients
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data}, label={self.label})"

    def __add__(self, other: Union["Value", float, int]) -> "Value":
        other = other if isinstance(other, Value) else Value(other, label=str(other))
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # Remember: when a value is the sum of two values, the local derivative is 1.
            # It just backpropagates the previous gradient. We multiply by out.grad because
            # of the chain rule
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __neg__(self):
        # The __neg__ operator acts in this case:
        #   a = Value(1.0)
        #   -a
        return self * -1

    def __sub__(self, other: Union["Value", float, int]) -> "Value":
        return self + (-other)

    def __mul__(self, other: Union["Value", float, int]) -> "Value":
        other = other if isinstance(other, Value) else Value(other, label=str(other))
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # Remember: when a value is the product of two values, the local derivative of
            # each operand is the other operand. We multiply by out.grad because of the
            # chain rule
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other: Union[float, int]):
        # Redirects to the __mul__ operation of thus class
        return self * other

    def __truediv__(self, other: Union["Value", float, int]): # self / other
        return self * other**-1

    def __pow__(self, other: Union[int, float]):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self, ), f'**{other}')

        def _backward():
            # Derivative of the power function
            # We multiply by out.grad because of the chain rule
            self.grad += other * self.data**(other-1) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            # Derivative of the f(x) = e^x function
            # We multiply by out.grad because of the chain rule
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        # We could have implemented the divison and the exponientiation but we want
        # to apply as few operations as possible
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            # Derivative of the tanh function
            # We multiply by out.grad because of the chain rule
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v: Value):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for node in reversed(topo):
            node._backward()
