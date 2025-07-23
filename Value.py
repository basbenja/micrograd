import math

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
        return f"Value(data={self.data})"

    def __add__(self, other: "Value") -> "Value":
        out = Value(self.data + other.data, (self, other), '+')

        # Remember: when a value is the sum of two values, the local derivative is 1
        # It just backpropagates the previous gradient
        def _backward():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other: "Value") -> "Value":
        out = Value(self.data * other.data, (self, other), '*')

        # Remember: when a value is the product of two values, the local derivative of
        # each operand is the other operand
        def _backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        # We could have implemented the divison and the exponientiation but we want
        # to apply as few operations as possible
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad = (1 - t**2) * out.grad
        out._backward = _backward

        return out
