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
        # Maintaints the derivative of the loss function with respect to this variable
        # 0 means no effect. This means that at initialization, we are assuming that
        # no variable affects the output
        self.grad = 0

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other: "Value") -> "Value":
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other: "Value") -> "Value":
        out = Value(self.data * other.data, (self, other), '*')
        return out

    def tanh(self):
        # We could have implemented the divison and the exponientiation but we want
        # to apply as few operations as possible
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        return out