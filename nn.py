import random

from engine import Value

class Neuron:
    def __init__(self, nin: int):
        """
        Args:
            nin (int): number of inputs of the neuron.
        """
        # We have to associate a weight to each input
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        # And the bias
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # w * x + b (dot product)
        activation = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        # tanh(w * x + b)
        out = activation.tanh()
        return out


class Layer:
    def __init__(self, nin: int, nout: int):
        """
        Args:
            nin (int): the dimensionality of each neuron (the size of the input of
                each neuron).
            nout (int): amount of neurons in the layer.
        """
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [neuron(x) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs


class MLP:
    def __init__(self, nin: int, nouts: list[int]):
        """
        Args:
            nin (int): amount of inputs of the network.
            nouts (list[int]): amount of neurons in each layer.
        """
        sizes = [nin] + nouts
        # len(nouts) = len(sizes) - 1
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x