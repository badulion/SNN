import numpy as np
from neuralnet.layer import *
class Optimizer:
    def __init__(self, name, model):
        self.model = model
        self.learning_rate = 0.1
        self.regularization = 0.01
        if name == "GradientDescent":
            self.optimize = self.GradientDescent_Optimize
        else:
            raise NameError('Use one of the provided optimizers')

    def GradientDescent_Optimize(self):
        for layer in reversed(self.model.networkLayers):
            if type(layer) is Layer:
                layer.weights = layer.weights - self.learning_rate*(layer.weightsGradient-self.regularization*layer.weights)
                layer.bias = layer.bias - self.learning_rate*(layer.biasGradient-self.regularization*layer.bias)

    def set_parameters(self, learning_rate, regularization):
        self.learning_rate = learning_rate
        self.regularization = regularization
