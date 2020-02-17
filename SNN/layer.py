import numpy as np
from .activation import getActivation, getActivationDerivative

class Layer:
    def __init__(self, inputShape, outputShape, activation="linear", dropout=0):
        self.inputShape = inputShape
        self.outputShape = outputShape
        self.dropout = dropout
        self.activation = getActivation(activation)
        self.activationDerivative = getActivationDerivative(activation)
        self.weights = np.empty([self.inputShape, self.outputShape])
        self.bias = np.empty([1, self.outputShape])

        self.outputActivation = None
        self.outputRaw = None
        self.input = None

        self.weightsGradient = np.empty([self.inputShape, self.outputShape])
        self.biasGradient = np.empty([1, self.outputShape])

    def initializeWeights(self):
        self.weights = np.random.randn(self.inputShape, self.outputShape)/np.sqrt(self.outputShape)
        self.bias = np.random.randn(1, self.outputShape)/np.sqrt(self.outputShape)

    def forward(self, input):
        self.input = input
        outputRaw = input @ self.weights + self.bias
        self.outputRaw = outputRaw
        outputActivation = self.activation(outputRaw)
        self.outputActivation = outputActivation
        return outputActivation

    def backward(self, error):
        activationError = error*self.activationDerivative(self.outputRaw)
        self.weightsGradient = (self.input.T @ activationError)/activationError.shape[0]
        self.biasGradient = activationError.mean(axis=0)
        inputError = activationError @ self.weights.T
        return inputError

    def predict(self, input):
        output = input @ self.weights + self.bias
        output = self.activation(output)
        return output
