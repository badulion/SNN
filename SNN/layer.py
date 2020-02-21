import numpy as np
from .activation import getActivation, getActivationDerivative

class Layer:
    def __init__(self, inputShape, outputShape, activation="linear", dropout=0):
        self.inputShape = inputShape
        self.outputShape = outputShape

        self.activation = getActivation(activation)
        self.activationDerivative = getActivationDerivative(activation)
        self.weights = np.empty([self.inputShape, self.outputShape])
        self.bias = np.empty([1, self.outputShape])

        self.dropout_probability = dropout
        self.dropout = np.empty(self.weights.shape)

        self.outputActivation = None
        self.outputRaw = None
        self.input = None

        self.weightsGradient = np.empty([self.inputShape, self.outputShape])
        self.biasGradient = np.empty([1, self.outputShape])

    def initializeWeights(self):
        self.weights = np.random.randn(self.inputShape, self.outputShape)/np.sqrt(self.inputShape)
        self.bias = np.random.randn(1, self.outputShape)/np.sqrt(self.inputShape)

    def forward(self, input):
        self.input = input
        self.outputRaw = input @ self.weights + self.bias
        self.outputActivation = self.activation(self.outputRaw)
        if self.dropout_probability > 0:
            self.dropout = np.random.choice([0,1], size=self.outputActivation.shape, p=[self.dropout_probability, 1-self.dropout_probability])
            self.outputActivation = self.dropout*self.outputActivation/(1-self.dropout_probability)
        return self.outputActivation

    def backward(self, error):
        if self.dropout_probability > 0:
            error = self.dropout*error
        activationError = error*self.activationDerivative(self.outputRaw)
        self.weightsGradient = (self.input.T @ activationError)/activationError.shape[0]
        self.biasGradient = activationError.mean(axis=0)
        inputError = activationError @ self.weights.T
        return inputError

    def predict(self, input):
        output = input @ self.weights + self.bias
        output = self.activation(output)
        return output
