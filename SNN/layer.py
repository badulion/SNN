import numpy as np
from .activation import getActivation, getActivationDerivative

class Layer:
    def __init__(self, inputShape, outputShape, activation="linear", dropout=0):
        #needed for the shape of the weight matrix
        self.inputShape = inputShape
        self.outputShape = outputShape

        #store the activation function
        #To Do: Implement passing your own activation functions vie:
        #tuple (activation, derivative)
        self.activation = getActivation(activation)
        self.activationDerivative = getActivationDerivative(activation)

        #define the weight matrix
        self.weights = np.empty([self.inputShape, self.outputShape])
        self.bias = np.empty([1, self.outputShape])

        #dropout parameters
        self.dropout_probability = dropout
        self.dropout = np.empty(self.weights.shape)

        #initialize the storage for input, intermediate and output activation
        self.outputActivation = None
        self.outputRaw = None
        self.input = None

        #gradients start empty
        self.weightsGradient = np.empty([self.inputShape, self.outputShape])
        self.biasGradient = np.empty([1, self.outputShape])

    def initializeWeights(self):
        self.weights = np.random.randn(self.inputShape, self.outputShape)/np.sqrt(self.inputShape)
        self.bias = np.random.randn(1, self.outputShape)/np.sqrt(self.inputShape)

    #forward propagate through the layer
    #pass the input through the weights, compute the activation function on that and dropout a random
    #set of neurons
    def forward(self, input):
        self.input = input
        self.outputRaw = input @ self.weights + self.bias
        self.outputActivation = self.activation(self.outputRaw)
        if self.dropout_probability > 0:
            self.dropout = np.random.choice([0,1], size=self.outputActivation.shape, p=[self.dropout_probability, 1-self.dropout_probability])
            self.outputActivation = self.dropout*self.outputActivation/(1-self.dropout_probability)
        return self.outputActivation

    #backprop - here dropout comes first, than backprop the error through the activation function and then the weights
    #gradients get updated here
    def backward(self, error):
        if self.dropout_probability > 0:
            error = self.dropout*error
        activationError = error*self.activationDerivative(self.outputRaw)
        self.weightsGradient = (self.input.T @ activationError)/activationError.shape[0]
        self.biasGradient = activationError.mean(axis=0)
        inputError = activationError @ self.weights.T
        return inputError

    #as in forward propagation, but do not compute the dropout (for test predictions)
    def predict(self, input):
        output = input @ self.weights + self.bias
        output = self.activation(output)
        return output
