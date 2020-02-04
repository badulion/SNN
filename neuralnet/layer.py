import numpy as np
class Layer:
    def __init__(self, inputDimension, outputDimension):
        self.inputDimension = inputDimension #number of input neurons
        self.outputDimension = outputDimension
        self.weights = np.random.rand(self.inputDimension, self.outputDimension)/10-0.05
        self.bias = np.random.rand(self.outputDimension)/10-0.05

        self.outputActivation = None
        self.inputActivation = None

        #backprop gradients
        self.weightsGradient = None
        self.biasGradient = None

    def forward_propagation(self, inputMatrix):
        if inputMatrix.shape[1] != self.inputDimension:
            raise NameError('Input Array does not have the right dimension')
        self.inputActivation = inputMatrix
        self.outputActivation = self.inputActivation @ self.weights + self.bias
        return self.outputActivation

    def backward_propagation(self, error):
        self.weightsGradient = (self.inputActivation.T @ error)/error.shape[0]
        self.biasGradient = error.mean(axis=0)
        backproperror = error @ self.weights.T
        return backproperror
