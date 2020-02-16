import numpy as np
class Activation:
    def __init__(self, name, parameter = 0.1):
        if name == "sigmoid":
            self.forward_propagation = self.activation_sigmoid
            self.derivative = self.derivative_sigmoid
        elif name == "identity":
            self.forward_propagation = self.activation_identity
            self.derivative = self.derivative_identity
        elif name == "ReLU":
            self.forward_propagation = self.activation_ReLU
            self.derivative = self.derivative_ReLU
        elif name == "leakyReLU":
            self.leak_parameter = parameter
            self.forward_propagation = self.activation_leakyReLU
            self.derivative = self.derivative_leakyReLU
        else:
            raise NameError('Use one of the provided activation functions')

        self.outputActivation = None
        self.inputActivation = None

    def activation_sigmoid(self, z):
        z = z *((z <= 20) & (z >= -20))+20*(z > 20)-20*(z < -20)
        self.inputActivation = z
        self.outputActivation = 1 / (1 + np.e**-z)
        return self.outputActivation

    def derivative_sigmoid(self, z):
        return self.activation_sigmoid(z)*(1-self.activation_sigmoid(z))

    def activation_identity(self, z):
        self.inputActivation = z
        self.outputActivation = z
        return self.outputActivation
    def derivative_identity(self, z):
        return np.ones(z)

    def activation_ReLU(self, z):
        self.inputActivation = z
        self.outputActivation = z*(z>0)
        return self.outputActivation
    def derivative_ReLU(self, z):
        return (z>0).astype(float)

    def activation_leakyReLU(self, z):
        self.inputActivation = z
        self.outputActivation = z*(z>0) + self.leak_parameter*z*(z<=0)
        return self.outputActivation
    def derivative_leakyReLU(self, z):
        return (z>0).astype(float) + self.leak_parameter*(z<=0)

    def backward_propagation(self, error):
        return error*self.derivative(self.inputActivation)
