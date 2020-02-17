import numpy as np

#activation functions
def linear(z):
    return z

def sigmoid(z):
    z = z *((z <= 20) & (z >= -20))+20*(z > 20)-20*(z < -20)
    return  1 / (1 + np.e**-z)

def ReLU(z):
    return z*(z>0)



#derivatives of activation functions
def linear_derivative(z):
        return z

def ReLU_derivative(z):
    return (z>0).astype(float)

def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))


def getActivation(activation):
    if activation == "linear":
        return linear
    elif activation == "sigmoid":
        return sigmoid
    elif activation == "ReLU":
        return ReLU

def getActivationDerivative(activation):
    if activation == "linear":
        return linear_derivative
    elif activation == "sigmoid":
        return sigmoid_derivative
    elif activation == "ReLU":
        return ReLU_derivative
