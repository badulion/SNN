import numpy as np

#activation functions
def linear(z):
    return z

def sigmoid(z):
    z = z *((z <= 20) & (z >= -20))+20*(z > 20)-20*(z < -20)
    return  1 / (1 + np.e**-z)

def ReLU(z):
    return z*(z>0)

def leakyReLU(z):
    return z*(z>0) + 0.01*z*(z<0)

def ELU(z):
    alpha = 1.67326
    lambd = 1.0507
    return lambd*(z*(z>0) + alpha*(np.e**z-1)*(z<0))



#derivatives of activation functions
def linear_derivative(z):
        return z

def ReLU_derivative(z):
    return (z>0).astype(float)

def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))

def leakyReLU_derivative(z):
    return 1.0*(z>0) + 0.01*(z<0)

def ELU_derivative(z):
    alpha = 1.67326
    lambd = 1.0507
    return lambd*((z>0) + alpha*(np.e**z)*(z<0))


def getActivation(activation):
    if activation == "linear":
        return linear
    elif activation == "sigmoid":
        return sigmoid
    elif activation == "ReLU":
        return ReLU
    elif activation == "leakyReLU":
        return leakyReLU
    elif activation == "ELU":
        return ELU

def getActivationDerivative(activation):
    if activation == "linear":
        return linear_derivative
    elif activation == "sigmoid":
        return sigmoid_derivative
    elif activation == "ReLU":
        return ReLU_derivative
    elif activation == "leakyReLU":
        return leakyReLU_derivative
    elif activation == "ELU":
        return ELU_derivative
