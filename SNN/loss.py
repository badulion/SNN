import numpy as np

class MSE:
    def error (self, y_true, y_pred):
        return((y_true - y_pred) ** 2).mean()

    def derivative(self, y_true, y_pred):
        return -2*y_pred*(y_true - y_pred)/(y_pred.shape[0]*y_pred.shape[1])

class BinaryCrossEntropy:
    def error(self, y_true, y_pred):
        return -(y_true*np.log(y_pred)).sum(axis=1).mean()-((1-y_true)*(np.log(1-y_pred))).sum(axis=1).mean()

    def derivative(self, y_true, y_pred):
        return -(y_true/y_pred)+((1-y_true)/(1-y_pred))


class CategoricalCrossEntropy:
    def error(self, y_true, y_pred):
        return -(y_true*np.log(y_pred)).sum(axis=1).mean()

    def derivative(self, y_true, y_pred):
        return -y_true/y_pred

class CategoricalCrossEntropyWithSoftmax:
    def error(self, y_true, y_pred):
        s = self.softmax(y_pred)
        return -(y_true*np.log(s)).sum(axis=1).mean()

    def derivative(self, y_true, y_pred):
        s = self.softmax(y_pred)
        return s - y_true

    def softmax(self, z):
        exponential = np.e ** z
        return exponential/exponential.sum(axis=1, keepdims=True)

def getLoss(loss):
    if loss=="MSE":
        return MSE()
    elif loss=="BinaryCrossEntropy":
        return BinaryCrossEntropy()
    elif loss=="CategoricalCrossEntropy":
        return CategoricalCrossEntropy()
    elif loss=="CategoricalCrossEntropyWithSoftmax":
        return CategoricalCrossEntropyWithSoftmax()
