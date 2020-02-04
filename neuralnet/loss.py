import numpy as np
class Loss:
    def __init__(self, name):
        if name == "CategoricalCrossEntropy":
            self.error = self.CategoricalCrossEntropy_Error
            self.derivative = self.CategoricalCrossEntropy_Derivative
        elif name == "BinaryCrossEntropy":
            self.error = self.BinaryCrossEntropy_Error
            self.derivative = self.BinaryCrossEntropy_Derivative
        else:
            raise NameError('Use one of the provided loss functions')
        self.error_history = []

    def CategoricalCrossEntropy_Error(self, ground_truth, hypothesis):
        backprop = -(ground_truth*np.log(hypothesis)).sum(axis=1).mean()
        self.error_history.append(backprop)
        return backprop

    def CategoricalCrossEntropy_Derivative(self, ground_truth, hypothesis):
        return -ground_truth/hypothesis

    def BinaryCrossEntropy_Error(self, ground_truth, hypothesis):
        backprop = -(ground_truth*np.log(hypothesis)).sum(axis=1).mean()-((1-ground_truth)*(np.log(1-hypothesis))).sum(axis=1).mean()
        self.error_history.append(backprop)
        return backprop

    def BinaryCrossEntropy_Derivative(self, ground_truth, hypothesis):
        return -(ground_truth/hypothesis)+((1-ground_truth)/(1-hypothesis))
