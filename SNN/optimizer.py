import numpy as np

class GradientDescent:
    def __init__(self, model):
        self.model = model
        self.learning_rate = 0.1
        self.regularization = 0
        self.beta = 0
    def updateWeights(self):
        for layer in self.model.networkLayers:
            layer.weights = layer.weights - self.learning_rate*(layer.weightsGradient+self.regularization*layer.weights)
            layer.bias = layer.bias - self.learning_rate*(layer.biasGradient+self.regularization*layer.bias)

class GradientDescentWithMomentum:
    def __init__(self, model):
        self.model = model
        self.weightsMomentums = []
        self.biasMomentums = []
        for layer in self.model.networkLayers:
            self.weightsMomentums.append(np.zeros_like(layer.weights))
            self.biasMomentums.append(np.zeros_like(layer.bias))

        self.learning_rate = 0.1
        self.regularization = 0
        self.beta = 0.8
    def updateWeights(self):
        for i in range(len(self.model.networkLayers)):
            layer = self.model.networkLayers[i]
            self.weightsMomentums[i] = self.beta*self.weightsMomentums[i]+(1-self.beta)*(layer.weightsGradient+self.regularization*layer.weights)
            self.biasMomentums[i] = self.beta*self.biasMomentums[i]+(1-self.beta)*(layer.biasGradient+self.regularization*layer.bias)
            layer.weights = layer.weights - self.learning_rate*self.weightsMomentums[i]
            layer.bias = layer.bias - self.learning_rate*self.biasMomentums[i]


def getOptimizer(optimizer, model):
    if optimizer=="GradientDescent":
        return GradientDescent(model)
    elif optimizer=="GradientDescentWithMomentum":
        return GradientDescentWithMomentum(model)
