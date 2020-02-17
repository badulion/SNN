from .loss import getLoss
from .optimizer import getOptimizer
import numpy as np

class Network:
    def __init__(self, layerList, loss, optimizer="GradientDescent"):
        #dimensions of input and output vectors
        self.inputShape = layerList[0].inputShape
        self.networkLayers = layerList

        for layer in self.networkLayers:
            layer.initializeWeights()

        #copile variables
        self.optimizer = getOptimizer(optimizer, self)
        self.loss = getLoss(loss)

    def forward(self, inputMatrix):
        #check the dimensions of the input matrix
        if inputMatrix.shape[1] != self.inputShape:
            raise NameError('Input Array does not have the right dimension')

        for layer in self.networkLayers:
            inputMatrix = layer.forward(inputMatrix)

        self.outputActivation = inputMatrix
        return self.outputActivation

    def predict(self, inputMatrix):
        #check the dimensions of the input matrix
        if inputMatrix.shape[1] != self.inputShape:
            raise NameError('Input Array does not have the right dimension')

        for layer in self.networkLayers:
            inputMatrix = layer.predict(inputMatrix)

        return inputMatrix

    def backward(self, error):
        for layer in reversed(self.networkLayers):
            error = layer.backward(error)

    def fit(self, X, y, epochs, batch_size=64):
        if X.shape[0] != y.shape[0]:
            raise NameError('Labels do not match the features!')

        data_size = X.shape[0]
        random_indices = np.arange(data_size)
        batch_divisions = np.arange(0, data_size, batch_size)

        #if the data size is not a perfect multiple of the batch size
        #last batch will be smaller:
        if batch_divisions[-1] < data_size:
            batch_divisions = np.append(batch_divisions, [data_size])



        for i in range(epochs):
            #randomly shuffle the data
            np.random.shuffle(random_indices)
            X = X[random_indices, :]
            y = y[random_indices, :]

            for j in range(len(batch_divisions)-1):
                #get the current batch
                batch_X = X[batch_divisions[j]:batch_divisions[j+1], :]
                batch_y = y[batch_divisions[j]:batch_divisions[j+1], :]

                #push forward through network:
                batch_y_pred = self.forward(batch_X)

                #compute loss and loss gradient
                loss = self.loss.error(batch_y, batch_y_pred)
                loss_gradient = self.loss.derivative(batch_y, batch_y_pred)

                #backpropagation (updates gradients)
                self.backward(loss_gradient)

                #call optimizer
                self.optimizer.updateWeights()

            y_pred = self.predict(X)
            loss = self.loss.error(y, y_pred)
            print("Epoch: %d, loss: %f" %(i+1, loss))

    def setParameters(self, lr = 0.1, regularization=0, beta=0.8):
        self.optimizer.learning_rate = lr
        self.optimizer.regularization = regularization
        self.optimizer.beta = beta
