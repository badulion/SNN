import numpy as np


class Network:
    def __init__(self, layerList):
        #dimensions of input and output vectors
        self.inputDimension = layerList[0].inputDimension
        self.networkLayers = layerList

        #copile variables
        self.compiled = False
        self.optimizer = None
        self.loss = None

    def compile(self, optimizer, loss):
        self.optimizer = Optimizer(optimizer, self)
        self.loss = Loss(loss)
        self.compiled=True

    def predict(self, inputMatrix):
        #check the dimensions of the input matrix
        if inputMatrix.shape[1] != self.inputDimension:
            raise NameError('Input Array does not have the right dimension')

        for layer in self.networkLayers:
            inputMatrix = layer.forward_propagation(inputMatrix)

        self.outputActivation = inputMatrix
        return self.outputActivation

    def backward_propagation(self, labels):
        if self.outputActivation.shape != labels.shape:
            raise NameError('Labels do not match the output.')
        if not self.compiled:
            raise NameError('Model has not been compiled yet. You need to run compile() first')
        #calculate error
        self.loss.error(labels, self.outputActivation)

        #initialize bagprop error
        error = self.loss.derivative(labels, self.outputActivation)
        for layer in reversed(self.networkLayers):
            error = layer.backward_propagation(error)

    def fit(self, data, labels, epochs, batch_size):
        if data.shape[0] != labels.shape[0]:
            raise NameError('Labels do not match the output.')
        if not self.compiled:
            raise NameError('Model has not been compiled yet. You need to run compile() first')

        data_size = data.shape[0]
        batch_size = np.min([data_size, batch_size])
        random_indices = np.arange(data_size)
        for epoch in range(epochs):
            #randomly shuffle the data
            np.random.shuffle(random_indices)
            data = data[random_indices, :]
            labels = labels[random_indices]

            #initialize the batch iterators
            batch_start = 0
            batch_end = batch_size

            while batch_start < data_size:
                batch_data = data[batch_start:batch_end, :]
                batch_labels = labels[batch_start:batch_end, :]

                #perform one gradient step
                self.predict(batch_data)
                self.backward_propagation(batch_labels)
                self.optimizer.optimize()

                #update batch iterators
                batch_start += batch_size
                batch_end = np.min([batch_end + batch_size, data_size])
            print('Epoch %d/%d   loss=%f' % (epoch+1,epochs,self.loss.error_history[-1]), sep=' ', end='\n')
