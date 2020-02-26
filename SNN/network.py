from .loss import getLoss
from .optimizer import getOptimizer
from .metrics import getMetrics
import numpy as np
import os

class Network:
    def __init__(self, layerList, loss, optimizer="GradientDescent"):
        #dimensions of input and output vectors
        self.inputShape = layerList[0].inputShape
        self.networkLayers = layerList

        for layer in self.networkLayers:
            layer.initializeWeights()

        #optimization variables
        self.optimizer = getOptimizer(optimizer, self)
        self.loss = getLoss(loss)

    def forward(self, inputMatrix):
        #check the dimensions of the input matrix
        if inputMatrix.shape[1] != self.inputShape:
            raise NameError('Input Array does not have the right dimension. Expected: %d but got %d.' % (self.inputShape, inputMatrix.shape[1]))

        for layer in self.networkLayers:
            inputMatrix = layer.forward(inputMatrix)

        self.outputActivation = inputMatrix
        return self.outputActivation

    def predict(self, inputMatrix):
        #check the dimensions of the input matrix
        if inputMatrix.shape[1] != self.inputShape:
            raise NameError('Input Array does not have the right dimension. Expected: %d but got %d.' % (self.inputShape, inputMatrix.shape[1]))

        for layer in self.networkLayers:
            inputMatrix = layer.predict(inputMatrix)

        return inputMatrix

    def backward(self, error):
        for layer in reversed(self.networkLayers):
            error = layer.backward(error)




    def fit(self, X, y, epochs, batch_size=64, validation_data=None, metrics=[], class_weights=None):
        if X.shape[0] != y.shape[0]:
            raise NameError('Labels do not match the features!')

        data_size = X.shape[0]
        random_indices = np.arange(data_size)
        batch_divisions = np.arange(0, data_size, batch_size)

        #class_weights:
        if not class_weights is None:
            class_weights = np.asarray(class_weights)
            classes = np.argmax(y, axis=1)
            weight_balance = np.expand_dims(class_weights[classes], axis=1)

        #metrics dictionary
        metrics_dictionary = getMetrics(metrics)

        #create history dictionary
        history = {
            'loss': [],
        }
        #store metrics values:
        for metric in metrics:
            history.update({metric: []})

        #if validation data provided:
        if not validation_data == None:
            history.update({'loss_val': []})

            #metrics on validation data
            for metric in metrics:
                history.update({metric+"_val": []})

        #if the data size is not a perfect multiple of the batch size
        #last batch will be smaller:
        if batch_divisions[-1] < data_size:
            batch_divisions = np.append(batch_divisions, [data_size])


        for i in range(epochs):
            #randomly shuffle the data
            np.random.shuffle(random_indices)
            X = X[random_indices, :]
            y = y[random_indices, :]

            #class_weights:
            if not class_weights is None:
                weight_balance = weight_balance[random_indices, :]

            for j in range(len(batch_divisions)-1):
                #get the current batch
                batch_X = X[batch_divisions[j]:batch_divisions[j+1], :]
                batch_y = y[batch_divisions[j]:batch_divisions[j+1], :]

                #class_weights:
                if not class_weights is None:
                    batch_weight_balance = weight_balance[batch_divisions[j]:batch_divisions[j+1], :]

                #push forward through network:
                batch_y_pred = self.forward(batch_X)

                #compute loss and loss gradient
                loss = self.loss.error(batch_y, batch_y_pred)
                loss_gradient = self.loss.derivative(batch_y, batch_y_pred)

                #class_weights:
                if not class_weights is None:
                    loss = loss*batch_weight_balance
                    loss_gradient = loss_gradient*batch_weight_balance

                #backpropagation (updates gradients)
                self.backward(loss_gradient)

                #call optimizer
                self.optimizer.updateWeights()

            y_pred = self.predict(X)

            #update history and print progress
            loss = self.loss.error(y, y_pred)
            history['loss'].append(loss)
            progress_message = "Epoch: %d/%d, loss: %.3f" % (i+1, epochs, loss)

            #update metrics history:
            for metric in metrics:
                metric_score = metrics_dictionary[metric].score(y, y_pred)
                history[metric].append(metric_score)
                progress_message = progress_message + (", %s: %.3f" % (metric, metric_score))

            if not validation_data == None:
                y_val_pred = self.predict(validation_data[0])
                val_loss = self.loss.error(validation_data[1], y_val_pred)
                history['loss_val'].append(val_loss)
                progress_message = progress_message + (", loss_val: %.3f" % val_loss)

                #update metrics history for val data:
                for metric in metrics:
                    metric_score = metrics_dictionary[metric].score(validation_data[1], y_val_pred)
                    history[metric+"_val"].append(metric_score)
                    progress_message = progress_message + (", %s: %.3f" % (metric+"_val", metric_score))

            print(progress_message)
        return history

    def setParameters(self, lr = 0.1, regularization=0, beta=0.8):
        self.optimizer.learning_rate = lr
        self.optimizer.regularization = regularization
        self.optimizer.beta = beta

    def save(self, path = "./"):
        if not os.path.exists(path):
            os.makedirs(path)
        for i in range(len(self.networkLayers)):
            bias_name = "bias"+"_layer_"+str(i)+".npy"
            weights_name = "weights"+"_layer_"+str(i)+".npy"
            np.save(path+weights_name, self.networkLayers[i].weights)
            np.save(path+bias_name, self.networkLayers[i].bias)

    def load(self, path = "./"):
        for i in range(len(self.networkLayers)):
            bias_name = "bias"+"_layer_"+str(i)+".npy"
            weights_name = "weights"+"_layer_"+str(i)+".npy"
            self.networkLayers[i].weights= np.load(path+weights_name)
            self.networkLayers[i].bias = np.load(path+bias_name)
