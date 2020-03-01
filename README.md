# Simple Neural Network

Simple Neural Network, or SNN for short, is a high-level neural networks API, written in Python and capable of training and evaluating simple multilayer perception networks. 
It was developed as part of the Gesture Recognition Project for the Lecture Machine Learning at the University of Würzburg. 
A big focus has been put on easiness-of-use to enable fast experimentation by intuitively defining the network. 

## Prerequisites

SNN requires a Python 3 environment with Numpy >= 1.17 to run the training of the network and Pandas >= 0.25.3 to run some of th evaluations.

## Getting Started

To use SNN in your project simply clone the repository and import SNN in your script.

You can create a SNN Model by passing a list of layer instances to the constructor:

```
import SNN
import numpy as np

myNetwork = SNN.Network([
n = 100 
    SNN.Layer(784, n, activation="ReLU", dropout=0.5),
    SNN.Layer(n, n, activation="ReLU", dropout=0.5),
    SNN.Layer(n, 10, activation="sigmoid")
])
loss="CategoricalCrossEntropyWithSoftmax",
optimizer="GradientDescentWithMomentum")
```

### Specifying the input shape 

The model needs to know what input shape it should expect. For this reason pass an `inputShape` argument to the first layer of the model. 
You have to also define the number of hidden and output layer neurons. Pass `inputShape` and `outputShape` arguments to the following layers of the model.

### Initialization of initial weights and bias 

The class `Layer` holds the method for initial weights and bias 
```
    def initializeWeights(self):
        self.weights = np.random.randn(self.inputShape, self.outputShape)/np.sqrt(self.inputShape)
        self.bias = np.random.randn(1, self.outputShape)/np.sqrt(self.inputShape)
```
 
### Specifying the neuron dropout 

Dropout is a regularization instrument to avoid overfitting of a NN. It randomly discarding a specified portion of neurons during training. To set the dropout probability to the model just pass the `dropout` argument to the layer. The dropout is set by default to 0.  

### Specifying the activation formula 

The activation of neurons is set by default to `linear`, but you can also define another activation formula for each layer by passing the `activation` argument. 

The method `forward` inside the class `Layer` uses the activation formula for the forward propagation. The return of this method is the `outputActivation` for the current layer. 

The method `backward` inside the class `Layer` uses the derivatives of the activation formula to compute the `activationError`. The `activationError` is then used to calculate the `inputError` for the backward propagation by applying the dot product of the `activationError` and the weights

The class `Layer` also holds the method `predict`. Inside this method the activation formula is used as in the forward propagation. The only difference between the method forward and predict is, that the predict method doesn't utilizes the dropout of neurons.

#### Available activations 
1. `linear`
2. `sigmoid` 
3. `ReLU`
4. `leakyReLU`
5. `ELU`

### Specifying the optimizer and the loss function

Before training, you need to specify the optimizer and the loss function for the learning process. Therefore pass the `optimizer` and the `loss` argument to your model. 

#### Available optimizers
1. `GradientDescent`
2. `GradientDescentWithMomentum`

The `updateWeights` method inside the optimizer's class updates the weights for the current layer by utilizing the defined `learning_rate`, `regularization` value and `beta` for the gradient descent with momentum.

The default value for `learning_rate` is set to 0.1, for `regularization` to 0 and the `beta` for gradient descent with momentum to 0.8.

You can also adjust this values by using the getParameter method.

```
myNetwork.setParameters(lr=1, regularization=0.005, beta=0.5) 
```
#### Available loss functions
1. `MSE`
2. `BinaryCrossEntropy`
3. `CategoricalCrossEntropy`
4. `CategoricalCrossEntropyWithSoftmax`

Each class of the loss function also holds a method for the derivative of the loss function. 

### Fitting the model

`fit` is a method of the `network` class, that allows to train the model to a fixed number of iterations on data set. It also holds some functions to store the `metrics` for the evaluation of the model. 
```
def fit(self, X, y, epochs, batch_size=64, validation_data=None, metrics=[], class_weights=None):
```
#### Arguments 
- X: input data 
- y: target data
- epochs: the number of times the training data set is used for training 
- batch_size: the number of training examples in one pass through the network 
- validation_data: specify your validation data, if you have some 
- metrics: list of metrics to be evaluated by the model during training and testing 
- class_weights: in case of imbalanced data samples, class_weight weights the loss function for under-represented classes (pay more attention on...) 

#### Available metrics 
1. `accuracy` = score of correct predictions
2. `f1 score` = the weighted average of precision and recall 
3. `precision` = positive predictive value. Measure of the classifier's exactness. Low precision indicates a high number of false positives 
4. `recall` = sensitivity or true positive rate. Measure of classifier's completeness. Low recall indicates a high number of false negatives
5. `r2` = coefficient of determination. Represents the proportion of variance that has been explained by the independent variable in the model. Goodness of fit.
6. `mse` = Mean squared error 

#### Returns 
A `history` dictionary with training loss values and metrics values at successive epochs as well as validation loss values and validation metrics values 

### Save and Load Data in network

### Evaluation =  classification_report





## Authors
* **Anrzej Dulny** 
* **Olga Strek**

