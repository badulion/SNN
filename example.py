import SNN
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report

# this may take a while
X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
y = y.astype(int)

#scaling the data
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std==0]=0.00001
X = (X-X_mean)/X_std


y_one_hot = np.eye(y.max() + 1)[y]

# determine size of data set
data_set_size = len(X)

# determine size of the training data set (80% of X)
train_size = int(data_set_size * 0.8)

# create array with indices for X ([0, 1, 2, ...])
idxs = np.arange(data_set_size)

# fixing the numpy's seed makes sure that random
# things (like np.random.shuffle) always produce
# the same output – this way we distribute each image
# to the same pseudo random dataset each time we
# execute this cell
np.random.seed = 42

# we must not shuffle X directly, otherwise
# we don't know anymore which entry in y belongs
# to what entry in X! The solution is to shuffle
# the indices, because this way we can select the
# according entries from both, X and y
np.random.shuffle(idxs)

# the first 80% of the samples belong to the
# training data set...
train_idxs = idxs[:train_size]
# ...and the rest belongs to the validation set
validation_idxs = idxs[train_size:]

# now the actual partitioning into train and
# validation set happens:
data = {
    "train": {
        "X": X[train_idxs],
        "y_one_hot": y_one_hot[train_idxs],
        "y": y[train_idxs],
    },
    "validation": {
        "X": X[validation_idxs],
        "y_one_hot": y_one_hot[validation_idxs],
        "y": y[validation_idxs],
    },
}

myNetwork = SNN.Network([
    SNN.Layer(784, 100, activation="ReLU"),
    SNN.Layer(100, 100, activation="ReLU"),
    SNN.Layer(100, 10, activation="sigmoid")
],
loss="CategoricalCrossEntropyWithSoftmax",
optimizer="GradientDescentWithMomentum")
myNetwork.setParameters(lr=1)

myNetwork.fit(data['train']['X'], data['train']['y_one_hot'], epochs=200, batch_size=1024)


y_test = data['validation']['y']
y_pred_onehot = myNetwork.predict(data['validation']['X'])
y_pred = np.argmax(y_pred_onehot, axis=1)
print(classification_report(y_test, y_pred))
