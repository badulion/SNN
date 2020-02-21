import numpy as np

class accuracy:
    def score (self, y_true, y_pred):
        #getting the true predictions:
        y_pred = (y_pred == y_pred.max(axis=1, keepdims=True))*1
        return (y_pred * y_true).sum()/len(y_true)

class f1_score:
    def score(self, y_true, y_pred):
        #getting the true predictions:
        y_pred = (y_pred == y_pred.max(axis=1, keepdims=True))*1
        precision = self.precision(y_true, y_pred)
        recall = self.recall(y_true, y_pred)
        #changing zeros to ones in denominator, since then 0 in numerator anyway
        recall[(recall == 0) * (precision == 0)]=1
        precision[(recall == 0) * (precision == 0)]=1
        return (2*(precision*recall)/(precision+recall)).mean()

    def precision(self, y_true, y_pred):
        prediction_counts = y_pred.sum(axis=0)

        #changing zeros to ones in denominator, since then 0 in numerator anyway
        prediction_counts[prediction_counts == 0]=1
        return ((y_pred * y_true).sum(axis=0)/prediction_counts)

    def recall(self, y_true, y_pred):
        label_counts = y_true.sum(axis=0)

        #changing zeros to ones in denominator, since then 0 in numerator anyway
        label_counts[label_counts == 0]=1
        return ((y_pred * y_true).sum(axis=0)/label_counts)

class r2:
    def score(self, y_true, y_pred):
        return -(y_true*np.log(y_pred)).sum(axis=1).mean()

class mse:
    def score (self, y_true, y_pred):
        return((y_true - y_pred) ** 2).mean()

def getMetrics(metric_list):
    metric_objects = [getOneMetric(metric) for metric in metric_list]
    metrics_dict = dict(zip(metric_list, metric_objects))
    return metrics_dict

def getOneMetric(metric):
    if metric=="accuracy":
        return accuracy()
    elif metric=="f1_score":
        return f1_score()
    elif metric=="r2":
        return r2()
    elif metric=="mse":
        return mse()
