from model.Model import Model
from sklearn.multiclass import OneVsRestClassifier
from sklearn.base import BaseEstimator

class OvRModel(Model):

    def __init__(self, base_sk_estimator, threshold=0.5):
        super().__init__(self)
        self.ovr_clf = OneVsRestClassifier(base_sk_estimator)
        self.threshold = threshold
        self.support_proba = hasattr(base_sk_estimator.__class__, 'predict_proba')
    
    def fit(self, X, y):
        # call parent method to get vector form of X and binarize y
        super().fit(X, y)
        self.ovr_clf.fit(X, y, verbose=True)

    def predict(self, X):
        # call parent method to vectorize X
        super().predict(X)

        # Apply threshold to prediction if supported by the estimator
        if self.support_proba:
            y_pred = self.ovr_clf.predict_proba(X)
            y_pred = (y_pred >= self.threshold).astype(int)
        else:
            y_pred = self.ovr_clf.predict(X)

        return y_pred
