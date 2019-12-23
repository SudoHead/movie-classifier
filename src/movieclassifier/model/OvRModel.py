from movieclassifier.model.Model import Model
from sklearn.multiclass import OneVsRestClassifier
from sklearn.base import BaseEstimator

class OvRModel(Model):

    def __init__(self, base_sk_estimator, threshold=0.5, test_mode=False):
        super().__init__()
        self.clf = OneVsRestClassifier(base_sk_estimator)
        self.threshold = threshold
        self.support_proba = hasattr(base_sk_estimator.__class__, 'predict_proba')
        self.test_mode = test_mode
    
    def fit(self, X, y):
        if not self.test_mode:
            # transform target variables
            self.binarizer.fit(y)
            y = self.binarizer.transform(y)

            # transform text to vector
            X = self.vectorizer.fit_transform(X.values.astype('U'))
        
        # train model
        self.clf.fit(X, y)

    def predict(self, X):
        if not self.test_mode:
            # transform text to vector
            if len(X) == 1:
                X = self.vectorizer.transform(X)
            else:
                X = self.vectorizer.transform(X.values.astype('U'))

        # Apply threshold to prediction if supported by the estimator
        if self.support_proba:
            y_pred = self.clf.predict_proba(X)
            y_pred = (y_pred >= self.threshold).astype(int)
            print(type(y_pred))
            # self.binarizer
        else:
            y_pred = self.clf.predict(X)

        return y_pred
