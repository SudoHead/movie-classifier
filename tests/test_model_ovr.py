import unittest
import pandas as pd
from movieclassifier.model.Model import Model
from movieclassifier.model.OvRModel import OvRModel
from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tests import PROJECT_ROOT

OVR_MODEL_PATH = PROJECT_ROOT + '/models/model.hal'

class TestModelOvR(unittest.TestCase):

    def setUp(self):
        X, y = make_multilabel_classification(n_samples=1000, n_classes=10, \
            n_labels=3, allow_unlabeled=False, random_state=42)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X,y, test_size=0.25)
        self.model = LogisticRegression(solver='saga', n_jobs=1, \
            max_iter=1000, verbose=False)

        self.model = OvRModel(self.model, threshold=0.3, test_mode=True)
        self.model.fit(self.x_train, self.y_train)

    def test_model_load(self):
        mod = Model.load(OVR_MODEL_PATH)
        self.assertIsInstance(mod, OvRModel)

    def test_trained_model(self):
        mod = Model.load(OVR_MODEL_PATH)
        title = 'Othello'
        desc = "The evil Iago pretends to be friend of Othello in order to manipulate him \
            to serve his own end in the film version of this Shakespeare classic."
        labels = mod.predict_single(title, desc)
        self.assertIn('Drama', labels)

    def test_model_predict(self):
        ypred = self.model.predict(self.x_test)
        score = f1_score(self.y_test, ypred, average='micro')
        self.assertGreater(score, 0.5)

