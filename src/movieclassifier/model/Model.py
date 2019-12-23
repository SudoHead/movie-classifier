from abc import ABC, abstractmethod
import pickle
import unicodedata
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from movieclassifier.preprocessing.text_preprocessing import process_text

class Model(ABC):

    def __init__(self):
        """Base constructor
        """
        self.binarizer = MultiLabelBinarizer()
        self.vectorizer = TfidfVectorizer(min_df=0.00009,\
            max_features=20000,\
            stop_words='english',\
            ngram_range=(1,3))
        self.clf = None

    @abstractmethod
    def fit(self, X, y):
        """This method trains the model with the input data.
        
        Arguments:
            X {pandas.Series} -- 1D array containing the text for each example
            y {numpy.ndarray} -- 1D array with the labels for each example
        """

    @abstractmethod
    def predict(self, X):
        """Predicts the labels (genres) of every observation in X.
        
        Arguments:
            X {pandas.Series} -- the data to predict 
        
        Returns:
            numpy.ndarray -- array with the predictions for each observation in X
        """

    def predict_single(self, title, description, verbose=False):
        """Predicts the genre tags of a movie given a title and a description.
        
        Arguments:
            title {string} -- title of the movie
            description {string} -- a short description of the movie
        
        Returns:
            tuple -- assigned genre tags
        """
        observation = []
        observation.append(process_text(title.lower() + description, show_times=verbose))
        pred = self.predict(observation)
        tags = self.binarizer.inverse_transform(pred)
        return tags[0]

    def save(self, file_path):
        """Saves the model to the specified file path.
        
        Arguments:
            file_path {string} -- file path to save the model to
        """
        file = open(file_path, 'wb')
        pickle.dump(self, file)
        file.close()

    @staticmethod
    def load(file_path):
        """Loads a model from file.
        
        Arguments:
            file_path {string} -- file path of the model to load
        
        Returns:
            Model -- the loaded model
        """
        file = open(file_path, 'rb')
        model = pickle.load(file)
        file.close()
        return model

    def get_stats(self, x_test, y_test):
        """Calculates the metrics of the model using the test set.
        
        Arguments:
            x_test {ndarray} -- arary of unprocessed text
            y_test {ndarray} -- array of raw labels
        
        Returns:
            dict -- values for corresponding metrics
        """
        # binarize labels and get predictions
        y_test = self.binarizer.transform(y_test)
        ypred = self.predict(x_test)

        prec, recall, f1, _ = metrics.precision_recall_fscore_support(y_test, ypred, average='micro')

        perf_metrics = {}
        perf_metrics['Precision'] = prec
        perf_metrics['Recall'] = recall
        perf_metrics['F1 score'] = f1
        return perf_metrics
        