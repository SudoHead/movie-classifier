from abc import ABC, abstractmethod, staticmethod
import unicodedata
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing.text_preprocessing import process_text

# use cPickle when available for better performance
try:
   import cPickle as pickle
except:
   import pickle

class Model(ABC):

    def __init__(self):
        """Base constructor
        """
        self.binarizer = MultiLabelBinarizer()
        self.vectorizer = TfidfVectorizer(min_df=0.00009,\
            max_features=20000,\
            stopwords='english',\
            ngram_range=(1,3))

    @abstractmethod
    def fit(self, X, y):
        """This method trains the model with the input data.
        
        Arguments:
            X {pandas.Series} -- 1D array containing the text for each example
            y {numpy.ndarray} -- 1D array with the labels for each example
        """
        # transform target variables
        self.binarizer.fit(y)
        y = self.binarizer.transform(y)

        # transform text to vector
        X = self.vectorizer.fit_transform(X.values.astype('U'))

    @abstractmethod
    def predict(self, X):
        """Predicts the labels (genres) of every observation in X.
        
        Arguments:
            X {pandas.Series} -- the data to predict 
        
        Returns:
            numpy.ndarray -- array with the predictions for each observation in X
        """
        X = self.vectorizer.transform(X.values.astype('U'))

    def predict_single(self, title, description):
        """Predicts the genre tags of a movie given a title and a description.
        
        Arguments:
            title {string} -- title of the movie
            description {string} -- a short description of the movie
        
        Returns:
            tuple -- assigned genre tags
        """
        observation = []
        observation.append(process_text(title.lower() + description))
        pred = self.predict(observation)
        tags = self.binarizer.inverse_transform(pred)
        return tags[0]

    def save(self, file_path):
        """Saves the model to the specified file path.
        
        Arguments:
            file_path {string} -- file path to save the model to
        """
        file = open(file_path, 'w')
        file.write(pickle.dumps(self.__dict__))
        file.close()

    @staticmethod
    def load(file_path):
        """Loads a model from file.
        
        Arguments:
            file_path {string} -- file path of the model to load
        
        Returns:
            Model -- the loaded model
        """
        file = open(file_path, 'r')
        model = pickle.load(file)
        file.close()
        return model

    def get_stats(self, x_test, y_test):
        pass
    