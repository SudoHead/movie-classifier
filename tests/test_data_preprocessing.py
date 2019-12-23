import unittest
import pandas as pd
import movieclassifier.preprocessing.data_preprocessing as datp

class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        self.file_path  = '../data/movies_metadata.csv'
        pass

    def test_load_data(self):
        df = datp.load_data(self.file_path)
        self.assertIsInstance(df, pd.DataFrame)

    def test_extract_genres(self):
        genres = "['Animation', 'Crime', 'Horror']"
        should = ['Animation', 'Crime', 'Horror']
        post = datp.extract_genres(genres)
        self.assertEqual(post, should)

    def test_data_preprocessing(self):
        # df = datp.load_data(self.file_path)
        # post = datp.process_data(df)
        # should
        pass

