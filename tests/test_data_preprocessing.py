import unittest
import pandas as pd
import movieclassifier.preprocessing.data_preprocessing as datp
from tests import PROJECT_ROOT

# THIS_PATH = os.path.dirname(os.path.realpath(__file__))
# PROJECT_ROOT = str(Path(THIS_PATH).parent)
FILE_PATH = PROJECT_ROOT + '/data/movies_metadata.csv'

class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        pass

    def test_load_data(self):
        df = datp.load_data(FILE_PATH)
        self.assertIsInstance(df, pd.DataFrame)

    def test_extract_genres(self):
        genres = "[{'id': 16, 'name': 'Animation'}, {'id': 35, 'name': 'Comedy'}, {'id': 10751, 'name': 'Family'}]"
        should = ['Animation', 'Comedy', 'Family']
        post = datp.extract_genres(genres)
        self.assertEqual(post, should)

    def test_data_preprocessing_df_colums(self):
        df = datp.load_data(FILE_PATH)
        post = datp.process_data(df)
        columns = list(post.columns)
        should = ['title', 'overview', 'genres']
        self.assertEqual(columns, should)

    def test_data_preprocessing_df_nan(self):
        df = datp.load_data(FILE_PATH)
        post = datp.process_data(df)
        is_nan = post.isnull().values.any()
        self.assertEqual(is_nan, False)

    def test_data_preprocessing_genres(self):
        df = datp.load_data(FILE_PATH)
        post = datp.process_data(df)
        genres = post['genres']
        self.assertIs(type(genres[0]), list)
