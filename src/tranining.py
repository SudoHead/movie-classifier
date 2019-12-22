import sys
import os
sys.path.append('./preprocessing')

import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = str(Path(os.getcwd()).parent)
DEFAULT_LOAD_PATH = PROJECT_ROOT + '/data/movies_metadata.csv'
DEFAULT_SAVE_PATH = PROJECT_ROOT + '/model/model'

def get_arg_parser():
    """Routine for parsing the flags
    
    Returns:
        arg_parser: the argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--savepath', default=DEFAULT_SAVE_PATH, help="specify where to save the trained model", action='store_true')
    parser.add_argument('-f', '--filepath', default=DEFAULT_LOAD_PATH, help='filepath cleaned data')
    parser.add_argument('-m', '--model', default="", help="model to train")
    return parser

def split_train_val_test(self, X, y, test_val_size=0.15, random_seed=42):
        """Splits the data in three sets: training, validation, test.
        
        Arguments:
            X {pandas.Series} -- 1D array containing the text for each example
            y {numpy.ndarray} -- 1D array with the labels for each example
        
        Keyword Arguments:
            test_val_size {float} -- proportion of the test and validation sets (default: {0.15})
            random_seed {int} -- seed for random shuffle (default: {42})
        
        Returns:
            Tuple -- split of X, y in x_train, x_val, x_test, y_train, y_val, y_test
        """
        x_train, x_test, y_train, y_test = train_test_split(X, y, \
                test_size=test_val_size, random_state=random_seed)

        validation_size_relative = test_val_size/(1-test_val_size)

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, \
                test_size=validation_size_relative, random_state=random_seed)
        return x_train, x_val, x_test, y_train, y_val, y_test

if __name__ == "__main__":
    argparser = get_arg_parser()
    args = vars(argparser.parse_args())

    
    print(args)