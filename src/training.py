import os
import time
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from movieclassifier.model.Model import Model
from movieclassifier.model.OvRModel import OvRModel
from movieclassifier.preprocessing.data_preprocessing import load_data
from movieclassifier.preprocessing.text_preprocessing import process_text

PROJECT_ROOT = str(Path(os.getcwd()).parent)
DEFAULT_LOAD_PATH = PROJECT_ROOT + '/data/movies_data_ready.csv'
DEFAULT_SAVE_PATH = PROJECT_ROOT + '/models/model.hal'
OVR = 'Ovr'

def get_arg_parser():
    """Routine for parsing the flags
    
    Returns:
        arg_parser: the argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--savepath', default=DEFAULT_SAVE_PATH, help="specify where to save the trained model", action='store_true')
    parser.add_argument('-f', '--filepath', default=DEFAULT_LOAD_PATH, help='filepath cleaned data')
    parser.add_argument('-m', '--model', default=OVR, help="model to train")
    parser.add_argument('--testsize', type=float, default=0.2, help="size of the test set")
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

    df = load_data(DEFAULT_LOAD_PATH)

    print("Getting the data ready...", end=' ')
    # add title to the text
    X = df['title'].apply(lambda x: x.lower()).astype(str) + ' ' + df['overview']
    y = df['genres'].apply(lambda x: x.split(','))

    # split the data into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(X, y, \
        test_size=args['testsize'], random_state=42)

    print('OK\n', 'Training...', sep='')

    model = None
    start_time = time.time()

    # train the OvR model
    if args['model'] == OVR:
        base_classifier = LogisticRegression(solver='saga', n_jobs=1, max_iter=100, verbose=True)
        model = OvRModel(base_classifier, threshold=0.3)
        model.fit(X, y)

    elapsed_sec = time.time() - start_time

    if model == None:
        raise ValueError("No such model:", args['model'])


    # Only calculate stats if there is some test data
    if args['testsize'] >= 0.01:
        stats = model.get_stats(x_test, y_test)
        print(stats)
    
    # Save the model as file
    model.save(args['savepath'])

    print("\nTotal training time:", round(elapsed_sec, 2), 's', end='\n')
    print("Model saved as \'", args['savepath'], "\'", sep='')