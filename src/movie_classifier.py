import argparse
from pathlib import Path
import os

PROJECT_ROOT = str(Path(os.getcwd()).parent)
DEFAULT_MODEL = PROJECT_ROOT + '/model/baseline'

def get_arg_parser():
    """Routine for parsing the flags
    
    Returns:
        arg_parser: the argument parser
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', default=DEFAULT_MODEL, help="model to use")
    parser.add_argument('-t', '--title', required=True, help="title of the movie")
    parser.add_argument('-d', '--description', required=True, help='description of the movie')
    return parser

def predict(title, desc):
    """Makes a prediction of a movie's genres based on the title and a description.
    
    Arguments:
        title {string} -- title of the movie
        desc {string} -- a short description of the movie
    
    Returns:
        list[string] -- a list of genre labels
    """
    labels = []
    return labels

if __name__ == "__main__":
    argparser = get_arg_parser()
    args = vars(argparser.parse_args())

    print(args)