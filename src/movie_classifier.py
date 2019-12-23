import argparse
from pathlib import Path
import os
import json
from movieclassifier.model.Model import Model
import time

PROJECT_ROOT = str(Path(os.getcwd()).parent)
DEFAULT_MODEL = PROJECT_ROOT + '/models/model.hal'

def get_arg_parser():
    """Routine for parsing the flags
    
    Returns:
        arg_parser: the argument parser
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', default=DEFAULT_MODEL, help="model to use")
    parser.add_argument('-t', '--title', required=True, help="title of the movie")
    parser.add_argument('-d', '--description', required=True, help='description of the movie')
    parser.add_argument('-v', '--verbose', help='verbose mode', action='store_true')
    return parser

def predict(title, desc, model, verbose=False):
    """Makes a prediction of a movie's genres based on the title and a description.
    
    Arguments:
        title {string} -- title of the movie
        desc {string} -- a short description of the movie
        model {Model} -- ML model
    
    Returns:
        list[string] -- a list of genre labels
    """
    labels = model.predict_single(title, desc, verbose)
    output = {\
        'title': title, \
        'description': desc, \
        'genre': list(labels)}
    jdata = json.dumps(output)
    return jdata

if __name__ == "__main__":
    argparser = get_arg_parser()
    args = vars(argparser.parse_args())

    start_time = time.time()
    model = Model.load(args['model'])

    loading_time = time.time()

    pred = predict(args['title'], args['description'], model, verbose=args['verbose'])
    pred_time = time.time() - loading_time

    sec2ms = lambda x: int(round(x * 1000))
    if args['verbose']:
        print('Model loading time:', sec2ms(loading_time - start_time), 'ms')
        print('Inference time:', sec2ms(pred_time), 'ms')
        print()

    print(pred)
    