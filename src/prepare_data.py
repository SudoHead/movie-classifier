import os

import argparse
from pathlib import Path
from tqdm import tqdm
import nltk
from movieclassifier.preprocessing.data_preprocessing import process_data, load_data, save_data
from movieclassifier.preprocessing.text_preprocessing import process_text

THIS_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = str(Path(THIS_PATH).parent)
DEFAULT_LOAD_PATH = PROJECT_ROOT + '/data/movies_metadata.csv'
DEFAULT_SAVE_PATH = PROJECT_ROOT + '/data/movies_data_ready.csv'

def get_arg_parser():
    """Routine for parsing the flags
    
    Returns:
        arg_parser: the argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--savepath', default=DEFAULT_SAVE_PATH, help="specify where to save the processed data")
    parser.add_argument('-f', '--filepath', default=DEFAULT_LOAD_PATH, help='filepath of the raw data')
    return parser

def ETL(df):
    """Transformes the raw dataframe, making it ready to used in the training stage.
    
    Arguments:
        df {pandas.DataFrame} -- unprocessed dataframe
    
    Returns:
        pandas.DataFrame -- transformed dataframe.
    """
    print('\nData cleaning...', end=' ')
    print('done.')
    df = process_data(df)

    print('\nText preprocessing and cleaning...')
    df['overview'] = df['overview'].progress_apply(process_text)

    # Make genres colum easy to separate as str
    df['genres'] = df['genres'].apply(lambda x: ','.join(x))
    return df

if __name__ == "__main__":
    argparser = get_arg_parser()
    args = vars(argparser.parse_args())

    path_load = args['filepath']
    path_save = args['savepath']

    # Check whether the load path exits
    if not Path(path_load).is_file():
        print('Error: \"', path_load, '\" is not a file!')
        exit()

    print("Using the following dataset: ", path_load)

    # Creates a new tqdm instance with pandas
    tqdm.pandas()

    # Load data, transform it and save it
    df = load_data(path_load)
    df = ETL(df)

    print("Saving processed data as", path_save, '...')
    save_data(df, path_save)



