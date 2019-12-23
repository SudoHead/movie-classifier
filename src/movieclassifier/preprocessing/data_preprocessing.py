import re
import json
import unicodedata
import nltk
import inflect
import numpy as np
import pandas as pd

def load_data(file_path):
    """Loads the data from the file path as a pandas dataframe without further processing.
    
    Arguments:
        file_path {string} -- data file path
    
    Returns:
        pandas.DataFrame -- loaded data as dataframe
    """
    return pd.read_csv(file_path)

def save_data(df, file_path):
    """Saves the dataframe as a csv file for later use.
    
    Arguments:
        df {pandas.DataFrame} -- dataframe to save to file
        file_path {str} -- default path (default: {'./data/movies_data_ready.csv'})
    """
    df.to_csv(file_path, index=False)

def extract_genres(genres_str):
    """Extracts the genres in string form as a list of genres
    
    Arguments:
        genres_str {string} -- string containing the genres
    
    Returns:
        list -- the extracted genres
    """
    genres_str = genres_str.replace("'", '\"')
    genres_json = json.loads(genres_str)
    genres_list = []
    for elem in genres_json:
        genres_list.append(elem['name'])
    return genres_list

def process_data(df):
    """Transformes the format of the raw dataframe and handles the missing values.
    
    Arguments:
        df {pandas.DataFrame} -- the raw input dataframe
    
    Returns:
        pandas.DataFrame -- transformed dataframe, ready for text preprocessing
    """
    df = pd.concat([df['release_date'], df['title'], df['overview'], df['genres']], axis=1)

    # Drop the NaN rows where either title or overview is NaN
    # remove duplicates
    duplicate_rows = df[df.duplicated()]
    df.drop(duplicate_rows.index, inplace=True)

    # convert empty string to NaN
    df['overview'].replace('', np.nan, inplace=True)
    df.dropna(subset=['release_date', 'title', 'overview'], inplace=True)

    # the release date is no longer necessary, because NaN are cleared
    del df['release_date']

    # Drop rows with no overview info or blank
    reg_404 = "^not available|^no overview"
    overview_not_found = df['overview'].str.contains(reg_404, regex=True, flags=re.IGNORECASE)
    df.drop(df[overview_not_found].index, inplace=True)

    overview_blank = df['overview'].str.isspace()
    df.drop(df[overview_blank].index, inplace=True)

    # Transform column genre
    # remove rows with no genres, since they don't provide any information
    df.drop(df[df['genres'] == '[]'].index, inplace=True)

    # transform genres from string to list
    temp_genre = df['genres'].apply(extract_genres)
    df['genres'] = temp_genre
    
    return df