import re
import json
import unicodedata
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import inflect
import pandas as pd
import time

def process_text(text, lemmatise=False, show_times=False):
    """Applies text processing techniques to the raw input text, which includes: 
        - convert to lower case;
        - replace non-unicode characters with valid counterparts
        - remove special characters 
        - tokenization of the text
        - stopwords removal
        - conversion of numbers to textual representation
        - lemmatisation
    
    Arguments:
        text {string} -- raw text as string
        lemmatise {bool} -- whether to lemmatise text
        show_times {bool} -- whether to show processing times
    
    Returns:
        string -- the transformed text
    """
    steps = ['1. to lowercase', '2. to ACII and utf8', '3. remove special chars', \
        '4. tokenization', '5. stop words', '6. num2words', '7. lemmatisation']
    step_times = [time.time()]

    # 1. Transform all characters in lowercase
    text = to_lower(text)
    step_times.append(time.time())

    # 2. Replace all compatibility characters with their equivalents (i.e. accented)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf8')
    step_times.append(time.time())

    # 3. Remove special characters (punctuation, extra spaces)
    text = remove_specials(text)
    step_times.append(time.time())

    # 4. Tokenization
    toks = nltk.word_tokenize(text)
    step_times.append(time.time())

    # 5. Stopwords removal
    toks = remove_stopwords(toks)
    step_times.append(time.time())

    # 6. Convert to number to text representation
    toks = replace_nums2words(toks)
    step_times.append(time.time())

    # 7. Lemmatisation
    if lemmatise:
        toks = lemmatisation(toks)
    step_times.append(time.time())

    if show_times:
        _print_times(steps, step_times)
    
    return ' '.join(toks)

def to_lower(text):
    return text.lower()

def remove_specials(sentence):
    """Removes special characters (non-alphanumeric).
    
    Arguments:
        sentence {string} -- input text
    
    Returns:
        string -- text with special characters removed.
    """
    sentence = sentence.replace('-', ' ')
    sentence = re.sub(r'[^\w\s]', '', sentence)
    return sentence

def remove_stopwords(tokens):
    """Removes common stopwords.
    
    Arguments:
        tokens {list[string]} -- tokenised text to be transformed
    
    Returns:
        list[string] -- transformed words
    """
    words = []
    for word in tokens:
        if word not in stopwords.words('english'):
            words.append(word)
    return words

def replace_nums2words(tokens):
    """Replaces the numbers with their textual counterpart.
    
    Arguments:
        tokens {list[string]} -- tokenised text to be transformed
    
    Returns:
        list[string] -- transformed words
    """
    e = inflect.engine()
    words = []
    for word in tokens:
        if word.isdigit():
            words.append(e.number_to_words(word).replace(',', ''))
        else:
            words.append(word)
    return words

def lemmatisation(tokens):
    """Apply lemmatisation on a tokenised text.
    
    Arguments:
        tokens {list[string]} -- tokenised text to be transformed
    
    Returns:
        list[string] -- lemmatised words
    """
    pos_tag = nltk.pos_tag(tokens)
    lemmatiser = nltk.WordNetLemmatizer()
    wornet_tags = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    words = []
    for word, tag in pos_tag:
        proper_tag = wornet_tags.get(tag[0].upper(), wordnet.NOUN)
        words.append(lemmatiser.lemmatize(word, proper_tag))
    return words

def _print_times(names, times):
    """Utility function to print the processing time for each step.
    
    Arguments:
        names {string} -- step name
        times {float} -- timestamp for each step
    """
    sec2ms = lambda x: round(x * 1000, 1)

    for i, e in reversed(list(enumerate(times))):
        times[i] = e - times[i-1]
    times = times[1:]

    name_time = dict(zip(names, times))

    df = pd.DataFrame.from_dict({'step':names, 'time (ms)':times})
    print(df)
    print('\nText processing performance: (total: ', sec2ms(sum(times)), ' ms)', sep='')
    for step, time in name_time.items():
        print('\t', step, ':', sec2ms(time), 'ms')
    print()