import re
import json
import unicodedata
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import inflect
import pandas as pd

def process_text(text):
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
    
    Returns:
        string -- the transformed text
    """
    # 1. Transform all characters in lowercase
    text = to_lower(text)
    # 2. Replace all compatibility characters with their equivalents (i.e. accented)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf8')
    # 3. Remove special characters (punctuation, extra spaces)
    text = remove_specials(text)
    # 4. Tokenization
    toks = nltk.word_tokenize(text)
    # 5. Stopwords removal
    toks = remove_stopwords(toks)
    # 6. Convert to number to text representation
    toks = replace_nums2words(toks)
    # 7. Lemmatisation
    toks = lemmatisation(toks)

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