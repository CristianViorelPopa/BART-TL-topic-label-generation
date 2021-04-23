import re

from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from joblib import Memory

mem = Memory('./cachedir', verbose=False)
nltk.download('stopwords')  # run once
stop_words = stopwords.words('english')

words_to_be_removed = stop_words + ['nbsp', 'would', 'whilst', 'whereas', 'although', 'often',
                                    'one']


def remove_html_tags(text):
    """Remove html tags from a string"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def preprocess_doc(doc):
    """
    Pre-processing of a document
    :param doc: input string
    :return: list of remaining words
    """
    doc = doc.lower()
    doc = remove_html_tags(doc)

    # remove punctuation and other unicode characters
    # bad_chars = '\n“‘’…→»║'
    # doc = (doc
    #        .translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    #        .translate(str.maketrans(bad_chars, ' ' * len(bad_chars))))

    doc = simple_preprocess(doc)
    # remove stopwords (and other words) in the document
    doc = [word for word in filter(len, doc) if word not in words_to_be_removed]
    # lemmatize the document
    lemmatizer = WordNetLemmatizer()
    doc = [lemmatizer.lemmatize(w) for w in doc]
    # TODO: decide if this is right to do or not
    # remove numbers as individual words
    doc = list(filter(lambda w: not w.lstrip("-+").isdigit(), doc))

    return doc


def remove_stopwords(docs):
    """
    Stopwords removal for a corpus
    :param docs: the corpus containing multiple docs in BOW format, but with n-grams
    :return: the corpus stripped of n-grams containing stopwords
    """
    for idx in range(len(docs)):
        docs[idx] = [ngram for ngram in docs[idx]
                     if len(set(stop_words).intersection(ngram.split('_'))) == 0]

    return docs
