
from gensim import corpora

import constants

from joblib import Memory

mem = Memory('./cachedir', verbose=False)


def preprocess_corpus(docs):
    """
    Pre-processing of a list of documents
    :param docs: list of lists of words
    :return: gensim dictionary and corpus
    """
    # remove too short documents
    left_docs = [idx for idx, doc in enumerate(docs) if len(doc) > constants.NUM_WORDS_THRESHOLD]
    docs = [doc for doc in docs if len(doc) > constants.NUM_WORDS_THRESHOLD]

    # Create the Inputs of LDA model: Dictionary and Corpus
    dct = corpora.Dictionary(docs)
    corpus = [dct.doc2bow(doc) for doc in docs]

    # count occurences of each token
    total_num_occurences = {id_: 0 for id_ in range(len(dct))}
    for doc in corpus:
        for token_data in doc:
            total_num_occurences[token_data[0]] += token_data[1]

    # remaining corpus after removing tokens with too few occurences
    cleaned_corpus = [[(id_, count) for id_, count in doc
                       if constants.OCCURENCES_MIN_THRESHOLD <= total_num_occurences[id_]
                       #]
                       <= constants.OCCURENCES_MAX_THRESHOLD]
                      for doc in corpus]

    return dct, cleaned_corpus, left_docs
