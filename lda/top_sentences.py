import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from doc_preprocessing import preprocess_doc, remove_html_tags
import constants


def rank_top_sentences_lda(sentences, lda_model, dct, topic_idx):
    # rank sentences in the document based on their relevance to the topic
    sentences_topics = [dict(lda_model[dct.doc2bow(preprocess_doc(sentence))])
                        for sentence in sentences]
    sentences_topic_prob = [sentence_topics[topic_idx] if topic_idx in sentence_topics
                            else 0.0 for sentence_topics in sentences_topics]
    ranked_sentences = np.argsort(np.array(sentences_topic_prob))

    return ranked_sentences


def get_top_sentences_regular(lda_model, corpus, num_topics, k, original_texts, left_docs, dct):
    # get the doc-topics distribution and populate a matrix for it
    docs_topics = lda_model.get_document_topics(corpus)
    topics_distribution = np.zeros((len(corpus), num_topics))
    for doc_idx in range(len(corpus)):
        for topic_idx, prob in docs_topics[doc_idx]:
            topics_distribution[doc_idx][topic_idx] = prob
    # sort the doc-topics distribution, in order to get the top K docs for every topic; shape of the
    # array will be K x num_topics
    topics_distribution = np.argsort(topics_distribution, axis=0)
    top_docs = topics_distribution[-k:, :]

    # the actual sentences, not their positions in the corpus
    # NOTE: the shape of this matrix is reversed: num_topics x K
    topics_top_sentences = []
    for topic_idx in range(num_topics):
        topics_top_sentences.append([])
        for k_idx in range(k):
            # reverse the sentences indices, so the best sentences are the first
            doc_idx = left_docs[top_docs[k - 1 - k_idx][topic_idx]]
            original_text = original_texts[doc_idx].attributes['Body'].value
            tokens_to_be_removed = ['&rdquo;', '&ldquo;', '&lt;', '&gt;', '&quot;', '&amp;',
                                    '&nbsp;', '&#39;', 'Possible Duplicate: ']
            text = (re.sub(' +', ' ', remove_html_tags(original_text)
                           .replace('\t', ' ').replace('\n', ' '))).strip()
            for token in tokens_to_be_removed:
                text = text.replace(token, '')
            # do some processing on the text to reduce the length, while preserving sentences
            if len(text) < constants.TOP_SENTENCE_LEN_UPPER_BOUND:
                paragraph = text
            else:
                tokens = re.split('(\.|!|\?)', text)
                tokens = list(filter(len, tokens))
                # sentence = ''
                # idx = 0
                # while len(sentence) < constants.TOP_SENTENCE_LEN_UPPER_BOUND \
                #         and idx + 1 < len(sentences):
                #     sentence += sentences[idx]
                #     if idx + 1 < len(sentences):
                #         sentence += sentences[idx + 1]
                #     idx += 2

                # merge real sentences with trailing punctuation that the split was done by
                sentences = [tokens[0]]
                for token in tokens[1:]:
                    if len(token) <= 1:
                        sentences[-1] += token
                    else:
                        sentences.append(token)

                # add a trailing whitespace to all sentences in case they get merged into text
                sentences = [sentence.strip() + ' ' for sentence in sentences]

                # rank sentences in the document based on their relevance to the topic
                ranked_indices = rank_top_sentences_lda(sentences, lda_model, dct, topic_idx)

                # get the best sentence
                paragraph = ''
                idx = 0
                while len(paragraph) < constants.TOP_SENTENCE_LEN_UPPER_BOUND \
                        and idx < len(sentences):
                    real_idx = ranked_indices[-idx - 1]
                    paragraph += sentences[real_idx]
                    idx += 1
                # paragraph = sentences[ranked_sentences[-1]].strip()
            topics_top_sentences[topic_idx].append(paragraph.strip())

    return topics_top_sentences


# this is inspired by the paper of Julien Velcin et al.:
# https://hal-lirmm.ccsd.cnrs.fr/lirmm-01910614/document
def rank_top_sentences_cos10(sentences, lda_model, dct, topic_idx):
    # rank sentences in the document based on the COS10 measure specified in the paper
    topic = lda_model.get_topic_terms(topic_idx, topn=constants.WORDS_PER_TOPIC_CSV)
    topic_terms = [lda_model.id2word[id_] for id_, _ in topic]
    sentence_vectors = np.zeros((len(sentences), len(topic_terms)))
    for sentence_idx in range(len(sentences)):
        for term_idx in range(len(topic_terms)):
            sentence_vectors[sentence_idx, term_idx] = \
                sentences[sentence_idx].split().count(topic_terms[term_idx]) \
                / len(sentences[sentence_idx])

    topic_vector = [prob for _, prob in topic]
    # actual computing of cosine similarity
    sentences_ranking = cosine_similarity(sentence_vectors, [topic_vector]).flatten()
    ranked_indices = np.flip(np.argsort(sentences_ranking))

    return ranked_indices


# this is inspired by the paper of Julien Velcin et al.:
# https://hal-lirmm.ccsd.cnrs.fr/lirmm-01910614/document
# the implementation here, however, differs a bit, as sentences are only taken from a single doc
def get_top_sentences_cos10(lda_model, corpus, num_topics, k, original_texts, left_docs, dct):
    # get the doc-topics distribution and populate a matrix for it
    docs_topics = lda_model.get_document_topics(corpus)
    topics_distribution = np.zeros((len(corpus), num_topics))
    for doc_idx in range(len(corpus)):
        for topic_idx, prob in docs_topics[doc_idx]:
            topics_distribution[doc_idx][topic_idx] = prob
    # sort the doc-topics distribution, in order to get the top K docs for every topic; shape of the
    # array will be K x num_topics
    topics_distribution = np.argsort(topics_distribution, axis=0)
    top_docs = topics_distribution[-k:, :]

    # the actual sentences, not their positions in the corpus
    # NOTE: the shape of this matrix is reversed: num_topics x K
    topics_top_sentences = []
    for topic_idx in range(num_topics):
        topics_top_sentences.append([])
        for k_idx in range(k):
            # reverse the sentences indices, so the best sentences are the first
            doc_idx = left_docs[top_docs[k - 1 - k_idx][topic_idx]]
            original_text = original_texts[doc_idx].attributes['Body'].value
            tokens_to_be_removed = ['&rdquo;', '&ldquo;', '&lt;', '&gt;', '&quot;', '&amp;',
                                    '&nbsp;', '&#39;', 'Possible Duplicate: ']
            text = (re.sub(' +', ' ', remove_html_tags(original_text)
                           .replace('\t', ' ').replace('\n', ' '))).strip()
            for token in tokens_to_be_removed:
                text = text.replace(token, '')
            # do some processing on the text to reduce the length, while preserving sentences
            if len(text) < constants.TOP_SENTENCE_LEN_UPPER_BOUND:
                paragraph = text
            else:
                tokens = re.split('(\.|!|\?)', text)
                tokens = list(filter(len, tokens))

                # merge real sentences with trailing punctuation that the split was done by
                sentences = [tokens[0]]
                for token in tokens[1:]:
                    if len(token) <= 1:
                        sentences[-1] += token
                    else:
                        sentences.append(token)

                # add a trailing whitespace to all sentences in case they get merged into text
                sentences = [sentence.strip() + ' ' for sentence in sentences]

                # rank sentences in the document based on their relevance to the topic
                ranked_indices = rank_top_sentences_cos10(sentences, lda_model, dct, topic_idx)

                # get the best sentence
                paragraph = ''
                idx = 0
                while len(paragraph) < constants.TOP_SENTENCE_LEN_UPPER_BOUND \
                        and idx < len(sentences):
                    real_idx = ranked_indices[-idx - 1]
                    paragraph += sentences[real_idx]
                    idx += 1
                # paragraph = sentences[ranked_sentences[-1]].strip()
            topics_top_sentences[topic_idx].append(paragraph.strip())

    return topics_top_sentences
