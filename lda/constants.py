
# ~~~ MODEL TRAINING ~~~

# minimum number of words per document
NUM_WORDS_THRESHOLD = 20  # prev: 20
# occurences thresholds for words
OCCURENCES_MIN_THRESHOLD = 10  # prev: 10
OCCURENCES_MAX_THRESHOLD = 50000  # prev: none
# number of topics to extract from the model
NUM_TOPICS = 100
# topics with a coherence lower than this threshold will be ignored from the model
COHERENCE_THRESHOLD = 0.30  # prev: 0.25
# how many top sentences to extract for each topic (to provide more insight into what a topic is
# about), raw and actual
NUM_TOP_SENTENCES_RAW = 10
NUM_TOP_SENTENCES = 2
# restricting the types of sentences that can be in the top
TOP_SENTENCE_LEN_UPPER_BOUND = 120
TOP_SENTENCE_LEN_LOWER_BOUND = 80  # NOTE: this only applied to non-raw sentences


# ~~~ TOPIC EXTRACTION ~~~

WORDS_PER_TOPIC_CSV = 10
WORDS_PER_TOPIC_JSON = 100


# ~~~ NOUN PHRASES EXTRACTION ~~~

TOPIC_DOC_SIMILARITY_THRESHOLD = 1500
NUM_NOUN_PHRASES_PER_TOPIC = 50
MIN_NOUN_PHRASE_LEN = 2
MAX_NOUN_PHRASE_LEN = 4
NOUN_PHRASES_OCCURRENCES_MIN_THRESHOLD = 25
