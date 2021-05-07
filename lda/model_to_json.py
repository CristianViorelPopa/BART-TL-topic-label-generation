import json

import constants


def save_topic_words_json(model, topics_indices, out_filename):
    topics = []
    for idx in topics_indices:
        topic = model.get_topic_terms(idx, topn=constants.WORDS_PER_TOPIC_JSON)

        topics.append([{
            'word': model.id2word[id_],
            'prob': str(prob)
        } for id_, prob in topic])

    with open(out_filename, 'w') as out_file:
        json.dump(topics, out_file, indent=4)
