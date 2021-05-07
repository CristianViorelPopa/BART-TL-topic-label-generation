import constants


def save_topic_words_simple_csv(model, topics_indices, out_filename):
    with open(out_filename, 'w') as out_file:
        out_file.write("topic_id,domain")
        for idx in range(constants.WORDS_PER_TOPIC_CSV):
            out_file.write(",term" + str(idx))
        out_file.write("\n")

        # count the topic number to write to the csv
        counter = 0
        for idx in topics_indices:
            topic = model.get_topic_terms(idx, topn=constants.WORDS_PER_TOPIC_CSV)

            # 'blogs' is a dummy domain
            out_file.write(str(counter) + ",blogs," +
                           ','.join([model.id2word[id_] for id_, _ in topic]) + "\n")
            counter += 1
