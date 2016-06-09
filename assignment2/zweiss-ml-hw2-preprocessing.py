__author__ = 'zweiss'


def get_statistics(in_file, out_file, writing_condition, language):
    """
    Gets the frequencies
    :param in_file: input file
    :param out_file: output file
    :param writing_condition: write (w) or append (a)
    :param language: current language
    """

    out_stream = open(out_file, writing_condition)
    if(writing_condition=="w"):
        out_stream.write("ADJ,ADP,ADV,AUX,CONJ,DET,NOUN,NUM,PART,PRON,PROPN,PUNCT,SCONJ,VERB,X,language,sent_length,sent_num\n")

    count = 1
    in_stream = open(in_file, 'r')
    for line in in_stream:

        cur_sent = line.strip().split(" ")
        length = len(cur_sent)

        # get unigram counts for sentence
        cur_row = {'ADJ':0, 'ADP':0, 'ADV':0, 'AUX':0, 'CONJ':0, 'DET':0, 'NOUN':0, 'NUM':0, 'PART':0, 'PRON':0,
                   'PROPN':0, 'PUNCT':0, 'SCONJ':0, 'VERB':0, 'X':0, 'language':language, "sent_length":length,
                   'sent_num':count}

        for token in cur_sent:
            cur_row[token] = cur_row[token] + 1

        # add new data row to rval string
        tmp = 1
        for key in sorted(cur_row.keys()):
            # get relative frequency for POS tags
            if key in ['language', 'sent_length', 'sent_num']:
                out_stream.write(str(cur_row.get(key)))
            else:
                out_stream.write(str(cur_row.get(key) / cur_row.get('sent_length')))
            if tmp < len(cur_row):
                out_stream.write(",")
            tmp = tmp + 1
        out_stream.write("\n")

        count = count + 1

    in_stream.close()
    out_stream.close()






if __name__ == '__main__':

    """
    Prepares data for processing in R for assignment 2 in the course "Machine learning for Computational Linguistics"
    in SS16 by Zarah Weiß
    """

    print("Start")

    out_file = "../data/language-stats.csv"

    # get statistics for English
    in_file_en = "../data/en-ud-pos.txt"
    get_statistics(in_file_en, out_file, "w", "English")

    # get statistics for German
    in_file_ge = "../data/de-ud-pos.txt"
    get_statistics(in_file_ge, out_file, "a", "German")

    # get statistics for English
    in_file_ja = "../data/ja-ud-pos.txt"
    get_statistics(in_file_ja, out_file, "a", "Japanese")

    print("Done.")









