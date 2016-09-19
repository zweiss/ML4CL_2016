__author__ = 'zweiss'

from nltk.tokenize import WordPunctTokenizer
import os
from sklearn import preprocessing


def read_all_files_as_gov_or_opp(cur_dir, index=5):
    """
    Read all files of political speeches in a directory recursively and distinguish between government or opposition
    :param cur_dir: directory with political speeches
    :param index: index of government / opposition information in the file name (1 or 0)
    :return: list of file names, government speeches, opposition speeches, speech dictionary
    """

    # get a listing of all plain txt files in the dir and all sub dirs
    txt_file_list = []
    for root, dirs, files in os.walk(cur_dir):
        for name in files:
            if name.endswith('.txt') and not name.startswith('.'):
                txt_file_list.append(os.path.join(root, name))

    # process all files
    type_occurrences = {}
    gov_content = []
    opp_content = []
    tokenizer = WordPunctTokenizer()
    for txt_file in txt_file_list:
        # read file content
        cur_content = read_file(txt_file)
        # count type occurrences
        for type in set(tokenizer.tokenize(cur_content)):
            if type in type_occurrences.keys():
                type_occurrences[type] += 1
            else:
                type_occurrences[type] = 1
        # save all speeches to their according list
        gov_info = txt_file[txt_file.rfind('/')+1:].split('_')[index]
        if gov_info == "NA":
            continue
        isGov = int(gov_info)
        if isGov:
            gov_content.append(cur_content)
        else:
            opp_content.append(cur_content)

    return txt_file_list, gov_content, opp_content, type_occurrences


def read_all_files_as_party(cur_dir, index=6):
    """
    Read all files of political speeches in a directory recursively and distinguish between parties
    :param cur_dir: directory with political speeches
    :param index: index of party information in the file name
    :return: list of file names, party speeches, speech dictionary
    """

    # get a listing of all plain txt files in the dir and all sub dirs
    txt_file_list = []
    for root, dirs, files in os.walk(cur_dir):
        for name in files:
            if name.endswith('.txt') and not name.startswith('.'):
                txt_file_list.append(os.path.join(root, name))

    # process all files
    type_occurrences = {}
    party_content = {}
    tokenizer = WordPunctTokenizer()
    for txt_file in txt_file_list:
        # read file content
        cur_content = read_file(txt_file)
        # count type occurrences
        for type in set(tokenizer.tokenize(cur_content)):
            if type in type_occurrences.keys():
                type_occurrences[type] += 1
            else:
                type_occurrences[type] = 1
        # save all speeches to their according list
        party_info = txt_file[txt_file.rfind('/')+1:].split('_')[index]
        if party_info == "NA":
            continue
        if party_info in party_content.keys():
            party_content[party_info].append(cur_content)
        else:
            party_content[party_info] = []

    return txt_file_list, party_content, type_occurrences


def read_file(input_file):
    """
    Retrieves the content of a file as a string object
    :param input_file: file to be read
    :return: file content
    """

    # get file content
    with open(input_file, 'r', encoding="utf-8") as input_stream:
        text = input_stream.read()
    input_stream.close()

    return text


def get_feature_set(table, regex, scale=False, scaler=preprocessing.MaxAbsScaler()):
    """

    :param table:
    :param regex:
    :param scale: scale between -1 and 1
    :return:
    """

    tmp = table.filter(regex=regex)
    if scale:
        matrix = scaler.fit_transform(tmp.as_matrix())
        # matrix = preprocessing.scale(tmp.as_matrix())
    else:
        matrix = tmp.as_matrix()
    return matrix
