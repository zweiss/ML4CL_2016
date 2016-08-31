__author__ = 'zweiss'

import xml.etree.ElementTree as et
import warnings
import os
import sys
import re

def process_all_bundesparse_files(cur_dir, out_dir):
    """
    Expand all bundesparser xml files in a directory and its subdirectories to plain text files of single speeches
    :param cur_dir: directory containing xml files
    :return: saves txt files to /cur_dir/plain_txt/
    """

    if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    for lp in range(13,18):
        if not os.path.exists(out_dir+'lp'+str(lp)+'/'):
            os.makedirs(out_dir+'lp'+str(lp)+'/')

    # get a listing of all xml files in the dir and all sub dirs
    xml_file_counter = 0
    txt_file_counter = 0
    for root, dirs, files in os.walk(cur_dir):
        for f in files:
            if f.endswith('.xml'):
                # expand file from xml to one plain txt file per speech
                print('Currently processed file: ' + f)
                txt_file_counter += process_bundesparse_file(os.path.join(root, f), out_dir)
                xml_file_counter += 1

    # report results
    print('Extracted ' + str(txt_file_counter) + ' speech(es) from ' + str(xml_file_counter) + ' xml file(s)')
    print(str(txt_file_counter) + ' plain text file(s) written to: ' + out_dir)
    print('File format: BT_legislativePeriod_protocolNumber_date_speechNumber_isGovernment_party_function_name.txt')


def process_bundesparse_file(f_xml, out_dir):
    """
    Expand bundesparser xml file to plain text files of single speeches and save them to a given output directory
    :param f_xml: xml file to be expanded
    :param out_dir: output directory for plain text speeches
    :return: number of plain text files created
    """

    # get speeches as plain text and meta information as file name
    out_file_name, out_file_content = get_bundesparse_as_txt(f_xml)

    # save speeches to files
    txt_file_counter = 0
    for i in range(0, len(out_file_name)):
        out_stream = open(out_dir + out_file_name[i], 'w+', encoding="utf-8-sig")
        out_stream.write(out_file_content[i])
        out_stream.close()
        txt_file_counter += 1

    return txt_file_counter


def get_bundesparse_as_txt(xml_file, doc_meta = ['legislative_period', 'protocol_number', 'date']):
    """
    Turns single xml file in list of plain text speeches and file names with meta document and speech meta information
    :param xml_file: xml file to be parsed
    :return: list of file names and list of plain text speech content without remarks
    """

    # 1. Set up
    # =================================================================================================================

    # define return variables
    meta_information_list = []
    speech_content_list = []

    tree = et.parse(xml_file)  # read speech in xml format
    root = tree.getroot()


    # a. sanity check root elements
    if not [child.tag for child in root] == ['header', 'body']:
        warnings.warn('Unexpected root structure: ' + xml_file)

    # b. process information
    header = root.find('header')
    body = root.find('body')
    # i.e. root has a header and a body child

    # 2. document meta information
    # =================================================================================================================

    # a. sanity check header elements
    if not [info.tag.lower() for info in header[2:-2]] == doc_meta:
        warnings.warn('Unexpected header structure: ' + xml_file)

    # b. process information
    infos = [header.find(info) for info in doc_meta]
    document_meta_information = '_'.join([get_pretty_file_name(info.text) for info in infos])

    # c. prepare government information
    legislative_period = int(header.find('legislative_period').text)
    government = []
    if legislative_period <= 13 or legislative_period == 17:
        government = ['cdu', 'fdp']
    elif legislative_period <= 15:
        government = ['spd', 'gruene']
    else:
        government = ['spd', 'cdu']

    # 3. individual speech information
    # =================================================================================================================

    # a. sanity check body elements
    if not ['speaker']*len(body) == [speaker.tag.lower() for speaker in body]:
        warnings.warn('Unexpected body structure: ' + xml_file)

    # get body information
    speech_counter = 0
    for speech in body:
        speech_counter += 1

        # a. sanity check header elements
        if not speech.attrib.keys() == set(['party', 'function', 'name']):
            warnings.warn('Unexpected speech structure: speech: ' + speech_counter + ': ' + xml_file)

        # only process speeches that are not held by the Bundes(Vize)präsident(in)
        function = get_pretty_file_name(speech.attrib['function'])
        if function != 'bundestagsvizepraesidentin':
            # get meta information
            party = get_pretty_party(speech.attrib['party'])
            name = get_pretty_file_name(speech.attrib['name'])
            speech_meta_information = party + '_' + function + '_' + name
            is_government = 'NA' if party == 'NA' else str(int(party in government))


            # get content without remarks
            remark_content = [remark.text for remark in speech]
            speaker_content = ''.join([text.strip() + ' ' if text not in remark_content else '' for text in speech.itertext()])

            # save results
            speech_num = str(speech_counter)
            stem = 'lp' + str(legislative_period) + '/BT'
            file_tmp = '_'.join([stem,  document_meta_information, speech_num, is_government, speech_meta_information])
            ending = '.txt'
            meta_information_list.append(file_tmp + ending)
            speech_content_list.append(speaker_content)

    return meta_information_list, speech_content_list


def get_pretty_party(party_info):

    # possible encodings for parties
    cdu = ['CDU', 'CSU', 'CDU/CSU', 'CDU/CDU/CSU']
    fdp = ['FDP', 'F.D.P.', 'F.D:P.']
    spd = ["SPD"]
    gruene = ["BÜNDNIS 90/DIE GRÜNEN", 'B├£NDNIS 90/DIE GR├£NEN']
    linke = ['PDS', 'DIE LINKE', 'linke']
    other = ['parteilos', 'fraktionslos']
    none = ['', 'unbekannt']

    # give notice, if some other party name is included
    if party_info.strip() not in cdu + spd + fdp + gruene + linke + other + none:
        print('WARNING: unknown party: ' + party_info)

    given_party = party_info.strip()
    if given_party in cdu:
        pretty_party = "cdu"
       # print('CDU!')
    elif given_party in fdp:
        pretty_party = "fdp"
      #  print('FDP!')
    elif given_party in spd:
        pretty_party = "spd"
      #  print('SPD!')
    elif given_party in gruene:
        pretty_party = "gruene"
       # print('GRUENE!')
    elif given_party in linke:
        pretty_party = "linke"
       # print('LINKE!')
    elif given_party in none:
        pretty_party = "NA"
       # print('NA!')
    elif given_party in other:
        pretty_party = "none"
        # print('NONE!)
    else:
        pretty_party = given_party.lower()
      #  print('OTHER!')

    return pretty_party


def get_pretty_file_name(string):
    """
    Gets rid of everythig that might make the file name problematic
    :param string: file name substring
    :return: cleansed file name substring
    """

    # hyphenate empty spaces and get rid of German special characters
    tmp = string.lower().replace(' ', '-').replace('ä', 'ae').replace('ü', 'ue').replace('ö', 'oe').replace('ß', 'ss')
    # skip slashes, brackets, and dots
    tmp = re.sub('\(|\)|\.|/|\[|\]|\{|\}', '', tmp)
    return tmp

if __name__ == '__main__':

    if not len(sys.argv) == 2:
        warnings.warn('Wrong number of arguments! Call:\n> python3 bundesparser2txt.py <input directory>')
        sys.exit(0)

    # sys.argv[1] = '/Users/zweiss/Documents/Uni/9_SS16/iscl-s9_ss16-machine_learning/project/data/bundesparser-xml/'

    print('Start')
    process_all_bundesparse_files(sys.argv[1], sys.argv[1] + 'plain_text/')
    print('Done.')
