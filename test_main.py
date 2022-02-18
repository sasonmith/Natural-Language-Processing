from flair.data import Sentence
from flair.embeddings import WordEmbeddings, FlairEmbeddings
from flair.embeddings import StackedEmbeddings
import pandas as pd
from difflib import SequenceMatcher
import json
from pathlib import Path


def matcher(string, pattern):
    """
    Return the start and end index of any pattern present in the text.
    """
    match_list = []
    # pattern = pattern.strip()
    seqMatch = SequenceMatcher(None, string, pattern, autojunk=False)
    match = seqMatch.find_longest_match(0, len(string), 0, len(pattern))
    if match.size == len(pattern):
        start = match.a
        end = match.a + match.size
        match_tup = (start, end)
        string = string.replace(pattern, "X" * len(pattern), 1)
        match_list.append(match_tup)

    return match_list, string


def mark_sentence(s, match_list):
    """
    Marks all the entities in the sentence as per the BIO scheme.
    """
    # dict containing all B-I
    word_dict = {}
    for word in s.split():
        word_dict[word] = 'O'

    for start, end, e_type in match_list:
        temp_str = s[start:end]
        tmp_list = temp_str.split()
        if len(tmp_list) > 1:
            word_dict[tmp_list[0]] = 'B-' + e_type
            for w in tmp_list[1:]:
                word_dict[w] = 'I-' + e_type
        else:
            word_dict[temp_str] = 'B-' + e_type
    return word_dict


def clean(text):
    """
    Just a helper function to add a space before the punctuations for better tokenization
    """
    filters = ["!", "#", "$", "%", "&", "(", ")", "/", "*", ":", ";", "<", "=", ">", "?", "@", "[",
               "\\", "]", "_", "`", "{", "}", "~", "'"]
    for i in text:
        if i in filters:
            text = text.replace(i, i + " ")

    return text


def create_data(df, filepath):
    """
    The function responsible for the creation of data in the said format.
    """
    with open(filepath, 'w', encoding="utf-8") as f:
        for text, annotation in zip(df.text, df.annotation):
            # text = clean(text)
            text_ = text
            match_list = []

            for i in annotation:
                a, text_ = matcher(text, i[0])
                match_list.append((a[0][0], a[0][1], i[1]))

            d = mark_sentence(text, match_list)

            for i in d.keys():
                f.writelines(i + ' ' + d[i] + '\n')
            f.writelines('\n')


def load_json(path):
    with open(path, "r") as f:
        dict = json.load(f)
    return dict


def get_answers(test_json_dict):
    results = {}
    ner_list = ["Document Name", "Parties", "Agreement Date", "Expiration Date", "Governing Law",
                "No-Solicit Of Employees", "Anti-Assignment", "License Grant", "Cap On Liability", "Insurance"]
    data = test_json_dict["data"]
    for contract_num, contract in enumerate(data):
        title = contract["title"]
        new_ner_list = []
        for ner in ner_list:
            title_appended = title + "__" + ner
            new_ner_list.append(title_appended)
        for para in contract["paragraphs"]:
            qas = para["qas"]
            for qa in qas:
                id = qa["id"]
                if id in new_ner_list:
                    answers = qa["answers"]
                    final_format = []
                    for i in range(len(answers)):
                        ans = [answers[i]["text"], answers[i]["answer_start"]]
                        final_format.append(ans)
                    results[id] = final_format
    return results


def get_titles(test_json_dict):
    data = test_json_dict["data"]
    titles = []
    for contract in data:
        title = contract['title']
        titles.append(title)
    return titles


def append_titles(titles):
    ner_list = ["Document Name", "Parties", "Agreement Date", "Expiration Date", "Governing Law",
                "No-Solicit Of Employees", "Anti-Assignment", "License Grant", "Cap On Liability", "Insurance"]
    appended_titles = []
    for title in titles:
        for ner in ner_list:
            title_appended = title + "__" + ner
            appended_titles.append(title_appended)

    return appended_titles


attributeDictionary = {
    "Document Name": "DN",
    "Parties": "P",
    "Agreement Date": "AD",
    "Expiration Date": "ED",
    "Governing Law": "GL",
    "No-Solicit of Employees": "NS",
    "Anti-Assignment": "AA",
    "License Grant": "LG",
    "Cap on Liability": "CL",
    "Insurance": "IN"
}


def get_cur_attr_by_title(title):
    attr_list = title.split("_")
    if attr_list[-1] == "Document Name":
        attr_list[-1] = "DN"
    elif attr_list[-1] == "Parties":
        attr_list[-1] = "P"
    elif attr_list[-1] == "Agreement Date":
        attr_list[-1] = "AD"
    elif attr_list[-1] == "Expiration Date":
        attr_list[-1] = "ED"
    elif attr_list[-1] == "Governing Law":
        attr_list[-1] = "GL"
    elif attr_list[-1] == "No-Solicit Of Employees":
        attr_list[-1] = "NS"
    elif attr_list[-1] == "Anti-Assignment":
        attr_list[-1] = "AA"
    elif attr_list[-1] == "License Grant":
        attr_list[-1] = "LG"
    elif attr_list[-1] == "Cap on Liability":
        attr_list[-1] = "CL"
    elif attr_list[-1] == "Insurance":
        attr_list[-1] = "IN"
    return attr_list[-1]


def answer_attribute_pair(ans, attr):
    pair = (ans, attr)
    return pair


def get_contexts(test_json_dict):
    results = {}
    data = test_json_dict['data']
    for contract in data:
        title = contract['title']
        for para in contract["paragraphs"]:
            results[title] = para["context"]
    return results


def get_only_title(title):
    newTitle = title.split('_')
    index = 0
    title_no_attr = ""
    while index < (len(newTitle) - 2):
        title_no_attr += newTitle[index]
        index += 1
    return title_no_attr


def main():

    # --------------------------------------------------------------------------------------- #
    # key:
    # a TITLE is the name of a specific attribute within a contract
    # an example of a title is 'LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT__Parties'
    # this title represents all the 'Parties' attributes that can be taken from CUAD_v1.json
    # a CONTRACT is the combination of 10 titles with every attribute included
    # contracts will mostly be plain text and used as context for NER recognition
    # --------------------------------------------------------------------------------------- #

    # path to save the txt file.
    filepath = 'train/'
    # read text for LIME ENERGY distribution contract from json
    cuad_json = load_json("json/CUAD_v1.json")
    # get all titles in a list, around 500 titles in this
    titles = get_titles(cuad_json)
    # append the attribute of the title to the end of the title
    # now of form 'contractName__attribute'
    titles_appended = append_titles(titles)
    # get full context of contract (contract as a str)
    context = get_contexts(cuad_json)
    # get all answers associated with different titles
    # key is title of contract
    answers = get_answers(cuad_json)

    # -------------------
    # testing
    # -------------------
    # print working title
    # print(titles_appended[1])
    # print working title answers (does not include text associated w)
    # print(answers[titles_appended[1]])
    # print only the text from the first element of answers[titles_appended[1]]
    # print("first of title[1]: " + answers[titles_appended[1]][1][0])
    # print first 10 titles of the first contract
    # print(titles_appended[0:10])
    # -------------------

    # create list of k,v pairs that will be input into the second half of the dataframe
    ans_attr_associated = []
    # change which title the for loop goes over, index one should output:
    # [('Distributor', 'Parties'), ('Electric City Corp.', 'Parties'), ('Electric City of Illinois L.L.C.', 'Parties'),
    # ('Company', 'Parties'), ('Electric City of Illinois LLC', 'Parties')]
    title_index = 0
    context_index = 0
    index = 0

    # get titles into new dict that associates answer with attribute pairing
    # loop through all titles (10 per contract with all attributes)
    for a in titles_appended:
        answer_index = 0
        # get current title
        cur_title = titles_appended[title_index]
        # print(cur_title)
        # loop through all answers in a given title:
        # example file="LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT__Parties"
        # returns five answers with the attribute of 'Parties'

        # next file
        if title_index % 10 == 0:
            # ans_attr_associated = []
            answer_index = 0
            # get new file name (next contract with 10 attrs)
            # get just name from title
            newTitle = get_only_title(cur_title)
            path = 'train/' + newTitle + '_train.txt'
            filepath = path
            # check if file already exists
            if Path('train/' + newTitle + '_train.txt').is_file():
                # clear old and overwrite if it does exist
                # print("File exist")
                file = open(path, "r+")
                file.truncate()
            else:
                # make new file if it doesnt exist
                # print("File not exist")
                f = open(path, "w+")

        for ans in answers[titles_appended[title_index]]:

            # if its the 10th title, it has gone through titles [0:9]
            # because theres only 10 attrs, we know the next file will start next itr
            if title_index % 10 == 0 and title_index != 0:
                break

            # - appends a k,v pair to the list ans_attr_associated
            # - answers[titles_appended[title_index]][answer_index][0] is the text of all answers inside a specific attr
            # - get_cur_attr_by_title(cur_title) gets the attr by splitting the current title and taking the last
            # element (the attribute)
            ans_attr_associated.append(answer_attribute_pair(answers[titles_appended[title_index]][answer_index][0],
                                                             get_cur_attr_by_title(cur_title)))
            answer_index += 1

        # create dataframe for the ten titles that have been gone through
        if title_index % 10 == 0 and title_index != 0:
            print(ans_attr_associated)
            print(len(ans_attr_associated))
            # the main data frame
            # includes the entire context for a contract and lists of every attribute answer given in CUAD_v1.json
            # made of the form text, annotation
            if title_index % 10 == 0 and title_index != 0:
                # print(ans_attr_associated)
                # print(len(ans_attr_associated))
                # the main data frame
                # includes the entire context for a contract and lists of every attribute answer given in CUAD_v1.json
                # made of the form text, annotation
                data = pd.DataFrame([[context[titles[context_index]],
                                      [ans_attr_associated[0]]],
                                     ],
                                    columns=['text', 'annotation'])

                context_frame = []
                answers_frame = ans_attr_associated
                data_dict = {'text': context[titles[context_index]], 'annotation': [ans_attr_associated]}

                data = pd.DataFrame(data_dict)
                print(data)

                # title_index += 1
                ans_attr_associated = []
                context_index += 1
                # creating the file.
                create_data(data, filepath)

        title_index += 1

    # print new answers with associated attribute pairings
    # print(ans_attr_associated)


if __name__ == '__main__':
    main()
