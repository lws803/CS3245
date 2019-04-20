#!/usr/bin/python
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import sys
import getopt
from os import listdir, path, walk
import string
import json
import struct
import csv
import string
import sys
import re
from math import log

LIMIT = 100
COUNT = 0
STEMMER = PorterStemmer()

def preprocess_title(titles):
    '''
    Clean data for titles before tokenizing.
    Remove court name, since it is present in court zone.
    '''
    titles = [t for t in titles if len(t) > 1]
    titles = [t for t in titles if not is_num(t)]
    titles = [t.lower() for t in titles]
    titles = split(titles)
    titles = [str(clean(w)) for w in titles]
    titles = [STEMMER.stem(t) for t in titles]
    return titles[:-1]    

def preprocess(data):
    '''
    Clean data for content before tokenizing
    '''
    data = [w for w in data if not is_num(w) and w not in string.punctuation]
    data = split(data)
    data = [str(clean(w)) for w in data]
    data = [w for w in data if not has_weird_chars(w)]
    data = [w for w in data if len(w) > 1]
    result_data = [str(STEMMER.stem(w)) for w in data]
    return result_data

def split(data):
    '''
    Splits words that are conjoined by punctuation
    '''
    result = []
    for word in data:
        result.extend(re.split('\\W+', word))
    return result

def is_num(number):
    '''
    Remove any numbers
    '''
    number = number.replace(",\\.", "")
    return number.isdigit()

def clean(original_word):
    '''
    Remove digits, punctuation, unicode chars in words
    '''
    word = re.sub("\\d", "", original_word)
    word = word.encode('utf-8', 'ignore')
    return word.lower()

def has_weird_chars(word):
    '''
    Remove illegal characters
    '''
    for i in word:
        if i not in string.printable:
            return True

def setup():
    '''
    Setting up parsing of csv files, and unicode reading
    '''
    # For reading unicode characters
    reload(sys)
    sys.setdefaultencoding('utf-8')

    # Handle large csv files
    maxInt = sys.maxsize
    while True:
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt/10)

def indexing(dataset_file, output_dictionary, output_postings):
    '''
    Weights for each doc and its court zone are written in docs.
    Title, content indexing and its positional indexes are written in postings.
    '''
    global LIMIT, COUNT

    setup()
    data_file = open(dataset_file)
    csvReader = csv.reader(data_file)
    doc_ids = []
    parsed_data = {}
    zoned_data = {}
    for row in csvReader:
        doc_id, title, content, date_posted, court = row
        if COUNT > 0:
            doc_ids.append(int(doc_id))
            parsed_data[int(doc_id)] = content
            zoned_data[int(doc_id)] = {'title': title, 'date': date_posted, 'court': court}

        COUNT += 1

        if LIMIT and COUNT == LIMIT:
            break
    doc_ids.sort()
    dictionary = {}
    doc_word_listing = {} # For Rocchio algorithm
    for doc_id in doc_ids:
        content = parsed_data[doc_id]
        content = content.decode('utf8')
        tokenized_words = word_tokenize(content)
        tokenized_words = preprocess(tokenized_words)
        tokenized_title = word_tokenize(zoned_data[doc_id]['title'].decode('utf8'))
        tokenized_title = preprocess_title(tokenized_title)

        doc_word_listing[doc_id] = set(tokenized_words+tokenized_title)

        # For (i, title/content), let 'title' = 0, 'content' = 1

        for i in range(len(tokenized_title)):
            title = tokenized_title[i]
            if len(title) == 0:
                continue
            if title not in dictionary:
                dictionary[title] = {}
            if doc_id not in dictionary[title]:
                dictionary[title][doc_id] = []
            dictionary[title][doc_id].append((i, 0))

        for i in range(len(tokenized_words)):
            term = tokenized_words[i]
            if len(title) == 0:
                continue
            if term not in dictionary:
                dictionary[term] = {}
            if doc_id not in dictionary[term]:
                dictionary[term][doc_id] = []    
            dictionary[term][doc_id].append((i, 1))

        
    dict_out = open(output_dictionary, 'w')
    postings_out = open(output_postings, "wb")
    postings = {}
    vector_space_model = {}
    byte_offset, byte_size = 0, 4

    for term, postings in dictionary.items():
        # Postings: [doc_id: [positional index, title/content]]
        term_doc_pindex = sorted(postings.items(), key=lambda x:(x[0], x[1][0]))
        df = len(term_doc_pindex)
        doc_post_length = 0
        extraBytes = 0
        for doc_id, pIndexes in term_doc_pindex:
            tf = len(pIndexes)
            doc_post_length += tf
            weight = 1 + log(tf, 10)
            if doc_id not in vector_space_model:
                vector_space_model[doc_id] = 0

            vector_space_model[doc_id] += weight**2
            for index in pIndexes:
                postings_out.write(struct.pack('III', doc_id, *index))
                extraBytes += (byte_size * 3)

        if len(term) > 0:
            dict_out.write(term + " " + str(df) + " " + str(doc_post_length) + " " + str(byte_offset) + "\n")
        byte_offset += extraBytes


    # Writing total doc weights and court-zone data to end of dictionary file
    vector_data = ""
    rochhio_data = ""
    vector_quantities = sorted(vector_space_model.items())
    for docId, value in vector_quantities:
        vector_data += str(docId) + ":" + str(value) + ":" + zoned_data[doc_id]['court'] + "\n"
        word_list = ""
        for word in doc_word_listing[docId]:
            word_list += word + "^"
        rochhio_data += str(docId)+ "^" + word_list  +"\n"
    dict_out.write("# BEGIN VECTOR DATA #\n")
    dict_out.write(vector_data)
    dict_out.write("# BEGIN ROCCHIO DATA #\n")
    dict_out.write(rochhio_data)
    dict_out.close()
    postings_out.close()

def usage():
    print ("usage: " + sys.argv[0] + " -i dataset_file -d dictionary-file -p postings-file")


if __name__ == "__main__":
    dataset_file = output_file_dictionary = output_file_postings = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
    except getopt.GetoptError as err:
        usage()
        sys.exit(2)

    for o, a in opts:
        if o == '-i': # csv file
            dataset_file = a
        elif o == '-d': # dictionary file
            output_file_dictionary = a
        elif o == '-p': # postings file
            output_file_postings = a
        else:
            assert False, "unhandled option"

    if dataset_file == None or output_file_postings == None or output_file_dictionary == None:
        usage()
        sys.exit(2)

    indexing(dataset_file, output_file_dictionary, output_file_postings)
