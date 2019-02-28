#!/usr/bin/python
import collections
import re
import nltk
import os
import sys
import getopt
import math
import struct
from nltk.stem.porter import PorterStemmer

FIXED_WIDTH = 4

def encoder(integer):
    return struct.pack('I', integer)

stemmer = PorterStemmer()

def usage():
    print ("usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file")

input_directory = output_file_dictionary = output_file_postings = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except (getopt.GetoptError):
    usage()
    sys.exit(2)
    
for o, a in opts:
    if o == '-i': # input directory
        input_directory = a
    elif o == '-d': # dictionary file
        output_file_dictionary = a
    elif o == '-p': # postings file
        output_file_postings = a
    else:
        assert False, "unhandled option"
        
if input_directory == None or output_file_postings == None or output_file_dictionary == None:
    usage()
    sys.exit(2)

index = {}
dictionary_data = open(output_file_dictionary, 'w')
postings_data = open(output_file_postings, 'wb')

count = 0
for filename in os.listdir(input_directory):
    f = open(input_directory + "/" + filename, 'r')
    whole_text = f.read()
    
    lines = nltk.sent_tokenize(whole_text)
    for individual_line in lines:
        words = nltk.word_tokenize(individual_line)
        for word in words:
            word = stemmer.stem(word.lower())
            # TODO: Take care of the stupid edge cases in punctuation and numbers
            if len(word) == 1 and (not word[0].isalpha()):
                continue
            else:
                if word in index:
                    index[word].add(int(filename))
                else:
                    index[word] = {int(filename)}
    count += 1
    # if (count >= 10): break


for key in index.keys():
    index[key] = sorted(index[key])

sorted_keys = sorted(index.keys())
docs = {}

for key in sorted_keys:
    for doc_id in index[key]:
        docs[doc_id] = True

for key in sorted(docs):
    dictionary_data.write(str(key) + ", ")

dictionary_data.write("\n")

for key in sorted_keys:
    # Print out to dictionary.txt and postings.txt
    starting_cursor = postings_data.tell()
    # Adds the key_term, size, starting cursor
    dictionary_data.write(key + ' ' + str(len(index[key])) + ' ' + str(starting_cursor) + '\n')

    skip_spaces = math.sqrt(len(index[key]))
    skip_spaces = int(skip_spaces)

    position = -1
    for doc_id in index[key]:
        position += 1
        
        if position == len(index[key]) - 1:
            postings_data.write(encoder(doc_id))
            postings_data.write(encoder(65535))

        else:
            postings_data.write(encoder(doc_id))
            if (skip_spaces >= 2 and not(position%skip_spaces) and position+skip_spaces < len(index[key])):
                postings_data.write(encoder(position+skip_spaces))
            else:
                postings_data.write(encoder(0))

dictionary_data.close()
postings_data.close()
