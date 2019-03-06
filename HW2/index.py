#!/usr/bin/python
import codecs
import collections
import re
import nltk
import os
import sys
import getopt
import math
import string
import struct
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

docs = {}
index = {}
stemmer = PorterStemmer()

FIXED_WIDTH = 4
IGNORE_NUMBERS = False # Adding an option to ignore any numerical terms
IGNORE_PUNCTUATION = True # Adding an option to strip away all punctuation in a term
IGNORE_SINGLE_CHARACTER_TERMS = True # Adding an option to ignore single character terms
IGNORE_STOPWORDS = False # Adding an option to ignore stopwords
PERFORM_STEMMING = True # Adding an option to perform stemming on the current term

def add_to_index(lines, filename):
    for individual_line in lines:
        words = nltk.word_tokenize(individual_line)

        for term in words:
            term = normalize(term)
            if term is None: continue
            if term == '': continue
            
            # After performing all normalization methods to the term, add it to the dictionary
            if term in index:
                index[term].add(int(filename))
            else:
                index[term] = {int(filename)}

def normalize(term):
    term = term.lower() # Perform case folding on the current term

    # Ignores any terms that are single characters if the bool is set to true
    if (IGNORE_SINGLE_CHARACTER_TERMS and len(term) == 1): return None
            
   # Remove all instances of punctuation if the bool is set to true.
    if IGNORE_PUNCTUATION: # Remove all instances of punctuation
        term = term.translate(str.maketrans('','',string.punctuation))
            
    # Ignores any terms which are stopwords if the bool is set to true
    if (IGNORE_STOPWORDS and term in stopwords.words('english')): return None

    """
    Ignore any numerical terms into the index. The following method
    is used to detect terms with preceding punctuation such as .09123 or even 
    fractions such as 1-1/2 as well as date representations such as 24/09/2018.

    Although, it is still not able to detect terms such as "July22nd", however we feel
    if the term also has alphabetical characters, and there are no spaces between them, 
    it is best not to treat the term as a numerical one.
    """
    if IGNORE_NUMBERS: 
        temporary = term.translate(str.maketrans('','',string.punctuation))
        if temporary.isdigit(): return None # only ignore if term is fully numerical

    # Perform stemming on the term
    if PERFORM_STEMMING: term = stemmer.stem(term)
    
    return term

def indexer(input_directory):
    # Term processing
    for filename in os.listdir(input_directory):
        f = codecs.open(input_directory + "/" + filename, encoding='utf-8')
        whole_text = f.read()
        lines = nltk.sent_tokenize(whole_text)
        add_to_index(lines, filename) # Perform term normalization and add terms to index

def sort_index():
    for key in index.keys(): index[key] = sorted(index[key])
    sorted_keys = sorted(index.keys())

    for key in sorted_keys:
        for doc_id in index[key]:
            docs[doc_id] = True
    
    return sorted_keys

def write_data(sorted_keys, dictionary_data, postings_data):
    # Adds all the document IDs to the first line seperated by commas
    for key in sorted(docs):
        dictionary_data.write(str(key) + ", ")
    dictionary_data.write("\n")

    # Adds all the dictionary terms as well as the postings list to a seperate postings file

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
            postings_data.write(encoder(doc_id))

def encoder(integer):
    return struct.pack('I', integer)

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

if __name__ == "__main__":
    dictionary_data = open(output_file_dictionary, 'w')
    postings_data = open(output_file_postings, 'wb')

    indexer(input_directory) # Index all terms in input directory
    sorted_keys = sort_index() # Sort the index
    write_data(sorted_keys, dictionary_data, postings_data) # write data to dictionary.txt and postings.txt
            
    dictionary_data.close()
    postings_data.close()