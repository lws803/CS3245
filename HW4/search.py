#!/usr/bin/python
import re
import nltk
import sys
import getopt
import os
import numpy as np
from queryReader import Query



# Parameters 
BYTE_WIDTH = 4 # Used in the unpacking of the postings list
IGNORE_PUNCTUATION = True # Adding an option to strip away all punctuation in a term
PERFORM_STEMMING = True # Adding an option to perform stemming on the current term

def usage():
    print ("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")

dictionary_file = postings_file = file_of_queries = output_file_of_results = None
	
try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
except (getopt.GetoptError):
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-d':
        dictionary_file  = a
    elif o == '-p':
        postings_file = a
    elif o == '-q':
        file_of_queries = a
    elif o == '-o':
        file_of_output = a
    else:
        assert False, "unhandled option"

if dictionary_file == None or postings_file == None or file_of_queries == None or file_of_output == None :
    usage()
    sys.exit(2)



if __name__ == "__main__":
    dictionary = open(dictionary_file, 'r')
    output = open(file_of_output, 'w')
    postings = os.open(postings_file, os.O_RDONLY)
    queries = open(file_of_queries, 'r')

    for line in queries.readlines():
        query = Query(line)
        print (query).is_boolean
        break



    queries.close()
    dictionary.close()
    output.close()
    os.close(postings)
