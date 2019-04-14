#!/usr/bin/python
import re
import nltk
import sys
import getopt
import os
import numpy as np
from queryReader import Query
from postingsReader import *
from math import log


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
    output = open(file_of_output, 'w')
    queries = open(file_of_queries, 'r')

    postings_file_ptr = read_dict(dictionary_file, postings_file)
    search = SearchBackend(postings_file_ptr)

    for line in queries.readlines():
        query = Query(line)
        print query.is_boolean
        if not query.is_boolean:
            print query.tf_q
            tf_idf = {}
            # Sample tf-idf of a query
            N = postings_file_ptr.get_number_of_docs() # Total number of docs
            for word in query.tf_q:
                try:
                    tf_idf[word] = (1+log(query.tf_q[word]))*search.get_idf(word)
                except KeyError:
                    tf_idf[word] = 0
            print tf_idf
        break

    queries.close()
    output.close()
