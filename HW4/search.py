#!/usr/bin/python
import re
import nltk
import sys
import getopt
import os
import numpy as np
import math
import string
import struct
from queryReader import Query
from nltk.stem.porter import PorterStemmer
from postingsReader import *
from math import log
from numpy import linalg as LA

# Parameters 
BYTE_WIDTH = 4 # Used in the unpacking of the postings list
IGNORE_PUNCTUATION = True # Adding an option to strip away all punctuation in a term
PERFORM_STEMMING = True # Adding an option to perform stemming on the current term

stemmer = PorterStemmer()

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

    # Obtaining docs for word "good"
    # for doc in postings_file_ptr.get_postings_list("good"):
    #     print doc # Format: DocID, position

    # print search.get_tf("good", [246400])
    
    for line in queries.readlines():
        query = Query(line)
        
        if not query.is_boolean:
            # LNC.LTC scheme
            tf_idf_query = {}
            # Sample tf-idf of a query
            N = postings_file_ptr.get_number_of_docs() # Total number of docs
            for word in query.tf_q:
                try:
                    tf_idf_query[word] = (1+log(query.tf_q[word]))*search.get_idf(word)
                except KeyError:
                    tf_idf_query[word] = 0
            doc_list = []
            doc_tf = {}
            for doc in deduplicate_results(search.free_text_query(line)):
                doc_list.append(doc[0])
            for word in query.tf_q:
                tf_doc = search.get_tf(word, doc_list)
                for doc in doc_list:
                    if doc not in doc_tf:
                        doc_tf[doc] = {}
                    if tf_doc[doc] != 0:
                        doc_tf[doc][word] = 1+log(tf_doc[doc])
                    else:
                        doc_tf[doc][word] = 0

            # Calculate tf_idn of docs
            scores = {}
            for doc in doc_tf:
                tf_idn_doc = np.array(doc_tf[doc].values())
                tf_idf_q = np.array(tf_idf_query.values())
                tf_idf_q = tf_idf_q/LA.norm(tf_idf_q)
                tf_idn_doc = tf_idn_doc/LA.norm(tf_idn_doc) 
                # TODO: Need to verify this, do we need to multiply by total number of words in that doc?
                tf_idn_doc = tf_idn_doc.reshape(len(tf_idn_doc), 1)
                score = np.dot(tf_idf_q, tf_idn_doc)[0]
                scores[doc] = score
            sorted_list = []
            for key in scores:
                sorted_list.append((-1*scores[key], key)) 
                # So that the scores can be sorted in descending order and the docs sorted in ascending order
            sorted_list.sort()
            
            for document in sorted_list:
                print document[1]

        #     print (query.tf_q)
        break


    queries.close()
    output.close()
