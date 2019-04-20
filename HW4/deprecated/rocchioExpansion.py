#!/usr/bin/python
import getopt
import sys
from struct import unpack
from index import preprocess
import nltk
from collections import Counter
import math

ALPHA = 1
BETA = 0.75


def generate_table (table, universal_vocab):
    vector = {}
    for term in universal_vocab:
        if (term in table):
            # TF no IDF
            vector[term] = 1 + math.log(table[term])
        else:
            vector[term]= 0
        
    return vector

def get_centroid(docs_list, score_table_docs):
    total_sum = {}
    for doc in docs_list:
        score_table = score_table_docs[doc]
        for term in score_table:
            if term in total_sum:
                total_sum[term] += score_table[term]
            else:
                total_sum[term] = score_table[term]
    
    centroid = {}
    for term in total_sum:
        centroid[term] = total_sum[term]/len(docs_list)
    
    return centroid


def get_rocchio_table(score_table_query, centroid_relevant, universal_vocab):
    original_query_space =  score_table_query # Obtain tf-idf of the query generated from queryReader
    query_modified = {}
    for term in universal_vocab:
        if term in original_query_space:
            query_modified[term] = ALPHA*original_query_space[term] + \
            BETA*centroid_relevant[term]
        else:
            query_modified[term] = BETA*centroid_relevant[term]
        
        # Set it to zero if it is negative
        if (query_modified[term] < 0):
            query_modified[term] = 0
        
    return query_modified

if __name__ == "__main__":
    dictionary_file = postings_file = file_of_queries = file_of_output = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
    except getopt.GetoptError as err:
        usage()
        sys.exit(2)

    for o, a in opts:
        if o == '-d':
            dictionary_file = a
        elif o == '-p':
            postings_file = a
        elif o == '-q':
            file_of_queries = a
        elif o == '-o':
            file_of_output = a
        else:
            assert False, "unhandled option"
