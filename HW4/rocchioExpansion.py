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

class RocchioExpansion:
    """
    Holds the interface for methods to obtain the rocchio table and suggested query.
    """
    def __init__ (self):
        pass

    def get_centroid(self, tf_doc, score_table_query):
        """
        Get the centroid of a document list. To be used for top_k relevant documents
        tf_doc is the term frequency for a doc for a term
        Need to obtain by calling get_tf(word, docs) for words in the query and docs in the doc_list returned by top_k
        """
        total_sum = {}
        for term in score_table_query:
            for doc in tf_doc:
                score = 0
                if term in tf_doc[doc]:
                    score = 1 + math.log(tf_doc[doc][term])

                if term in total_sum:
                    total_sum[term] += score
                else:
                    total_sum[term] = score

        centroid = {}
        for term in total_sum:
            centroid[term] = total_sum[term]/len(tf_doc)
    
        return centroid


    def get_rocchio_table(self, tf_doc, score_table_query):
        original_query_space =  score_table_query # Obtain tf-idf of the query generated from queryReader
        centroid_relevant = self.get_centroid (tf_doc, score_table_query)
        query_modified = {}
        for term in score_table_query:
            query_modified[term] = ALPHA*original_query_space[term] + \
            BETA*centroid_relevant[term]
            
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


    

