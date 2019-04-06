#!/usr/bin/python
import getopt
import sys
from struct import unpack
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from index import preprocess
import math
import re

def chomp(x):
    if x.endswith("\r\n"): return x[:-2]
    if x.endswith("\n") or x.endswith("\r"): return x[:-1]
    return x

class Query:
    """
    Holds information for a query (a single line of query)
    """
    def __init__ (self, query):
        self.orig_query = chomp(query)
        self.tf_q = {}
        self.processed_queries = None

        self.is_boolean, out = self.__identify_query(chomp(query))
        if (self.is_boolean == True):
            self.processed_queries = out


    def __identify_query(self, query_string):
        """
        identify the query type and process the queries
        """        
        if ("AND" in query_string):
            # Consider as list of queries
            # Pre process if need be
            out = []
            split_word = query_string.split(' ')
            
            i = 0
            while (i < len(split_word)):
                if (split_word[i][0] == '"'):
                    combined = ""
                    d = i
                    while (d < len(split_word)):
                        combined += split_word[d].replace('"', '') + " "
                        d += 1
                        if split_word[d][-1] == '"':
                            i = d + 1
                            combined += split_word[d].replace('"', '')
                            break
                    out.append(combined)
                else:
                    if (split_word[i] != "AND"):
                        out.append(split_word[i])
                    i += 1
            return True, preprocess(out)
        else:
            # pre process as per normal
            self.__get_tf(query_string)
            return False, None



    def __get_tf (self, query_string):
        term_list = preprocess(word_tokenize(query_string))
        for i in range(0, len(term_list)):
            token = str(term_list[i])
            if token not in self.tf_q:
                self.tf_q[token] = 1
            else:
                self.tf_q[token] += 1


    def get_tf_idf(self, dictionary, total_docs):
        tf_idf = {}
        for token in self.tf_q.keys():
            if (token in dictionary):
                tf_idf[token] = ((1+math.log10(tf_idf[token]))* \
                    (math.log10(total_docs/float(dictionary[token][0]))))
        return tf_idf


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

    query_file = open(file_of_queries, 'r')
    for query in query_file.readlines():
        processed_query = Query(query)

        if (processed_query.is_boolean == True):
            print (processed_query.processed_queries)
        else:
            print (processed_query.tf_q)
            # TODO: Test out tf_idf as well
        break
