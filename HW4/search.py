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

# Parameters 
BYTE_WIDTH = 4 # Used in the unpacking of the postings list
IGNORE_PUNCTUATION = True # Adding an option to strip away all punctuation in a term
PERFORM_STEMMING = True # Adding an option to perform stemming on the current term

stemmer = PorterStemmer()

def usage():
    print ("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")

# Find cosine similarity between query string and document terms
def findCosineSimilarity (query_string):
    tf_doc = {}
    tf_q = {}

    for token in nltk.word_tokenize(query_string):
        token = normaliseTerm(token)
        if token not in tf_q:
            tf_q[token] = 1
        else:
            tf_q[token] += 1

    # We only process terms that are within the dictionary and document
    # If its not in either one of them we just ignore
    for token in tf_q.keys():
        if (token not in terms): continue
        offset = terms[token][1]
        size = terms[token][0] # Also the document frequency per term

        os.lseek(postings, offset, 0)
        for i in range(0, size):
            unpacked_value = struct.unpack('I', os.read(postings, BYTE_WIDTH))[0]
            term_freq = struct.unpack('I', os.read(postings, BYTE_WIDTH))[0]
            if (unpacked_value not in tf_doc):
                tf_doc[unpacked_value] = {token: term_freq}
            else:
                tf_doc[unpacked_value][token] = term_freq


    # Main score counting loop
    scores = {}
    # We will end up with some terms without similar documents, if that's the case just put 0
    for doc in tf_doc.keys():
        tf_idn_doc = []
        tf_idn_q = []
        for token in tf_q.keys():
            if (token not in terms): continue 

            # Iterate over all the documents and start scoring them
            if (token not in tf_doc[doc]):
                tf_idn_doc.append(0)
            else:
                tf_idn_doc.append((1+math.log10(tf_doc[doc][token]))*1.0)

            tf_idn_q.append((1+math.log10(tf_q[token]))*(math.log10(len(documents)/float(terms[token][0]))))

        # After setting the matrix based on the vector space for query
        tf_idn_q = np.array(tf_idn_q)
        tf_idn_q = tf_idn_q/LA.norm(tf_idn_q)

        tf_idn_doc = np.array(tf_idn_doc)
        tf_idn_doc = tf_idn_doc*documents[doc]
        tf_idn_doc = tf_idn_doc.reshape(len(tf_idn_doc), 1)

        score = np.dot(tf_idn_q, tf_idn_doc)[0]
        scores[doc] = score

    sorted_list = []
    for key in scores:
        sorted_list.append((-1*scores[key], key)) 
        # So that the scores can be sorted in descending order and the docs sorted in ascending order
    sorted_list.sort()

    # Print the first 10
    count = 0
    for i in sorted_list:
        count += 1
        if (count == 11):
            break
        print(i)
        if (count == 10 or sorted_list[-1] == i):
            output.write(str(i[1]))
        else:
            output.write(str(i[1]) + " ")

    output.write("\n")
    print("=====================================")

# normaliseTerm user search query terms
def normaliseTerm (token):
    token = token.lower() # Perform case folding on the current term

   # Remove all instances of punctuation if the bool is set to true.
    if IGNORE_PUNCTUATION: # Remove all instances of punctuation
        token = token.translate(str.maketrans('','',string.punctuation))
            
    # Perform stemming on the term
    if PERFORM_STEMMING: token = stemmer.stem(token)
    
    return token

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
    for doc in postings_file_ptr.get_postings_list("good"):
        print doc # Format: DocID, position

    print search.get_tf("good", [246400])
    
    for line in queries.readlines():
        query = Query(line)
        print (query.is_boolean)
        if not query.is_boolean:
            print (query.tf_q)
            tf_idf = {}
            # Sample tf-idf of a query
            N = postings_file_ptr.get_number_of_docs() # Total number of docs
            for word in query.tf_q:
                try:
                    tf_idf[word] = (1+log(query.tf_q[word]))*search.get_idf(word)
                except KeyError:
                    tf_idf[word] = 0
            # print tf_idf
            for word in query.tf_q:
                for doc in postings_file_ptr.get_postings_list(word):
                    print word, search.get_tf(word, [doc])
        break


    queries.close()
    output.close()
