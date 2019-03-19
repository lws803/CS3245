#!/usr/bin/python
import re
import nltk
import sys
import getopt
import os
import operator
import struct
import math
import string
import numpy as np
from numpy import linalg as LA
from collections import Counter


from nltk.stem.porter import PorterStemmer

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

# Global variables that represent the input files, as well as global lists such as the universe list
dictionary = None
output = None
postings = None
queries = None
operators = {"AND": 4, "OR": 3}
stemmer = PorterStemmer()
terms = {} # Document frequencies for each term
documents = {}

# Class of logical functions processed in each query
class Logic:
    def cosineSimilarity (query):
        tf_q = Counter(nltk.word_tokenize(query))
        
        # For file among all files
        # Find the cosine similarity with the query term
        return []



def mainQuery (query_string):
    tf_doc = {}
    tf_q = Counter(nltk.word_tokenize(query))

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
                tf_idn_doc.append((1+math.log10(tf_doc[doc][token]))*(math.log10(len(documents)/float(terms[token][0]))))

            tf_idn_q.append((1+math.log10(tf_q[token]))*(math.log10(len(documents)/float(terms[token][0]))))

        # After we're done setting the matrix based on the vector space for query
        tf_idn_q = np.array(tf_idn_q)
        tf_idn_q = tf_idn_q/LA.norm(tf_idn_q)

        tf_idn_doc = np.array(tf_idn_doc)
        tf_idn_doc = tf_idn_doc/LA.norm(tf_idn_doc)
        tf_idn_doc = tf_idn_doc.reshape(len(tf_idn_doc), 1)

        score = np.dot(tf_idn_q, tf_idn_doc)[0]
        scores[doc] = score

    # Print the first 10
    count = 0
    for i in sorted(scores.items(), key=operator.itemgetter(1), reverse=True):
        print (i)
        count += 1
        if (count == 10):
            break
    # print(len(documents))
    # print (tf_doc)

# normaliseTerm user search query terms
def normaliseTerm (token):
    token = token.lower() # Perform case folding on the current term

   # Remove all instances of punctuation if the bool is set to true.
    if IGNORE_PUNCTUATION: # Remove all instances of punctuation
        token = token.translate(str.maketrans('','',string.punctuation))
            
    # Perform stemming on the term
    if PERFORM_STEMMING: token = stemmer.stem(token)
    
    return token


# Initialise the dictionary and universe set for all documents
def populateDictionaryAndUniverse (lines):
    firstLine = True
    for line in lines:
        if (firstLine):
            tokens = line.split(",")
            for i in range(0, len(tokens), 2):
                if (tokens[i] == ' \n'):
                    break
                documents[int(tokens[i])] = int(tokens[i+1])

            firstLine = False
        else:
            # Term -> term_frequency, postings list address
            terms[line.split()[0]] = (int(line.split()[1]), int(line.split()[2]))



if __name__ == "__main__":
    dictionary = open(dictionary_file, 'r')
    output = open(file_of_output, 'w')
    postings = os.open(postings_file, os.O_RDONLY)
    queries = open(file_of_queries, 'r')

    # Store terms in memory with their frequencies and starting byte offsets
    populateDictionaryAndUniverse(dictionary.readlines())
    query = "the man"
    mainQuery(query)

    # Process each query in the queries file
    # query(queries.readlines())

    queries.close()
    dictionary.close()
    output.close()
    os.close(postings)
