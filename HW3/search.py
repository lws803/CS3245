#!/usr/bin/python
import re
import nltk
import sys
import getopt
import os
import struct
import math
import string

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
terms = {}
universe = {}

"""
Node class for skip list, holds a value that represents the docID as well as a skip index 
(i.e. which position to skip to)
"""
class Node:
    def __init__ (self, value, skip_index=None):
        self.value = value
        self.skip_index = skip_index

    def getValue (self):
        return self.value

    def getSkip(self):
        return self.skip_index

# Class of logical functions processed in each query
class Logic:
    # Performs merge AND operation between right and left list
    def and_operation(right_list, left_list):
        common_docs = []

        right_pos = 0
        left_pos = 0

        while (right_pos < len(right_list) and left_pos < len(left_list)):
            doc_right = right_list[right_pos]
            doc_left = left_list[left_pos]

            if (doc_right.getValue() < doc_left.getValue()):
                if (doc_right.getSkip() is not None and right_list[doc_right.getSkip()].getValue() <= doc_left.getValue()):
                    right_pos = doc_right.getSkip()
                else:
                    right_pos += 1
            elif (doc_right.getValue() > doc_left.getValue()):
                if (doc_left.getSkip() is not None and left_list[doc_left.getSkip()].getValue() <= doc_right.getValue()):
                    left_pos = doc_left.getSkip()
                else:
                    left_pos += 1
            elif (doc_right.getValue() == doc_left.getValue()):
                common_docs.append(doc_right.getValue())
                right_pos += 1
                left_pos += 1
        
        return generateSkipList(common_docs)

    # Performs merge AND NOT operation between right and left list
    def and_not_operation(right_list, left_list):
        resulting_docs = []

        right_pos = 0
        left_pos = 0
        
        while (right_pos < len(right_list) and left_pos < len(left_list)):
            doc_right = right_list[right_pos]
            doc_left = left_list[left_pos]

            if (doc_right.getValue() == doc_left.getValue()):
                right_pos += 1
                left_pos += 1
            elif (doc_right.getValue() < doc_left.getValue()):
                resulting_docs.append(doc_right.getValue())
                right_pos += 1
            else:
                if (doc_left.getSkip() is not None and left_list[doc_left.getSkip()].getValue() <= doc_right.getValue()):
                    left_pos = doc_left.getSkip()
                else:
                    left_pos += 1
        
        while (right_pos < len(right_list)):
            doc_right = right_list[right_pos]
            resulting_docs.append(doc_right.getValue())
            right_pos += 1
        
        return generateSkipList(resulting_docs)

    # Performs merge OR operation between right and left list
    def or_operation(right_list, left_list):
        union_set = {}

        right_pos = 0
        left_pos = 0

        while (right_pos < len(right_list) and left_pos < len(left_list)):
            doc_right = right_list[right_pos]
            doc_left = left_list[left_pos]

            union_set[doc_right.getValue()] = True
            right_pos += 1
            union_set[doc_left.getValue()] = True
            left_pos += 1

        while (right_pos < len(right_list)):
            doc_right = right_list[right_pos]

            union_set[doc_right.getValue()] = True 
            right_pos += 1
        
        while (left_pos < len(left_list)):
            doc_left = left_list[left_pos]

            union_set[doc_left.getValue()] = True
            left_pos += 1
        
        return generateSkipList(sorted(union_set))

    """
    Performs merge NOT operation between right and left list, where the right list is a 
    universal list consisting of all documents in the dataset
    """
    def not_operation(list, universe):
        inverse = universe.copy()
        for items in list:
            del inverse[items.getValue()]

        return generateSkipList(sorted(inverse))

# Shunting yard algorithm to process infix to postfix expression
def shunting_yard(query):
    stack = []
    queue = []

    # Shunting yard algorithm to generate a postfix expression to execute
    for token in nltk.word_tokenize(query):

        if (token == "("):
            stack.append(token)
        elif (token == ")"):
            while (stack[-1] != "("):
                queue.append(stack.pop())
            stack.pop()

        # If an operator is found
        elif (token in operators):
            if (len(stack) != 0 and stack[-1] != "(" and operators[stack[-1]] >= operators[token]):
                queue.append(stack.pop())
                stack.append(token)
            else:
                stack.append(token)
        
        # If a term is found
        else:
            if (len(queue) > 0 and queue[-1] == "NOT" and token == "NOT"):
                queue.pop()
            else:
                queue.append(token)

    while (len(stack) > 0):
        queue.append(stack.pop())

    return queue

# Generates skip list in memory from a given array before performing an operation
def generateSkipList (data=[]):
    length = len(data)
    skips = math.sqrt(length)
    skip_spaces = int(skips)
    skip_list = []

    position = -1
    for index in data:
        position += 1
        if (skip_spaces >= 2 and not(position%skip_spaces) and position+skip_spaces < length):
            skip_list.append(Node(index, position+skip_spaces))
        else:
            skip_list.append(Node(index, None))
    return skip_list

# Generates skip list by reading the postings list file, DIFFERENT than generateSkipList
def retriever (token):
    if (token not in terms): return []
    offset = terms[token][1]
    size = terms[token][0]
    skips = math.sqrt(size)
    skip_spaces = int(skips)

    os.lseek(postings, offset, 0)
    skip_list = []
    for i in range(0, size):
        skip = None
        unpacked_value = struct.unpack('I', os.read(postings, BYTE_WIDTH))[0]
        if (skip_spaces >= 2 and not(i%skip_spaces) and i+skip_spaces < size):
            skip = i+skip_spaces
        skip_list.append(Node(unpacked_value, skip))

    return skip_list

# Normalize user search query terms
def normalize (token):
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
            for ids in line.split(','):
                if (ids == ' \n'):
                    break
                universe[int(ids)] = True
            firstLine = False
        else:
            terms[line.split()[0]] = (int(line.split()[1]), int(line.split()[2]))

# Process each query using the stack expression generated by the shunting yard algorithm
def query (lines):
    for line in lines:
        postfix_expression = shunting_yard(line)
        processing_stack = [] # stack to process the postfixes, this stack holds postings list
        
        for token in postfix_expression:
            if token not in operators and token != "NOT":
                token = normalize(token) # normalize token as per normalization techniques used in index.py
                processing_stack.append(retriever(token)) # retrieve skip list of token's skip list
            elif token == "AND": # perform AND operation
                left_list = processing_stack.pop()
                right_list = processing_stack.pop()

                if (right_list == "NOT"): 
                    third_list = processing_stack.pop()
                    processing_stack.append(Logic.and_not_operation(third_list, left_list)) #optimisation for AND NOT operation
                else:
                    processing_stack.append(Logic.and_operation(right_list, left_list))
            elif token == "OR": # perform OR operation
                left_list = processing_stack.pop()
                right_list = processing_stack.pop()

                if (right_list == "NOT"):
                    third_list = processing_stack.pop()

                    left_list = Logic.not_operation(left_list, universe)
                    processing_stack.append(Logic.or_operation(third_list, left_list))
                else:
                    processing_stack.append(Logic.or_operation(right_list, left_list))
            elif token == "NOT": # treat NOT as a normal term first, process with either "AND" or "OR" later
                processing_stack.append("NOT")
        
        for node in processing_stack[-1]:
            output.write(str(node.getValue()) + ' ') # output resulting postings list to output file
        output.write('\n')

if __name__ == "__main__":
    dictionary = open(dictionary_file, 'r')
    output = open(file_of_output, 'w')
    postings = os.open(postings_file, os.O_RDONLY)
    queries = open(file_of_queries, 'r')

    # Store terms in memory with their frequencies and starting byte offsets
    populateDictionaryAndUniverse(dictionary.readlines())

    # Process each query in the queries file
    query(queries.readlines())

    queries.close()
    dictionary.close()
    output.close()
    os.close(postings)
