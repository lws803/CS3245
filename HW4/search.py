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
from postingsReader import *
from math import log
from numpy import linalg as LA
from nltk.corpus import wordnet as wn
from collections import Counter
from rocchioExpansion import generate_table, get_centroid, get_rocchio_table

# Parameters 
K_PSEUDO_RELEVANT = 5
ROCCHIO_SCORE_THRESH = 0.5
PSEUDO_RELEVANCE_FEEDBACK = True

postings_file_ptr = None
search = None

def get_tf_idf_query(query):
    # LNC.LTC scheme
    tf_idf_query = {}

    # Sample tf-idf of a query
    for word in query.tf_q:
        try:
            tf_idf_query[word] = (1+log(query.tf_q[word]))*search.get_idf(word)
        except KeyError:
            tf_idf_query[word] = 0
    
    return tf_idf_query

def get_doc_tf(doc_list, query):
    doc_tf = {}
    for word in query.tf_q:
        tf_doc = search.get_tf(word, doc_list)
        for doc in doc_list:
            if doc not in doc_tf:
                doc_tf[doc] = {}
            if tf_doc[doc] != 0:
                doc_tf[doc][word] = 1+log(tf_doc[doc])
            else:
                doc_tf[doc][word] = 0

    return doc_tf

def calculate_scores(doc_tf, tf_idf_query):
    scores = {}

    for doc in doc_tf:
        tf_idn_doc = np.array(doc_tf[doc].values())
        tf_idf_q = np.array(tf_idf_query.values())
        
        normalise_tf_idf_q = LA.norm(tf_idf_q)
        normalise_tf_idn_doc = search.get_document_length(doc)


        if normalise_tf_idf_q != 0:
            tf_idf_q /= normalise_tf_idf_q
        # if normalise_tf_idn_doc != 0:
            tf_idn_doc /= normalise_tf_idn_doc
        tf_idn_doc *= len(search.get_words_in_doc(doc)) # TODO: Verify the correctness of this
        # TODO: Do we have to normalise this?
        
        tf_idn_doc = tf_idn_doc.reshape(len(tf_idn_doc), 1)
        score = np.dot(tf_idf_q, tf_idn_doc)[0]
        scores[doc] = score
    
    return scores

def calculate_ranked_scores(doc_list, tf_idf_query, query):
    doc_tf = get_doc_tf(doc_list, query)

    # Calculate tf_idn of docs
    scores = calculate_scores(doc_tf, tf_idf_query)

    sorted_list = []
    for key in scores:
        sorted_list.append((-1*scores[key], key)) 
        # So that the scores can be sorted in descending order and the docs sorted in ascending order
    sorted_list.sort()

    return sorted_list

def ranked_retrieval(query, query_line):
    tf_idf_query = get_tf_idf_query(query)
    
    doc_list = []
    for doc in deduplicate_results(search.free_text_query(query_line)):
        doc_list.append(doc[0])

    sorted_list = calculate_ranked_scores(doc_list, tf_idf_query, query)
    return sorted_list

# TODO: Fix issue with retrieving tf_idf_query values
def ranked_retrieval_boolean(doc_list, query):
    sorted_list = calculate_ranked_scores(doc_list, get_tf_idf_query(query), query)
    return sorted_list

def handle_query(query_line):
    """
    Returns an unranked query after passing through boolean retrieval
    :param query:
    :return:
    """
    print query_line
    query = Query(query_line)

    # For testing purposes: write all queries to the output file
    output.write(str(query_line))

    relevant_docs = []

    if query.is_boolean:
        for subquery in query.processed_queries:
            if len(subquery) > 1:
                if len(relevant_docs) == 0:
                    relevant_docs = search.phrase_query(subquery)
                relevant_docs = two_way_merge(relevant_docs, search.phrase_query(subquery))
            else:
                if len(relevant_docs) == 0:
                    relevant_docs = deduplicate_results(search.free_text_query(subquery[0]))
                relevant_docs = two_way_merge(relevant_docs, deduplicate_results(search.free_text_query(subquery[0])))
        flattened_docs = set([])
        for doc in relevant_docs:
            flattened_docs.add(doc[0])
        
        relevant_docs = []
        for doc in ranked_retrieval_boolean(list(flattened_docs), query):
            print doc[1], -doc[0]
            relevant_docs.append(doc[1])
    else:
        ranked_list = ranked_retrieval(query, query_line)
        for doc in ranked_list:
            print doc[1], -doc[0]
            relevant_docs.append(doc[1])

        """
        # Beginning of rocchio expansion
        index = 0
        universal_vocab = set(query.tf_q.keys())
        score_table_docs = {}

        if PSEUDO_RELEVANCE_FEEDBACK:
            while index < len(ranked_list) and index < K_PSEUDO_RELEVANT:
                curr_doc = ranked_list[index][1]
                relevant_docs.append(curr_doc)
                doc_words = set(Counter(search.get_words_in_doc(curr_doc)).keys())
                universal_vocab |= doc_words
                index += 1

        # Second round to get the score table
        for curr_doc in relevant_docs:
            doc_words = Counter(search.get_words_in_doc(curr_doc))
            score_table_docs[curr_doc] = generate_table(doc_words, universal_vocab)


        # Obtain table and calculate the scores
        rocchio_table = get_rocchio_table(get_tf_idf_query(),
            get_centroid(relevant_docs, score_table_docs),
            universal_vocab)

        for term in sorted(rocchio_table.items(), key = lambda kv:(kv[1], kv[0]), reverse=True):
            print (term)
            if (term[1] < ROCCHIO_SCORE_THRESH): break
        """

    return relevant_docs

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

    postings_file_ptr = read_dict(dictionary_file, postings_file)
    search = SearchBackend(postings_file_ptr)

    query_string = None
    relevant_docs = []
    query_result = []

    with open(file_of_queries, 'r') as queries:
        line_count = 0
        for line in queries:
            if line_count == 0:
                query_result = handle_query(line)
                line_count = 1
            else:
                relevant_docs.append(int(line))

    # Reorder and add the relevant docs at the top
    for doc in relevant_docs:
        if doc in query_result:
            query_result.remove(doc)
        query_result.append(doc)
    print len(query_result)
    for doc in query_result:
        output.write(str(doc) + " ")
    """
    for line in queries:
        if query is not None:
            # relevant_docs.append(int(line)) # TODO: Uncomment this only when you have the full postings list
            pass
        else:
            query = Query(line)
            query_string = line

    if query.is_boolean: # Boolean retrieval
        # TODO: For boolean retrieval, decide if we need to consider the initial given docs as well,
        # but if so, they do not contain any offset value (not a tuple)
        for subquery in query.processed_queries:
            if len(subquery) > 1:
                if len(relevant_docs) == 0:
                    relevant_docs = search.phrase_query(subquery)

                relevant_docs = two_way_merge(relevant_docs, search.phrase_query(subquery))
            else:
                if len(relevant_docs) == 0:
                    relevant_docs = deduplicate_results(search.free_text_query(subquery[0]))
                relevant_docs = two_way_merge(relevant_docs, deduplicate_results(search.free_text_query(subquery[0])))

        print relevant_docs
        print query.tf_q
        print ranked_retrieval_boolean(relevant_docs) 
        # TODO: Need to verift this, it outputs a score of zero for a case where it shouldnt be

        # After performing AND operations on the query, rank the relevant_docs list generated

    #(Arjo's comment) There should be no diff between boolean and
    else: # Free text retrieval
        ranked_list = ranked_retrieval(query, query_string)
        for doc in ranked_list:
            print doc[1]

        # Beginning of rocchio expansion
        index = 0
        universal_vocab = set(query.tf_q.keys())
        score_table_docs = {}

        if PSEUDO_RELEVANCE_FEEDBACK:
            while index < len(ranked_list) and index < K_PSEUDO_RELEVANT:
                curr_doc = ranked_list[index][1]
                relevant_docs.append(curr_doc)
                doc_words = set(Counter(search.get_words_in_doc(curr_doc)).keys())
                universal_vocab |= doc_words
                index += 1

        # Second round to get the score table
        for curr_doc in relevant_docs:
            doc_words = Counter(search.get_words_in_doc(curr_doc))
            score_table_docs[curr_doc] = generate_table(doc_words, universal_vocab)


        # Obtain table and calculate the scores
        rocchio_table = get_rocchio_table(get_tf_idf_query(),
            get_centroid(relevant_docs, score_table_docs),
            universal_vocab)

        for term in sorted(rocchio_table.items(), key = lambda kv:(kv[1], kv[0]), reverse=True):
            print (term)
            if (term[1] < ROCCHIO_SCORE_THRESH): break

        # TODO: Reindex and lift the limit
        # TODO: Output baseline ranked retrieval results without the rocchio expansion to file and upload to CS3245 site

    
    """
    output.close()