#!/usr/bin/python
import re
import sys
import getopt
import os
import math
import string
from struct import unpack
from math import log

from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
from numpy import linalg as LA

from collections import Counter

# Parameters 
K_PSEUDO_RELEVANT = 10
ROCCHIO_SCORE_THRESH = 0.7
PSEUDO_RELEVANCE_FEEDBACK = False
ALPHA = 1
BETA = 0.75
STEMMER = PorterStemmer()

postings_file_ptr = None
search = None
# -- Synset  --

def get_adjusted_tf(docs, synset, expanded_query, backend):
    """
    Calculates TF vector based on synonyms.
    I.E collapses the doc vector dimensions
    :param docs:
    :param synset:
    :param expanded_query:
    :param backend:
    :return:
    """
    inv_synset = {}
    for word in synset:
        for d in synset[word]:
            for w in preprocess([d]):
                inv_synset[w] = preprocess([word])[0]
    terms = list(set(preprocess(expanded_query.split())))
    print terms
    doc_tfs = {}
    for term in terms:
        for doc, tf in backend.get_tf(term, sorted(docs)).items():
            if doc not in doc_tfs:
                doc_tfs[doc] = {}
            if tf != 0:
                if inv_synset[term] not in doc_tfs[doc]:
                    doc_tfs[doc][inv_synset[term]] = tf
                else:
                    doc_tfs[doc][inv_synset[term]] += tf

    for doc in doc_tfs:
        doc_tfs[doc] = log_vector(doc_tfs[doc])

    return doc_tfs

def log_vector(vec):
    res = {}
    for word in vec:
        res[word] = 1 + log(vec[word])
    return res

def query_norm(query):
    norm = 0
    for word in query:
        norm += query[word]**2
    return norm**0.5

def rank_documents(doc_vectors, query_vector, backend):
    """
    Rank documents and sort them by a query vector
    :param doc_vectors:
    :param query_vector:
    :param backend
    :return:
    """
    results = []
    q_vec = log_vector(query_vector.tf_q)
    for doc in doc_vectors:
        # get dot product
        dot_product = 0
        print doc_vectors[doc], q_vec
        for word in doc_vectors[doc]:
            if word in q_vec:
                dot_product += doc_vectors[doc][word]*q_vec[word]*backend.get_idf(word)
        dot_product /= backend.get_document_length(doc)
        dot_product /= query_norm(q_vec)
        results.append((-dot_product, doc))
    return sorted(results)

def synset_expansion(query, backend):
    """
    Performs a synset expansion based TF-IDF ranked retrival
    :param query: A Query object
    :param backend: A SearchBackend object
    :return:
    """
    new_query = query.query_line
    for synset in query.synonyms:
        for synonym in query.synonyms[synset]:
            new_query += " " + synonym

    res = backend.free_text_query(new_query)
    res = list(set([x[0] for x in res]))
    doc_vectors = get_adjusted_tf(res, query.synonyms, new_query, backend)
    return rank_documents(doc_vectors, query, backend)

# -- Rocchio --
def generate_table (table, universal_vocab):
    vector = {}
    for term in universal_vocab:
        if (term in table):
            # TF no IDF
            vector[term] = 1 + log(table[term])
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

# -- Index --
def preprocess(data):
    '''
    Clean data for content before tokenizing
    '''
    data = [w for w in data if not is_num(w) and w not in string.punctuation]
    data = split(data)
    data = [str(clean(w)) for w in data]
    data = [w for w in data if not has_weird_chars(w)]
    data = [w for w in data if len(w) > 1]
    result_data = [str(STEMMER.stem(w)) for w in data]
    return result_data

def split(data):
    '''
    Splits words that are conjoined by punctuation
    '''
    result = []
    for word in data:
        result.extend(re.split('\\W+', word))
    return result

def is_num(number):
    '''
    Remove any numbers
    '''
    number = number.replace(",\\.", "")
    return number.isdigit()

def clean(original_word):
    '''
    Remove digits, punctuation, unicode chars in words
    '''
    word = re.sub("\\d", "", original_word)
    word = word.encode('utf-8', 'ignore')
    return word.lower()

def has_weird_chars(word):
    '''
    Remove illegal characters
    '''
    for i in word:
        if i not in string.printable:
            return True

# -- Query --
def chomp(x):
    if x.endswith("\r\n"): return x[:-2]
    if x.endswith("\n") or x.endswith("\r"): return x[:-1]
    return x

class Query:
    """
    Holds information for a query (a single line of query)
    """
    def __init__ (self, query):
        self.query_line = chomp(query)
        self.tf_q = {}
        self.processed_queries = None
        self.synonyms = {}
        self.is_boolean, self.processed_queries = self.__identify_query(chomp(query))

    def __identify_query(self, query_string):
        """
        identify the query type and process the queries
        """        
        if ("AND" in query_string or (query_string[0] == '"' and query_string[-1] == '"')):
            # Consider as list of queries
            # Pre process if need be
            out = []
            split_word = query_string.split(' ')
            
            i = 0
            while (i < len(split_word)):
                if (split_word[i][0] == '"'):
                    combined = []
                    d = i
                    while (d < len(split_word)):
                        combined.append(split_word[d].replace('"', ''))
                        d += 1
                        if split_word[d][-1] == '"':
                            i = d + 1
                            combined.append(split_word[d].replace('"', ''))
                            break
                    out.append(preprocess(combined))
                else:
                    if (split_word[i] != "AND"):
                        out.append(preprocess([split_word[i]]))
                    i += 1

            # Flattens the out list so we can get a tf_q
            flat_term_list = [item for sublist in out for item in sublist]
            self.__get_tf(flat_term_list)
            return True, out
        else:
            # pre process as per normal
            self.__get_tf(self.__get_term_list(query_string))
            return False, query_string

    def __get_term_list (self, query_string):
        term_list = preprocess(word_tokenize(query_string))
        return term_list

    def __get_tf (self, term_list):
        """
        Obtains the term frequency for preprocessed query terms. Use this to find the idf as well.
        Since we can extract the different words of this query from that tf_q
        """
        for i in range(0, len(term_list)):
            token = str(term_list[i])
            if token not in self.tf_q:
                self.tf_q[token] = 1
                # Process synonyms for terms in the query words
            else:
                self.tf_q[token] += 1

        for word in self.query_line.split():
            for syn in wordnet.synsets(word):
                for l in syn.lemmas():
                    if word not in self.synonyms:
                        self.synonyms[word] = set()
                    self.synonyms[word].add(str(l.name()).replace("_", " "))

    def add_suggestions (self, new_terms):
        """
        Allows addition of new terms from relevance feedback
        """
        for term in new_terms:
            if term not in self.tf_q and term != "\n" \
                and term not in preprocess(stopwords.words('english')):
                
                self.tf_q[term] = 1 # Add them as one
                print "accepted:", term
                self.query_line += " " + term


# -- Postings --
class PostingsList:
    """
    An iterable that iterates through the on-disk postings list
    """
    def __init__(self, base_address, length, postings_fileptr):
        self.file = postings_fileptr
        self.base_address = base_address
        self.length = length
        self.offset = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.offset >= self.length:
            raise StopIteration

        self.file.seek(self.base_address + self.offset*12)
        self.offset += 1
        data_fields = self.file.read(12)
        return unpack("III", data_fields)

    def __len__(self):
        return self.length

    next = __next__


class PostingsFilePointers:
    """
    Holds information from the postings file. Do not use outside this module. 
    See SearchBackend for methods which you should use.
    """
    def __init__(self, postings_file):
        self.pointers = {}
        self.doc_freqs = {}
        self.postings_length = {}
        self.metadata = {}
        self.words_in_docs = {}
        self.postings_file = open(postings_file, "rb")

    def add_word(self, word, pointer, freq, postings_length):
        """
        Adds a word to the dictionary pointers. Used by read_dictionary method.
        Should not be called by anything else.
        :param word: Word to store
        :param pointer: pointer to the start address in the postings file
        :param freq: document frequency
        :param postings_length: The length of the postings for the word
        :return:
        """
        self.pointers[word] = int(pointer)
        self.doc_freqs[word] = int(freq)
        self.postings_length[word] = int(postings_length)

    def get_doc_freq(self, word):
        """
        Returns the document frequency of the word otherwise raises a KeyError
        :param word:
        :return: An integer containing the doc freq
        """
        return self.doc_freqs[word]

    def get_postings_list_base_address(self, word):
        """
        Returns the base address of the postings list.
        :param word:
        :return:
        """
        return self.pointers[word]

    def get_postings_list(self, word):
        """
        Retrieves a postings list for a given word.
        :param word:
        :return: An Iterable that contains pairs of (doc_id, position_index)
        """
        if word not in self.pointers:
            return []
        return PostingsList(self.pointers[word], 
                            self.postings_length[word], self.postings_file)

    def add_metadata(self, doc_id, length, court):
        self.metadata[int(doc_id)] = {"length": float(length), "court": court}

    def get_meta_data(self, doc_id):
        """
        Returns metadata
        :param word:
        :return:
        """
        return self.metadata[doc_id]

    def get_number_of_docs(self):
        """
        Returns number of docs
        """
        return len(self.metadata)

    def get_words_in_doc(self, doc):
        return self.words_in_docs[doc]

    def add_words_to_doc(self, doc, words):
        self.words_in_docs[doc] = words

    def __del__(self):
        self.postings_file.close()


def read_dict(dictionary, postings_file):
    """
    Reads a dictionary file and outputs the postings data.
    :param dictionary: A string containing the file path to the dictionary
    :return: A PostingsFilePointers object containing individual
    """
    postings_file_ptrs = PostingsFilePointers(postings_file)
    with open(dictionary) as f:
        mode = "POSTINGS"
        for line in f:
            data = line.split()
            try:
                if line == "# BEGIN VECTOR DATA #\n":
                    print "switching to vector data"
                    mode = "VSM"
                elif line == "# BEGIN ROCCHIO DATA #\n":
                    mode = "ROCHIO"
                elif mode == "POSTINGS":
                    word, doc_freq, postings_length, postings_location = data
                    postings_file_ptrs.add_word(word, postings_location,  doc_freq, postings_length)
                elif mode == "ROCHIO":
                    words_in_docs = line.split("^")
                    postings_file_ptrs.add_words_to_doc(int(words_in_docs[0]), words_in_docs[1:])
                elif mode == "VSM":
                    doc_id, length, court = line.split(":")
                    postings_file_ptrs.add_metadata(doc_id, length, court)

            except Exception as E:
                print "Warning: Incorrect parsing occured"
                print E
                print line
                print mode
    return postings_file_ptrs


def two_way_merge(array1, array2, offset=0, use_offset=False):
    """
    Performs a conjunction (and operation) of pairs of ordered indices with positional index.
    :param array1: A sorted list consisting of tuples with format (doc_id, pos_index, ...)
    :param array2: Another sorted list with tuples in the same format as array 1
    :param offset: The difference in positional index between the item in array1 
        and array2 (array1 - array2)
    :param use_offset: Tells the function whether to use the offset for merging or not. If set to false, then
        will simply ignore positional index
    :return: the merged list
    """
    result = []
    self_left, other_left = True, True
    array1 = iter(array1)
    array2 = iter(array2)
    try:
        self_val = next(array1)
    except StopIteration as s:
        self_left = False

    try:
        other_val = next(array2)
    except StopIteration as s:
        other_left = False

    while self_left and other_left:

        desired_val = (other_val[0], other_val[1]+offset)
        if (self_val < desired_val and use_offset) or (not use_offset and self_val[0] < other_val[0]):
            try:
                self_val = next(array1)
            except StopIteration as s:
                self_left = False

        elif (self_val[0] == other_val[0] and self_val[1] == other_val[1]+offset and use_offset) or (not use_offset and self_val[0] == other_val[0]):
            result.append(self_val)
            try:
                self_val = next(array1)
            except StopIteration as s:
                self_left = False
        else:
            try:
                other_val = next(array2)
            except StopIteration as s:
                other_left = False
    return result


def union(array1, array2):
    """
    Performs union of two arrays
    :param array1:
    :param array2:
    :return: the list union.
    """
    result = []
    self_left, other_left = True, True

    array1 = iter(array1)
    array2 = iter(array2)

    try:
        self_val = next(array1)
    except StopIteration as s:
        self_left = False

    try:
        other_val = next(array2)
    except StopIteration as s:
        other_left = False

    while self_left or other_left:
        if self_left and other_left:
            if self_val < other_val:
                try:
                    if (len(result) > 0 and result[-1][0] != self_val[0]) or len(result) == 0: result.append(self_val)
                    self_val = next(array1)
                except StopIteration as s:
                    self_left = False
            else:
                try:
                    if (len(result) > 0 and result[-1][0] != other_val[0]) or len(result) == 0: result.append(other_val)
                    other_val = next(array2)
                except StopIteration as s:
                    other_left = False

        elif self_left:
            try:
                if (len(result) > 0 and result[-1][0] != self_val[0]) or len(result) == 0: result.append(self_val)
                self_val = next(array1)
            except StopIteration as s:
                self_left = False

        elif other_left:
            try:
                if (len(result) > 0 and result[-1][0] != other_val[0]) or len(result) == 0: result.append(other_val)
                other_val = next(array2)
            except StopIteration as s:
                other_left = False

    return result


def deduplicate_results(results):
    prev_doc_id = -1
    out = []
    for res in results:
        if res[0] != prev_doc_id:
            out.append(res)
            prev_doc_id = res[0]
    return out


class SearchBackend:
    """
    This is the class of interest. It exposes the postings list to a set of high level commands.
    """
    def __init__(self, postings_file_pointer):
        """
        Constructor for SearchBackend
        :param postings_file_pointer: An object of type Posi
        """
        self.postings = postings_file_pointer

    def phrase_query(self, words):
        """
        Retrieve an *exact* phrase using the postings list
        :param phrase: A string with *at least* two words
        :return: a sorted list consisting of tuples of type (doc_id, pos_index) [pos index may be ignored after this operation]
        """
        # words = word_tokenize(phrase)
        # words = preprocess(words)
        postings_lists = {}
        word_offset = {}
        offset = 0
        execution_plan = []
        for word in words:
            word_offset[word] = offset
            offset += 1
            postings_lists[word] = self.postings.get_postings_list(word)
            execution_plan.append((len(postings_lists), word))

        # optimize and operation by picking the smallest set first
        execution_plan = sorted(execution_plan)
        _, first = execution_plan.pop(0)
        res = postings_lists[first]
        while len(execution_plan) > 0:
            _, second = execution_plan.pop(0)
            res = two_way_merge(res, postings_lists[second], use_offset=True, offset=word_offset[first] - word_offset[second])
        return deduplicate_results(res)

    def free_text_query(self, query):
        """
        Retrieve documents from query.
        :param query: A string with something. Please have something,
        :return:
        """
        words = word_tokenize(query)
        words = preprocess(words)
        if len(words) == 0:
            return []
        res = self.postings.get_postings_list(words.pop())
        while len(words) > 0:
            posting = self.postings.get_postings_list(words.pop())
            res = union(res, posting)
        return res

    def get_document_length(self, doc_id):
        return self.postings.get_meta_data(doc_id)["length"]

    def get_tf(self, term, documents):
        """
        Returns TF of term in document list
        :param term: term after preprocessing to be looked up
        :param documents: A sorted list of doc ids
        :return: dict containing the tf of document {<doc_id>: <tf>}
        """
        postings = iter(self.postings.get_postings_list(term))
        counts = {}
        for doc in documents:
            counts[doc] = 0
        
        end_postings = False
        doc_index = 0

        try:
            postings_item = next(postings)
        except StopIteration as s:
            end_postings = True
        
        try:
            while doc_index < len(documents) and not end_postings:
                if documents[doc_index] == postings_item[0]:
                    counts[documents[doc_index]] += 1
                    postings_item = next(postings) 
                elif documents[doc_index] > postings_item[0]:
                    postings_item = next(postings)
                elif documents[doc_index] < postings_item[0]:
                    doc_index += 1
        except StopIteration as s:
            pass
        return counts

    def get_idf(self, term):
        """
        Returns the idf of a term
        """
        return log(self.postings.get_number_of_docs()) - log(self.postings.get_doc_freq(term))

    def get_words_in_doc(self, doc_id):
        """
        Returns the words in a doc
        """
        return self.postings.get_words_in_doc(doc_id)

# -- Search --
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
        tf_idn_doc = tf_idn_doc.astype(float) # Turn the current 2D array to a float array 
        tf_idf_q = np.array(tf_idf_query.values())
        
        # normalise_tf_idf_q = LA.norm(tf_idf_q)
        # normalise_tf_idn_doc = search
        # Cosine normalization requires you to consider *EVERY TF* which is what document_length considers.
        # Read the notes properly
        normalise_tf_idn_doc = search.get_document_length(doc)

        # tf_idf_q /= normalise_tf_idf_q + 1e-9
        # tf_idn_doc /= normalise_tf_idn_doc +1e-9
        
        tf_idn_doc = tf_idn_doc.reshape(len(tf_idn_doc), 1)
        score = np.dot(tf_idf_q, tf_idn_doc)[0]
        score /= normalise_tf_idn_doc # Simplified step for obtain scoring

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
    doc_list.sort()
    sorted_list = calculate_ranked_scores(doc_list, tf_idf_query, query)
    return sorted_list

def ranked_retrieval_boolean(doc_list, query):
    doc_list.sort()
    sorted_list = calculate_ranked_scores(doc_list, get_tf_idf_query(query), query)
    return sorted_list

def rocchio_expansion (query, relevant_docs, legit_relevant_docs):
    # Beginning of rocchio expansion
    universal_vocab = set([])
    score_table_docs = {}

    print "Performing ROCCHIO expansion..."
    pseudo_relevant_docs = []
    index = 0
    for doc in relevant_docs:
        if (index > K_PSEUDO_RELEVANT):
            break
        pseudo_relevant_docs.append(doc)
        doc_words = set(Counter(search.get_words_in_doc(doc)).keys())
        
        # Take only recurring terms among a few documents
        if len(universal_vocab) == 0:
            universal_vocab = doc_words
        universal_vocab &= doc_words
        index += 1
    universal_vocab |= set(query.tf_q.keys())
    
    # Second round to get the score table
    for curr_doc in pseudo_relevant_docs:
        doc_words = Counter(search.get_words_in_doc(curr_doc))
        score_table_docs[curr_doc] = generate_table(doc_words, universal_vocab)

    # Obtain table and calculate the scores
    rocchio_table = get_rocchio_table(get_tf_idf_query(query),
        get_centroid(pseudo_relevant_docs, score_table_docs),
        universal_vocab)

    accepted_terms = []
    for term in rocchio_table:
        if rocchio_table[term] > ROCCHIO_SCORE_THRESH:
            accepted_terms.append(term)

    query.add_suggestions(accepted_terms)
    print query.tf_q
    print query.query_line

    # TODO: Fix extremely slow process, find out what is wrong, maybe only take recurrent terms among a few documents
    # TODO: Problem might be from a slow ranked retrieval format. Find out how to fix it
    relevant_docs = legit_relevant_docs
    ranked_list = ranked_retrieval(query, query.query_line)

    for doc in ranked_list:
        if doc[1] not in legit_relevant_docs:
            print doc[1], -doc[0]
            relevant_docs.append(doc[1])

    return relevant_docs


def handle_query(query_line, legit_relevant_docs):
    """
    Returns an unranked query after passing through boolean retrieval
    :param query:
    :return:
    """
    print query_line
    query = Query(query_line)
    print query.processed_queries
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

        relevant_docs = legit_relevant_docs
        for doc in ranked_retrieval_boolean(list(flattened_docs), query):
            # Handle duplicates
            if doc[1] not in legit_relevant_docs:
                # print doc[1], -doc[0]
                relevant_docs.append(doc[1])
    else:

        ranked_list = ranked_retrieval(query, query.query_line)
        relevant_docs = legit_relevant_docs
        
        for doc in ranked_list:
            if doc[1] not in legit_relevant_docs:
                # print doc[1], -doc[0]
                relevant_docs.append(doc[1])

        if PSEUDO_RELEVANCE_FEEDBACK:
            relevant_docs = rocchio_expansion(query, relevant_docs, legit_relevant_docs)

        docs = synset_expansion(query, search)
        for scor, doc in docs[0:10]:
            print doc, scor


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
                query_string = line
                line_count = 1
            else:
                relevant_docs.append(int(line))

    query_result = handle_query(query_string, relevant_docs)    
    for doc in query_result:
        print doc
        output.write(str(doc) + " ")
    """
    for line in queries:
        if query is not None:
            # relevant_docs.append(int(line))
            pass
        else:
            query = Query(line)
            query_string = line

    if query.is_boolean: # Boolean retrieval
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
    """
    output.close()