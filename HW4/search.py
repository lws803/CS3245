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
THESAURUS_ENABLED = True
K_PSEUDO_RELEVANT = 10
K_PROMINENT_WORDS = 10
ROCCHIO_SCORE_THRESH = 0.5
PSEUDO_RELEVANCE_FEEDBACK = False
ROCCHIO_EXPANSION = False
ALPHA = 1
BETA = 0.75
STEMMER = PorterStemmer()

postings_file_ptr = None
search = None
_stopwords = stopwords.words('english') + [
            "court",
            "case",
            "would",
            "also", 
            "one", 
            "two", 
            "three", 
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten"]

_spaces = ["\n", "", " "]

def parse_query(query_string):
    parsed_query = []
    STATE = "EXPECT-QUOTE"
    buff = ""
    tmp = ""
    for c in query_string:
        if STATE == "EXPECT-QUOTE" and c == "\"":
            STATE = "PHRASE-QUERY"
        elif STATE == "EXPECT-QUOTE":
            buff += c
            STATE = "FREE-TEXT"
        elif STATE == "PHRASE-QUERY" and c == "\"":
            parsed_query.append({"text":str(buff), "type":"phrase_query"})
            buff = ""
            STATE = "EXPECT_AND"
        elif STATE == "PHRASE-QUERY":
            buff +=c
        elif STATE == "FREE-TEXT":
            buff += c
            if buff[-5:] == " AND ":
                parsed_query.append({"text": str(buff[:-5]), "type": "free-text"})
                buff = ""
                STATE = "EXPECT-QUOTE"
        elif STATE == "EXPECT_AND":
            tmp += c
            if tmp == " AND ":
                STATE = "EXPECT-QUOTE"
            if len(tmp) > 5:
                raise Exception
        else:
            raise Exception
    if STATE == "FREE-TEXT":
        parsed_query.append({"text":buff, "type": "free-text"})
    elif STATE != "EXPECT_AND":
        raise Exception
    else:
        parsed_query.append({"text": buff, "type": "phrase_query"})
    return parsed_query

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
            print preprocess([d]), d
            for w in preprocess([d]):
                try:
                    inv_synset[w] = preprocess([word])[0]
                except IndexError:
                    print "Error adding ",w,preprocess([word])
                    pass
    terms = list(set(preprocess(expanded_query.split())))
    # print terms
    doc_tfs = {}
    for term in terms:
        for doc, tf in backend.get_tf(term, sorted(docs)).items():
            if doc not in doc_tfs:
                doc_tfs[doc] = {}
            if tf != 0:
                try:
                    if inv_synset[term] not in doc_tfs[doc]:
                        doc_tfs[doc][inv_synset[term]] = tf
                    else:
                        doc_tfs[doc][inv_synset[term]] += tf
                except KeyError:
                    pass
                    #print "Warning can't find term in invsynset "+term
                    #print inv_synset

    for doc in doc_tfs:
        doc_tfs[doc] = log_vector(doc_tfs[doc])

    return doc_tfs

def log_vector(vec):
    """
    Gets log value of a Vector
    :param vec:
    :return:
    """
    res = {}
    for word in vec:
        res[word] = 1 + log(vec[word])
    return res

def query_norm(query):
    """
    Normalizes a query vector
    :param query:
    :return:
    """
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
        # print doc_vectors[doc], q_vec
        for word in doc_vectors[doc]:
            if word in q_vec:
                dot_product += doc_vectors[doc][word]*q_vec[word]*backend.get_idf(word)
        dot_product /= backend.get_document_length(doc)
        dot_product /= query_norm(q_vec)+1e-9
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
    data = [w for w in data if w not in string.punctuation]
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
    # word = re.sub("\\d", "", original_word)
    word = original_word
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
        query = query.decode('utf-8')
        self.query_line = chomp(query)
        self.tf_q = {}
        self.processed_queries = None
        self.synonyms = {}
        self.__identify_query(chomp(query))

    def __identify_query(self, query_string):
        """
        identify the query type and process the queries
        """
        try:
            self.processed_queries = parse_query(query_string)
        except:
            self.processed_queries = [{"text": query_string, "type":"free-text"}]

        self.__get_tf(self.__get_term_list(query_string))


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

        try:
            if THESAURUS_ENABLED:
                for word in self.query_line.split():
                    for syn in wordnet.synsets(word):
                        for l in syn.lemmas():
                            if word not in self.synonyms:
                                self.synonyms[word] = set()
                            self.synonyms[word].add(str(l.name()).replace("_", " "))
        except UnicodeDecodeError:
            print "no syn found for temr"


    def add_suggestions (self, new_terms):
        """
        Allows addition of new terms from relevance feedback
        """
        for term in new_terms:                
            self.tf_q[term] = 1 # Add them as one
            print "accepted"
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
        :param term: term after pre-processing to be looked up
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
_stopwords = preprocess(_stopwords)


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
    universal_vocab = []
    score_table_docs = {}

    print "Performing ROCCHIO expansion..."
    pseudo_relevant_docs = []

    if legit_relevant_docs is not None:
        print "Performing relevance feedback..."
        for doc in legit_relevant_docs:
            pseudo_relevant_docs.append(doc)
            doc_words = search.get_words_in_doc(doc)            
            universal_vocab += doc_words

    index = 0
    if relevant_docs is not None:
        print "Performing pseudo relevance feedback..."
        for doc in relevant_docs:
            if (index >= K_PSEUDO_RELEVANT):
                break
            pseudo_relevant_docs.append(doc)
            doc_words = search.get_words_in_doc(doc)            
            universal_vocab += doc_words
            index += 1

    universal_vocab += query.tf_q.keys()
    counted_terms = Counter(universal_vocab)
    prominent_list = []
    universal_vocab = set([])
    for term in counted_terms:
        if term not in _stopwords and \
            term not in _spaces and \
            term not in query.tf_q:

            prominent_list.append((-1*counted_terms[term], term))
    
    prominent_list.sort()
    # Pick out K top most prominent terms
    index = 0
    for item in prominent_list:
        if index >= K_PROMINENT_WORDS:
            break
        universal_vocab.add(item[1])
        index += 1

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
    # print query.query_line

    relevant_docs = legit_relevant_docs
    ranked_list = ranked_retrieval(query, query.query_line)

    for doc in ranked_list:
        if doc[1] not in legit_relevant_docs:
            # print doc[1], -doc[0]
            relevant_docs.append(doc[1])

    return relevant_docs


def handle_query(query_line, legit_relevant_docs):
    """
    Returns an unranked query after passing through boolean retrieval
    :param query:
    :return:
    """
    # print query_line
    query = Query(query_line.encode("ascii", "ignore"))
    # print query.processed_queries
    relevant_docs = []
    is_boolean = len(query.processed_queries) > 1 or (len(query.processed_queries) > 0 and query.processed_queries[0]["type"] == "phrase_query")
    print query.processed_queries
    print is_boolean
    if is_boolean:
        for q in query.processed_queries:
            words = preprocess(word_tokenize(q["text"]))
            print words
            if len(words) > 0:
                docs = search.phrase_query(words)
                if len(relevant_docs) == 0:
                    relevant_docs = docs
                else:
                    relevant_docs = two_way_merge(docs, relevant_docs)
        relevant_docs = list(map(lambda x: x[0], relevant_docs))
        relevant_docs = list(map(lambda x: x[1], ranked_retrieval_boolean(relevant_docs, query)))
    else:
        relevant_docs = list(map(lambda x: x[1], synset_expansion(query, search)))
        if ROCCHIO_EXPANSION:
            if PSEUDO_RELEVANCE_FEEDBACK:
                relevant_docs = rocchio_expansion(query, relevant_docs, None)
            else:
                relevant_docs = rocchio_expansion(query, None, legit_relevant_docs)

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
        output.write(str(doc) + " ")
    output.close()