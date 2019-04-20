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
from struct import unpack
from nltk import word_tokenize
from math import log
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
from math import log
from numpy import linalg as LA
from nltk.corpus import wordnet as wn
from collections import Counter

# Parameters 
K_PSEUDO_RELEVANT = 5
ROCCHIO_SCORE_THRESH = 0.5
PSEUDO_RELEVANCE_FEEDBACK = True
ALPHA = 1
BETA = 0.75
STEMMER = PorterStemmer()

postings_file_ptr = None
search = None

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
        self.synonyms = {}
        self.is_boolean, self.processed_queries = self.__identify_query(chomp(query))

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
                self.synonyms[token] = set([])
                for syn in wordnet.synsets(token):
                    for l in syn.lemmas():
                        self.synonyms[token].add(l.name())
            else:
                self.tf_q[token] += 1

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
        #print list(postings)
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
        
        normalise_tf_idf_q = LA.norm(tf_idf_q)
        normalise_tf_idn_doc = search.get_document_length(doc)


        if normalise_tf_idf_q != 0:
            tf_idf_q /= normalise_tf_idf_q
        if normalise_tf_idn_doc != 0:
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
        query_result = [doc] + query_result
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