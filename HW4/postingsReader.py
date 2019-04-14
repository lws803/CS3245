#!/usr/bin/python
import getopt
import sys
from struct import unpack
from nltk import word_tokenize
from index import preprocess
from math import log

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
        self.metadata = {}
        self.postings_file = open(postings_file, "rb")

    def add_word(self, word, pointer, freq):
        """
        Adds a word to the dictionary pointers. Used by read_dictionary method.
        Should not be called by anything else.
        :param word: Word to store
        :param pointer: pointer to the start address in the postings file
        :param freq: document frequency
        :return:
        """
        self.pointers[word] = int(pointer)
        self.doc_freqs[word] = int(freq)

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
                            self.doc_freqs[word], self.postings_file)

    def add_metadata(self, doc_id, length, court):
        self.metadata[doc_id] = {"length": length, "cout": court}

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
        for line in f:
            data = line.split()
            if ":" not in line:
                word, doc_freq, postings_location = data
                postings_file_ptrs.add_word(word, postings_location, doc_freq)
            else: 
                doc_id, length, court = line.split(":")
                postings_file_ptrs.add_metadata(doc_id, length, court)
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

    def phrase_query(self, phrase):
        """
        Retrieve an *exact* phrase using the postings list
        :param phrase: A string with *at least* two words
        :return: a sorted list consisting of tuples of type (doc_id, pos_index) [pos index may be ignored after this operation]
        """
        words = word_tokenize(phrase)
        words = preprocess(words)
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
        res = self.postings.get_postings_list(words.pop())
        while len(words) > 0:
            res = union(res, self.postings.get_postings_list(words.pop()))
        return res

    def get_tf(self, term, documents):
        """
        Returns TF of term in document list
        :param term: term after preprocessing to be looked up
        :param documents: A sorted list of doc ids
        :return: dict containing the tf of document {<doc_id>: <tf>}
        """
        postings = self.postings.get_postings_list(term)
        counts = {}
        for doc in documents:
            counts[doc] = 0
        
        postings_index = 0
        doc_index = 0
        while doc_index < len(documents) and postings_index < len(postings):
            if documents[doc_index] == postings[postings_index][0]:
                counts[documents[doc_index]] += 1
                postings_index += 1
            elif documents[doc_index] > postings[postings_index][0]:
                postings_index += 1
            elif documents[doc_index] < postings[postings_index][0]:
                doc_index += 1
        return counts

    def get_idf(self, term):
        """
        Returns the idf of a term
        """
        return log(self.postings.get_number_of_docs()) - log(self.postings.get_doc_freq(term))

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

    postings_file_ptr = read_dict(dictionary_file, postings_file)
    search = SearchBackend(postings_file_ptr)
    print(list(search.phrase_query("Sim Cheng Ho")))
