#!/usr/bin/python3
from struct import unpack


class PostingsList:
    """
    An iterable that iterates through the on-disk postings
    """
    def __init__(self, base_address, length, postings_fileptr):
        self.file = postings_fileptr
        self.base_address = base_address
        self.length = length
        self.offset = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.offset < self.length:
            raise StopIteration

        self.file.seek(self.base_address + self.offset*8)
        self.offset += 1
        data_fields = self.file.read(8)
        return unpack("II", data_fields)

    def __len__(self):
        return self.length


class PostingsFilePointers:
    def __init__(self, postings_file):
        self.pointers = {}
        self.doc_freqs = {}
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
        self.doc_freqs = int(freq)

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
        return PostingsList(self.pointers[word], self.doc_freqs[word], self.postings_file)

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
            if len(data) == 3:
                word, doc_freq, postings_location = data
                postings_file_ptrs.add_word(word, postings_location, doc_freq)
    return postings_file_ptrs


def two_way_merge(array1, array2, offset=0, use_offset=False):
    """
    Performs a conjunction of pairs of ordered indices with positional index.
    :param array1: A sorted list consisting of tuples with format (doc_id, pos_index)
    :param array2: Another sorted list with tuples in the same format as array 1
    :param offset: The difference in positional index between the item in array1 and array2 (array1 - array2)
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
        words = phrase.split()

        assert len(words) >= 2

        postings_lists = {}
        word_offset = {}
        offset = 0
        execution_plan = []
        for word in words:
            word_offset[word] = offset
            offset += 1
            postings_lists[word] = self.postings.get_postings_list(word)
            execution_plan.append((len(postings_lists), word))

        execution_plan = sorted(execution_plan)
        _, first = execution_plan.pop(0)
        res = postings_lists[first]
        while len(execution_plan) > 0:
            _, second = execution_plan.pop(0)
            res = two_way_merge(res, postings_lists[second], use_offset=True, offset=word_offset[first] - word_offset[second])

        return res
