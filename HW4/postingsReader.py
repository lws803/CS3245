#!/usr/bin/python3
from struct import unpack


class PostingsList:
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
    :return:
    """
    postings_file_ptrs = PostingsFilePointers(postings_file)
    with open(dictionary) as f:
        for line in f:
            data = line.split()
            if len(data) == 3:
                word, doc_freq, postings_location = data
                postings_file_ptrs.add_word(word, postings_location, doc_freq)
    return postings_file_ptrs


