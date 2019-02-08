#!/usr/bin/python
import re
import nltk
import sys
import getopt
import math
import numpy as np

N_GRAMS = 4

# Processes probability if the ngram exists in the freq table
def processProbability(pr, freq, total, ngram, vocab):
    if (ngram in freq):

        pr += math.log(freq[ngram]/float(total+len(vocab)))

    return pr

# Performs add one smoothing to the freq table and vocab for the queries
def addOne (freq, total, ngram, vocab):
    vocab[ngram] = 1
    
    if (ngram not in freq):
        total += 1
        freq[ngram] = 1
    else:
        total += 1
        freq[ngram] += 1

    return total, freq, vocab

# Count the number of misses (ngrams not in vocab)
def countMisses (ngram, vocab):
    misses = 0
    hits = 0
    if (ngram not in vocab):
        misses += 1
    else:
        hits += 1
    return hits, misses

# Removes new line
def chomp(x):
    if x.endswith("\r\n"): return x[:-2]
    if x.endswith("\n") or x.endswith("\r"): return x[:-1]
    return x


def build_LM(in_file):
    print 'building language models...'

    freq = {"malaysian": {}, "tamil": {}, "indonesian": {}}
    total = {"malaysian": 0, "tamil": 0, "indonesian": 0}
    vocab = {}

    text = file(in_file)

    for line in text:

        split_words = nltk.word_tokenize(line)
        language = split_words[0]

        # Removal of newline and addition of padding
        line = chomp(line)
        line = "___" + line
        line = line + "___"

        # Generate four gram and storing in general vocab + frequency table of that language
        fourGram = nltk.ngrams(line, N_GRAMS)
        for gram in fourGram:
            ngram = ''.join(gram)
            vocab[ngram] = 1
            if (ngram not in freq[language]):
                freq[language][ngram] = 1 # add one smoothing
                total[language] += 1
            else:
                freq[language][ngram] += 1
                total[language] += 1
                
    return [freq, total, vocab]



def test_LM(in_file, out_file, LM):
    print "testing language models..."

    freq = LM[0]
    total = LM[1]

    text = file(in_file)

    for line in text:
        line_original = line[:]

        # Removal of newline and addition of padding
        line = chomp(line)
        line = "___" + line
        line += "___"

        pr_malaysian = 0
        pr_indonesian = 0
        pr_tamil = 0

        # For each query, we wish to reset the current frequency table and vocab (remove smoothing from prev queries)
        total_malaysian = total["malaysian"]
        total_indonesian = total["indonesian"]
        total_tamil = total["tamil"]
        freq_malaysian = freq["malaysian"]
        freq_indonesian = freq["indonesian"]
        freq_tamil = freq["tamil"]
        vocab = LM[2]

        hits = 0
        misses = 1 # Intialised to 1 to prevent divide by zero
        
        # Add one smoothing first for this particular query
        for gram in nltk.ngrams(line, N_GRAMS):
            ngram = ''.join(gram)

            hits_, misses_ = countMisses(ngram, vocab)
            hits += hits_
            misses += misses_

            total_malaysian, freq_malaysian, vocab = addOne(freq_malaysian, total_malaysian, ngram, vocab)
            total_indonesian, freq_indonesian, vocab = addOne(freq_indonesian, total_indonesian, ngram, vocab)
            total_tamil, freq_tamil, vocab = addOne(freq_tamil, total_tamil, ngram, vocab)

        output_line = ""

        if (hits/float(misses) < 1):
            output_line = "other" + " " + line_original
        else:
            # Calculate probability after smoothing
            for gram in nltk.ngrams(line, N_GRAMS):
                ngram = ''.join(gram)

                pr_malaysian = processProbability(pr_malaysian, freq_malaysian, total_malaysian, ngram, vocab)
                pr_indonesian = processProbability(pr_indonesian, freq_indonesian, total_indonesian, ngram, vocab)
                pr_tamil = processProbability(pr_tamil, freq_tamil, total_tamil, ngram, vocab)

            prediction = [[pr_malaysian, "malaysian"], [pr_indonesian, "indonesian"], [pr_tamil, "tamil"]]
            prediction.sort(reverse = True)

            output_line = prediction[0][1] + " " + line_original

        out_file.write(output_line)


def usage():
    print "usage: " + sys.argv[0] + " -b input-file-for-building-LM -t input-file-for-testing-LM -o output-file"

input_file_b = input_file_t = output_file = None
try:
    opts, args = getopt.getopt(sys.argv[1:], 'b:t:o:')
except getopt.GetoptError, err:
    usage()
    sys.exit(2)
for o, a in opts:
    if o == '-b':
        input_file_b = a
    elif o == '-t':
        input_file_t = a
    elif o == '-o':
        output_file = a
    else:
        assert False, "unhandled option"
if input_file_b == None or input_file_t == None or output_file == None:
    usage()
    sys.exit(2)

LM = build_LM(input_file_b)
with open(output_file, 'a') as f:
    test_LM(input_file_t, f, LM)
    f.close()
