#!/usr/bin/python
import re
import nltk
import sys
import getopt
import math

N_GRAMS = 4

def filter (split_words):
    line = ""
    # Filter and append
    for word in split_words:
        if (word.isalpha()):
            line += word.lower()
            line += " "

    return line

def processProbability(pr, freq, total, ngram, vocab):
    if (ngram in freq):

        pr += math.log(freq[ngram]/float(total+len(vocab)))

    return pr


def addOne (freq, total, ngram, vocab):
    if (ngram not in freq):
        total += 1
        freq[ngram] = 1
        vocab[ngram] = 1

    return total, freq, vocab


def chomp(x):
    if x.endswith("\r\n"): return x[:-2]
    if x.endswith("\n") or x.endswith("\r"): return x[:-1]
    return x

# TODO: Refactor and create a giant vocab space


def build_LM(in_file):
    """
    build language models for each label
    each line in in_file contains a label and a string separated by a space
    """
    print 'building language models...'

    freq = {"malaysian": {}, "tamil": {}, "indonesian": {}}
    total = {"malaysian": 0, "tamil": 0, "indonesian": 0}
    vocab = {}

    text = file(in_file)

    for line in text:
        # print line
        split_words = nltk.word_tokenize(line)
        language = split_words[0]

        line = chomp(line)
        line = "___" + line
        line = line + "___"

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
    """
    test the language models on new strings
    each line of in_file contains a string
    you should print the most probable label for each string into out_file
    """
    print "testing language models..."

    freq = LM[0]
    total = LM[1]
    vocab = LM[2]

    text = file(in_file)

    count = 0
    actual = ["malaysian", "tamil", "indonesian", "other", "indonesian", "malaysian", "tamil", "tamil", 
        "tamil", "indonesian", "other", "indonesian", "malaysian", "malaysian", "indonesian", "malaysian", "indonesian", "tamil", "malaysian", "indonesian"]

    for line in text:
        line_original = line[:]

        line = chomp(line)
        line = "___" + line
        line += "___"

        pr_malaysian = 0
        pr_indonesian = 0
        pr_tamil = 0

        total_malaysian = total["malaysian"]
        total_indonesian = total["indonesian"]
        total_tamil = total["tamil"]
        freq_malaysian = freq["malaysian"]
        freq_indonesian = freq["indonesian"]
        freq_tamil = freq["tamil"]

        # TODO: Add a boolean to check if there is really no hit at all. If not hit at all then just call it as others

        # Add one smoothing first
        for gram in nltk.ngrams(line, N_GRAMS):
            ngram = ''.join(gram)

            total_malaysian, freq_malaysian, vocab = addOne(freq_malaysian, total_malaysian, ngram, vocab)
            total_indonesian, freq_indonesian, vocab = addOne(freq_indonesian, total_indonesian, ngram, vocab)
            total_tamil, freq_tamil, vocab = addOne(freq_tamil, total_tamil, ngram, vocab)

        # Calculate probability after normalising
        for gram in nltk.ngrams(line, N_GRAMS):
            ngram = ''.join(gram)

            pr_malaysian = processProbability(pr_malaysian, freq_malaysian, total_malaysian, ngram, vocab)

            pr_indonesian = processProbability(pr_indonesian, freq_indonesian, total_indonesian, ngram, vocab)

            pr_tamil = processProbability(pr_tamil, freq_tamil, total_tamil, ngram, vocab)

        prediction = [[pr_malaysian, "malaysian"], [pr_indonesian, "indonesian"], [pr_tamil, "tamil"]]
        prediction.sort(reverse = True)
        # print prediction[0][1], " ", actual[count], " ", prediction[0][0]
        # print prediction[0][1]
        count += 1
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


# Note:
# Use log addition for the probabilities instead, follow instructions here: 
# http://practicalcryptography.com/miscellaneous/machine-learning/tutorial-automatic-language-identification-ngram-b/

