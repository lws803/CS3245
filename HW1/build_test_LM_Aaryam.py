#!/usr/bin/python
import re
from nltk import ngrams
import math
import sys
import getopt

"""
Cleans the string of any new line characters, and append underscores at the front and back of the string to 
act as padding (i.e. the start and end tokens)
"""
def clean_string(sentence):
    if sentence.endswith('\n') or sentence.endswith('\r'): sentence = sentence[:-1]
    if sentence.endswith('\r\n'): sentence = sentence[:-2]
    sentence = '___' + sentence + '___'
    return sentence

"""
Perform add-one smoothing to finish building the language model for each of the languages
"""
def add_one_smoothing(language_LM, language_count, overall_LM):
    for key in language_LM:
        language_LM[key] += 1
        language_count += 1

    for key in overall_LM:
        if key not in language_LM:
            language_LM[key] = 1
            language_count += 1

    return language_LM, language_count

def build_LM(in_file):
    """
    build language models for each label
    each line in in_file contains a label and a string separated by a space
    """
    print ('building language models...')

    # Dictionaries to store the frequencies of all terms found in each respective LM
    indonesian_LM = {}
    malaysian_LM = {}
    tamil_LM = {}
    overall_LM = {}

    # Variables to hold the total count of each language model
    indonesian_unigram_count = 0
    malaysian_unigram_count = 0
    tamil_unigram_count = 0

    training_data = open(in_file, 'r')

    # Extract each line in input.train.txt and build the respective language models
    for line in training_data:
        language, sentence = line.split(maxsplit=1)

        sentence = clean_string(sentence)
        
        # Split the sentence into four-grams using the nltk library
        four_grams = ngrams(sentence, 4)

        for four_gram in four_grams:

            # Add to the overall LM
            if four_gram in overall_LM:
                overall_LM[four_gram] += 1
            else:
                overall_LM[four_gram] = 1

            # Add to the respective language's LM
            if language.lower() == 'indonesian':
                if four_gram in indonesian_LM:
                    indonesian_LM[four_gram] += 1
                    indonesian_unigram_count += 1
                else:
                    indonesian_LM[four_gram] = 1
                    indonesian_unigram_count += 1
            elif language.lower() == 'malaysian':
                if four_gram in malaysian_LM:
                    malaysian_LM[four_gram] += 1
                    malaysian_unigram_count += 1
                else:
                    malaysian_LM[four_gram] = 1
                    malaysian_unigram_count += 1
            elif language.lower() == 'tamil':
                if four_gram in tamil_LM:
                    tamil_LM[four_gram] += 1
                    tamil_unigram_count += 1
                else:
                    tamil_LM[four_gram] = 1
                    tamil_unigram_count += 1

    # Perform add-one smoothing to finish building the language model

    # 1) Perform add-one smoothing to the indonesian LM
    indonesian_LM, indonesian_unigram_count = add_one_smoothing(indonesian_LM, indonesian_unigram_count, overall_LM)

    # 2) Perform add-one smoothing to the malaysian LM
    malaysian_LM, malaysian_unigram_count = add_one_smoothing(malaysian_LM, malaysian_unigram_count, overall_LM)

    # 3) Perform add-one smoothing to the tamil LM
    tamil_LM, tamil_unigram_count = add_one_smoothing(tamil_LM, tamil_unigram_count, overall_LM)
    
    training_data.close()

    # Create a list to hold all the LM's term frequencies and total counts
    LM = [indonesian_LM, indonesian_unigram_count, malaysian_LM, malaysian_unigram_count, tamil_LM, tamil_unigram_count, overall_LM]
    return LM
    
def test_LM(in_file, out_file, LM):
    """
    test the language models on new strings
    each line of in_file contains a string
    you should print the most probable label for each string into out_file
    """
    print ("testing language models...")

    testing_data = open(in_file, 'r')
    output_data = open(out_file, 'w')

    for line in testing_data:

        # These variables hold the probabiliteis of the sentence belonging to the respective language
        indonesian_prob = 0.0
        malaysian_prob = 0.0
        tamil_prob = 0.0

        # Counters to keep track of the 4-grams found in the language model, and those that are not
        hit = 0
        miss = 1

        clean_line = clean_string(line)

        # Split the sentence into four-grams using the nltk library
        four_grams = ngrams(clean_line, 4)

        for four_gram in four_grams:

            # If four-gram not in overall LM, skip
            if four_gram not in LM[6]:
                miss += 1
                continue

            # Otherwise calculate the respective probability of seeing this four-gram in each language
            else:
                hit += 1

                """
                 Instead of multipying the probabilities as fractions, the logarithmic representations are added instead
                 to prevent underflow of floating point values. I.e. a probability of three four-grams (a/total) * (b/total) * (c/total)
                 will instead of calculated as (log(a) - log(total)) + (log(b) - log(total)) + (log(c) - log(total)). The logarithmic values
                 (in base 10) will prevent any form of underflow caused by the small probablities involved in this homework.
                """
                indonesian_prob += (math.log10(LM[0][four_gram]) - math.log10(LM[1]))
                malaysian_prob += (math.log10(LM[2][four_gram]) - math.log10(LM[3]))
                tamil_prob += (math.log10(LM[4][four_gram]) - math.log10(LM[5]))

        """
         If the hit/miss ratio is less than 1, more than half of the four-grams were not found in the language model
         and hence it is more likely that this sentence belongs to another language
        """
        ratio = hit/miss

        if ratio < 1.0:
            output_data.write('other ' + line)
        else:
            maxprobability = max(indonesian_prob, malaysian_prob, tamil_prob)

            if maxprobability == indonesian_prob:
                output_data.write('indonesian ' + line)
            elif maxprobability == malaysian_prob:
                output_data.write('malaysian ' + line)
            elif maxprobability == tamil_prob:
                output_data.write('tamil ' + line)

    testing_data.close()
    output_data.close()

def usage():
    print ("usage: " + sys.argv[0] + " -b input-file-for-building-LM -t input-file-for-testing-LM -o output-file")

input_file_b = input_file_t = output_file = None
try:
    opts, args = getopt.getopt(sys.argv[1:], 'b:t:o:')
except getopt.GetoptError as err:
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
test_LM(input_file_t, output_file, LM)
