import nltk
import csv
import sys
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import re
import math


args = sys.argv

stop_words = set(stopwords.words('english')) 


vocab_unigram = {}
vocab_bigram = {}
total_count_bigram = 0
total_count_unigram = 0
documents = {}
count = 0

with open(args[1]) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        statement = row[1]
        response = row[4]
        statement = re.sub(r"[^a-zA-Z0-9]+", ' ', statement)
        response = re.sub(r"[^a-zA-Z0-9]+", ' ', response)
        statement = statement.lower()
        response = response.lower()
        combined = statement + " " + response

        word_tokens = combined.split(" ")
        for term in word_tokens:
            total_count_unigram += 1
            if (term in vocab_unigram):
                vocab_unigram[term] += 1
            else:
                vocab_unigram[term] = 1


        bigrm = nltk.bigrams(word_tokens)
        for gram in bigrm:
            combined = gram[0] + " " + gram[1]
            total_count_bigram += 1
            if (combined in vocab_bigram):
                vocab_bigram[combined] += 1
            else:
                vocab_bigram[combined] = 1



# print total_count_bigram, total_count_unigram


# Assuming query is a term which exists in unigram
query = "tzeguang"
results = []

for bigram in vocab_bigram.keys():
    results.append((math.log10(vocab_bigram[bigram]) - math.log10(vocab_unigram[query]), bigram))

results.sort(reverse=True)

count = 0
for result in results:
    print result
    if (count >= 20):
        break
    count += 1

# TODO: Find out why the distribution is so skewed towards the first few words.