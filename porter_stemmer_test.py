from nltk.stem import *

from nltk.probability import FreqDist
import nltk

stemmer = PorterStemmer()

text = "penyetted penyet"

for word in nltk.word_tokenize(text):
    print stemmer.stem(word) # Output: penyet, penyet


# Porter stemmer can remove the -ed and -s etc etc
