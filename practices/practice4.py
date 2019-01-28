from nltk.stem import *
from nltk.book import text1
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist

stemmer = PorterStemmer()

stem_list = []
word_list = []

for word in text1:
    stem_list.append(stemmer.stem(word))
    word_list.append(word)


fdist1 = FreqDist(stem_list)
fdist2 = FreqDist(word_list)

filtered_stem_list = dict((word, freq) for word, freq in fdist1.items() if word.isalpha()) # filter the words
filtered_word_list = dict((word, freq) for word, freq in fdist2.items() if word.isalpha()) # filter the words

filtered_stem_list = sorted(filtered_stem_list.items(), key=lambda item: item[1], reverse = True) # Sort the words
filtered_word_list = sorted(filtered_word_list.items(), key=lambda item: item[1], reverse = True) # Sort the words

for x in xrange(0,5):
    print "word: ", filtered_word_list[x]
    print "stem: ", filtered_stem_list[x]

