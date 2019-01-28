import sys
import nltk


f = file(sys.argv[1])

words = []

for line in f:
    split_words = nltk.word_tokenize(line)
    words.extend(split_words)

fdist1 = nltk.FreqDist(words)

filtered_word_freq = dict((word, freq) for word, freq in fdist1.items() if not word.isdigit()) # filter the words
filtered_word_freq = sorted(filtered_word_freq.items(), key=lambda item: item[1], reverse = True) # Sort the words

print(filtered_word_freq)

f.close()
