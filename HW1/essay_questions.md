### In the homework assignment, we are using character-based ngrams, i.e., the gram units are characters. Do you expect token-based ngram models to perform better?

No. The probability of finding an exact word match is much harder as compared to character-based grams. Therefore, there might be a higher chance of false predictions.


### What do you think will happen if we provided more data for each category for you to build the language models? What if we only provided more data for Indonesian?

Indonesian texts will have a higher prediction accuracy whereas the rest could potentially be mislabeled as indonesian as well given that the indonesian frequency table has a higher count.

### What do you think will happen if you strip out punctuations and/or numbers? What about converting upper case characters to lower case?

Might reduce accuracy as structure is reduced since punctuations and upper cases provide some sort of indication as to what is a start of sentence or end of sentence.

### We use 4-gram models in this homework assignment. What do you think will happen if we varied the ngram size, such as using unigrams, bigrams and trigrams?

With a lower-gram, we will get a reduced accuracy as there is a higher probability of finding a sequence of characters in a that particular order thus resulting in a wrong langauge match.
