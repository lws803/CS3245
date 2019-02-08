import nltk

sentence = "April is the cruelest month"
dictionary = {"":[]}

for word in nltk.word_tokenize(sentence):
    processed_word = "$" + word.lower() + "$"
    for gram in nltk.ngrams(processed_word, 2):
        print "".join(gram)
        kgram = "".join(gram)
        if (kgram in dictionary):
            dictionary[kgram].append(word.lower())
        else:
            dictionary[kgram] = [word.lower()]


print (dictionary["ap"])
print (dictionary["th"])