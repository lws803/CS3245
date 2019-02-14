import nltk

tokens = nltk.word_tokenize("how are you doing?")

print (tokens)

print (nltk.pos_tag(tokens))