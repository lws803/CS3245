import math
import nltk

NGRAM = 2
# True ngram method is better to be used when there's more data
# Working: P("hello world") = P("world"| "hello") * P("hello")
# Working: P("hello how are you") = P("you"| "hello how are") * P("hello how are")
# We use the probability of the previous result. Basically store a (n-1)-gram freq table as well


text_collection_PC = ["I Don't Want To Go", "A Groovy Kind Of Love", "You Can't Hurry Love", "This Must Be Love", "Take Me With You"]
text_collection_AS = ["All Out Of Love", "Here I Am", "I Remember Love", "Love Is All", "Don't Tell Me"]


vocab = {}
text_collection_AS_freq = {}
total_text_collection_AS_freq = 0


for titles in text_collection_AS:
    titles = "<START> " + titles
    titles += " <END>"

    for ngram in nltk.ngrams(titles.split(), NGRAM):
        combined_string = ngram[0] + " " + ngram[1]

        vocab[combined_string] = 1
        total_text_collection_AS_freq += 1

        if (combined_string not in text_collection_AS_freq):
            text_collection_AS_freq[combined_string] = 1
        else:
            text_collection_AS_freq[combined_string] += 1


print (len(vocab))
print "Total for AS, including total vocab size: ", (total_text_collection_AS_freq + len(vocab))
print "=========================="


# Find probability of a text "Here I" in text_collection_PC
# TODO: Make this use the conditional probability technique instead


probability = 1.0
query = "<START> Here I Am<END>"
for ngram in nltk.ngrams(query.split(), NGRAM):
    combined_string = ngram[0] + " " + ngram[1]
    if (combined_string in vocab and combined_string in text_collection_AS):
        probability *= text_collection_AS_freq[combined_string]/float(total_text_collection_AS_freq + len(vocab))
    else:
        print ("Ngram not found")
print probability

