import math
import nltk


text_collection_PC = ["I Don't Want To Go", "A Groovy Kind Of Love", "You Can't Hurry Love", "This Must Be Love", "Take Me With You"]
text_collection_AS = ["All Out Of Love", "Here I Am", "I Remember Love", "Love Is All", "Don't Tell Me"]


vocab = {}
text_collection_PC_freq = {}
text_collection_AS_freq = {}
total_text_collection_PC_freq = 0
total_text_collection_AS_freq = 0


for titles in text_collection_AS:
    for word in titles.split():
        vocab[word] = 1
        total_text_collection_AS_freq += 1
        if (word not in text_collection_AS_freq):
            text_collection_AS_freq[word] = 1
        else:
            text_collection_AS_freq[word] += 1


for titles in text_collection_PC:
    for word in titles.split():
        vocab[word] = 1
        total_text_collection_PC_freq += 1

        if (word not in text_collection_PC_freq):
            text_collection_PC_freq[word] = 1
        else:
            text_collection_PC_freq[word] += 1


print (len(vocab))
print "Total for AS, including total vocab size: ", (total_text_collection_AS_freq + len(vocab))
print "Total for PC, including total vocab size: ", (total_text_collection_PC_freq + len(vocab))
print "=========================="


# Find probability of a text "I Don't Love You" in text_collection_PC


# Add one smoothing before querying

query = "I Remember You"

for word in query.split():
    vocab[word] = 1

    if (word not in text_collection_PC_freq):
        text_collection_PC_freq[word] = 1
        total_text_collection_PC_freq += 1


for word in query.split():
    vocab[word] = 1

    if (word not in text_collection_AS_freq):
        text_collection_AS_freq[word] = 1
        total_text_collection_AS_freq += 1



# print "Total for PC (NEW), including total vocab size: ", (total_text_collection_PC_freq + len(vocab))

probability_AS = 1.0
probability_PC = 1.0


for word in query.split():
    probability_PC *= text_collection_PC_freq[word]/float(total_text_collection_PC_freq + len(vocab))

for word in query.split():
    probability_AS *= text_collection_AS_freq[word]/float(total_text_collection_AS_freq + len(vocab))


if (probability_PC > probability_AS):
    print "This title is from Air Supply"
else:
    print "This title is from Phil Collins"




