import math
import nltk


text_collection_PC = ["I Don't Want To Go", "A Groovy Kind Of Love", "You Can't Hurry Love", "This Must Be Love", "Take Me With You"]
text_collection_AS = ["All Out Of Love", "Here I Am", "I Remember Love", "Love Is All", "Don't Tell Me"]

vocab = {} # We store vocab here to count the universal number of different vocabs available****
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
print ("Total for AS, including total vocab size: ", (total_text_collection_AS_freq + len(vocab)))
print ("Total for PC, including total vocab size: ", (total_text_collection_PC_freq + len(vocab)))
print ("==========================")


query = "I Remember You"


probability_AS = 0
probability_PC = 0


for word in query.split():
    if (word in text_collection_PC_freq):
        probability_PC += math.log(text_collection_PC_freq[word]/float(total_text_collection_PC_freq + len(vocab)))

for word in query.split():
    if (word in text_collection_AS_freq):
        probability_AS += math.log(text_collection_AS_freq[word]/float(total_text_collection_AS_freq + len(vocab)))
        
        
print ("Air Supply", probability_AS)
print ("Phil Collins", probability_PC)


# ## Final result


if (probability_PC > probability_AS):
    print ("This title is from Air Supply")
else:
    print ("This title is from Phil Collins")

