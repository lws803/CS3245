from nltk.stem import *

from nltk.probability import FreqDist
import nltk

stemmer = PorterStemmer()

text = "penyetted penyet"
text1 = "penyet test helloed"
text2 = "penyetted hello"

texts = [text, text1, text2]

dictionary = {}

for i in range(0, 3):
    for word in nltk.word_tokenize(texts[i]):
        word = stemmer.stem(word) # Stem it first
        if (word not in dictionary):
            dictionary[word] = [i]
        else:
            if (i not in dictionary[word]):
                dictionary[word].append(i)


# Porter stemmer can remove the -ed and -s etc etc

for items in dictionary:
    print items, " ", dictionary[items]

# Texts are ordered by their index in increasing order

query1 = "penyet"
query2 = "hello"

query1 = stemmer.stem(query1)
query2 = stemmer.stem(query2)


queries = [[len(dictionary[query1]), query1], [len(dictionary[query2]), query2]]
queries.sort() # Sort the queries so we tackle the smallest one first

# We want to find a text which contains both penyet and hello
p1 = 0
p2 = 0
foundTexts = {}
# We can check both of them at the same time as their arrays are sorted
while (p1 < len(dictionary[queries[0][1]]) and p2 < len(dictionary[queries[1][1]])):
    index1 = dictionary[queries[0][1]][p1]
    index2 = dictionary[queries[1][1]][p2]


    if (index1 < index2):
        p1 += 1 # If index1 < index2 then we move p1 up
    elif (index2 < index1):
        p2 += 1 # vice versa for this one as well
    elif (index1 == index2): 
        foundTexts[index1] = True
        p1 += 1
        p2 += 1

print foundTexts.keys()


# We want to find a text which contains penyet but not hello
foundTexts = {}
p1 = 0
p2 = 0
while (p1 < len(dictionary["penyet"]) and p2 < len(dictionary["hello"])):
    index1 = dictionary["penyet"][p1]
    index2 = dictionary["hello"][p2]
    if (index1 < index2):
        foundTexts[index1] = True # Here we use a set instead as the index could be added in multiple times
        # If index1 < index2, means index2 does not contain the article we want, else they would have both been matched
        # In this case, the second condition of !"hello" has to be found from the second pointer
        p1 += 1
    elif (index1 > index2):
        p2 += 1
    elif (index1 == index2):
        p1 += 1
        p2 += 1
        foundTexts.pop(index1, None) # If found later on in the list, then just pop it off

print foundTexts.keys()
