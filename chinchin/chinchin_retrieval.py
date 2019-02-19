import nltk
import csv
import sys
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import re


args = sys.argv

stop_words = set(stopwords.words('english')) 


# Word freq
# with open(args[1]) as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     for row in csv_reader:
#         this_row = row[1]
#         this_row = re.sub(r"[^a-zA-Z0-9]+", ' ', this_row)
#         this_row = this_row.lower()
#         word_tokens = word_tokenize(this_row)

#         filtered_sentence = [w for w in word_tokens if not w in stop_words]
#         for term in filtered_sentence:
#             if (term not in vocab):
#                 vocab[term] = 1
#             else:
#                 vocab[term] += 1

#         this_row = row[4]
#         this_row = re.sub(r"[^a-zA-Z0-9]+", ' ', this_row)
#         this_row = this_row.lower()
#         word_tokens = word_tokenize(this_row)
        
#         filtered_sentence = [w for w in word_tokens if not w in stop_words]
#         for term in filtered_sentence:
#             if (term not in vocab):
#                 vocab[term] = 1
#             else:
#                 vocab[term] += 1

# sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
# sorted_vocab.reverse()

# for it in xrange(0,20):
#     print sorted_vocab[it]


vocab = {}
documents = {}
count = 0
with open(args[1]) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        statement = row[1]
        response = row[4]
        statement = re.sub(r"[^a-zA-Z0-9]+", ' ', statement)
        response = re.sub(r"[^a-zA-Z0-9]+", ' ', response)
        statement = statement.lower()
        response = response.lower()
        combined = "<START> " + statement + " " + response + " <END>"
        documents[count] = combined
        tokens = combined.split(' ')

        filtered_sentence = [w for w in tokens if not (w is "<START>" or w is "<END>")]
        for term in filtered_sentence:
            if (term not in vocab):
                vocab[term] = [count]
            else:
                vocab[term].append(count)

        count += 1

query = "justin"
query = re.sub(r"[^a-zA-Z0-9]+", ' ', query)
query = query.lower()

query2 = "dick"
query2 = re.sub(r"[^a-zA-Z0-9]+", ' ', query2)
query2 = query2.lower()


p1 = 0
p2 = 0
foundTexts = []

while (p1 < len(vocab[query]) and p2 < len(vocab[query2])):
    index1 = vocab[query][p1]
    index2 = vocab[query2][p2]
    
    if (index1 < index2):
        p1 += 1 # If index1 < index2 then we move p1 up
    elif (index1 > index2):
        p2 += 1 # if index2 < index1 then we move p2 up
    elif (index1 == index2): 
        foundTexts.append(index1)
        p1 += 1
        p2 += 1


for IDs in foundTexts:
    print (documents[IDs])
