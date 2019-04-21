This is the README file for A0167342E-A0177603A-A0164515H-A0164682X's submission 
Email address for A0167342E: e0175775@u.nus.edu
Email address for A0177603A: e0260109@u.nus.edu
Email address for A0164515H: e0148557@u.nus.edu
Email address for A0164682X: e0148724@u.nus.edu

== Python Version ==
I'm using Python Version <2.7.15> for
this assignment.

== General Notes about this assignment ==

=== Indexing: ===
1. CSV file is parsed through to extract the zoned data, using Python's csvreader.
2. Index the data into a dictionary for further processing.
3. For every indexed document ID, its content and title are tokenized.
4. Preprocess the words by removing punctuation and numbers, casefolding, single letters, unicode characters,
and finally stemming.
5. Then insert it into the dictionary, along with its document ID, positional index, and whether it originated from content or title.
6. For each doc ID, sort the postings by docId and positional index. Then convert these postings into byte and write out. Update the current position of byte offset and the document frequency for each term and write them into documents.txt.
7. Weights for each document is summed up. Together with each doc_id's court information, they are written into the documents.txt.

=== Format of dictionary: ===
The dictionary contains 3 regions. The first region is the list of words, followed by document metadata. Lastly the
third section contains a list of words in each document. This section is used by the Rocchio algorithm to perform
query refinement. For the first region the format is:
   <term> <document frequency> <postings length> <postings address>
The term refers to the stemmed term. Postings length is the number of entries in the posting list for the term and postings offset is the byte offset of the starting of the postings.
An example of this section would be:
   telephon 254 1074 5688716
The second region consists of the format:
   <docid>:<document vector length>:<court>
Here the document vector length is the square sum of all the term frequencies. This is useful for ranking later. An example would be
   1231:4567:SG HIGH COURT
The third region consists of a word list.
   doc_id^word1^word2^...^wordn
An example would be
   1231^chicken^tastes^good

=== Format of postings: ===
Postings were stored in sorted arrays which took up contiguous space. Each posting entry has 12bytes which are structured like so:

    +-------------+-------------+-------------+
    |    doc_id   |  pos_index  |    zone     |
    +-------------+-------------+-------------+
    0             3             7             11
The pos_index refers to the position of the term occurence in the document. The zone refers to where the term was found. Whether in the
title or in the document body.

=== Originality ===
While the creating of the index, searching, and the algorithms implemented were quite similar to those taught in class, our team decided 
to experiment by using a state machine to read the postings list in an efficient manner and switch between modes of retrieving data. This is
akin to the format of the dictionary, which was explained earlier. 

=== Searching: ===
There are two modes which are supported, namely boolean and free text. 
For the relevant methods, we use the two pointer method to perform intersections and unions, and after experimenting
with the query refinement methods, we decided to perform both query expansion as well as pseudo relevance feedback in order
to ensure that the number of docs retrieved is sufficient to perform accurate ranking.

To add to this, for each list of relevant docIDs, any documents that were provided to us in the query were appended to the
front of the 'positive' list, to indicate that these documents are 100% relevant and hence need to be the highest up in the ranking list.

For phrase queries we used position indexinf gor retrieving them.

=== Query Refinement: ===
1) Query Expansion using thesaurus
We used Wordnet to find synonyms to words in the query. The hits were then retrieved from the postings list and returned
for ranking. When ranking, synonyms were bunched together in the vector bins. I.e. 'Telephone' and 'phone' would be
merged together into one dimension.

2) Ranked Retrieval using rocchio
The general pattern of the rocchio algorithm is highlighted below:
1. AND the terms discovered among doc terms in relevance/ pseudo relevance feedback
2. Set threshold for rocchio score to ignore terms with low score (< 0.7)
3. Combine the relevant and pseudo relevant docs together to provide a smaller range of common terms among the docs
4. But this method is not representative as even if the term is prominent in a few documents but not in some of them, it will be left out
5. Hence, we then decided to use the top 10 prominent terms for rocchio instead
6. We have also decided to create our own stopword list as the documents consistently contain words such as court, case etc etc

=== Configuring all features for user use ===
The above can be configured using the following lines
```
    THESAURUS_ENABLED = True # Enable synonym based query expansion
    K_PSEUDO_RELEVANT = 10
    K_PROMINENT_WORDS = 10
    ROCCHIO_SCORE_THRESH = 0.5
    PSEUDO_RELEVANCE_FEEDBACK = False
    ROCCHIO_EXPANSION = True # Enable rocchio expansion
    ALPHA = 1
    BETA = 0.75
```
== Files included with this submission ==
1. dictionary_final.txt - the dictionary part of our index
2. postings_final.txt - the postings list of our index
3. index.py - python script used to create our index from the Intelllex CSV data
4. search.py - python script that generates a list of documents that are ranked by relevance in relation to
			   the query. The query can be either free text, or can be a Boolean query with phrasal queries
			   included. In order to refine the query, query expansion and pseudo relevant techniques were
			   also implemented. 
5. BONUS.docx - contains information about query refinement procedures that we implemented, including pseudo 
                relevance feedback using rocchio as well as thesaurus expansion using wordnet.

== Statement of individual work ==
Please initial one of the following statements.

[X] I, A0167342E-A0177603A-A0164515H-A0164682X, certify that I have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, I
expressly vow that I have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.  

== References ==
=== Websites ===
1. https://nlp.stanford.edu/IR-book/html/htmledition/rocchio-classification-1.html (To gain a better understanding of the Rocchio algorithm)
2. https://www.guru99.com/wordnet-nltk.html (Understanding the uses of wordnet further for thesaurus expansion)
