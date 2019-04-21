This is the README file for A0167342E-A0177603A-A0164515H-A0164682X's submission 
Email address for A0167342E: e0175775@u.nus.edu
Email address for A0177603A: e0260109@u.nus.edu
Email address for A0164515H: 
Email address for A0164682X: 

== Python Version ==

I'm using Python Version <2.7.15> for
this assignment.

== General Notes about this assignment ==

Indexing:
1. CSV file is parsed through to extract the zoned data, using Python's csvreader.
2. Index the data into a dictionary for further processing.
3. For every indexed document ID, its content and title are tokenized.
4. Preprocess the words by removing punctuation and numbers, casefolding, single letters, unicode characters,
and finally stemming.
5. Then insert it into the dictionary, along with its document ID, positional index, and whether it originated from content or title.
6. For each doc ID, sort the postings by docId and positional index. Then convert these postings into byte and write out. Update the current position of byte offset and the document frequency for each term and write them into documents.txt.
7. Weights for each document is summed up. Together with each doc_id's court information, they are written into the documents.txt.

Searching:



Query Refinement:

1) Query Expansion using thesaurus

2) Pseudo Ranked Retrieval using rocchio 


== Files included with this submission ==
1. dictionary_final.txt - the dictionary part of our index
2. postings_final.txt - the postings list of our index
3. index.py - python script used to create our index from the Intelllex CSV data
4. search.py - python script that generates a list of documents that are ranked by relevance in relation to
			   the query. The query can be either free text, or can be a Boolean query with phrasal queries
			   included. In order to refine the query, query expansion and pseudo relevant techniques were
			   also implemented. 

== Statement of individual work ==

Please initial one of the following statements.

[X] I, A0167342E-A0177603A-A0164515H-A0164682X, certify that I have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, I
expressly vow that I have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.  

== References ==



