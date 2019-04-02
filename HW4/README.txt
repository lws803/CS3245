This is the README file for 's submission 

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


== Files included with this submission ==


== Statement of individual work ==

Please initial one of the following statements.

[ ] I, A0000000X, certify that I have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, I
expressly vow that I have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.  

[ ] I, A0000000X, did not follow the class rules regarding homework
assignment, because of the following reason:

<Please fill in>

I suggest that I should be graded as follows:

<Please fill in>

== References ==
