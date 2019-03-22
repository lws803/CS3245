This is the README file for A0167342E-A0177603A's submission
Email address for A0167342E: e0175775@u.nus.edu
Email address for A0177603A: e0260109@u.nus.edu

== Python Version ==

We're using Python Version 3.6.5 for this assignment.

== General Notes about this assignment ==

=== index.py ===

- For the indexer, we allow 5 parameters for customisations on what to discard from the dictionary. 
These paramters are fed into the normalize function as the indexing is done. Indexing is done by 
reading the files one by one, then extracting the lines within the documents. 
- These lines are then sentence tokenized using (sent_tokenize) before tokenizing with word_tokenize 
to obtain the terms. The terms are then processed in the normalize function previously mentioned before 
storing into the dictionary in memory. 
- This dictionary, a (term: listOfDocIDs) pair is then sorted before writing to an actual file 
(postings.txt and dictionary.txt). 

- For dictionary.txt, we have decided to reserve the first line for all the documment IDs that were 
indexed. This is so that it will be easier to obtain the universal set during searching. Subsequent 
lines are added in the form of tuples where elements (term, frequency, docID) are seperated by a 
single space. The frequency for each term would later facilitate the retrieval method in search.py. 
- As for postings.txt, we have decided to use byte encoding to store the postings lists as it will take 
up less space compared to storing as characters in a text document. Since the maximum docID number is 
within the integer limit for a 4 byte unsigned integer, it would make more sense to pack it in a binary 
file. These docID are written to file using struct.pack() method.

Experimental notes:
We have tried to store the skip pointers in the postings list as well in the first place but quickly 
realise that it will not help in reducing the time complexity at all. By storing the skip pointers in 
the postings file, it will double up the size of the postings file since now each docID would be 
accompanied by a skip pointer (if any, else it will be stored as 0). This will actually be detrimental 
to the retrieval operation in search as it will now have to read twice as many bytes iteratively. We 
have decided to leave this up to search.py to generate the skip pointers instead during retrieval for 
the required term.

=== search.py ===
The first step for search is to generate the universal terms set from dictionary.txt. Our first line consists of all the document IDs which will make up our universal documents set and subsequent lines are terms with their doc frequency and starting offset to the postings list.

After generating the universal terms and documents set, we process the queries from specified query input file. These queries will be processed line by line and we will find the cosine similarity with lnc.ltc scheme. In this function, we just consider the terms present in the query as we will ultimately set terms which are not present in either query or document with tf-idf score of zero. The terms are of course processed with normalisation function before using it to obtain the starting offset of the postings list for that term. Then we begin storing the docIDs together with their term frquencies in a dictionary `tf_doc`. 
We also obtain the term frequencies of the query string before proceeding to the main bulk of this function - finding tf-idf scores for the query vector space and document terms vector space.

After obtaining the two arrays from generating the vector spaces for query and document terms, we proceed to creating a numpy array and normalizing it.

Finally, we perform a dot product using numpy operation to obtain the final vector (1x1 vector) which is the cosine similarity result.

To sort the documents with respect to their cosine similarity results, we multiplied the cosine similarities by -1 and stored it together with the document ID as a pair in an array. This is because we wish to sort the cosine similarity score by desceding order (larger the more relevant) and document IDs by ascending order. After sorting, we just have to output the first 10 results or less.


== Files included with this submission ==

1. dictionary_final.txt - the dictionary part of our index
2. postings_final.txt - the postings list of our index
3. index.py - python script used to create our index from the Reuters data
4. search.py - python script that generates a list of documents that satisfy input queries using
               the shunting yard algorithm

== Statement of individual work ==

[X] We, A0167342E-A0177603A, certify that we have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, we
expressly vow that we have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.  

== References ==
Websites:
1. https://en.wikipedia.org/wiki/Shunting-yard_algorithm (To understand the Shunting Yard algo better)
2. https://www.tutorialspoint.com/python/file_seek.htm (To learn how Python Files I/O functions work)

== Essay questions ==
