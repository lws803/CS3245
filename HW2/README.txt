This is the README file for A0167342E-A0177603A's submission

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
- The first step was to create a universal list out of all the terms founding in the dataset. This 
universal set was used to compute the negation (i.e. NOT) of a term when processing the boolean query.

- Next, every query in the file-of-queries is processed by first generating a queue of terms generated
from the shunting yard algorithm. In our implementation, the operator "NOT" was treated like a normal
term rather than an operator, and processed based on if there was a boolean query consisting of the 
operator "OR NOT", in which case the negation of the NOT term was processed, and "AND NOT", in which case
an optimisation was used (for A NOT B simply remove postings with term B in postings list in A).

- The processing for each query is as follows:

(i) Shunting yard algorithm generates a queue of tokens for each query

(ii) Each token in the queue is processed from left to right using a processing stack. If the token is
a normal term in the dictionary, we first normalized the token as per normalization techniques used in 
index.py and then add its postings list to the stack. If the token is a recognised operator, the 
respective operation from our "Logic" class will be performed (i.e. either one of AND, NOT, OR, AND NOT). 
OR NOT is processed by first doing a NOT operation on the respective term and then doing an OR with 
the other term.

(iii) However, as mentioned above, before performing the operation on the postings list, a skip list was
generated in memory during retrieval for the required term in order to increase processing speed as well
as reduce the size of the postings file dramatically (by 50% as each term had an associated 4 byte skip
pointer encoding in our original implementation!).

(iv) After processing each operation in the query, the resulting list of docID terms are written to the
output file as the answer for that particular query. 

== Files included with this submission ==

1. dictionary.txt - the dictionary part of our index
2. postings.txt - the postings list of our index
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
- Q1: Do you think it is a good idea to remove these number entries from the dictionary and the 
      postings lists? Can you propose methods to normalize these numbers? How many percentage of 
      reduction in disk storage do you observe after removing/normalizing these numbers?

      We believe it is not beneficial to remove number entries from the dictionary. As the reuters data
      set consists of news articles, many queries may contain numerical values (e.g. news about an event
      on a specific date) and hence keeping numbers in the dictionary is important. Moreover, the postings
      list of such numberical terms will likely be small, and hence not very beneficial even if they are
      reduced from the dictionary.
      
      To normalize numbers, all punctuation is stripped from the term string (i.e. a term such as 
      150,029,398 is reduced to 150029398 and treated like a number). This approach does mean that 
      terms such as June22nd will not be treated like a numerical term, and rightfully so since even 
      though the term represents a date it is not a pure numerical entry.

      Thanks to the customisation (boolean that can be set to ignore stopwords, numbers, etc.) we've 
      implemented in index.py, we calculated the percentage reduction in disk storage using our method
      of number removal using our method of normalization:

      Disk Storage WITHOUT numerical terms: 1,996,444 bytes
      Disk Storage WITH numerical terms: 2,202,932 bytes

      Percentage reduction after removing numerical terms: 9.373%

      Since the percentage reduction is relatively small, as well as the fact that queries related to news
      stories may contain numerical values, we have decided to keep number entries in the dictionary. 

- Q2: What do you think will happen if we remove stop words from the dictionary and postings file? 
      How does it affect the searching phase?

      If we remove stopwords, we will not be able to search for terms which include specifically those 
      with stopwords. Eg. if we wish to search for titles such as `The Awakening`. We will instead just 
      turn up with results from `Awakening`. However, it could speed up the searching phase as there 
      will be less terms in the dictionary search tree. Since these stopwords usually exist in most 
      documents, we would also save the trouble of querying such terms which could potentially result 
      in a larger postings list to merge with. Also, the postings and dictionary file will be much 
      smaller as we no longer have to store these stopwords together with their postings lists.



- Q3: The NLTK tokenizer may not correctly tokenize all terms. What do you observe from the resulting 
      terms produced by sent_tokenize() and word_tokenize()? Can you propose rules to further refine 
      these results?

      The sent_tokenize function tokenizes sentences whereas word_tokenize tokenizes words. 
      On the other hand, word_tokenize will only split off periods that are at the end of the line. 
      It assumes that the user has already performed sent_tokenize on the text.
