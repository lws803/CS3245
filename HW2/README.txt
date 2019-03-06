This is the README file for A0167342E-A0177603A's submission

== Python Version ==

We're using Python Version 3.6.5 for this assignment.

== General Notes about this assignment ==

Give an overview of your program, describe the important algorithms/steps 
in your program, and discuss your experiments in general.  A few paragraphs 
are usually sufficient.

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
