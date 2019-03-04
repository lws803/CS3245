This is the README file for A0000000X's submission

== Python Version ==

I'm (We're) using Python Version <2.7.x or replace version number> for
this assignment.

== General Notes about this assignment ==

Give an overview of your program, describe the important algorithms/steps 
in your program, and discuss your experiments in general.  A few paragraphs 
are usually sufficient.

== Files included with this submission ==

List the files in your submission here and provide a short 1 line
description of each file.  Make sure your submission's files are named
and formatted correctly.

== Statement of individual work ==

Please initial one of the following statements.

[X] I, A0000000X, certify that I have followed the CS 3245 Information
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

<Please list any websites and/or people you consulted with for this
assignment and state their role>


== Essay questions ==
- Q1:

- Q2: What do you think will happen if we remove stop words from the dictionary and postings file? How does it affect the searching phase?

If we remove stopwords, we will not be able to search for terms which include specifically those with stopwords. Eg. if we wish to search for titles such as `The Awakening`. We will instead just turn up with results from `Awakening`. However, it could speed up the searching phase as there will be less terms in the dictionary search tree. Since these stopwords usually exist in most documents, we would also save the trouble of querying such terms which could potentially result in a larger postings list to merge with. Also, the postings and dictionary file will be much smaller as we no longer have to store these stopwords together with their postings lists.

- Q3: The NLTK tokenizer may not correctly tokenize all terms. What do you observe from the resulting terms produced by sent_tokenize() and word_tokenize()? Can you propose rules to further refine these results?

sent_tokenize tokenizes sentences whereas word_tokenize tokenizes words. word_tokenize will only split off periods that are at the end of the line. It assumes that the user has already performed sent_tokenize on the text.
