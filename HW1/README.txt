This is the README file for A0167342E's submission

== Python Version ==

I'm (We're) using Python Version 2.7 for
this assignment.

== General Notes about this assignment ==

Summary:

This program was made to identify query texts of 3 different languages based on training text data.
The program will first create a 4-gram character model of the training data with add-one smoothing and then calculate the probability, 
PR(4-gram|language_model) from each languages before sorting them to find the language with the highest probability.

The program can also validate and determine if the predicted language is accurate 
by performing baye's theorem on the probability above to find PR(language_model|4-gram). 
If the probability is below a certain threshold, we consider it a language of unknown origin (output "other").

Experiments/ problems encoutered:

Tokenisation and case folding was done at first before concatenating and recreating a sentence free of punctuation and upper cases.
Then 4-gram was performed on the newly filtered text.
However, the result was not satisfactory as the probality calculated was inconsistent.
This goes to show that the capitalisation of words and puntuations play an important role in definining the sentence structure.

There were also issues in storing float numbers that are very small due to the multiplication of probabilities.
Probabilities which are very small usually get truncated to 0. Hence, we have to perform all multiplication of probabilities
in logarithmic additions.

Lastly, there were some difficulties in identifying if a language for a text should be considered as "other".
Initially I wanted to use the probabilities generated PR(4-gram|language_model), but quickly realise that this probability
can not be depended as they cannot be compared among different queries. Hence a specific threshold could not be used.
It was eventually revamped and now using baye's theorem to find PR(language_model|4-gram) before deciding.


== Files included with this submission ==

- build_test_LM.py - to build then language model and generate output based on the language model predictions
- eval.py - to evaluate the output with a correct output and determine percentage accuracy.

== Statement of individual work ==

Please initial one of the following statements.

[X] I, A0167342E, certify that I have followed the CS 3245 Information
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

logarithmic addition:
http://practicalcryptography.com/miscellaneous/machine-learning/tutorial-automatic-language-identification-ngram-b/
