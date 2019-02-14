from nltk.book import text1

text1.concordance("monstrous") # Show a concordance view of a word with its context

text1.similar("monstrous") # Show words that appear in similar context

text1.common_contexts(["monstrous", "very"]) # Examine the context shared by two or more words

print len(text1) # Count number of words and punctuations

print len(set(text1)) # Print vocab size of the text

print text3.count("smote") # Print num word occurence

