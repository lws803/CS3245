import struct
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import codecs
import nltk

stemmer = PorterStemmer()

IGNORE_NUMBERS = False # Adding an option to ignore any numerical terms
IGNORE_PUNCTUATION = True # Adding an option to strip away all punctuation in a term
IGNORE_SINGLE_CHARACTER_TERMS = True # Adding an option to ignore single character terms
IGNORE_STOPWORDS = False # Adding an option to ignore stopwords
PERFORM_STEMMING = True # Adding an option to perform stemming on the current term


def normalize(term):
    term = term.lower() # Perform case folding on the current term

    # Ignores any terms that are single characters if the bool is set to true
    if (IGNORE_SINGLE_CHARACTER_TERMS and len(term) == 1): return None
            
   # Remove all instances of punctuation if the bool is set to true.
    if IGNORE_PUNCTUATION: # Remove all instances of punctuation
        term = term.translate(str.maketrans('','',string.punctuation))
            
    # Ignores any terms which are stopwords if the bool is set to true
    if (IGNORE_STOPWORDS and term in stopwords.words('english')): return None

    """
    Ignore any numerical terms into the index. The following method
    is used to detect terms with preceding punctuation such as .09123 or even 
    fractions such as 1-1/2 as well as date representations such as 24/09/2018.

    Although, it is still not able to detect terms such as "July22nd", however we feel
    if the term also has alphabetical characters, and there are no spaces between them, 
    it is best not to treat the term as a numerical one.
    """
    if IGNORE_NUMBERS: 
        temporary = term.translate(str.maketrans('','',string.punctuation))
        if temporary.isdigit(): return None # only ignore if term is fully numerical

    # Perform stemming on the term
    if PERFORM_STEMMING: term = stemmer.stem(term)
    
    return term



f = codecs.open("/Users/wilson/nltk_data/corpora/reuters/training/1", encoding='utf-8')
whole_text = f.read()
lines = nltk.sent_tokenize(whole_text)
dictionary = {}
count = 0

for line in lines:
    words = nltk.word_tokenize(line)
    for word in words:
        processed_word = normalize(word)
        if (processed_word != ""):
            count += 1
            dictionary[processed_word] = True

print (count)
print (len(dictionary))