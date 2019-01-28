from nltk.book import text6

print [str(w) for w in text6 if (w.lower().endswith('ize') or 'a' in w.lower() or 'pt' in w.lower() or (w[0:1].isupper() and w[1:].islower()))]
