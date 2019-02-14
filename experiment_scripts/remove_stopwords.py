from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sentence = "Ima Despo, a usually diligent and highly motivated student, suddenly found herself horrified by the 2040C relationship graph on a balmy crisp morning..."

stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example_sentence)

filtered_sentence = [w for w in word_tokens if not w in stop_words]

filtered_sentence = []

for w in word_tokens:
	if w not in stop_words:
		filtered_sentence.append(w)

print(word_tokens)
print(filtered_sentence)