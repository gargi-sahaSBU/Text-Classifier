from nltk.stem.porter import *
from sklearn.datasets import load_files
from nltk.corpus import stopwords

#porter stemmer and stop words

#categories = ['rec.sport.hockey','sci.med','soc.religion.christian','talk.religion.misc']
#twenty_train = load_files(container_path='Training',categories=categories,load_content=True,encoding='latin-1')
def config3(categories,data):
	cachedStopWords = stopwords.words("english")

	for i in range(len(data)):
		#print(twenty_train.data[i])
		#print("xxxxxxxxxxxxxxxxxx")
		stemmer = PorterStemmer()
		singles = []
		for word in data[i].split():
			if word.lower() not in cachedStopWords:
				singles.append(stemmer.stem(word.lower()))

		data[i] = []
		#print()
		#print()
		data[i] = ' '.join([word for word in singles])
	#print(twenty_train.data[i])
	return data