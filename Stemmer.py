from nltk.stem.porter import *
from sklearn.datasets import load_files

#porter stemmer and lower case

#categories = ['rec.sport.hockey','sci.med','soc.religion.christian','talk.religion.misc']
#twenty_train = load_files(container_path='Training',categories=categories,load_content=True,encoding='latin-1')

#print(twenty_train.data[1])
#print("xxxxxxxxxxxxxxx")
def config2(categories,data):
	for i in range(len(data)):
		stemmer = PorterStemmer()
		singles = []
		for word in data[i].split():
			singles.append(stemmer.stem(word.lower()))

		data[i] = []
		#print()
		#print()
		data[i] = ' '.join([word for word in singles])
		#print(twenty_train.data[i])

	return data