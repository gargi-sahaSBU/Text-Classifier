from nltk.corpus import stopwords
#from sklearn.datasets import load_files
#stop words lower case
def config1(categories,data):
	cachedStopWords = stopwords.words("english")

	#categories = ['rec.sport.hockey','sci.med','soc.religion.christian','talk.religion.misc']
	#twenty_train = load_files(container_path='Training',categories=categories,load_content=True,encoding='latin-1')

	for i in range(len(data)):
		data[i] = ' '.join([word.lower() for word in data[i].split() if word.lower() not in cachedStopWords])

	return data