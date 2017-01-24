from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import sys
import time
import random
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import f1_score
import copy
import numpy

def compare_classifiers(classifier,training_data_size,categories,twenty_test):

	twenty_train = load_files(container_path='Training',categories=categories,load_content=True,encoding='latin-1')
	random_indices=[]
	random.seed(time.time())
	size = len(twenty_train.data)
	
	for i in range(1,training_data_size+1):
		random_indices.append(random.randrange(0,size))
	
	data = []
	target = []
	filenames = []
	for i in random_indices:
		data.append(twenty_train.data[i])
		target.append(twenty_train.target[i])
		filenames.append(twenty_train.filenames[i])

	twenty_train.data = copy.copy(data)
	twenty_train.target = copy.copy(target)
	twenty_train.filenames  = copy.copy(filenames)

	count_vect = CountVectorizer()
	X_train_counts = count_vect.fit_transform(twenty_train.data)

	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

	X_new_counts = count_vect.transform(twenty_test.data)
	X_new_tfidf = tfidf_transformer.transform(X_new_counts)
	predicted = []

	if classifier == 1:
		clf = MultinomialNB().fit(X_train_counts, twenty_train.target)
		predicted = clf.predict(X_new_counts)
	elif classifier == 2:
		#vary parameters
		clf = LogisticRegression(penalty='l1').fit(X_train_counts, twenty_train.target)
		predicted = clf.predict(X_new_counts)
	elif classifier == 3:
		#vary parameters
		clf = SVC(kernel = 'linear', probability = True).fit(X_train_tfidf, twenty_train.target)
		predicted = clf.predict(X_new_tfidf)
	elif classifier == 4:
		#vary parameters
		clf = RandomForestClassifier().fit(X_train_tfidf, twenty_train.target)
		predicted = clf.predict(X_new_tfidf)


	return f1_score(twenty_test.target,predicted,average='macro')


categories = ['rec.sport.hockey','sci.med','soc.religion.christian','talk.religion.misc']

twenty_test = load_files(container_path='Test',categories=categories,encoding='latin-1',load_content=True)

for i in range(1,3):
	output = open(str(i)+"gp_plot.txt",'w')
	for data_size in range(71,2172,20):
		output.write(str(data_size)+" ")
		output.write(str(compare_classifiers(i,data_size,categories,twenty_test)))
		output.write("\n")
	output.close()

