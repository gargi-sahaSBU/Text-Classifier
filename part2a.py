from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import sys
import time
import random
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import f1_score
import copy
import numpy
from StopWords import config1
from Stemmer import config2
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE

def SVM(feature_rep, l1orl2, kerneltype, C, gamma ):
	categories = ['rec.sport.hockey','sci.med','soc.religion.christian','talk.religion.misc']
	twenty_train = load_files(container_path='Training',categories=categories,load_content=True,encoding='latin-1')
	if feature_rep == "Remove Stop Words":
		twenty_train.data = config1(categories,twenty_train.data)
	elif feature_rep == "Porter Stemmer":
		twenty_train.data = config2(categories,twenty_train.data)

	count_vect = CountVectorizer()
	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(count_vect.fit_transform(twenty_train.data))

	twenty_test = load_files(container_path='Test',categories=categories,encoding='latin-1',load_content=True)
	X_new_counts = count_vect.transform(twenty_test.data)
	X_new_tfidf = tfidf_transformer.transform(X_new_counts)

	if kerneltype == 'linear':
		if l1orl2 == 'l1':
			clf = LinearSVC(penalty= l1orl2,dual= False).fit(X_train_tfidf, twenty_train.target)
		else :
			clf = LinearSVC(penalty= l1orl2,dual= True).fit(X_train_tfidf, twenty_train.target)
	else:
		clf = SVC(kernel = kerneltype, probability = True).fit(X_train_tfidf, twenty_train.target)

	predicted = clf.predict(X_new_tfidf)

	print(f1_score(twenty_test.target,predicted,average='macro'))

'''
1) stopwords with lower case
2) stemmer with lower case

1) l1
2) l2

1) linear
2) poly
3) rbf


'''
feature_rep = ["Remove Stop Words","Porter Stemmer"]
feature_sel = ['l1','l2'] #for linearsvc
kernels = ['linear','poly','rbf']
C = [1,100,1000]
gamma = [0.001,0.0001]

print('Linear kernel:')
#C,feature_rep,feature_sel
for i in feature_rep:
	for j in feature_sel:
		for k in C:
			print(i,",",j,",",k,":")
			SVM(i,j,'linear',k,None)

print('RBF:')
for i in C:
	for j in gamma:
		print(i,",",j,":")
		SVM(None,None,'rbf',i,j)

print('Polynomial:')
for i in C:
	for j in gamma:
		print(i,",",j,":")
		SVM(None,None,'poly',i,j)
