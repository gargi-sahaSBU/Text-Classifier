from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import sys
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def func(classifier,ngram,output_file,features):
	categories = ['rec.sport.hockey','sci.med','soc.religion.christian','talk.religion.misc']
	twenty_train = load_files(container_path='Training',categories=categories,load_content=True,encoding='latin-1')
	
	count_vect = CountVectorizer(ngram_range=(ngram,ngram))
	X_train_counts = count_vect.fit_transform(twenty_train.data)
	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)



	twenty_test = load_files(container_path='Test',categories=categories,encoding='latin-1',load_content=True)
	X_new_counts = count_vect.transform(twenty_test.data)
	X_new_tfidf = tfidf_transformer.transform(X_new_counts)
	predicted = []

	if classifier == 1:
		output_file.write("NaiveBayes"+"\t")
		if features == 1: #count vector
			output_file.write("count vector"+"\t")
			clf = MultinomialNB().fit(X_train_counts, twenty_train.target)
			output_file.write(str(ngram)+"\t")
			predicted = clf.predict(X_new_counts)
		else: #tfidf
			output_file.write("tfidf"+"\t")
			clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
			output_file.write(str(ngram)+"\t")
			predicted = clf.predict(X_new_tfidf)

	elif classifier == 2:
		#vary parameters
		output_file.write("LogisticRegression"+"\t")
		if features == 1: #count vector
			output_file.write("count vector"+"\t")
			clf = LogisticRegression(penalty='l1').fit(X_train_counts, twenty_train.target)
			output_file.write(str(ngram)+"\t")
			predicted = clf.predict(X_new_counts)
		else: #tfidf
			output_file.write("tfidf"+"\t")
			clf = LogisticRegression(penalty='l1').fit(X_train_tfidf, twenty_train.target)
			output_file.write(str(ngram)+"\t")
			predicted = clf.predict(X_new_tfidf)

		

	elif classifier == 3:
		#vary parameters
		output_file.write("SVM"+"\t")
		if features == 1: #count vector
			output_file.write("count vector"+"\t")
			clf = SVC(kernel = 'linear', probability = True).fit(X_train_counts, twenty_train.target)
			output_file.write(str(ngram)+"\t")
			predicted = clf.predict(X_new_counts)
		else: #tfidf
			output_file.write("tfidf"+"\t")
			clf = SVC(kernel = 'linear', probability = True).fit(X_train_tfidf, twenty_train.target)
			output_file.write(str(ngram)+"\t")
			predicted = clf.predict(X_new_tfidf)
		
		
	elif classifier == 4:
		output_file.write("RandomForest"+"\t")
		if features == 1: #count vector
			output_file.write("count vector"+"\t")
			clf = RandomForestClassifier().fit(X_train_counts, twenty_train.target)
			output_file.write(str(ngram)+"\t")
			predicted = clf.predict(X_new_counts)
		else: #tfidf
			output_file.write("tfidf"+"\t")
			clf = RandomForestClassifier().fit(X_train_tfidf, twenty_train.target)
			output_file.write(str(ngram)+"\t")
			predicted = clf.predict(X_new_tfidf)
		
		
	
	#print(np.mean(predicted == twenty_test.target))
	#print("macro average of precision")
	output_file.write(str(precision_score(twenty_test.target,predicted,average='macro')) +"\t")
	#print("macro average of recall")
	output_file.write(str(recall_score(twenty_test.target,predicted,average='macro')) + "\t")
	#print("macro average of f1 score")
	output_file.write(str(f1_score(twenty_test.target,predicted,average='macro')))
	output_file.write("\n")





