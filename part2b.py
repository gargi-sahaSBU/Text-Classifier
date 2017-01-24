'''
Export the best configuration training model and give us
code that we can run to get performance of your model on a test set. Make sure
to write a brief documentation on how to run your code in your report. The
organization of the test data we use is the same as the test data you are given,
you can design an interface accordingly.
'''
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import sys
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.svm import LinearSVC
from StopWords import config1
from sklearn.externals import joblib

categories = ['rec.sport.hockey','sci.med','soc.religion.christian','talk.religion.misc']


twenty_train = load_files(container_path='Training',categories=categories,load_content=True,encoding='latin-1')
twenty_train.data = config1(categories,twenty_train.data)


count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(count_vect.fit_transform(twenty_train.data))



clf = LinearSVC(penalty= 'l2',dual= True).fit(X_train_tfidf, twenty_train.target)



joblib.dump([clf,count_vect,tfidf_transformer],'classifier.pkl')
