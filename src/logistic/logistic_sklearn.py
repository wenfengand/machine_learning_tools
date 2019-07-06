import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report 

from os.path import dirname, abspath, join
PROJECT_ROOT = dirname(dirname(dirname(abspath(__file__))))
INPUT_ROOT = join(PROJECT_ROOT, 'input')
SMS_FILE = join(INPUT_ROOT, 'sms', 'SMSSpamCollection')

df = pd.read_csv(SMS_FILE, delimiter='\t', header=None) 

x = df[1].values
y = df[0].values 
x_train_raw, x_test_raw, y_train, y_test = train_test_split(x,y) 
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(x_train_raw)
x_test  = vectorizer.transform(x_test_raw) 

lb = LabelBinarizer()
y_train_binarized = lb.fit_transform(y_train) 
y_test_binarized = lb.transform(y_test)

classifier = LogisticRegression() 
classifier.fit(x_train, y_train_binarized) 

predictions = classifier.predict(x_test) 

precisions = cross_val_score(classifier, x_train, y_train_binarized, cv=5, scoring='precision')
print('Precisions from cross_val_score', precisions) 

report = classification_report(y_test_binarized, predictions,\
    target_names=['ham', 'spam'], labels=lb.transform(['ham','spam']).reshape(-1))
print('Report from classification_report\n', report)

