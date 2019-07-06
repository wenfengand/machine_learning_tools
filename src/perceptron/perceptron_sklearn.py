from sklearn.datasets import fetch_20newsgroups 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import Perceptron 
from sklearn.metrics import f1_score, classification_report 

categories = ['rec.sport.hockey', 'rec.sport.baseball', 
    'rec.autos'] 
news_groups_trian = fetch_20newsgroups(subset='train',
    categories=categories,
    remove=('headers','footers', 'quotes'))
news_groups_test = fetch_20newsgroups(subset='test',
    categories=categories,
    remove=('headers','footers','quotes')) 
vectorizer = TfidfVectorizer() 
x_train = vectorizer.fit_transform(news_groups_trian.data) 
x_test  = vectorizer.transform(news_groups_test.data) 

clf = Perceptron(random_state=11) 
clf.fit(x_train, news_groups_trian.target) 
predictions = clf.predict(x_test) 
print(classification_report(news_groups_test.target, predictions)) 

