import pandas as pd 
from sklearn.datasets import load_breast_cancer 
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split

x, y = load_breast_cancer(return_X_y=True) 
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y,
    test_size=0.2, random_state=31) 

nb = GaussianNB()

nb.fit(x_train, y_train) 
nb_score = nb.score(x_test, y_test)
print(nb_score)