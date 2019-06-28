import numpy as np 
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, \
    precision_score, recall_score, f1_score, \
    classification_report 

K = 3 

x_train = np.array([
    [158, 64],
    [170, 86],
    [183, 84],
    [191, 80],
    [155, 49],
    [163, 59],
    [180, 67],
    [158, 54],
    [170, 67]
])
x_test = np.array([
    [168, 65],
    [180, 96],
    [160, 52],
    [169, 67]
])
y_train = np.array([
    'male',
    'male',
    'male',
    'male',
    'female',
    'female',
    'female',
    'female',
    'female'
])
y_test = np.array([
    'male',
    'male',
    'female',
    'female'
])

lb = LabelBinarizer()
y_train_binarized = lb.fit_transform(y_train) 
y_test_binarized = lb.fit_transform(y_test) 

clf = KNeighborsClassifier(n_neighbors=K) 
clf.fit(x_train, y_train_binarized.reshape(-1))

predictions_binarized = clf.predict(x_test) 
predictions_labels = lb.inverse_transform(predictions_binarized) 

# individual metrics 
acc = accuracy_score(y_test_binarized, predictions_binarized)
precision = precision_score(y_test_binarized, predictions_binarized) 
recall    = recall_score(y_test_binarized, predictions_binarized) 
f1        = f1_score(y_test_binarized, predictions_binarized) 

# report
report    = classification_report(y_test_binarized, predictions_binarized, \
    target_names=['male', 'female'], labels=lb.transform(['male','female']).reshape(-1))



print('Accuracy %f - precision %f - recall %f - f1 %f'  %(acc, precision, recall, f1))

print(report)