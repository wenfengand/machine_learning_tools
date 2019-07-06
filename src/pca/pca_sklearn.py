import os 
import numpy as np 
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.preprocessing import scale 
from sklearn.decomposition import PCA 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report 
from PIL import Image 

x = []
y = [] 

for dirpath, _, filenames in os.walk('../../input/att_faces'):
    for filename in filenames:
        if filename[-3:] == 'pgm':
            img = Image.open(os.path.join(dirpath, filename)).convert('L')
            arr = np.array(img).reshape(10304).astype('float32')  / 255.
            x.append(arr) 
            y.append(dirpath) 
x = scale(x) 

x_train, x_test, y_train, y_test = train_test_split(x, y) 
pca = PCA(n_components=150) 

x_train_reduced = pca.fit_transform(x_train) 
x_test_reduced = pca.transform(x_test) 

print(x_train.shape) 
print(x_train_reduced.shape)

classifier = LogisticRegression()
accuracies = cross_val_score(classifier, x_train_reduced, y_train) 
print('Cross validation accuracy: %s ' %np.mean(accuracies))
classifier.fit(x_train_reduced, y_train) 
predictions = classifier.predict(x_test_reduced) 
print(classification_report(y_test, predictions))