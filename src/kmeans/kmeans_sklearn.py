import numpy as np 
import mahotas as mh 
from mahotas.features import surf 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import * 
from sklearn.cluster import MiniBatchKMeans 
import glob 
import os 
import random 

all_instance_filenames = []
all_instance_targets = [] 

for f in glob.glob('../../input/dogs_vs_cats/train/*.jpg'):
    target = 1 if 'cat' in os.path.split(f)[1] else 0 
    all_instance_filenames.append(f) 
    all_instance_targets.append(target) 

# use a part of dataset to save time
n_examples = len(all_instance_filenames)
idx = np.random.choice(range(0,n_examples), size=200, replace=False)

random.shuffle(idx)

all_instance_filenames = np.array(all_instance_filenames)[idx]
all_instance_targets = np.array(all_instance_targets)[idx]

surf_features = [] 
for f in all_instance_filenames:
    image = mh.imread(f, as_grey=True) 
    surf_features.append(surf.surf(image)[:, 5:])

train_len = int(len(all_instance_filenames) * .60)
x_train_surf_features = np.concatenate(surf_features[:train_len])
x_test_surf_features  = np.concatenate(surf_features[train_len:])
y_train = all_instance_targets[:train_len] 
y_test  = all_instance_targets[train_len:] 

n_clusters = 300 
estimator = MiniBatchKMeans(n_clusters=n_clusters) 
estimator.fit_transform(x_train_surf_features) 

x_train = [] 
for instance in surf_features[:train_len]:
    clusters = estimator.predict(instance) 
    features = np.bincount(clusters)
    if len(features) < n_clusters:
        features = np.append(features, np.zeros((1, n_clusters-len(features))))
    x_train.append(features) 

x_test = [] 
for instance in surf_features[train_len:]:
    clusters = estimator.predict(instance) 
    features = np.bincount(clusters)
    if len(features) < n_clusters:
        features = np.append(features, np.zeros((1, n_clusters-len(features))))
    x_test.append(features) 

clf = LogisticRegression(C=0.001, penalty='l2') 
clf.fit(x_train, y_train) 
predictions = clf.predict(x_test) 
print(classification_report(y_test, predictions))