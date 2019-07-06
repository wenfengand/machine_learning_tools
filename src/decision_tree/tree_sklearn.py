import pandas as pd 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.metrics import classification_report 
from sklearn.pipeline import Pipeline 

df = pd.read_csv('../../input/ad/ad.data', header=None) 

explanatory_variable_columns = set(df.columns.values) 
explanatory_variable_columns.remove(len(df.columns.values) - 1) 
response_variable_column = df[len(df.columns.values) - 1]

y = [1 if e == 'ad.' else 0 for e in response_variable_column] 
x = df[list(explanatory_variable_columns)].copy() 
x.replace(to_replace=' *?', value=-1, regex=True, inplace=True) 
x_train, x_test, y_train, y_test = train_test_split(x,y) 

pipeline = Pipeline([
    ('clf', DecisionTreeClassifier(criterion='entropy')) 
])

parameters = {
    'clf__max_depth': (150, 155, 160),
    'clf__min_samples_split': (2,3),
    'clf__min_samples_leaf': (1,2,3)
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, 
    scoring='f1') 
grid_search.fit(x_train, y_train) 

best_parameters = grid_search.best_estimator_.get_params() 
print('Best score: %0.3f' % grid_search.best_score_) 
print('Best parameters set:') 
for param_name in sorted(parameters.keys()):
    print('t%s: %r' %(param_name, best_parameters[param_name])) 

predictions = grid_search.predict(x_test) 
print(classification_report(y_test, predictions)) 
