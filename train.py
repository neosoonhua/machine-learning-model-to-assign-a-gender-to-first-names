import re

def text_cleaning_and_split(text):
    #Clean
    text = re.sub(r"[^a-zA-Z\d]", " ", text) #Remove special Charecters
    text = re.sub(' +', ' ', text) #Remove Extra Spaces
    text = text.strip() # remove spaces at the beginning and at the end of string
    
    #Split
    text = " ".join(text)

    return text


'''
Baseline model at https://realpython.com/python-keras-text-classification/#defining-a-baseline-model.
'''

import pandas as pd
import numpy as np

# df = pd.read_csv("Data/name_gender_10_rows.csv")
df = pd.read_csv("Data/name_gender.csv")
# print(df['name'])
df['name'] = df['name'].apply(text_cleaning_and_split)
# print(df.name)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=0, lowercase=False, token_pattern = r"(?u)\b\w+\b") #token_pattern = r"(?u)\b\w+\b" to include 1 letter words.
vectorizer.fit(df.name)
vectorizer.vocabulary_
vectorizer.transform(df.name).toarray()

from sklearn.model_selection import train_test_split

sentences = df.name.values
y = df.gender.values

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)
# print(sentences_test)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(token_pattern = r"(?u)\b\w+\b") #token_pattern = r"(?u)\b\w+\b" to include 1 letter words.
vectorizer.fit(sentences_train)

x_train = vectorizer.transform(sentences_train)
x_test  = vectorizer.transform(sentences_test)
# print(x_test)
x_train

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))
score = model.score(x_test, y_test)

print("Accuracy for test set:", score)

#### RandomTreeClassifier hyperparameter tuning by RandomizedSearchCV from https://www.kaggle.com/code/mohitsital/random-forest-hyperparameter-tuning
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
# n_estimators = [int(x) for x in range(200,2000,900)]
# Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in range(10,110,50)]
max_depth.append(None)
# Minimum number of samples required to split a node
# min_samples_split = [2, 10]
# Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 4]
# Method of selecting samples for training each tree
# bootstrap = [True, False]
# Create the random grid
random_grid = {#n_estimators': n_estimators,
               #'max_features': max_features,
                'max_depth': max_depth,
               # 'min_samples_split': min_samples_split,
               # 'min_samples_leaf': min_samples_leaf,
               # 'bootstrap': bootstrap
               }
print(random_grid)

#### Training parameter grid by RandomizedSearchCV
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 2-fold cross validation, 
# search across 2 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 2, cv = 2, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(x_train, y_train)

#### Best parameters
rf_random.best_params_

#### model accuracy
from sklearn import metrics

def evaluate(model, test_features, test_labels):
    y_pred = model.predict(test_features)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print (accuracy)
    print(confusion_matrix(y_test,y_pred))
    

best_random = rf_random.best_estimator_
evaluate(best_random, x_test, y_test)

import pickle
model_filename = "model4a1_fewer_combinations.pkl"
with open(model_filename, 'wb') as file:
    pickle.dump((vectorizer, x_train, model), file)

score = best_random.score(x_test, y_test)

print("Accuracy for test set:", score)
