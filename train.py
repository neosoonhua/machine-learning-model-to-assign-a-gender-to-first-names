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

X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)
# print(X_test)
X_train

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

import pickle
model_filename = "model.pkl"
with open(model_filename, 'wb') as file:
    pickle.dump((vectorizer, X_train, model), file)

score = model.score(X_test, y_test)

print("Accuracy for test set:", score)