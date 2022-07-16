import re

def text_cleaning_and_split(text):
    #Clean
    text = re.sub(r"[^a-zA-Z\d]", " ", text) #Remove special Charecters
    text = re.sub(' +', ' ', text) #Remove Extra Spaces
    text = text.strip() # remove spaces at the beginning and at the end of string
    
    #Split
    text = " ".join(text)

    return text


import pandas as pd
import pickle
model_filename = "model.pkl"
with open(model_filename, 'rb') as file:
    vectorizer, X_train, model = pickle.load(file)
    
# make predictions
unseen_names = pd.read_csv("Data/unseen_names.csv")
# print(unseen_names)
unseen_names_cleaned = unseen_names['name'].apply(text_cleaning_and_split)
X_unseen  = vectorizer.transform(unseen_names_cleaned)
# print(X_unseen)
y_unseen = unseen_names['gender']

score_unseen = model.score(X_unseen, y_unseen)
print("Accuracy for unseen set:", score_unseen)

y_pred_unseen = model.predict(X_unseen)
# print(y_pred_unseen)