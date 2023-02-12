import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# Load data relationally and remove dupes
data = pd.read_csv("emails.csv")
data.drop_duplicates(inplace=True)

# Download stopwords package for language processing
#nltk.download("stopwords")

# Tidy the text and return the stop words
def process_sw(text):
    not_punct = [char for char in text if char not in string.punctuation]
    not_punct = ''.join(not_punct)
    tidy = [word for word in not_punct.split() if word.lower()]
    return tidy

# Convert text to matrix token counts
message = CountVectorizer(analyzer=process_sw).fit_transform(data['text'])
# create train and test splits
xtrain, xtest, ytrain, ytest = train_test_split(message,data['spam'], test_size=0.20, random_state=0)
# Train the multinomial naive bayes - classifies discrete features
classifier = MultinomialNB().fit(xtrain, ytrain)
# Evaluate the model
pred = classifier.predict(xtrain)
print(classification_report(ytrain, pred))
print("Confusion Matrix: \n", confusion_matrix(ytrain, pred))
print("Accuracy:", "{:.2f}%".format((accuracy_score(ytrain, pred)*100)))
