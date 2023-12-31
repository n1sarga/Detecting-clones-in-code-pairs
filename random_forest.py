# -*- coding: utf-8 -*-
"""Random Forest

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Um74flHtObpyI9yFzb7ssPpyHav46yjI
"""

# Importing Important Libraries

import pandas as pd

# Installing datasets library for loading big-clone-bench dataset

!pip install datasets

# Loading the Big-Clone-Bench dataset

from datasets import load_dataset
dataset = load_dataset("code_x_glue_cc_clone_detection_big_clone_bench")

# Printing the keys in the dataset

print(dataset.keys())

# Copying the model training data from the dataset into a dataframe

df = pd.DataFrame(dataset['train'])

df = df.head(50000)

# Printing the dataframe

df.head()

# Dropping the unnecessary columns

column = ['id', 'id1', 'id2']

df = df.drop(column, axis = 1)

# Converting the boolean labels into 0 and 1

df['label'] = df['label'].astype(int)

# Printing the dataframe after deleting the columns

df.head()

# Printing the size of the dataframe

df.shape

# Checking if there are any null values [if null values exist, further processing is required]

df.isnull().sum().sort_values(ascending=False)

# Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

code_snippets = (df['func1'] + ' ' + df['func2']).tolist()

labels = df['label'].values

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(code_snippets)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))