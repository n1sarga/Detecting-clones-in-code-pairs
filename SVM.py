# Importing Important Libraries
import pandas as pd

'''
The following section works better with the Google Colab
So I am just commenting the following files to avoid any errors
'''
# Installing datasets library for loading big-clone-bench dataset
# pip install datasets

# Loading the Big-Clone-Bench dataset
from datasets import load_dataset
dataset = load_dataset("code_x_glue_cc_clone_detection_big_clone_bench")

# Printing the keys in the dataset
print(dataset.keys())

# Copying the model training data from the dataset into a dataframe
df = pd.DataFrame(dataset['train'])

# Adding 50K rows into the dataframe
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

# ANN
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

code_snippets = (df['func1'] + ' ' + df['func2']).tolist()

labels = df['label'].values

tokenizer = Tokenizer()
tokenizer.fit_on_texts(code_snippets)
vocab_size = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(code_snippets)

padded_sequences = pad_sequences(sequences)

X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, labels, test_size=0.2, random_state=42
)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=8, input_length=padded_sequences.shape[1]),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_split=0.1)

y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred_binary))
print("Classification Report:\n", classification_report(y_test, y_pred_binary))

# Token Matrix
token_matrix = pad_sequences(sequences)
print("Token Matrix:")
print(token_matrix)