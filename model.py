import pandas as pd
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib

# Load dataset
df = pd.read_csv('UpdatedResumeDataSet.csv')

# Preprocess the text data
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_words = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_words)

df['processed_text'] = df['Resume'].apply(preprocess_text)

# Check if Personality_Trait column exists
if 'Personality_Trait' not in df.columns:
    raise ValueError("The dataset must contain a 'Personality_Trait' column.")

label_encoder = LabelEncoder()
df['Personality_Trait'] = label_encoder.fit_transform(df['Personality_Trait'])

# Extract features and labels
X = df['processed_text']
y = df['Personality_Trait']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the textual data into numerical features using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Create a neural network model for multi-class classification
model = Sequential()
model.add(Dense(512, input_dim=5000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Output layer for multi-class classification

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train_tfidf.toarray(), y_train, epochs=20, batch_size=32, validation_data=(X_test_tfidf.toarray(), y_test))

# Save the model and the vectorizer
model.save('personality_model.h5')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Model Evaluation
loss, accuracy = model.evaluate(X_test_tfidf.toarray(), y_test)
print(f"Test Accuracy: {accuracy}")
