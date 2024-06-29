from flask import Flask, request, render_template
import joblib
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)
model = tf.keras.models.load_model('personality_model.h5')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_words = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        cv_text = request.form['cv_text']
        processed_text = preprocess_text(cv_text)
        tfidf_text = tfidf_vectorizer.transform([processed_text])
        prediction = model.predict(tfidf_text.toarray())
        predicted_class = label_encoder.inverse_transform([prediction.argmax()])[0]
        return render_template('index.html', prediction=predicted_class)

if __name__ == "__main__":
    app.run(debug=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        cv_text = request.form['cv_text']
        processed_text = preprocess_text(cv_text)
        tfidf_text = tfidf_vectorizer.transform([processed_text])
        prediction = model.predict(tfidf_text.toarray())
        predicted_class = label_encoder.inverse_transform([prediction.argmax()])[0]
        return render_template('index.html', prediction=predicted_class)

if __name__ == "__main__":
    app.run(debug=True)