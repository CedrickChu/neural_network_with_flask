import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
import joblib

# Load the trained model
model = load_model('spam_classifier_model.h5')

# Load the CountVectorizer used during training
with open('vectorizer.joblib', 'rb') as file:
    vectorizer = joblib.load(file)

# User input text
user_input_text = input("Enter Message: ")

# Preprocess the user input using the loaded vectorizer
user_input_numeric = vectorizer.transform([user_input_text]).toarray()

# Make predictions
predictions = model.predict(user_input_numeric)

# Output the prediction
if predictions[0] > 0.5:
    print("The message is spam.")
    print(f"Spam Probability: {predictions[0][0]:.4f}")
else:
    print("The message is not spam.")
    print(f"Spam Probability: {predictions[0][0]:.4f}")