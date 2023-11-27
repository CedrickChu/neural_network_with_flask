from flask import Flask, render_template, request
import joblib
import pickle
import base64
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import numpy as np
import tensorflow as tf
from flask_socketio import SocketIO
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from io import BytesIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler

warnings.filterwarnings('ignore')
tf.autograph.set_verbosity(0)

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result')
def show_result():
    # Load the training history
    with open('training_history.pkl', 'rb') as file:
        history = pickle.load(file)

    # Calculate accuracy from the last epoch
    last_epoch = len(history['loss'])
    train_accuracy = history['accuracy'][last_epoch - 1]
    val_accuracy = history['val_accuracy'][last_epoch - 1]

    return render_template('result.html', train_accuracy=train_accuracy, val_accuracy=val_accuracy)

@app.route('/visualize_history')
def visualize_history():
    # Load training history from the pickle file
    with open('training_history.pkl', 'rb') as file:
        history = pickle.load(file)

    # Plot training history
    plt.plot(history['accuracy'], label='accuracy')
    plt.plot(history['val_accuracy'], label='val_accuracy')
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title('Training History')
    plt.legend()

    # Save the plot to a BytesIO object
    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)

    # Convert the BytesIO object to a base64-encoded string
    img_base64 = base64.b64encode(img_data.getvalue()).decode('utf-8')

    # Pass the base64 string to the template
    return render_template('visualize_history.html', img_base64=img_base64)


def data_generator(data, labels, batch_size):
    num_samples = len(data)
    while True:
        indices = np.random.choice(num_samples, batch_size, replace=False)
        yield data[indices], np.array(labels[indices])[:, np.newaxis]


@socketio.on('connect')
def handle_connect():
    print('Client connected')
    socketio.emit('message', {'data': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('update')
def send_update(data):
    socketio.emit('message', {'data': data})

@app.route('/run_model', methods=['POST'])
def run_model():
    df = pd.read_csv('spam_dataset.csv', skiprows=1, names=['label', 'text'])
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})

    def clean(text):
        text = text.lower()
        return text

    df['text'] = df['text'].apply(clean)

    train_data, test_data, train_labels, test_labels = train_test_split(
        df['text'], df['label'], test_size=0.3, random_state=100
    )

    vectorizer = CountVectorizer(max_features=400)
    train_data_numeric = vectorizer.fit_transform(train_data).toarray()
    test_data_numeric = vectorizer.transform(test_data).toarray()

    ros = RandomOverSampler(random_state=42)
    train_data_resampled, train_labels_resampled = ros.fit_resample(train_data_numeric, train_labels)

    # Save the vectorizer
    joblib.dump(vectorizer, 'vectorizer.joblib')

    # Normalize the data
    scaler = MinMaxScaler()
    train_data_normalized = scaler.fit_transform(train_data_resampled)
    test_data_normalized = scaler.transform(test_data_numeric)

    class_weights = {
        0: 1.0,  
        1: 1.5   
    }

    tf.random.set_seed(1234)
    model = Sequential([
        Dense(128, activation='relu', input_shape=(400,), name="L1"),
        Dropout(0.5), 
        Dense(64, activation='relu', name="L2"),
        Dropout(0.3), 
        Dense(32, activation='relu', name="L3"),
        Dropout(0.3),
        Dense(1, activation='sigmoid', name="Output"),
    ], name="my_model")

    # Adjust the learning rate
    initial_learning_rate = 0.0001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=10000, decay_rate=0.9, staircase=True
    )

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=['accuracy']
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model_checkpoint = ModelCheckpoint(
        'spam_classifier_model.h5',
        save_best_only=True,
        save_weights_only=False,  
        monitor='val_loss',     
        mode='min',              
        verbose=1
    )
    
    batch_size = 65
    steps_per_epoch = len(train_data_normalized) // batch_size

    epochs = 10

    for epoch in range(epochs):
        update_data = f'Epoch {epoch + 1}: training...'
        send_update(update_data)

        history = model.fit(
            data_generator(train_data_normalized, train_labels_resampled, batch_size),
            epochs=1,
            steps_per_epoch=steps_per_epoch,
            validation_data=(test_data_normalized, np.array(test_labels.values)[:, np.newaxis]),
            callbacks=[early_stopping, model_checkpoint],
            class_weight=class_weights
        )

    # Save the training history
    with open('training_history.pkl', 'wb') as file:
        pickle.dump(history.history, file)

    test_data_resampled, test_labels_resampled = ros.fit_resample(test_data_normalized, test_labels)

    train_pred_prob = model.predict(train_data_resampled)
    test_pred_prob = model.predict(test_data_resampled)

    train_pred = (train_pred_prob > 0.5).astype(int)
    test_pred = (test_pred_prob > 0.5).astype(int)

    train_accuracy = accuracy_score(train_labels_resampled, train_pred)
    test_accuracy = accuracy_score(test_labels_resampled, test_pred)
    
    socketio.emit('status', {'data': 'Training completed'})
    socketio.emit('accuracy', {'train_accuracy': train_accuracy})
    socketio.emit('accuracy', {'train_accuracy': test_accuracy})
    
    return render_template('result.html', train_accuracy=train_accuracy, test_accuracy=test_accuracy)
    

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', debug=True)