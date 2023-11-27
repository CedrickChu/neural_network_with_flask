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
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from io import BytesIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
import matplotlib

matplotlib.use('Agg')


warnings.filterwarnings('ignore')
tf.autograph.set_verbosity(0)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


# Load the trained model
model = load_model('spam_classifier_model.h5')

with open('vectorizer.joblib', 'rb') as file:
    vectorizer = joblib.load(file)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        user_input_text = request.form['user_input_text']

        user_input_numeric = vectorizer.transform([user_input_text]).toarray()

        predictions = model.predict(user_input_numeric)

        if predictions[0] > 0.5:
            result = {
                'prediction': 'The message is spam.',
                'spam_probability': f"Spam Probability: {predictions[0][0]:.4f}"
            }
        else:
            result = {
                'prediction': 'The message is not spam.',
                'spam_probability': f"Spam Probability: {predictions[0][0]:.4f}"
            }

        socketio.emit('prediction_result', result)

        return render_template('predict.html', result=result)

    return render_template('predict.html')

@app.route('/')
def dashboard():
    df = pd.read_csv('spam_dataset.csv')
    spam_count = len(df[df['label'] == 'spam'])
    ham_count = len(df[df['label'] == 'ham'])

    total_messages = len(df)
    spam_percentage = (spam_count / total_messages) * 100
    ham_percentage = (ham_count / total_messages) * 100

    labels = ['Spam', 'Regular (ham)']
    sizes = [spam_percentage, ham_percentage]
    colors = ['#ff9999', '#66b3ff']
    explode = (0.1, 0)

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Percentage of Spam Messages to Regular Messages')
    plt.axis('equal')

    img_data_pie = BytesIO()
    plt.savefig(img_data_pie, format='png')
    img_data_pie.seek(0)

    img_base64_pie = base64.b64encode(img_data_pie.getvalue()).decode('utf-8')
    
    df['number_of_characters_in_the_message'] = df['text'].apply(len)
    plt.figure(figsize=(8, 6))
    df['number_of_characters_in_the_message'].hist(bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('Number of Characters in the Message')
    plt.ylabel('Frequency')
    plt.title('Distribution of Number of Characters in Messages')

    img_data_hist = BytesIO()
    plt.savefig(img_data_hist, format='png')
    img_data_hist.seek(0)
    img_base64_hist = base64.b64encode(img_data_hist.getvalue()).decode('utf-8')
    plt.close()  

    return render_template('dashboard.html', img_base64_pie=img_base64_pie, img_base64_hist=img_base64_hist)


@app.route('/train')
def train():
    return render_template('train_data.html')



@app.route('/result')
def show_result():
    with open('training_history.pkl', 'rb') as file:
        history_list = pickle.load(file)

    last_epoch_history = history_list[-1]
    train_accuracy = last_epoch_history['accuracy'][0]  
    val_accuracy = last_epoch_history['val_accuracy'][0] 

    return render_template('result.html', train_accuracy=train_accuracy, val_accuracy=val_accuracy)

@app.route('/visualize_history')
def visualize_history():
    with open('training_history.pkl', 'rb') as file:
        history_list = pickle.load(file)

    accuracy = [acc for epoch_history in history_list for acc in epoch_history['accuracy']]
    val_accuracy = [val_acc for epoch_history in history_list for val_acc in epoch_history['val_accuracy']]
    loss = [l for epoch_history in history_list for l in epoch_history['loss']]
    val_loss = [val_l for epoch_history in history_list for val_l in epoch_history['val_loss']]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(accuracy, label='accuracy')
    plt.plot(val_accuracy, label='val_accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='loss')
    plt.plot(val_loss, label='val_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)

    img_base64 = base64.b64encode(img_data.getvalue()).decode('utf-8')

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

    train_data, test_data, train_labels, test_labels = train_test_split(
        df['text'], df['label'], test_size=0.3, random_state=100
    )

    vectorizer = CountVectorizer(max_features=400)
    train_data_numeric = vectorizer.fit_transform(train_data).toarray()
    test_data_numeric = vectorizer.transform(test_data).toarray()

    ros = RandomOverSampler(random_state=42)
    train_data_resampled, train_labels_resampled = ros.fit_resample(train_data_numeric, train_labels)

    joblib.dump(vectorizer, 'vectorizer.joblib')

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

    epochs = 40
    history_list = []

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

        history_list.append(history.history)

    with open('training_history.pkl', 'wb') as file:
        pickle.dump(history_list, file)

    test_data_resampled, test_labels_resampled = ros.fit_resample(test_data_normalized, test_labels)

    train_pred_prob = model.predict(train_data_resampled)
    test_pred_prob = model.predict(test_data_resampled)

    train_pred = (train_pred_prob > 0.5).astype(int)
    test_pred = (test_pred_prob > 0.5).astype(int)

    train_accuracy = accuracy_score(train_labels_resampled, train_pred)
    test_accuracy = accuracy_score(test_labels_resampled, test_pred)

    socketio.emit('status', {'data': 'Training completed'})
    socketio.emit('accuracy', {'train_accuracy': train_accuracy})
    socketio.emit('accuracy', {'test_accuracy': test_accuracy})

    return render_template('result.html', train_accuracy=train_accuracy, test_accuracy=test_accuracy)

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000)