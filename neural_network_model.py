import pandas as pd
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import callbacks, models, layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
import joblib

warnings.filterwarnings('ignore')
tf.autograph.set_verbosity(0)


def data_generator(data, labels, batch_size):
    num_samples = len(data)
    while True:
        indices = np.random.choice(num_samples, batch_size, replace=False)
        yield data[indices], np.array(labels[indices])[:, np.newaxis]

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
    0: 1.0,  # Weight for class 0 (not spam)
    1: 1.5   # Weight for class 1 (spam)
}

tf.random.set_seed(1234)
model = Sequential([
    Dense(64, activation='relu', input_shape=(400,), name="L1"),
    Dropout(0.3), 
    Dense(25, activation='relu', name="L2"),
    Dropout(0.3), 
    Dense(1, activation='sigmoid', name="Output"),
], name="my_model")

initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=10000, decay_rate=0.9, staircase=True
)

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=['accuracy']
)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

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

history = model.fit(
    data_generator(train_data_normalized, train_labels_resampled, batch_size),
    epochs=40,
    steps_per_epoch=steps_per_epoch,
    validation_data=(test_data_normalized, np.array(test_labels.values)[:, np.newaxis]),
    callbacks=[early_stopping, model_checkpoint],
    class_weight=class_weights
)

# Save the training history
with open('training_history.pkl', 'wb') as file:
    import pickle
    pickle.dump(history.history, file)


train_pred_prob = model.predict(train_data_normalized)
test_pred_prob = model.predict(test_data_normalized)

train_pred = (train_pred_prob > 0.5).astype(int)
test_pred = (test_pred_prob > 0.5).astype(int)

print('Train Accuracy - ', accuracy_score(train_labels_resampled, train_pred))
print("Test Accuracy - ", accuracy_score(test_labels, test_pred))