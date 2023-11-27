import matplotlib.pyplot as plt
import pickle

with open('training_history.pkl', 'rb') as file:
    history = pickle.load(file)

plt.plot(history['accuracy'], label='accuracy')
plt.plot(history['val_accuracy'], label='val_accuracy')
plt.plot(history['loss'], label='loss')
plt.plot(history['val_loss'], label='val_loss')
plt.legend()
plt.show()