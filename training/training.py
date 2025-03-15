import json
import random
import pickle
import numpy as np
import nltk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents file
intents = json.loads(open('intents.json').read())

words = []
labels = []
documents = []
ignore = ['?', '!', '.', ',']

# Preprocess the intents data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in labels:
            labels.append(intent['tag'])

# Lemmatize and remove duplicates from words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore]
words = sorted(list(set(words)))

labels = sorted(list(set(labels)))

# Save words and labels for later use
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(labels, open('labels.pkl', 'wb'))

train_set = []
output_empty = [0] * len(labels)

# Create bag of words and output row
for doc in documents:
    bag = [0] * len(words)  # Initialize bag with zeros
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for i, word in enumerate(words):
        if word in word_patterns:
            bag[i] = 1
    output_row = list(output_empty)
    output_row[labels.index(doc[1])] = 1
    train_set.append([bag, output_row])

# Shuffle the training set
random.seed(42)
np.random.seed(42)
random.shuffle(train_set)
train_set = np.array(train_set, dtype=object)

train_x = np.array(train_set[:, 0].tolist(), dtype=np.float32)
train_y = np.array(train_set[:, 1].tolist(), dtype=np.int32)

# Define the model
model = Sequential(
    [
        Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
        Dropout(0.3),  # Adjusted dropout rate
        Dense(64, activation='relu'),
        Dropout(0.3),  # Adjusted dropout rate
        Dense(len(train_y[0]), activation='softmax')
    ]
)

# Compile the model with Adam optimizer
optimizer = Adam(learning_rate=0.001)  # Lowered learning rate
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Set up early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
hist = model.fit(train_x, train_y, epochs=500, batch_size=5, verbose=1, validation_split=0.2, callbacks=[early_stop])

# Save the trained model
model.save('chatbot_model.h5')
print("Model trained and saved successfully.")

# Print the final training accuracy
print(f"Final training accuracy: {hist.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {hist.history['val_accuracy'][-1]:.4f}")
