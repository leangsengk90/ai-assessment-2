import json
import random
import pickle
import numpy as np
import nltk
import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------
# NLP and Chatbot Preparation
# ------------------------------

def prepare_chatbot_data(intents_path='intents.json'):
    nltk.download('punkt')
    nltk.download('wordnet')
    
    lemmatizer = WordNetLemmatizer()
    intents = json.loads(open(intents_path).read())
    
    words, labels, documents = [], [], []
    ignore_chars = ['?', '!', '.', ',']

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens)
            documents.append((tokens, intent['tag']))
            if intent['tag'] not in labels:
                labels.append(intent['tag'])

    words = sorted(set(lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_chars))
    labels = sorted(set(labels))

    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(labels, open('labels.pkl', 'wb'))

    training_data = []
    output_empty = [0] * len(labels)

    for doc in documents:
        bag = [0] * len(words)
        token_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
        for i, w in enumerate(words):
            if w in token_words:
                bag[i] = 1
        output_row = output_empty[:]
        output_row[labels.index(doc[1])] = 1
        training_data.append([bag, output_row])

    random.seed(42)
    np.random.seed(42)
    random.shuffle(training_data)
    training_data = np.array(training_data, dtype=object)

    train_x = np.array(training_data[:, 0].tolist(), dtype=np.float32)
    train_y = np.array(training_data[:, 1].tolist(), dtype=np.int32)

    return train_x, train_y, len(words), len(labels)

# ------------------------------
# Build and Train Chatbot Model
# ------------------------------

def train_chatbot_model(train_x, train_y, input_dim, output_dim):
    model = Sequential([
        Dense(128, input_shape=(input_dim,), activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(output_dim, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        train_x, train_y,
        epochs=500,
        batch_size=5,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    model.save('chatbot_model.h5')
    print("✅ Chatbot model trained and saved successfully.")
    print(f"📈 Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"📊 Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

# ------------------------------
# Linear Regression Section
# ------------------------------

def run_linear_regression():
    np.random.seed(42)
    x = 2 * np.random.rand(100, 1)
    y = 4 + 3 * x + np.random.randn(100, 1)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n📉 --- Linear Regression Results ---")
    print(f"Slope: {model.coef_[0][0]:.4f}")
    print(f"Intercept: {model.intercept_[0]:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Plot results
    plt.figure(figsize=(8, 5))
    plt.scatter(x_test, y_test, color='blue', label='Actual')
    plt.plot(x_test, y_pred, color='red', label='Predicted')
    plt.title('Linear Regression Fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------------------
# Main Execution
# ------------------------------

if __name__ == '__main__':
    # Chatbot Training
    x_train, y_train, input_size, output_size = prepare_chatbot_data()
    train_chatbot_model(x_train, y_train, input_size, output_size)

    # Linear Regression Analysis
    run_linear_regression()
