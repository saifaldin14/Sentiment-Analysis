import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist

from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Dense, GRU, Embedding
from tensorflow.compat.v1.keras.optimizers import Adam
from tensorflow.compat.v1.keras.preprocessing.text import Tokenizer
from tensorflow.compat.v1.keras.preprocessing.sequence import pad_sequences

import imdb
#imdb.maybe_download_and_extract()

x_train_text, y_train = imdb.load_data(train= True)
x_test_text, y_test = imdb.load_data(train= False)

data_text = x_train_text + x_test_text

#Tokenizer
num_words = 10000
tokenizer = Tokenizer(num_words = num_words)
tokenizer.fit_on_texts(data_text)

x_train_tokens = tokenizer.texts_to_sequences(x_train_text)
x_test_tokens = tokenizer.texts_to_sequences(x_test_text)

#Padding and Truncating Data
num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)

max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)

pad = 'pre'
x_train_pad = pad_sequences(x_train_tokens, maxlen= max_tokens, padding= pad, truncating = pad)
x_test_pad = pad_sequences(x_test_tokens, maxlen= max_tokens, padding= pad, truncating = pad)

#Tokinzer Inverse Map
idx = tokenizer.word_index
inverse_map = dict(zip(idx.values(), idx.keys()))


def tokens_to_string(tokens):
    # Map from tokens back to words.
    words = [inverse_map[token] for token in tokens if token != 0]

    # Concatenate all words.
    text = " ".join(words)

    return text
#Create the RNN
model = Sequential()
embedding_size = 8
model.add(Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    input_length=max_tokens,
                    name='layer_embedding'))
model.add(GRU(units=16, return_sequences=True))
model.add(GRU(units=8, return_sequences=True))
model.add(GRU(units=4))
model.add(Dense(1, activation='sigmoid'))
optimizer = Adam(lr=1e-3)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

x = np.array(x_train_pad)
y = np.array(y_train)

model.fit(x, y, validation_split=0.06, epochs=3, batch_size=64)
result = model.evaluate(x_test_pad, y_test)
print("Accuracy: {0:.2%}".format(result[1]))

