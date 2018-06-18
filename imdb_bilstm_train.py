import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, BatchNormalization
from keras.datasets import imdb
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
np.random.seed(7)



max_features = 5000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 500
batch_size = 64

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
model.add(Embedding(max_features, 300, input_length=maxlen))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(64,return_sequences=True)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(64,return_sequences=True)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(64)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print(model.summary())
print('Train...\n')
checkpointer = ModelCheckpoint(filepath='model/model-{epoch:02d}.hdf5', verbose=1)
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=20,
          validation_data=[x_test, y_test],
          callbacks=[checkpointer])
