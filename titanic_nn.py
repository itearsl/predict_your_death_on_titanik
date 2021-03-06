

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Sequential 
import tensorflow as tf 

epochs = 300
val_loss = []
loss = []
acc = []
val_acc = []
prec = []
val_prec = []
k = 3

def get_model():
  model = Sequential([
    Dense(10, activation='relu', input_shape = (10,)),
    Dropout(0.2),
    Dense(7, activation='relu'),
    Dropout(0.2),
    Dense(5, activation='relu'),
    Dense(1, activation='sigmoid'),
  ])
  model.compile(optimizer='adam',
                metrics=['Precision', 'acc'],
                loss='binary_crossentropy')
  return model

data = pd.read_csv('train.csv')
data.drop(['Ticket', 'Cabin', 'PassengerId', 'Name'], axis=1, inplace=True)
data = data.sample(frac=1)


mean_age = data['Age'].mean()
data['Age'].fillna(mean_age, inplace=True)

X_data = data.drop(['Survived'], axis=1)
y_data = data['Survived']


X_data = pd.get_dummies(X_data)

X_data = X_data.to_numpy()
y_data = y_data.to_numpy()

X_test = X_data[:200]
y_test = y_data[:200]

X_data = X_data[200:]
y_data = y_data[200:]

pack_size = len(X_data) // k

for i in range(0, k):
  val_x = X_data[i*pack_size:(i+1)*pack_size]
  val_y = y_data[i*pack_size:(i+1)*pack_size]
  train_x = np.concatenate([
    X_data[:i*pack_size],
    X_data[(i+1)*pack_size:]
  ], axis=0)
  train_y = np.concatenate([
    y_data[:i*pack_size],
    y_data[(i+1)*pack_size:]
  ], axis=0)

  model = get_model()
  history = model.fit(train_x, train_y, epochs=epochs, validation_data=(val_x, val_y))
  history = history.history
  loss.append(history['loss'])
  val_loss.append(history['val_loss'])
  acc.append(history['acc'])
  val_acc.append(history['val_acc'])
  prec.append(history['precision'])
  val_prec.append(history['val_precision'])

average_loss = [np.mean([x[i] for x in loss]) for i in range(epochs)]
average_acc = [np.mean([x[i] for x in acc]) for i in range(epochs)]
average_prec = [np.mean([x[i] for x in prec]) for i in range(epochs)]

average_val_loss = [np.mean([x[i] for x in val_loss]) for i in range(epochs)]
average_val_acc = [np.mean([x[i] for x in val_acc]) for i in range(epochs)]
average_val_prec = [np.mean([x[i] for x in val_prec]) for i in range(epochs)]

steps = range(1, epochs+1)
plt.figure(figsize=(10, 10))
plt.plot(steps, average_loss)
plt.plot(steps, average_val_loss)
plt.legend(['loss', 'val_loss'])
plt.figure(figsize=(10, 10))
plt.plot(steps, average_acc)
plt.plot(steps, average_val_acc)
plt.legend(['acc', 'val_acc'])
plt.figure(figsize=(10, 10))
plt.plot(steps, average_prec)
plt.plot(steps, average_val_prec)
plt.legend(['prec', 'val_prec'])
plt.show()

model = get_model()
history_f = model.fit(X_data, y_data, epochs=epochs)
history_f = history_f.history

plt.figure(figsize=(10, 10))
plt.plot(steps, history_f['precision'])
plt.show()

final = model.evaluate(X_test, y_test)