from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
import pandas as pd
import tqdm.auto

tqdm.tqdm = tqdm.auto.tqdm

#prepare data
data = pd.read_csv('train.csv')
data_c = data[['Pclass', 'Age', 'Sex', 'Survived']]
data_c = data_c.dropna()
data_c.replace('female', 0, inplace=True)
data_c.replace('male', 1, inplace=True)

pclass_in = np.array(data_c["Pclass"], dtype='float')
age_in = np.array(data_c['Age'], dtype='float')
sex_in = np.array(data_c['Sex'], dtype='float')

age_in = age_in/80

data_in = np.array([23, 0, 1])

for i, c in enumerate(pclass_in):
    example = np.array([age_in[i], sex_in[i], pclass_in[i]], dtype='float')
    data_in = np.vstack((data_in, example))


data_out_f = [[1],]
data_out = np.array(data_c['Survived'], dtype='float')

for i in data_out:
    example = np.array(i, dtype='float')
    data_out_f = np.vstack((data_out_f, example))


#create layers
l0 = tf.keras.layers.Dense(units=16, activation='relu', input_shape=[3])
l1 = tf.keras.layers.Dense(units=16, activation='relu')
l3 = tf.keras.layers.Dense(units=1, activation='sigmoid')



#create model
model = tf.keras.Sequential([l0,l1,l3])
model.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

history = model.fit(data_in, data_out_f, epochs=47)

while True:
    x = input('''Input your age(<=80), sex(0-female, 1-male), pclass(1, 2 or 3) as an example:
    35 1 1
    ''')
    x = x.split(' ')
    age = float(x[0])/80
    sex = float(x[1])
    pclass = float(x[2])
    result = model.predict([[age, sex, pclass]])
    print("Chance that you stay alive on Titanik = {:.1f}%".format(result[0][0]*100))




