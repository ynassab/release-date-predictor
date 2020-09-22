
# Predict the year a song was released based on the characteristics of its sound
# e.g. tempo, "speechiness", "acousticness", etc.
# Yahia Nassab
# September 2020
# *dataset provided by the Kaggle user Yamac Eren Ay under a Community Data License Agreement

import tensorflow as tf
from tensorflow import keras
import pandas as pd

# import data
df = pd.read_csv('data.csv', engine='python')

# temporarily reduce size of data to make training faster during testing
# df = df.sample(frac=0.1)

# reduce the number of fit parameters
df.pop('artists')
df.pop('id')
df.pop('name')
df.pop('release_date')

target = df.pop('year') # extract target parameter
dataset_size = df.shape[0] # get number of entries in dataset

# prepare to split into appropriate sets
train_size = int(0.70 * dataset_size)
test_size  = int(0.30 * dataset_size)

# convert to TF tensor
dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

# split into appropriate sets
dataset = dataset.shuffle(dataset_size).batch(1)
train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)

# initialize model
model = keras.Sequential()
# first hidden layer with a neuron for each input parameter (14), each activated by a rectified linear unit
model.add(keras.layers.Dense(14, activation='relu'))
# second hidden layer
model.add(keras.layers.Dense(14, activation='relu'))
# output a 0-rank tensor showing the predicted year
model.add(keras.layers.Dense(1))

# train model
# minimize the mean squared error (MSE) of year predictions rather than maximize the accuracy of guessing specific years
model.compile(optimizer='adam', 
			loss='mse',
			metrics=['mean_squared_error'])
model.fit(train_dataset, epochs=5)

# test for model accuracy (as measured by MSE)
loss, mse = model.evaluate(test_dataset)

print('\n\nLoss: ' + str(loss))
print('Mean squared error: ' + str(mse))

# show a few predictions
small_test_ds = test_dataset.take(20)
predictions = model.predict(small_test_ds)

# output an integer year
i = 0
while i < len(predictions):
	predictions[i] = int(predictions[i])
	i += 1

print('\nreal values\n=========================\n')
for params, year in small_test_ds:
	year = year.numpy() # convert to array
	print(year)
print('\n\npredictions\n=========================\n')
print(predictions)
