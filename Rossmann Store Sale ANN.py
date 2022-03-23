import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dropout, Activation, Dense
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import pandas as pd
from sklearn import preprocessing
import time

class Network():
	def __init__(self, input_shape):
		self.LEARNING_RATE = 0.0001
		self.MINIBATCH_SIZE = 64
		self.NUM_EPOCHS = 5
		self.MODEL_NAME = 'MyModel.model'
		self.input_shape = input_shape
		self.tensorboard = TensorBoard(log_dir=f"logs/{self.MODEL_NAME}-{int(time.time())}")
		self.model = self.get_model()

	def get_model(self):
		input_ = keras.Input(shape=self.input_shape, name='input')

		first_layer = Dense(64, activation='relu', name='test_name')(input_)
		first_layer = Dropout(0.2)(first_layer)

		second_layer = Dense(32, activation='relu')(first_layer)
		second_layer = Dropout(0.2)(second_layer)

		third_layer = Dense(16, activation='relu')(second_layer)
		third_layer = Dropout(0.2)(third_layer)

		output_layer = Dense(1, activation='sigmoid')(third_layer)

		model = keras.Model(inputs=[input_], outputs=[output_layer])

		model.summary()

		model.compile(loss='mse', optimizer=Adam(lr=self.LEARNING_RATE))

		return model

	def train(self, X, Y):
		self.model.fit(X, Y, batch_size = self.MINIBATCH_SIZE, 
						epochs = self.NUM_EPOCHS, shuffle=True, 
						validation_split=0, callbacks=[self.tensorboard])

	def save(self):
		self.model.save_weights(self.MODEL_NAME)



print('Starting')
# Load the training data
df = pd.read_csv('train.csv')
# Convert datetime to float
df['Date'] = pd.to_datetime(df['Date'])
min_train_date = df['Date'].min() # Remember the minimum date for the test predictions
df['Date'] = (df['Date'] - df['Date'].min())  / np.timedelta64(1,'D')

# One-hot encode the 'StateHoliday' column
dummies = pd.get_dummies(df['StateHoliday'], drop_first = True)
new_df = []
new_df.insert(0, {'0': 1, 'a': 0, 'b': 0, 'c': 0})
pd.concat([pd.DataFrame(new_df), dummies], ignore_index=True)
for i in ['0', 'a', 'b', 'c']:
	df[i] = dummies[i]
# And get rid of the old one
df = df.drop(['StateHoliday'], axis=1)

# Shuffle the data
df = df.sample(frac=1).reset_index(drop=True)

# Separate the data into input columns
x = df
x = x.drop(['Sales'], axis=1)
x = x.drop(['Customers'], axis=1)
x = x.values
# And targets
y = df['Sales'].values

# Rescale the data to [0,1]
input_scaler = preprocessing.MinMaxScaler()
data = input_scaler.fit_transform(x)

target_scaler = preprocessing.MinMaxScaler()
target = target_scaler.fit_transform(np.reshape(y,(-1,1)))

# Init the model
model = Network(input_shape=(data.shape[1],))

# Train the model
model.train(data, target)

# Save the model 
# model.save()

# Load the test data just the same way as train data
df_test = pd.read_csv('test.csv')
# Convert datetime to float
df_test['Date'] = pd.to_datetime(df_test['Date'])
# Here we use 'min_train_date' to match the training data
df_test['Date'] = (df_test['Date'] - min_train_date)  / np.timedelta64(1,'D')

# One-hot encode the 'StateHoliday' column
dummies_test = pd.get_dummies(df_test['StateHoliday'])
new_df_test = []
new_df_test.insert(0, {'0': 1, 'a': 0})
pd.concat([pd.DataFrame(new_df_test), dummies_test], ignore_index=True)
for i in ['0', 'a']:
	df_test[i] = dummies_test[i]

for i in ['b', 'c']: # The test set doesn't have any 'b's and 'c's
	df_test[i] = 0

# And get rid of the old one
df_test = df_test.drop(['StateHoliday'], axis=1)

# Separate the ID column since we don't predict based on it
df_test = df_test.drop(['Id'], axis=1)

# Prepare the test data
x_test = df_test.values
data_test = input_scaler.fit_transform(x_test)

# Predict on the scaled test data
scaled_prediction = model.model.predict(data_test)
# Scale the data back
prediction = target_scaler.inverse_transform(scaled_prediction)

# Convert it to pandas dataframe
submission = pd.DataFrame(prediction)
# Index from 1 like in 'sample_submission.csv'
submission.index += 1
# Rename the column
submission.rename(columns={0: 'Sales'}, inplace=True)

# Save the final result
submission.to_csv('submission.csv')
