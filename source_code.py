import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

train_data = pd.read_csv("https://raw.githubusercontent.com/gokulgkn/test/master/WIPRO_TrainData.csv")
train_set = train_data.iloc[:,4:5].values

sc = MinMaxScaler(feature_range=(0,1))
train_set_scaled = sc.fit_transform(train_set)

x_train = []
y_train = []

for i in range(60, 4729):
    x_train.append(train_set_scaled[i-60:i, 0])
    y_train.append(train_set_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()

model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 100, batch_size = 32)

test_data = pd.read_csv("https://raw.githubusercontent.com/gokulgkn/test/master/WIPRO_TestData.csv")
test_set = test_data.iloc[:,4:5].values

dataset = pd.concat((train_data['Open'], test_data['Open']), axis = 0)

inputs = dataset[len(dataset) - len(test_data) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

x_test = []
for i in range(60,428):
    x_test.append(inputs[i-60:i,0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction = model.predict(x_test)
prediction = sc.inverse_transform(prediction)

plt.plot(test_set, color = 'green', label = 'WIPRO Stock Price')
plt.plot(prediction, color = 'red', label = 'WIPRO Predicted Stock Price')
plt.title('WIPRO Stock Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()