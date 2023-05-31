import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Define the data directory and a list to hold all the data
data_dir = 'venv/data'
data = []

# Loop over each file in the directory
for filename in os.listdir(data_dir):
    if filename.endswith('.csv'):
        # Load the data from the file and append it to the list
        df = pd.read_csv(os.path.join(data_dir, filename))
        data.append(df[['Close']].values)

# Concatenate all the data into a single array
close_prices = np.concatenate(data)

# We will use past 60 days data to predict the next day
look_back = 60

# Normalize the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_prices)

# Create sequences of length `look_back` and corresponding labels
X = []
y = []
for i in range(look_back, len(scaled_data)):
    X.append(scaled_data[i-look_back:i, 0])
    y.append(scaled_data[i, 0])

# Convert X and y to numpy arrays
X, y = np.array(X), np.array(y)

# Split the data into training and testing sets
split = int(0.8 * len(X))
X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]

# Reshape X to fit the LSTM input shape [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # prediction of the next closing value

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Use the model to make predictions on the testing data
predicted_close_prices = model.predict(X_test)

# Un-normalize the predicted close prices
predicted_close_prices = scaler.inverse_transform(predicted_close_prices)

# Un-normalize the actual close prices
actual_close_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate the difference between the predicted and actual prices
difference = actual_close_prices - predicted_close_prices

# Plot the actual close prices, predicted close prices, and the difference
plt.figure(figsize=(14,5))
plt.plot(actual_close_prices, color='red', label='Real Stock Price')
plt.plot(predicted_close_prices, color='blue', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend(loc='upper left')

# Create a second y-axis to plot the difference
plt.twinx()
plt.plot(difference, color='green', label='Difference')
plt.ylabel('Difference')
plt.legend(loc='upper right')

plt.show()
