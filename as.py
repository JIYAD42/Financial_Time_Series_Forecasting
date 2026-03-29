import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras import models, layers
from scipy.fft import fft

# -----------------------------
# STEP 1: DATA COLLECTION
# -----------------------------
stocks = ["RELIANCE.NS", "INFY.NS", "HDFCBANK.NS"]  # removed TCS

data = yf.download(stocks, start="2018-01-01", end="2024-01-01", threads=False)

# Extract Close prices
data = data['Close']

# Drop missing values
data = data.dropna()

data.plot(title="Stock Prices Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

print(data.head())
print("Data shape:", data.shape)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

signal = data_scaled[:, 0]  # first stock

fft_vals = np.abs(fft(signal))

plt.plot(fft_vals)
plt.title("Frequency Spectrum")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()

# -----------------------------
# STEP 3: STFT + SPECTROGRAM
# -----------------------------
spectrograms = []
targets = []

window_size = 128

for i in range(len(data_scaled) - window_size - 1):
    segment = data_scaled[i:i+window_size]

    # Flatten multivariate signal
    signal = segment[:,0]

    f, t, Zxx = stft(signal, nperseg=64)
    S = np.abs(Zxx)**2

    if i==0:
        plt.pcolormesh(t, f, S)
        plt.title("Spectrogram")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar()
        plt.show()

    spectrograms.append(S)

    # Target = next time step price (first stock)
    targets.append(data_scaled[i+window_size][0])

spectrograms = np.array(spectrograms)
targets = np.array(targets)

# -----------------------------
# STEP 4: PREPARE DATA FOR CNN
# -----------------------------
spectrograms = spectrograms[..., np.newaxis]

X_train, X_test, y_train, y_test = train_test_split(
    spectrograms, targets, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 5: CNN MODEL
# -----------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=X_train.shape[1:]),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

model.summary()

model.compile(optimizer='adam', loss='mse')

# -----------------------------
# STEP 6: TRAINING
# -----------------------------
model.fit(X_train, y_train, epochs=30, batch_size=32)

# -----------------------------
# STEP 7: EVALUATION
# -----------------------------
predictions = model.predict(X_test)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predictions)

print("MSE:", mse)

# -----------------------------
# STEP 8: VISUALIZATION
# -----------------------------
pred_smooth = pd.Series(predictions.flatten()).rolling(5).mean()
plt.plot(y_test, label="Actual")
plt.plot(pred_smooth, label="Predicted")
plt.legend()
plt.title("Prediction vs Actual")
plt.show()