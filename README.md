# Financial Time Series Forecasting using CNN

**Name:** MUHAMMED JIYAD U
**University Registration Number:** LTCR24CS074

## Overview

This project uses **signal processing (STFT)** and **deep learning (CNN)** to predict stock prices.
Financial time series data is converted into a **spectrogram**, which is then used as input to a CNN model.

---

## Features

* Stock data collection (Yahoo Finance)
* Time series normalization
* Spectrogram generation using STFT
* CNN-based prediction
* Visualization of results

---

## Outputs

* Time Series Plot
* Frequency Spectrum
* Spectrogram
* CNN Architecture
* Prediction vs Actual Graph

---

## How to Run

### 1. Clone Repository

```bash
git clone https://github.com/JIYAD42/Financial_Time_Series_Forecasting.git
cd Financial_Time_Series_Forecasting
```

### 2. Install dependencies

```bash id="h3c0r3"
pip install yfinance numpy pandas matplotlib scipy scikit-learn tensorflow
```

### 3. Run the script

```bash id="5m5czt"
python as.py
```

### 4. Output

* Graphs will appear one by one
* Model training logs will be shown in terminal
* CNN architecture will be printed

---

## Files

* `as.py` → Main code
* `README.md` → Documentation
* `Timeseriesplot.png` → Time series plot
* `Frequencyspectrum.png` → Frequency spectrum
* `Spectrogram.png` → Spectrogram
* `CNNarchitecturediagram.png` → CNN architecture diagram
* `Finalgraph.png` → Final Ouput
---

## Result

* The CNN model is able to capture **general trends** in stock prices
* Predictions are smoother compared to actual values
* High-frequency fluctuations are harder to predict due to noise

---
---

## ⭐ If you found this useful, consider giving a star!
