import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow import keras


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


def preprocess_data(data):
    data.dropna(inplace=True)

    data['date'] = pd.to_datetime(data['date'])

    data.sort_values(by=['Name', 'date'], inplace=True)

    data = data[['open', 'high', 'low', 'close']]

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    return scaled_data, scaler


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 3])
    return np.array(X), np.array(y)


def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=MeanSquaredError(),
                  metrics=['accuracy'])
    return model


def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    model = build_model((X_train.shape[1], X_train.shape[2]))
    history = model.fit(X_train, y_train, epochs=4, batch_size=32, validation_split=0.2, verbose=1)

    model.save(model_path)
    print(f"Модель сохранена в файл: {model_path}")

    joblib.dump(scaler, scaler_path)
    print(f"Scaler сохранен в файл: {scaler_path}")

    loss, acc = model.evaluate(X_test, y_test)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))

    predictions = model.predict(X_test)
    plt.figure(figsize=(14, 5))
    plt.plot(y_test, label='True Values')
    plt.plot(predictions, label='Predictions')
    plt.legend()
    plt.show()


model_path = "stock_price_model_ver_2_60.h5"
scaler_path = "scaler_ver_2_60.save"
file_path = "all_stocks_5yr.csv"
seq_length = 60

# Загрузка данных
data = load_data(file_path)

# Предобработка данных
scaled_data, scaler = preprocess_data(data)

# Создание временных рядов
X, y = create_sequences(scaled_data, seq_length)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Обучение и оценка модели
train_and_evaluate_model(X_train, y_train, X_test, y_test)