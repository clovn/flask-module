from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

app = Flask(__name__)

# Загрузка модели и scaler
MODEL_PATH = 'stock_price_model.h5'
SCALER_PATH = 'scaler.save'

model = keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Параметры
SEQ_LENGTH = 60  # Длина последовательности


# Функция для предобработки входных данных
def preprocess_input_data(data, scaler):
    # Преобразуем данные в DataFrame
    df = pd.DataFrame(data)

    # Выбираем нужные столбцы
    df = df[['open', 'high', 'low', 'close', 'volume']]

    # Нормализуем данные
    scaled_data = scaler.transform(df)

    # Возвращаем последнюю последовательность длиной SEQ_LENGTH
    return np.array([scaled_data[-SEQ_LENGTH:]])


# Endpoint для предсказаний будущих значений
@app.route('/predict', methods=['POST'])
def predict_future():
    try:
        # Получение данных из запроса
        data = request.json.get('data')
        print(len(data))
        if not data or len(data) < SEQ_LENGTH:
            return jsonify({'error': f'Provide at least {SEQ_LENGTH} days of data'}), 400

        # Предобработка данных
        X_new = preprocess_input_data(data, scaler)

        # Предсказание
        prediction = model.predict(X_new)

        # Обратная нормализация (чтобы получить реальное значение)
        prediction_rescaled = scaler.inverse_transform(
            np.concatenate((np.zeros((prediction.shape[0], 4)), prediction), axis=1)
        )[:, -1]

        # Возвращаем результат
        return jsonify({'predicted_price': prediction_rescaled[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Запуск приложения
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=True)