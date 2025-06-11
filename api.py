from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

app = Flask(__name__)

MODEL_PATH = 'stock_price_model.h5'
SCALER_PATH = 'scaler.save'

model = keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

SEQ_LENGTH = 60


def preprocess_input_data(data, scaler):
    df = pd.DataFrame(data)

    df = df[['open', 'high', 'low', 'close', 'volume']]

    scaled_data = scaler.transform(df)

    return np.array([scaled_data[-SEQ_LENGTH:]])


@app.route('/predict', methods=['POST'])
def predict_future():
    try:
        data = request.json.get('data')
        print(len(data))
        if not data or len(data) < SEQ_LENGTH:
            return jsonify({'error': f'Provide at least {SEQ_LENGTH} days of data'}), 400

        X_new = preprocess_input_data(data, scaler)

        prediction = model.predict(X_new)

        prediction_rescaled = scaler.inverse_transform(
            np.concatenate((np.zeros((prediction.shape[0], 4)), prediction), axis=1)
        )[:, -1]

        prediction_rescaled[0] /= 337832

        return jsonify({'predicted_price': prediction_rescaled[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    X = np.random.random((1, 60, 5))
    predictions = model.predict(X)
    prediction_rescaled = scaler.inverse_transform(
        np.concatenate((np.zeros((predictions.shape[0], 4)), predictions), axis=1)
    )[:, -1]
    print(np.concatenate((np.zeros((predictions.shape[0], 4)), predictions), axis=1))
    print(prediction_rescaled)
    app.run(host='0.0.0.0', port=8082, debug=True)