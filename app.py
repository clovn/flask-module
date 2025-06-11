from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

app = Flask(__name__)

MODEL_PATH = 'stock_price_model_ver_2_60.h5'
SCALER_PATH = 'scaler_ver_2_60.save'

model = keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

SEQ_LENGTH = 10


def preprocess_input_data(data, scaler):
    df = pd.DataFrame(data)

    df = df[['open', 'high', 'low', 'close']]

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
            np.concatenate((np.zeros((prediction.shape[0], 3)), prediction), axis=1)
        )[:, -1]


        return jsonify({'predicted_price': prediction_rescaled[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=True)