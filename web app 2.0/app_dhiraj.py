from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)

# Load the pre-trained LSTM model
model_path = os.path.join('lstm_model', 'NABIL_model.keras')
model = load_model(model_path)

# Define the path to your CSV file
csv_file_path = os.path.join('lstm_model', 'NABIL.csv')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    symbol = data['stock_name']
    days = int(data['days'])
    
    # Load the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Preprocess the data
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    
    close_prices = df['Close'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    def create_sequences(data, seq_length=60):
        x = []
        y = []
        for i in range(seq_length, len(data)):
            x.append(data[i - seq_length:i, 0])
            y.append(data[i, 0])
        return np.array(x), np.array(y)
    
    x_train, y_train = create_sequences(train_data)
    x_test, y_test = create_sequences(test_data)
    
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    # Make predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Future predictions
    last_sequence = x_train[-1]
    future_predictions = []

    for _ in range(days):
        next_price = model.predict(last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1]))
        future_predictions.append(next_price[0, 0])
        last_sequence = np.append(last_sequence[1:], next_price, axis=0)

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    future_days_range = np.arange(1, days + 1)
    future_prices_df = pd.DataFrame({
        'Day': future_days_range,
        'Predicted Price': future_predictions.flatten()
    })

    df.index = df.index.strftime('%Y-%m-%d')

    return jsonify({
        'prediction': future_prices_df.to_dict(orient='records'),
        'actual_prices': y_test_actual.flatten().tolist(),
        'predicted_prices': predictions.flatten().tolist(),
        'dates': df.index[-len(y_test):].tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
