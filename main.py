# Import necessary libraries
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Load historical stock data
def load_data(stock_symbol, period='10y'):
    stock_data = yf.download(stock_symbol, period=period)
    return stock_data

# Feature engineering
def create_features(data):
    # Calculate moving averages
    data['MA1'] = data['Close']
    data['MA7'] = data['Close'].rolling(window=7).mean()
    
    # Create target: 1 if price goes up the next day, 0 otherwise
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    # Drop NaN values created by rolling
    data = data.dropna()
    
    return data

# Prepare data for LSTM
def prepare_lstm_data(data, sequence_length=10):
    # Select features and scale them
    feature_columns = ['Close', 'MA1', 'MA7']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[feature_columns])
    
    # Create sequences of data
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])  # Add sequence of 10 days
        y.append(data['Target'].iloc[i + sequence_length])  # Target for next day
    
    X, y = np.array(X), np.array(y)
    
    # Split into training and testing sets
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X_train, X_test, y_train, y_test, scaler

# Build the LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification (up or down)
    
    model.compile(optimizer=Adam(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate LSTM model
def train_lstm_model(X_train, X_test, y_train, y_test):
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=35, batch_size=15, validation_data=(X_test, y_test), verbose=1)
    
    # Evaluate the model on test data
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy:.2f}')
    
    return model

# Predict future direction based on latest data
def predict_future_direction_lstm(model, data, scaler, sequence_length=10):
    # Get the last sequence of data for prediction
    latest_data = data[['Close', 'MA1', 'MA7']].iloc[-sequence_length:]
    scaled_data = scaler.transform(latest_data)  # Scale latest data
    scaled_data = scaled_data.reshape((1, sequence_length, 3))  # Reshape for model input
    
    # Predict with the LSTM model
    prediction = model.predict(scaled_data)
    if prediction[0][0] > 0.5:
        confidence = str(round(prediction[0][0]*100,1))
        print("The model predicts the stock price will go up.")
        print("Confidence level: "+ confidence + "%")
    else:
        confidence = str(round((1-prediction[0][0])*100,1))
        print("The model predicts the stock price will go down.")
        print("Confidence level: "+ confidence + "%")
    
    return prediction[0][0] 

def main(stock):
    # Putting it all together

    # Load and preprocess data
    try:
        stock_symbol = stock  # Example stock symbol 
        stock_data = load_data(stock_symbol)
    except:
        return "Failed to find stock"
    
    # Feature engineering
    stock_data = create_features(stock_data)
    
    # Prepare data for LSTM
    X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(stock_data)
    
    # Train and evaluate LSTM model
    model = train_lstm_model(X_train, X_test, y_train, y_test)
    
    # Predict future direction
    return predict_future_direction_lstm(model, stock_data, scaler)
