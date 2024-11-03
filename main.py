# Import necessary libraries
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load historical stock data
def load_data(stock_symbol, period='10y'):
    stock_data = yf.download(stock_symbol, period=period)
    return stock_data

# Feature engineering
def create_features(data):
    # Create moving averages as features
    data['MA1'] = data['Close']
    data['MA3'] = data['Close'].rolling(window=3).mean()  # 3-day moving average
    
    # Create a target column: 1 if price goes up, 0 if price goes down
    data['Target'] = data['Close'].shift(-1) > data['Close']
    data['Target'] = data['Target'].astype(int)
    
    # Drop NaN values created by rolling
    data = data.dropna()
    
    return data

# Prepare data for training
def prepare_data(data):
    X = data[['MA1', 'MA3']]  # Feature columns
    y = data['Target']          # Target column
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Train and evaluate model
def train_model(X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))
    
    return model

# Predict future price direction based on latest data
def predict_future_direction(model, data):
    # Use the latest available data point for prediction
    latest_data = data[['MA1', 'MA3']].iloc[-1:].values  # Select last row and convert to numpy array
    
    # Predict with the model
    prediction = model.predict(latest_data)
    
    # Interpret the prediction
    if prediction[0] == 1:
        print("The model predicts the stock price will go up.")
    else:
        print("The model predicts the stock price will go down.")
    
    return prediction[0]

# Putting it all together
if __name__ == "__main__":
    # Load and preprocess data
    stock_symbol = 'MCD'  # Example stock symbol (Apple)
    stock_data = load_data(stock_symbol)
    
    # Feature engineering
    stock_data = create_features(stock_data)
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(stock_data)
    
    # Train and evaluate model
    model = train_model(X_train, X_test, y_train, y_test)

    # Predict future day's price:
    predict_future_direction(model, stock_data)
