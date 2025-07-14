"""
Simple Weather Predictor Demo
=============================

This script demonstrates how to build a basic weather prediction model
using linear regression to forecast tomorrow's temperature based on
historical weather data.

Educational purpose: Show how ML can learn patterns from data to make predictions.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class WeatherPredictor:
    """
    A simple weather prediction model using linear regression.
    
    Uses the past N days of temperature data to predict tomorrow's temperature.
    """
    
    def __init__(self, num_days=3):
        """
        Initialize the weather predictor.
        
        Args:
            num_days (int): Number of past days to use for prediction (default: 3)
        """
        self.num_days = num_days
        self.model = LinearRegression()
        self.is_trained = False
        
    def prepare_features(self, data):
        """
        Prepare features and targets from temperature data.
        
        Args:
            data (pd.DataFrame): DataFrame with 'date' and 'temperature' columns
            
        Returns:
            tuple: (features, targets) where features are past N days and targets are next day
        """
        temperatures = data['temperature'].values
        
        # Create features: sliding window of past N days
        features = []
        targets = []
        
        for i in range(self.num_days, len(temperatures)):
            # Take the past N days as features
            past_days = temperatures[i-self.num_days:i]
            features.append(past_days)
            
            # The current day is the target
            targets.append(temperatures[i])
        
        return np.array(features), np.array(targets)
    
    def train(self, data):
        """
        Train the model on historical weather data.
        
        Args:
            data (pd.DataFrame): DataFrame with 'date' and 'temperature' columns
        """
        print("Training the model...")
        
        # Prepare features and targets
        X, y = self.prepare_features(data)
        
        if len(X) < 10:
            raise ValueError(f"Not enough data for training. Need at least {self.num_days + 10} days.")
        
        # Split data for training and validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Done! The model has learned the patterns.")
        print(f"Model performance - R² score: {r2:.3f}, RMSE: {rmse:.1f}°F")
        
        self.is_trained = True
        
    def predict_tomorrow(self, data):
        """
        Predict tomorrow's temperature based on recent days.
        
        Args:
            data (pd.DataFrame): DataFrame with recent temperature data
            
        Returns:
            float: Predicted temperature for tomorrow
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        
        # Get the last N days of temperature data
        recent_temps = data['temperature'].tail(self.num_days).values
        
        # Make prediction
        prediction = self.model.predict([recent_temps])[0]
        
        return prediction
    
    def analyze_trend(self, recent_temps):
        """
        Analyze the trend in recent temperatures.
        
        Args:
            recent_temps (array): Array of recent temperatures
            
        Returns:
            str: Description of the trend
        """
        if len(recent_temps) < 2:
            return "Not enough data for trend analysis"
        
        # Calculate the average change
        changes = np.diff(recent_temps)
        avg_change = np.mean(changes)
        
        if avg_change > 0.5:
            return "rising"
        elif avg_change < -0.5:
            return "falling"
        else:
            return "stable"

def main():
    """
    Main demo function that loads data, trains model, and makes predictions.
    """
    print("=== Simple Weather Predictor Demo ===\n")
    
    # Load weather data
    try:
        data = pd.read_csv('weather_history.csv')
        print("Our data:")
        print(data.head(10))
        print(f"\nWe have {len(data)} examples to learn from\n")
    except FileNotFoundError:
        print("Error: weather_history.csv not found!")
        print("Please make sure the data file is in the same directory as this script.")
        return
    
    # Convert date column to datetime
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date')
    
    # Create and train the weather predictor
    predictor = WeatherPredictor(num_days=3)
    
    try:
        predictor.train(data)
        
        # Make a prediction for tomorrow
        recent_temps = data['temperature'].tail(3).values
        prediction = predictor.predict_tomorrow(data)
        
        # Analyze trend
        trend = predictor.analyze_trend(recent_temps)
        
        # Display results
        print(f"\nLast 3 days: {recent_temps}")
        print(f"Tomorrow's predicted temperature: {prediction:.1f}°F")
        
        if trend == "rising":
            print("Temperatures have been rising, so prediction is higher")
        elif trend == "falling":
            print("Temperatures have been falling, so prediction is lower")
        else:
            print("Temperatures have been stable, so prediction is similar")
            
        print(f"\nTrend analysis: Temperatures are {trend}")
        
        # Show some additional insights
        print(f"\nAdditional insights:")
        print(f"- Average temperature in data: {data['temperature'].mean():.1f}°F")
        print(f"- Temperature range: {data['temperature'].min():.1f}°F to {data['temperature'].max():.1f}°F")
        print(f"- Most recent temperature: {data['temperature'].iloc[-1]:.1f}°F")
        
    except Exception as e:
        print(f"Error during training or prediction: {e}")
        return

if __name__ == "__main__":
    main() 