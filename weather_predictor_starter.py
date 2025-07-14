#!/usr/bin/env python3
"""
Weather Predictor Starter Code
==============================

Complete the TODOs below to build your own weather prediction model!

This is a hands-on exercise to learn how machine learning can predict
tomorrow's weather based on historical temperature data.
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
    
    Your task: Complete the methods below to make weather predictions!
    """
    
    def __init__(self, num_days=3):
        """
        Initialize the weather predictor.
        
        Args:
            num_days (int): Number of past days to use for prediction
        """
        self.num_days = num_days
        self.model = LinearRegression()
        self.is_trained = False
        
    def prepare_features(self, data):
        """
        TODO: Prepare features and targets from temperature data.
        
        Hint: You need to create a sliding window of past temperatures.
        - Features: past N days of temperatures
        - Targets: the next day's temperature
        
        Args:
            data (pd.DataFrame): DataFrame with 'date' and 'temperature' columns
            
        Returns:
            tuple: (features, targets) as numpy arrays
        """
        temperatures = data['temperature'].values
        
        features = []
        targets = []
        
        # TODO: Create sliding window of past N days
        # Loop through the data starting from index self.num_days
        for i in range(self.num_days, len(temperatures)):
            # TODO: Extract past N days as features
            past_days = None  # Replace with: temperatures[i-self.num_days:i]
            features.append(past_days)
            
            # TODO: Current day is the target
            targets.append(None)  # Replace with: temperatures[i]
        
        return np.array(features), np.array(targets)
    
    def train(self, data):
        """
        TODO: Train the model on historical weather data.
        
        Steps:
        1. Prepare features and targets
        2. Split data for training and testing
        3. Fit the model
        4. Evaluate performance
        
        Args:
            data (pd.DataFrame): DataFrame with 'date' and 'temperature' columns
        """
        print("Training the model...")
        
        # TODO: Prepare features and targets
        X, y = None, None  # Replace with: self.prepare_features(data)
        
        if len(X) < 10:
            raise ValueError(f"Not enough data for training. Need at least {self.num_days + 10} days.")
        
        # TODO: Split data for training and validation
        X_train, X_test, y_train, y_test = None, None, None, None
        # Hint: Use train_test_split(X, y, test_size=0.2, random_state=42)
        
        # TODO: Train the model
        # Hint: Use self.model.fit(X_train, y_train)
        
        # TODO: Evaluate the model
        y_pred = None  # Replace with: self.model.predict(X_test)
        r2 = None      # Replace with: r2_score(y_test, y_pred)
        rmse = None    # Replace with: np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Done! The model has learned the patterns.")
        print(f"Model performance - R² score: {r2:.3f}, RMSE: {rmse:.1f}°F")
        
        self.is_trained = True
        
    def predict_tomorrow(self, data):
        """
        TODO: Predict tomorrow's temperature based on recent days.
        
        Steps:
        1. Get the last N days of temperature data
        2. Use the trained model to make a prediction
        
        Args:
            data (pd.DataFrame): DataFrame with recent temperature data
            
        Returns:
            float: Predicted temperature for tomorrow
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        
        # TODO: Get the last N days of temperature data
        recent_temps = None  # Replace with: data['temperature'].tail(self.num_days).values
        
        # TODO: Make prediction
        prediction = None  # Replace with: self.model.predict([recent_temps])[0]
        
        return prediction
    
    def analyze_trend(self, recent_temps):
        """
        TODO: Analyze the trend in recent temperatures.
        
        Calculate if temperatures are rising, falling, or stable.
        
        Args:
            recent_temps (array): Array of recent temperatures
            
        Returns:
            str: "rising", "falling", or "stable"
        """
        if len(recent_temps) < 2:
            return "Not enough data for trend analysis"
        
        # TODO: Calculate the average change between consecutive days
        changes = None      # Replace with: np.diff(recent_temps)
        avg_change = None   # Replace with: np.mean(changes)
        
        # TODO: Determine trend based on average change
        if avg_change > 0.5:
            return "rising"
        elif avg_change < -0.5:
            return "falling"
        else:
            return "stable"

def main():
    """
    Main function - complete the TODOs to make it work!
    """
    print("=== Weather Predictor Student Exercise ===\n")
    
    # Load weather data
    try:
        # TODO: Load the CSV file
        data = None  # Replace with: pd.read_csv('weather_history.csv')
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
    
    # TODO: Create and train the weather predictor
    # Try different values for num_days (2, 3, 5, 7) to see how it affects predictions!
    predictor = None  # Replace with: WeatherPredictor(num_days=3)
    
    try:
        # TODO: Train the model
        # Hint: predictor.train(data)
        
        # TODO: Make a prediction for tomorrow
        prediction = None  # Replace with: predictor.predict_tomorrow(data)
        
        # TODO: Analyze trend
        recent_temps = data['temperature'].tail(3).values
        trend = None  # Replace with: predictor.analyze_trend(recent_temps)
        
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
        print("Check your TODO implementations above!")
        return

# TODO: Experiment with different num_days values!
# Try changing the num_days parameter in WeatherPredictor(num_days=X)
# What happens when you use:
# - num_days=2: Only 2 days of history
# - num_days=5: 5 days of history  
# - num_days=7: A full week of history
# Which gives the best predictions?

if __name__ == "__main__":
    main() 