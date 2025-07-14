#!/usr/bin/env python3
"""
Weather Predictor Module
========================

A reusable weather prediction class that can be integrated into larger applications.
Uses linear regression to predict tomorrow's temperature based on historical data.

Example usage:
    from weather_predictor import WeatherPredictor
    
    # Create predictor
    predictor = WeatherPredictor(num_days=3)
    
    # Train on your data
    predictor.train(data)
    
    # Make predictions
    tomorrow_temp = predictor.predict_tomorrow(data)
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
    
    This class can be used to predict tomorrow's temperature based on
    historical temperature data using a sliding window approach.
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
        self.training_stats = {}
        
    def prepare_features(self, data):
        """
        Prepare features and targets from temperature data.
        
        Creates a sliding window of past temperatures as features and
        the next day's temperature as the target.
        
        Args:
            data (pd.DataFrame): DataFrame with 'temperature' column
            
        Returns:
            tuple: (features, targets) as numpy arrays
        """
        if 'temperature' not in data.columns:
            raise ValueError("Data must contain a 'temperature' column")
        
        temperatures = data['temperature'].values
        
        if len(temperatures) < self.num_days + 1:
            raise ValueError(f"Need at least {self.num_days + 1} days of data")
        
        features = []
        targets = []
        
        # Create sliding window features
        for i in range(self.num_days, len(temperatures)):
            # Past N days as features
            past_days = temperatures[i-self.num_days:i]
            features.append(past_days)
            
            # Current day as target
            targets.append(temperatures[i])
        
        return np.array(features), np.array(targets)
    
    def train(self, data):
        """
        Train the model on historical weather data.
        
        Args:
            data (pd.DataFrame): DataFrame with 'temperature' column
            
        Raises:
            ValueError: If insufficient data for training
        """
        X, y = self.prepare_features(data)
        
        if len(X) < 10:
            raise ValueError(f"Need at least {self.num_days + 10} days of data for training")
        
        # Split data for training and validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate performance
        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Store training statistics
        self.training_stats = {
            'r2_score': r2,
            'rmse': rmse,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        self.is_trained = True
        
    def predict_tomorrow(self, data):
        """
        Predict tomorrow's temperature based on recent days.
        
        Args:
            data (pd.DataFrame): DataFrame with recent temperature data
            
        Returns:
            float: Predicted temperature for tomorrow
            
        Raises:
            ValueError: If model not trained or insufficient recent data
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if len(data) < self.num_days:
            raise ValueError(f"Need at least {self.num_days} days of recent data")
        
        # Get the last N days of temperature data
        recent_temps = data['temperature'].tail(self.num_days).values
        
        # Make prediction
        prediction = self.model.predict([recent_temps])[0]
        
        return prediction
    
    def predict_multiple_days(self, data, days_ahead=1):
        """
        Predict multiple days ahead (experimental).
        
        Note: Accuracy decreases significantly for predictions beyond 1-2 days.
        
        Args:
            data (pd.DataFrame): DataFrame with recent temperature data
            days_ahead (int): Number of days to predict ahead
            
        Returns:
            list: List of predicted temperatures
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        current_data = data.copy()
        
        for _ in range(days_ahead):
            # Predict next day
            pred = self.predict_tomorrow(current_data)
            predictions.append(pred)
            
            # Add prediction to data for next iteration
            next_date = current_data.index[-1] + pd.Timedelta(days=1)
            new_row = pd.DataFrame({'temperature': [pred]}, index=[next_date])
            current_data = pd.concat([current_data, new_row])
        
        return predictions
    
    def analyze_trend(self, recent_temps):
        """
        Analyze the trend in recent temperatures.
        
        Args:
            recent_temps (array-like): Array of recent temperatures
            
        Returns:
            dict: Dictionary with trend analysis
        """
        if len(recent_temps) < 2:
            return {'trend': 'insufficient_data', 'change': 0, 'direction': 'unknown'}
        
        # Calculate daily changes
        changes = np.diff(recent_temps)
        avg_change = np.mean(changes)
        
        # Determine trend
        if avg_change > 0.5:
            trend = 'rising'
            direction = 'warmer'
        elif avg_change < -0.5:
            trend = 'falling'
            direction = 'cooler'
        else:
            trend = 'stable'
            direction = 'similar'
        
        return {
            'trend': trend,
            'direction': direction,
            'avg_change': avg_change,
            'total_change': recent_temps[-1] - recent_temps[0],
            'recent_temps': recent_temps
        }
    
    def get_model_info(self):
        """
        Get information about the trained model.
        
        Returns:
            dict: Model information and training statistics
        """
        if not self.is_trained:
            return {'trained': False}
        
        return {
            'trained': True,
            'num_days': self.num_days,
            'model_type': 'LinearRegression',
            'training_stats': self.training_stats,
            'coefficients': self.model.coef_.tolist(),
            'intercept': self.model.intercept_
        }
    
    def save_model(self, filename):
        """
        Save the trained model to a file.
        
        Args:
            filename (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        import pickle
        
        model_data = {
            'model': self.model,
            'num_days': self.num_days,
            'training_stats': self.training_stats,
            'is_trained': self.is_trained
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filename):
        """
        Load a trained model from a file.
        
        Args:
            filename (str): Path to the saved model
        """
        import pickle
        
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.num_days = model_data['num_days']
        self.training_stats = model_data['training_stats']
        self.is_trained = model_data['is_trained'] 