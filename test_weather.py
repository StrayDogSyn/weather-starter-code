#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print("=== Testing Weather Predictor ===")

# Load data
print("Loading data...")
data = pd.read_csv('weather_history.csv')
print(f"Loaded {len(data)} rows")
print(data.head())

# Test the WeatherPredictor class
from weather_predictor_starter import WeatherPredictor

print("\nCreating predictor...")
predictor = WeatherPredictor(num_days=3)

print("Converting dates...")
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date')

print("Training model...")
predictor.train(data)

print("Making prediction...")
prediction = predictor.predict_tomorrow(data)

print(f"Prediction: {prediction:.1f}°F")
print("Done!")
