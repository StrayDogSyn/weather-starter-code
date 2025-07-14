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
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('weather_predictor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
        self.training_stats = {}
        self.feature_names = [f"temp_minus_{i}_days" for i in range(num_days, 0, -1)]
        
        logger.info(f"Initialized WeatherPredictor with {num_days} days lookback")
        
    def prepare_features(self, data):
        """
        Prepare features and targets from temperature data.
        
        Args:
            data (pd.DataFrame): DataFrame with 'date' and 'temperature' columns
            
        Returns:
            tuple: (features, targets) where features are past N days and targets are next day
        """
        if 'temperature' not in data.columns:
            raise ValueError("Data must contain a 'temperature' column")
            
        temperatures = data['temperature'].values
        
        if len(temperatures) < self.num_days + 1:
            raise ValueError(f"Need at least {self.num_days + 1} days of data, got {len(temperatures)}")
        
        # Create features: sliding window of past N days
        features = []
        targets = []
        
        for i in range(self.num_days, len(temperatures)):
            # Take the past N days as features
            past_days = temperatures[i-self.num_days:i]
            features.append(past_days)
            
            # The current day is the target
            targets.append(temperatures[i])
        
        logger.info(f"Prepared {len(features)} training examples from {len(temperatures)} temperature readings")
        return np.array(features), np.array(targets)
    
    def train(self, data):
        """
        Train the model on historical weather data.
        
        Args:
            data (pd.DataFrame): DataFrame with 'date' and 'temperature' columns
        """
        logger.info("Starting model training...")
        
        # Prepare features and targets
        X, y = self.prepare_features(data)
        
        if len(X) < 10:
            raise ValueError(f"Not enough data for training. Need at least {self.num_days + 10} days, got {len(X)} examples.")
        
        # Split data for training and validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Store training statistics
        self.training_stats = {
            'r2_score': r2,
            'rmse': rmse,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_importance': dict(zip(self.feature_names, self.model.coef_)),
            'intercept': self.model.intercept_
        }
        
        logger.info(f"Model training complete:")
        logger.info(f"  - Training samples: {len(X_train)}")
        logger.info(f"  - Test samples: {len(X_test)}")
        logger.info(f"  - R² score: {r2:.3f}")
        logger.info(f"  - RMSE: {rmse:.1f}°F")
        
        print("Training the model...")
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
        
        if len(recent_temps) < self.num_days:
            raise ValueError(f"Need at least {self.num_days} recent temperatures, got {len(recent_temps)}")
        
        # Make prediction
        prediction = self.model.predict([recent_temps])[0]
        
        # Log the prediction details
        logger.info(f"Prediction made:")
        logger.info(f"  - Input temperatures: {recent_temps}")
        logger.info(f"  - Predicted temperature: {prediction:.1f}°F")
        
        return prediction
    
    def analyze_trend(self, recent_temps):
        """
        Analyze the trend in recent temperatures.
        
        Args:
            recent_temps (array): Array of recent temperatures
            
        Returns:
            dict: Detailed trend analysis
        """
        if len(recent_temps) < 2:
            return {
                'trend': 'insufficient_data',
                'description': 'Not enough data for trend analysis',
                'avg_change': 0,
                'total_change': 0
            }
        
        # Calculate the average change
        changes = np.diff(recent_temps)
        avg_change = np.mean(changes)
        total_change = recent_temps[-1] - recent_temps[0]
        
        # Determine trend
        if avg_change > 0.5:
            trend = "rising"
            description = f"Rising trend: avg {avg_change:.1f}°F/day"
        elif avg_change < -0.5:
            trend = "falling"
            description = f"Falling trend: avg {avg_change:.1f}°F/day"
        else:
            trend = "stable"
            description = f"Stable trend: avg {avg_change:.1f}°F/day"
        
        return {
            'trend': trend,
            'description': description,
            'avg_change': avg_change,
            'total_change': total_change,
            'daily_changes': changes.tolist()
        }
    
    def explain_prediction(self, recent_temps, prediction):
        """
        Explain how the model made its prediction in understandable terms.
        
        Args:
            recent_temps (array): Recent temperature inputs
            prediction (float): Model prediction
            
        Returns:
            str: Detailed explanation of prediction
        """
        if not self.is_trained:
            return "Model not trained"
        
        # Calculate contribution of each day
        contributions = []
        total_contribution = 0
        
        for i, (temp, coef) in enumerate(zip(recent_temps, self.model.coef_)):
            contrib = temp * coef
            total_contribution += contrib
            
            # Explain what the coefficient means and WHY
            if coef > 0:
                influence = "pushes prediction UP"
                strength = "strongly" if abs(coef) > 0.5 else "moderately" if abs(coef) > 0.2 else "weakly"
                reason = "the model learned that warmer temperatures on this day usually lead to warmer tomorrows"
            else:
                influence = "pushes prediction DOWN"
                strength = "strongly" if abs(coef) > 0.5 else "moderately" if abs(coef) > 0.2 else "weakly"
                reason = "the model learned that warmer temperatures on this day usually lead to cooler tomorrows"
            
            day_name = f"{i+1} days ago" if i > 0 else "yesterday"
            contributions.append(f"  {day_name}: {temp:.1f}°F × {coef:.3f} = {contrib:.1f}°F")
            contributions.append(f"    → This day {strength} {influence} the prediction")
            contributions.append(f"    → WHY? Because {reason}")
        
        intercept_contrib = self.model.intercept_
        
        explanation = f"""
🤖 HOW THE MODEL THINKS:

The model learned patterns from {self.training_stats['train_samples']} historical examples.
It discovered that each of the past {self.num_days} days influences tomorrow's temperature differently.

📊 PREDICTION CALCULATION:
{chr(10).join(contributions)}

🏠 Baseline temperature: {intercept_contrib:.1f}°F
   → This is the "average" temperature the model expects

🧮 MATH SUMMARY:
   Total from recent days: {total_contribution:.1f}°F
   + Baseline: {intercept_contrib:.1f}°F
   = Final prediction: {prediction:.1f}°F

💡 WHAT THE COEFFICIENTS ACTUALLY MEAN:

🔍 HOW THE MODEL DISCOVERED THESE PATTERNS:
The model analyzed {self.training_stats['train_samples']} historical examples and found:
"When the temperature X days ago was high, what usually happened to tomorrow's temperature?"

📈 POSITIVE COEFFICIENTS (>0): 
- Meaning: "Warmer temperatures on this day historically led to warmer tomorrows"
- Weather reason: Could be due to persistent weather patterns (high pressure systems, 
  seasonal warming trends, etc.)

📉 NEGATIVE COEFFICIENTS (<0):
- Meaning: "Warmer temperatures on this day historically led to cooler tomorrows"  
- Weather reason: Could be due to weather fronts moving through, seasonal transitions,
  or other cyclical patterns (e.g., warm days followed by storms)

🤔 WHY THIS MAKES SENSE:
Weather isn't just "if today is warm, tomorrow will be warm." Real weather has complex patterns:
- A warm day 3 days ago might signal a weather front is coming
- A warm day yesterday might mean the pattern is continuing
- The model learned these actual relationships from real historical data

🎯 WHY PREDICTIONS MIGHT SEEM WRONG:
The model doesn't just follow trends - it learned complex patterns from history.
It might predict cooling after warm days if that's what usually happened historically
(maybe due to weather fronts, seasonal patterns, etc.).
"""
        
        return explanation

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
        logger.info(f"Loaded {len(data)} weather records from weather_history.csv")
    except FileNotFoundError:
        print("Error: weather_history.csv not found!")
        print("Please make sure the data file is in the same directory as this script.")
        logger.error("weather_history.csv not found")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        logger.error(f"Error loading data: {e}")
        return
    
    # Convert date column to datetime
    try:
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date')
    except Exception as e:
        print(f"Error processing dates: {e}")
        logger.error(f"Error processing dates: {e}")
        return
    
    # Create and train the weather predictor
    predictor = WeatherPredictor(num_days=3)
    
    try:
        predictor.train(data)
        
        # Make a prediction for tomorrow
        recent_temps = data['temperature'].tail(3).values
        prediction = predictor.predict_tomorrow(data)
        
        # Analyze trend
        trend_analysis = predictor.analyze_trend(recent_temps)
        
        # Display results
        print(f"\n{'='*50}")
        print("PREDICTION RESULTS")
        print(f"{'='*50}")
        
        print(f"\nRecent temperatures: {recent_temps}")
        print(f"Tomorrow's predicted temperature: {prediction:.1f}°F")
        print(f"Most recent temperature: {recent_temps[-1]:.1f}°F")
        
        # Compare prediction to recent temperature
        diff = prediction - recent_temps[-1]
        if abs(diff) > 0.5:
            direction = "higher" if diff > 0 else "lower"
            print(f"Prediction is {abs(diff):.1f}°F {direction} than today")
        else:
            print("Prediction is similar to today's temperature")
        
        # Show trend analysis
        print(f"\nTrend analysis: {trend_analysis['description']}")
        print(f"Temperature change over period: {trend_analysis['total_change']:.1f}°F")
        
        # Show model explanation
        print(f"\n{'='*50}")
        print("MODEL EXPLANATION")
        print(f"{'='*50}")
        explanation = predictor.explain_prediction(recent_temps, prediction)
        print(explanation)
        
        # Show additional insights
        print(f"\n{'='*50}")
        print("ADDITIONAL INSIGHTS")
        print(f"{'='*50}")
        print(f"- Average temperature in data: {data['temperature'].mean():.1f}°F")
        print(f"- Temperature range: {data['temperature'].min():.1f}°F to {data['temperature'].max():.1f}°F")
        print(f"- Model accuracy (R²): {predictor.training_stats['r2_score']:.3f}")
        print(f"- Typical error (RMSE): {predictor.training_stats['rmse']:.1f}°F")
        
        # Show feature importance
        print(f"\nFeature importance (how much each day matters):")
        for feature, importance in predictor.training_stats['feature_importance'].items():
            print(f"  - {feature}: {importance:.3f}")
        
        logger.info("Demo completed successfully")
        
    except Exception as e:
        print(f"Error during training or prediction: {e}")
        logger.error(f"Error during training or prediction: {e}")
        return

if __name__ == "__main__":
    main() 