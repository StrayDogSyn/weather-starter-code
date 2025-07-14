#!/usr/bin/env python3
"""
Breakout Activity: Build Your Weather Predictor
===============================================

Time: 10 minutes
Format: Work in pairs
Goal: Implement a temperature predictor using the starter code

Instructions:
1. Load your weather data
2. Try different numbers of days (3, 5, 7)
3. Make a prediction and discuss if it seems reasonable
4. Bonus: Try predicting something else like humidity!
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def prepare_data_simple(df, num_days=3):
    """
    Prepare data using 'num_days' to predict next day
    
    Args:
        df: DataFrame with temperature data
        num_days: Number of past days to use for prediction
    
    Returns:
        features, targets as numpy arrays
    """
    features = []
    targets = []
    
    print(f"Preparing data using {num_days} days to predict the next day...")
    
    for i in range(num_days, len(df)):
        # TODO: Get past temperatures
        # Hint: Use df['temperature'].iloc[i-num_days:i].values
        past_temps = df['temperature'].iloc[i-num_days:i].values  # Fill this in!
        
        # TODO: Get next day temperature  
        # Hint: Use df['temperature'].iloc[i]
        next_temp = df['temperature'].iloc[i]  # Fill this in!
        
        features.append(past_temps)
        targets.append(next_temp)
    
    return np.array(features), np.array(targets)

def main():
    """
    Main breakout activity function
    """
    print("=" * 60)
    print("  BREAKOUT ACTIVITY: Build Your Weather Predictor")
    print("=" * 60)
    print()
    
    # TODO: Load your weather data
    print("📊 Step 1: Loading weather data...")
    try:
        weather_data = pd.read_csv('weather_history.csv')
        print(f"✅ Loaded {len(weather_data)} days of weather data")
        print("\nFirst few rows:")
        print(weather_data.head())
    except FileNotFoundError:
        print("❌ Error: weather_history.csv not found!")
        print("Make sure the file is in the same directory as this script.")
        return
    
    print("\n" + "="*40)
    print("🎯 YOUR TASK: Complete the following steps")
    print("="*40)
    
    # Experiment with different numbers of days
    for num_days in [3, 5, 7]:
        print(f"\n🔬 Trying with {num_days} days of history...")
        
        # TODO: Prepare the data
        try:
            X, y = prepare_data_simple(weather_data, num_days)
            print(f"   Created {len(X)} training examples")
            
            # TODO: Create and train your model
            model = LinearRegression()
            model.fit(X, y)
            print(f"   ✅ Model trained successfully!")
            
            # TODO: Make a prediction for tomorrow
            recent_temps = weather_data['temperature'].tail(num_days).values
            prediction = model.predict([recent_temps])[0]
            
            print(f"   📈 Recent {num_days} days: {recent_temps}")
            print(f"   🔮 Tomorrow's prediction: {prediction:.1f}°F")
            
            # Analyze the prediction
            if len(recent_temps) >= 2:
                trend = recent_temps[-1] - recent_temps[0]
                if trend > 1:
                    print(f"   📊 Analysis: Rising trend (+{trend:.1f}°F)")
                elif trend < -1:
                    print(f"   📊 Analysis: Falling trend ({trend:.1f}°F)")
                else:
                    print(f"   📊 Analysis: Stable temperatures")
            
        except Exception as e:
            print(f"   ❌ Error with {num_days} days: {e}")
    
    print("\n" + "="*40)
    print("🤔 DISCUSSION QUESTIONS (for pairs):")
    print("="*40)
    
    discussion_questions = [
        "Does your prediction make sense given recent weather?",
        "What happened when you used more days? Why might 30 days be worse than 7?",
        "What other information might help? (Hint: time of year, humidity, pressure)",
        "When might this simple approach fail? (Hint: seasonal changes, storms)"
    ]
    
    for i, question in enumerate(discussion_questions, 1):
        print(f"{i}. {question}")
    
    print("\n" + "="*40)
    print("🎯 BONUS CHALLENGE:")
    print("="*40)
    print("If you have extra time, try predicting humidity or another weather variable!")
    print("Just replace 'temperature' with your chosen variable in the code above.")
    
    print("\n" + "="*40)
    print("📝 REFLECTION:")
    print("="*40)
    print("Discuss with your partner:")
    print("- Which number of days worked best?")
    print("- What surprised you about the predictions?")
    print("- How would you improve this predictor?")

if __name__ == "__main__":
    main() 