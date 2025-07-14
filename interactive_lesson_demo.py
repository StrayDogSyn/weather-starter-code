#!/usr/bin/env python3
"""
Interactive Weather Prediction Lesson Demo
==========================================

This script follows the lesson plan structure for teaching ML concepts
using relatable examples and step-by-step progression.

Designed for classroom use with instructor guidance.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

def lemonade_stand_example():
    """
    Demonstrate predictive modeling with the lemonade stand analogy
    from the lesson plan.
    """
    print("=== LEMONADE STAND EXAMPLE ===")
    print("Imagine you run a lemonade stand and notice patterns:")
    print()
    
    # Our lemonade stand data
    lemonade_data = [
        {"temperature": 65, "cups_sold": 8},
        {"temperature": 72, "cups_sold": 22},
        {"temperature": 85, "cups_sold": 51},
        {"temperature": 90, "cups_sold": 63},
        {"temperature": 78, "cups_sold": 28}
    ]
    
    print("Your summer observations:")
    for day in lemonade_data:
        temp = day["temperature"]
        cups = day["cups_sold"]
        if temp >= 85:
            category = "Hot day"
        elif temp >= 70:
            category = "Warm day"
        else:
            category = "Cool day"
        print(f"  {temp}°F ({category}) → Sold {cups} cups")
    
    print()
    print("You naturally start predicting:")
    print("  'Tomorrow will be 88°F, so I should make extra lemonade!'")
    print()
    print("This is EXACTLY what a predictive model does, but more systematically!")
    print()
    
    # Show the pattern
    temps = [d["temperature"] for d in lemonade_data]
    cups = [d["cups_sold"] for d in lemonade_data]
    
    print("🤔 DISCUSSION QUESTION:")
    print("What pattern do you notice between temperature and lemonade sales?")
    input("Press Enter after discussing...")
    
    print("\n💡 INSIGHT: Higher temperatures → More lemonade sold!")
    print("A model would find the mathematical relationship!")
    print()

def yesterday_vs_trend_example():
    """
    Show why using trends is better than just copying yesterday's weather.
    """
    print("=== WHY NOT JUST USE YESTERDAY'S WEATHER? ===")
    print()
    
    # Example temperature trend
    week_temps = [68, 70, 72, 74]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday"]
    
    print("Recent temperature trend:")
    for day, temp in zip(days, week_temps):
        print(f"  {day}: {temp}°F")
    
    print()
    print("Two prediction approaches:")
    print(f"  📋 Simple copying: Friday will be {week_temps[-1]}°F (same as Thursday)")
    
    # Calculate trend
    trend = week_temps[-1] - week_temps[0]  # 74 - 68 = 6°F increase over 4 days
    trend_prediction = week_temps[-1] + (trend / 3)  # Continue the trend
    
    print(f"  📈 Trend-based: Friday will be {trend_prediction:.1f}°F (following the rising pattern)")
    print()
    print("🤔 DISCUSSION QUESTION:")
    print("Which approach do you think would be more accurate over time?")
    input("Press Enter after discussing...")
    
    print("\n💡 INSIGHT: Trends often continue, so the model can spot these patterns!")
    print()

def build_model_step_by_step():
    """
    Build the weather prediction model with detailed explanations
    matching the lesson plan progression.
    """
    print("=== BUILDING OUR WEATHER PREDICTOR ===")
    print()
    
    # Step 1: Load data
    print("📊 STEP 1: Load our historical weather data")
    print("Think of this as giving our model a 'textbook' to study from!")
    print()
    
    try:
        weather_data = pd.read_csv('weather_history.csv')
        print("Our data (first 5 rows):")
        print(weather_data.head())
        print(f"\nWe have {len(weather_data)} days of weather history to learn from!")
    except FileNotFoundError:
        print("⚠️  weather_history.csv not found. Please run this in the project directory.")
        return
    
    input("\nPress Enter to continue to Step 2...")
    print()
    
    # Step 2: Prepare data
    print("🔧 STEP 2: Prepare the data for learning")
    print("We'll create 'flashcards' for our model to study:")
    print("  Front side: Past 3 days of temperature")
    print("  Back side: What the next day's temperature actually was")
    print()
    
    def prepare_data_with_explanation(df, num_days=3):
        """Prepare data with detailed explanations"""
        features = []
        targets = []
        
        print(f"Creating flashcards using {num_days} days to predict the next day...")
        
        for i in range(num_days, len(df)):
            # Past N days of temperature
            past_temps = df['temperature'].iloc[i-num_days:i].values
            features.append(past_temps)
            
            # Tomorrow's temperature (what we want to predict)
            targets.append(df['temperature'].iloc[i])
            
            # Show first few examples
            if i < num_days + 3:
                print(f"  Flashcard {i-num_days+1}: {past_temps} → {targets[-1]:.1f}°F")
        
        print(f"  ... and {len(features)-3} more flashcards!")
        return np.array(features), np.array(targets)
    
    X, y = prepare_data_with_explanation(weather_data)
    print(f"\n✅ Created {len(X)} training examples!")
    
    input("\nPress Enter to continue to Step 3...")
    print()
    
    # Step 3: Train the model
    print("🧠 STEP 3: Train the model (The magic moment!)")
    print("This is like a student studying all the flashcards to find patterns...")
    print()
    
    model = LinearRegression()
    
    print("Training in progress... 🤖")
    model.fit(X, y)
    print("✅ Done! The model has learned the patterns!")
    print()
    
    # Show what the model learned
    print("🔍 What did the model learn?")
    coefficients = model.coef_
    print(f"  The model found these weights for each day:")
    for i, coef in enumerate(coefficients):
        print(f"    Day {i+1} (oldest): {coef:.3f}")
    print(f"  Base temperature: {model.intercept_:.1f}°F")
    print()
    
    input("Press Enter to continue to Step 4...")
    print()
    
    # Step 4: Make a prediction
    print("🔮 STEP 4: Make a prediction!")
    print("Let's predict tomorrow's temperature...")
    print()
    
    # Get last 3 days
    last_3_days = weather_data['temperature'].tail(3).values
    prediction = model.predict([last_3_days])[0]
    
    print(f"Recent temperatures: {last_3_days}")
    print(f"Tomorrow's predicted temperature: {prediction:.1f}°F")
    print()
    
    # Explain the prediction
    print("🤔 Why this prediction makes sense:")
    if len(last_3_days) >= 2:
        trend = last_3_days[-1] - last_3_days[0]
        if trend > 1:
            print("  📈 Temperatures have been rising, so prediction continues the trend")
        elif trend < -1:
            print("  📉 Temperatures have been falling, so prediction continues the trend")
        else:
            print("  📊 Temperatures have been stable, so prediction is similar")
    
    print()
    print("🎉 Congratulations! You've built your first AI weather predictor!")
    print()
    
    return model, last_3_days, prediction


def main():
    """
    Main lesson flow following the lesson plan structure.
    """
    print("=" * 60)
    print("    WEATHER PREDICTION WITH MACHINE LEARNING")
    print("         Interactive Lesson Demo")
    print("=" * 60)
    print()
    
    # Start with relatable example
    lemonade_stand_example()
    
    # Show why trends matter
    yesterday_vs_trend_example()
    
    # Build the actual model
    model, last_temps, prediction = build_model_step_by_step()
    
    
    print("🎓 LESSON COMPLETE!")
    print("Next steps: Integrate this into your weather application!")
    print()
    print("Key takeaways:")
    print("  ✅ AI/ML is about finding patterns in data")
    print("  ✅ Simple models can still be very useful")
    print("  ✅ Training means showing the model examples")
    print("  ✅ Your weather app now has prediction capabilities!")

if __name__ == "__main__":
    main() 