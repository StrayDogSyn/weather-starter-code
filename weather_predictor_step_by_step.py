
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

print("=== Weather Predictor - Step by Step Guide ===\n")

# Step 1: Load your weather data
print("STEP 1: Loading Weather Data")
print("=" * 40)

weather_data = pd.read_csv('weather_history.csv')
print("Our data:")
print(weather_data.head())
print(f"\nWe have {len(weather_data)} days of weather history!")


# Step 2: Prepare the data for the model
print("STEP 2: Preparing Data for the Model")
print("=" * 40)
# Sarah: "We'll use the past 3 days to predict the next day"
def prepare_data(df):
    """
    Create features (past 3 days) and target (next day)
    Think of it like: Mon, Tue, Wed temps → Predict Thursday
    """
    features = []
    targets = []
    
    # Loop through the data
    for i in range(3, len(df)):
        # Past 3 days of temperature
        past_temps = df['temperature'].iloc[i-3:i].values
        features.append(past_temps)
        
        # Tomorrow's temperature (what we want to predict)
        targets.append(df['temperature'].iloc[i])
    
    return np.array(features), np.array(targets)

X, y = prepare_data(weather_data)
print(f"We have {len(X)} examples to learn from")
print(f"Each example uses 3 days to predict the next day")

# Show a few examples
print("\nExample training data:")
for i in range(min(3, len(X))):
    print(f"Example {i+1}: Past 3 days {X[i]} → Next day {y[i]:.1f}°F")


# Step 3: Create and train the model
print("STEP 3: Training the Model")
print("=" * 40)
model = LinearRegression()

# This is where the learning happens!
print("Training the model...")
model.fit(X, y)


# Let's check how well the model learned
predictions = model.predict(X)
accuracy = np.mean((predictions - y) ** 2) ** 0.5  # Root Mean Square Error
print(f"Model training accuracy: {accuracy:.1f}°F average error")


# Step 4: Make a prediction
print("STEP 4: Making a Prediction")
print("=" * 40)
# Use the last 3 days to predict tomorrow
last_3_days = weather_data['temperature'].tail(3).values
tomorrow_prediction = model.predict([last_3_days])

print(f"Last 3 days: {last_3_days}")
print(f"Tomorrow's predicted temperature: {tomorrow_prediction[0]:.1f}°F")

if len(last_3_days) >= 2:
    trend = last_3_days[-1] - last_3_days[0]
    if trend > 0:
        print("Temperatures have been rising, so prediction is higher")
    elif trend < 0:
        print("Temperatures have been falling, so prediction is lower")
    else:
        print("Temperatures have been steady")

# Bonus: Let's add some extra insights
print("BONUS: Additional Insights")
print("=" * 40)
print(f"Average temperature in our data: {weather_data['temperature'].mean():.1f}°F")
print(f"Temperature range: {weather_data['temperature'].min():.1f}°F to {weather_data['temperature'].max():.1f}°F")
print(f"Most recent temperature: {weather_data['temperature'].iloc[-1]:.1f}°F")

# Calculate and show recent trend
recent_5_days = weather_data['temperature'].tail(5).values
daily_changes = np.diff(recent_5_days)
avg_change = np.mean(daily_changes)

print(f"\nRecent 5-day trend: {avg_change:.1f}°F average change per day")
if avg_change > 0.5:
    print("Recent trend: Getting warmer! 🌡️↗️")
elif avg_change < -0.5:
    print("Recent trend: Getting cooler! 🌡️↘️")
else:
    print("Recent trend: Staying stable! 🌡️→")

print("\n" + "="*60)
print("Congratulations! You've just built your first weather prediction model!")
print("Try changing the number of days (currently 3) to see how it affects predictions!")
print("="*60) 