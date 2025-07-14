# Simple Weather Predictor

This project is a beginner-friendly demonstration of building a predictive model using Machine Learning (ML) to forecast tomorrow's temperature based on historical weather data. It uses linear regression to learn patterns from past temperatures and make predictions. This is designed for educational purposes, tying into a lesson on demystifying AI/ML with a weather project.

## Description

The script loads historical weather data from a CSV file, prepares features (e.g., past 3 days' temperatures) and targets (next day's temperature), trains a Linear Regression model, and predicts tomorrow's temperature. It includes explanations and trend analysis to make the process intuitive.

Key concepts covered:
- Loading and preparing data with Pandas and NumPy.
- Training a model with scikit-learn's LinearRegression.
- Making predictions and interpreting trends.

This is a simplified example—real weather prediction uses more features (e.g., humidity, pressure) and advanced models.

## Installation

1. Clone the repository:
   ```
   git clone git@github.com:SarahE-Dev/weather-starter-code.git
   cd weather-starter-code
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your data: Place your `weather_history.csv` in the project directory. The CSV should have columns: `date` (YYYY-MM-DD) and `temperature` (in °F).

   Example CSV snippet:
   ```
   date,temperature
   2025-01-01,67.0
   2025-01-02,68.2
   ...
   ```

2. Run the demo script (`demo_weather_predictor.py`):
   ```
   python demo_weather_predictor.py
   ```

   Output example:
   ```
   Our data:
             date  temperature
   0  2025-01-01         67.0
   1  2025-01-02         68.2
   ...
   We have 190 examples to learn from
   Training the model...
   Done! The model has learned the patterns.
   Last 3 days: [53.1 57.8 60.4]
   Tomorrow's predicted temperature: 59.2°F
   Temperatures have been rising, so prediction is higher
   ```

3. For the student breakout: Use `weather_predictor_starter.py` to fill in the TODOs and experiment with different `num_days`.

4. Integrate into a larger app: Use the `WeatherPredictor` class in your weather application for ongoing predictions.

## Data

- The provided `weather_history.csv` is synthetic data with seasonal patterns (colder winters, warmer summers).
- You can replace it with real data from sources like NOAA or OpenWeatherMap APIs.
- Minimum data: At least 30 days for meaningful training.

## Project Structure

- `demo_weather_predictor.py`: Full demo script for live coding.
- `breakout_activity.py`: Starter code for breakout activity.
- `weather_predictor_starter.py`: Starter code for hands-on activity.
- `weather_predictor.py`: Sample weather predictor class.
- `weather_history.csv`: Sample data file.
- `requirements.txt`: Dependencies.
- `README.md`: This file.

## Limitations and Improvements

- This model assumes linear trends; it may not handle sudden changes (e.g., storms).
- Ideas to improve:
  - Add more features (e.g., humidity, season indicators).
  - Use advanced models like RandomForestRegressor.
  - Include error handling for insufficient data.
  - Visualize predictions with Matplotlib (add `matplotlib` to requirements if needed).

## License

MIT License. Feel free to use and modify for educational purposes.
