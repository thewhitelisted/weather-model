import pandas as pd
import glob
import numpy as np
from sklearn.linear_model import RidgeCV
from datetime import datetime
from sklearn.metrics import mean_absolute_error

# Load all weather CSVs
all_files = glob.glob("data/*.csv")
dfs = [pd.read_csv(f, parse_dates=['Date/Time']) for f in all_files]
weather = pd.concat(dfs, ignore_index=True)

# Keep relevant columns and drop NA
weather = weather[['Date/Time', 'Max Temp (°C)', 'Min Temp (°C)']].dropna()

# Feature engineering
weather = weather.sort_values('Date/Time')
weather['day_of_year'] = weather['Date/Time'].dt.dayofyear
weather['year'] = weather['Date/Time'].dt.year
weather['sin_doy'] = np.sin(2 * np.pi * weather['day_of_year'] / 365)
weather['cos_doy'] = np.cos(2 * np.pi * weather['day_of_year'] / 365)

# Add lagged and rolling features
weather['MaxTemp_lag1'] = weather['Max Temp (°C)'].shift(1)
weather['MinTemp_lag1'] = weather['Min Temp (°C)'].shift(1)
weather['MaxTemp_lag2'] = weather['Max Temp (°C)'].shift(2)
weather['MinTemp_lag2'] = weather['Min Temp (°C)'].shift(2)
weather['MaxTemp_roll7'] = weather['Max Temp (°C)'].rolling(7).mean().shift(1)
weather['MinTemp_roll7'] = weather['Min Temp (°C)'].rolling(7).mean().shift(1)

weather = weather.dropna()

# Define features and targets
features = ['sin_doy', 'cos_doy', 'year',
            'MaxTemp_lag1', 'MinTemp_lag1',
            'MaxTemp_lag2', 'MinTemp_lag2',
            'MaxTemp_roll7', 'MinTemp_roll7']

X = weather[features]
y_max = weather['Max Temp (°C)']
y_min = weather['Min Temp (°C)']

# Train models with RidgeCV
ridge_alphas = np.logspace(-3, 3, 50)
model_max = RidgeCV(alphas=ridge_alphas).fit(X, y_max)
model_min = RidgeCV(alphas=ridge_alphas).fit(X, y_min)

# Evaluate model performance on year
for eval_year in range(2015,2024):
    train = weather[weather['year'] <= eval_year - 1]
    test = weather[weather['year'] == eval_year]

    X_train = train[features]
    y_train_max = train['Max Temp (°C)']
    X_test = test[features]
    y_test_max = test['Max Temp (°C)']

    model = RidgeCV(alphas=ridge_alphas).fit(X_train, y_train_max)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test_max, preds)
    max_error = np.max(np.abs(y_test_max.values - preds))
    min_error = np.min(np.abs(y_test_max.values - preds))

    print(f"MAE on {eval_year}: {mae:.2f}°C")
    print(f"Max absolute error on {eval_year}: {max_error:.2f}°C")
    print(f"Min absolute error on {eval_year}: {min_error:.2f}°C")