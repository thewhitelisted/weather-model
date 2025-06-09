import pandas as pd
import glob
import numpy as np
from sklearn.linear_model import RidgeCV
from datetime import datetime
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

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

# Add climatology features
climatology = weather.groupby('day_of_year')[['Max Temp (°C)', 'Min Temp (°C)']].mean()
climatology.columns = ['MaxTemp_clim', 'MinTemp_clim']
weather = weather.join(climatology, on='day_of_year')

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
            'MaxTemp_roll7', 'MinTemp_roll7',
            'MaxTemp_clim', 'MinTemp_clim']

X = weather[features]
y_max = weather['Max Temp (°C)']
y_min = weather['Min Temp (°C)']

# Train models with RidgeCV
ridge_alphas = np.logspace(-3, 3, 50)
model_max = RidgeCV(alphas=ridge_alphas).fit(X, y_max)
model_min = RidgeCV(alphas=ridge_alphas).fit(X, y_min)

# Evaluate model performance on year
for eval_year in range(2015, 2025):
    train = weather[weather['year'] <= eval_year - 1]
    test = weather[weather['year'] == eval_year]

    X_train = train[features]
    y_train_max = train['Max Temp (°C)']
    X_test = test[features]
    y_test_max = test['Max Temp (°C)']
    
    y_test_min = test['Min Temp (°C)']
    y_train_min = train['Min Temp (°C)']
    X_train_min = train[features]
    X_test_min = test[features]

    model_max = RidgeCV(alphas=ridge_alphas).fit(X_train, y_train_max)
    preds_max = model_max.predict(X_test)

    mae = mean_absolute_error(y_test_max, preds_max)
    max_error = np.max(np.abs(y_test_max.values - preds_max))
    min_error = np.min(np.abs(y_test_max.values - preds_max))

    model_min = RidgeCV(alphas=ridge_alphas).fit(X_train_min, y_train_min)
    preds_min = model_min.predict(X_test_min)
    mae_min = mean_absolute_error(y_test_min, preds_min)

    mae_min = mean_absolute_error(y_test_min, preds_min)
    max_error_min = np.max(np.abs(y_test_min.values - preds_min))
    min_error_min = np.min(np.abs(y_test_min.values - preds_min))

    print(f"MAE (max) on {eval_year}: {mae:.2f}°C")
    print(f"Max absolute error on {eval_year}: {max_error:.2f}°C")
    print(f"Min absolute error on {eval_year}: {min_error:.2f}°C")
    print(f"MAE (min) on {eval_year}: {mae_min:.2f}°C")
    print(f"Max absolute error (min) on {eval_year}: {max_error_min:.2f}°C")
    print(f"Min absolute error (min) on {eval_year}: {min_error_min:.2f}°C")

    # large_errors = np.where(np.abs(y_test_max.values - preds_max) > 10)[0]
    # for i in large_errors:
    #     print(test.iloc[i][['Date/Time', 'Max Temp (°C)']], '→ Predicted:', preds_max[i])


    # Visualization for last eval year only
    if eval_year == 2024:
        dates = test['Date/Time']
        errors = y_test_max.values - preds_max

        # Plot true vs predicted
        plt.figure(figsize=(12, 5))
        plt.plot(dates, y_test_max.values, label='Actual', linewidth=2)
        plt.plot(dates, preds_max, label='Predicted', linewidth=2)
        plt.title(f"Max Temp: True vs Predicted ({eval_year})")
        plt.xlabel("Date")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot true vs predicted for min temp
        plt.figure(figsize=(12, 5))
        plt.plot(dates, y_test_min.values, label='Actual Min Temp', linewidth=2)
        plt.plot(dates, preds_min, label='Predicted Min Temp', linewidth=2)
        plt.title(f"Min Temp: True vs Predicted ({eval_year})")
        plt.xlabel("Date")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot error histogram
        plt.hist(errors, bins=30, edgecolor='black')
        plt.title(f"Prediction Error Distribution ({eval_year})")
        plt.xlabel("Prediction Error (°C)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

        # Plot error histogram for min temp
        plt.hist(y_test_min.values - preds_min, bins=30, edgecolor='black')
        plt.title(f"Min Temp Prediction Error Distribution ({eval_year})")
        plt.xlabel("Prediction Error (°C)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

        # Plot true vs predicted scatter
        plt.scatter(y_test_max.values, preds_max, alpha=0.5)
        plt.plot([min(y_test_max), max(y_test_max)], [min(y_test_max), max(y_test_max)], 'r--')
        plt.xlabel("Actual Max Temp (°C)")
        plt.ylabel("Predicted Max Temp (°C)")
        plt.title(f"Actual vs Predicted Max Temp ({eval_year})")
        plt.grid(True)
        plt.show()
        
        # Plot true vs predicted scatter
        plt.scatter(y_test_min.values, preds_min, alpha=0.5)
        plt.plot([min(y_test_min), max(y_test_min)], [min(y_test_min), max(y_test_min)], 'r--')
        plt.xlabel("Actual Min Temp (°C)")
        plt.ylabel("Predicted Min Temp (°C)")
        plt.title(f"Actual vs Predicted Min Temp ({eval_year})")
        plt.grid(True)
        plt.show()