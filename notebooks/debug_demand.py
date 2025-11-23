import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import pickle

# Load Data
try:
    df = pd.read_csv('../data/sales_data.csv')
except FileNotFoundError:
    # Try absolute path if relative fails (just in case)
    df = pd.read_csv('/Users/praveenkumardevamane/Downloads/Dynamic pricing/Dynamic-Pricing/data/sales_data.csv')

# Feature Engineering
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.weekday

# Encode Categoricals
cat_cols = ['Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality', 'Promotion']
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Define Features and Target
features = ['Store ID', 'Product ID', 'Category', 'Region', 'Inventory Level', 
            'Price', 'Discount', 'Weather Condition', 'Promotion', 
            'Competitor Pricing', 'Seasonality', 'Epidemic', 'Month', 'Day', 'Weekday']
target = 'Demand'

X = df[features]
y = df[target]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse

# Run Optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1) # Just 1 trial to check for errors

print('Best trial:', study.best_trial.params)
