import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("Loading feature matrix...")
df = pd.read_csv('data/processed/h3_feature_matrix.csv')

# Ensure chronological sorting to prevent lookahead bias
df['time_window'] = pd.to_datetime(df['time_window'])
df = df.sort_values(by='time_window').reset_index(drop=True)

# 1. Feature Selection
features = ['hour', 'dayofweek', 'lag_15m', 'lag_60m', 'neighborhood_demand']
target = 'surge_multiplier'

X = df[features]
y = df[target]

# 2. Chronological 80/20 Train-Test Split
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Training on {len(X_train)} time windows | Testing on {len(X_test)} time windows...")

# 3. Train XGBoost Model
model = XGBRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# 4. Model Evaluation
predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)

print("\n" + "="*40)
print("     MODEL PERFORMANCE EVALUATION")
print("="*40)
print(f"R² Score: {r2:.4f}")
print(f"RMSE:     {rmse:.4f}")
print(f"MAE:      {mae:.4f}")
print("="*40)

# Feature Importances
print("\nFeature Importances:")
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
for feat, imp in importances.items():
    print(f"  - {feat:20s}: {imp*100:.2f}%")

# 5. Save Model Artifact
os.makedirs('models', exist_ok=True)
model_path = 'models/xgboost_surge_model.pkl'
joblib.dump(model, model_path)
print(f"\nSUCCESS: Trained model saved to '{model_path}'!")