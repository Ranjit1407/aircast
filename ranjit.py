import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.inspection import PartialDependenceDisplay

# ========== DATA LOADING & CLEANING ==========

# Load your data (change path as needed)
df = pd.read_csv(r"D:\Sem7\aircast\Clean15YearChennaiWeather.csv")

# Extract the date (dd Month yyyy) part only
df['date_no_day'] = df['date'].str.extract(r'(\d{1,2} [A-Za-z]+ \d{4})')[0]
df['parsed_date'] = pd.to_datetime(df['date_no_day'], format='%d %B %Y')

# Features to use (add more if available in your CSV):
chosen_features = []
for f in ['wind', 'baro', 'humidity', 'dew', 'precip', 'vis']:
    if f in df.columns:
        chosen_features.append(f)
print("Using features:", chosen_features)

cols_for_model = chosen_features + ['temp']

# Drop rows with NaN in features or target
df_clean = df.dropna(subset=cols_for_model)

# Add month column for monthly plots if not already present
if 'month' not in df_clean.columns:
    df_clean['month'] = df_clean['parsed_date'].dt.month

# ========== VISUALIZATIONS ==========

# 1. Temperature Trend
plt.figure(figsize=(12, 5))
plt.plot(df_clean['parsed_date'], df_clean['temp'], color='orange')
plt.title('Chennai Temperature Trend')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.tight_layout()
plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df_clean[chosen_features + ['temp']].corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.tight_layout()
plt.show()

# 3. Monthly Temperature Variation (Boxplot)
plt.figure(figsize=(10, 5))
sns.boxplot(x='month', y='temp', data=df_clean)
plt.title('Monthly Temperature Variation')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')
plt.tight_layout()
plt.show()

# ========== MACHINE LEARNING ==========

# Feature matrix X, target vector y
X = df_clean[chosen_features]
y = df_clean['temp']

# Train/test split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# ========== MODEL EVALUATION ==========

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print('RMSE:', rmse)


print('Mean Absolute Error (MAE):', mae)
print('R^2 Score:', r2)
print('Root Mean Squared Error (RMSE):', rmse)

# ========== ADDITIONAL VISUALIZATIONS ==========

# 4. Scatter plot: Actual vs. Predicted Temperature
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel('Actual Temperature (°C)')
plt.ylabel('Predicted Temperature (°C)')
plt.title('Actual vs. Predicted Temperature')
plt.tight_layout()
plt.show()

# 5. Distribution of Residuals (Errors)
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, bins=30)
plt.title('Distribution of Prediction Residuals')
plt.xlabel('Residual (Actual - Predicted Temperature)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 6. Actual vs. Predicted Temperature Over Test Samples (Time or Index)
plt.figure(figsize=(12, 5))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted', alpha=0.7)
plt.title('Actual vs Predicted Temperature (Test Set)')
plt.xlabel('Sample Index')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.tight_layout()
plt.show()

# 7. Residuals vs. Predicted Values (Check homoscedasticity)
plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Temperature (°C)')
plt.ylabel('Residual')
plt.title('Residuals vs. Predicted Temperature')
plt.tight_layout()
plt.show()

# 8. Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = [chosen_features[i] for i in indices]

plt.figure(figsize=(8, 5))
sns.barplot(x=importances[indices], y=feature_names)
plt.title('Feature Importances')
plt.xlabel('Relative Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# 9. Partial Dependence Plots for Top 2 Features
top_features_idx = indices[:2]  # indices of top 2 features
fig, ax = plt.subplots(figsize=(12, 6))
PartialDependenceDisplay.from_estimator(model, X_train, top_features_idx, ax=ax)
plt.suptitle('Partial Dependence Plots for Top Features', y=1.02)
plt.tight_layout()
plt.show()

# 10. Pairplot of Features and Target
sns.pairplot(df_clean[chosen_features + ['temp']])
plt.suptitle('Pairplot of Features and Temperature', y=1.02)
plt.show()
