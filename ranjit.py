import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ========== DATA LOADING & CLEANING ==========

# Load your data (change path as needed)
df = pd.read_csv(r"D:\Sem7\aircast\Clean15YearChennaiWeather.csv")

# Extract the date (dd Month yyyy) part only
df['date_no_day'] = df['date'].str.extract(r'(\d{1,2} [A-Za-z]+ \d{4})')[0]
df['parsed_date'] = pd.to_datetime(df['date_no_day'], format='%d %B %Y')

# ----------- ENSURE NO NaN in features/target -----------
# Features to use (add more if available in your CSV):
chosen_features = []
for f in ['wind', 'baro', 'humidity', 'dew', 'precip', 'vis']:
    if f in df.columns:
        chosen_features.append(f)
print("Using features:", chosen_features)

cols_for_model = chosen_features + ['temp']
df_clean = df.dropna(subset=cols_for_model)

# ========== VISUALIZATIONS ==========

# 1. Temperature Trend
plt.figure(figsize=(12, 5))
plt.plot(df_clean['parsed_date'], df_clean['temp'])
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

# 3. Monthly Temperature Variation
if 'month' not in df_clean.columns:
    df_clean['month'] = df_clean['parsed_date'].dt.month
plt.figure(figsize=(10, 5))
sns.boxplot(x='month', y='temp', data=df_clean)
plt.title('Monthly Temperature Variation')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')
plt.tight_layout()
plt.show()

# ========== MACHINE LEARNING ==========

# Train/test split
X = df_clean[chosen_features]
y = df_clean['temp']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('MAE:', mean_absolute_error(y_test, y_pred))
print('R2 Score:', r2_score(y_test, y_pred))

# 4. Scatter plot: Actual vs. Predicted
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel('Actual Temperature (°C)')
plt.ylabel('Predicted Temperature (°C)')
plt.title('Actual vs. Predicted Temperature')
plt.tight_layout()
plt.show()

# 5. Feature Importance
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
