import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("="*60)
print("ELECTRICITY CONSUMPTION PREDICTION USING XGBOOST")
print("="*60)

# -----------------------------
# LOAD DATASET
# -----------------------------

df = pd.read_csv(
    "dataset/household_power_consumption.txt",
    sep=";",
    low_memory=False
)

print("\nDataset Loaded Successfully!")

print(df.head())

print("\nShape :", df.shape)
# -----------------------------
# SAMPLE DATA
# -----------------------------

df = df.sample(
    n=100000,
    random_state=42
)

print("\nSampled Dataset Shape :", df.shape)
# ==========================
# DATA PREPROCESSING
# ==========================

# Merge Date & Time

df["DateTime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    dayfirst=True
)

# Drop old columns

df.drop(["Date", "Time"], axis=1, inplace=True)

# Convert all columns to numeric

for col in df.columns:
    if col != "DateTime":
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove missing values

df.dropna(inplace=True)

print("\nAfter Cleaning :", df.shape)

# ==========================
# Create Time Features
# ==========================

df["Year"] = df["DateTime"].dt.year
df["Month"] = df["DateTime"].dt.month
df["Day"] = df["DateTime"].dt.day
df["Hour"] = df["DateTime"].dt.hour

print("\nData Preprocessing Completed Successfully!")
# ==========================
# FEATURES & TARGET
# ==========================

X = df.drop(
    ["DateTime", "Global_active_power"],
    axis=1
)

y = df["Global_active_power"]

print("\nFeatures Shape :", X.shape)

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining :", X_train.shape)
print("Testing  :", X_test.shape)
# ==========================
# XGBOOST MODEL
# ==========================

model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")
# ==========================
# PREDICTION
# ==========================

y_pred = model.predict(X_test)

print("\nR2 Score :", r2_score(y_test, y_pred))
print("MAE :", mean_absolute_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))
plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred, alpha=0.5)

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Electricity Consumption")

plt.savefig("actual_vs_predicted.png")

plt.show()
plt.figure(figsize=(10,8))

sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")

plt.title("Correlation Heatmap")

plt.savefig("correlation_heatmap.png")

plt.show()
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10,6))

sns.barplot(data=importance, x="Importance", y="Feature")

plt.title("Feature Importance")

plt.savefig("feature_importance.png")

plt.show()
errors = y_test - y_pred

plt.figure(figsize=(8,5))

plt.hist(errors, bins=40)

plt.title("Prediction Error Distribution")

plt.xlabel("Error")

plt.savefig("prediction_error.png")

plt.show()
pickle.dump(
    model,
    open("electricity_xgboost_model.pkl", "wb")
)

print("\nModel Saved Successfully!")
output = X_test.copy()

output["Actual"] = y_test.values
output["Predicted"] = y_pred

output.to_csv(
    "Electricity_Predictions.csv",
    index=False
)

print("\nPrediction CSV Saved Successfully!")