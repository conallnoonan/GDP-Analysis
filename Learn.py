import pandas as pd # NOQA F401
from sklearn.model_selection import train_test_split as tts # NOQA F401
from sklearn.ensemble import RandomForestRegressor # NOQA F401
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # NOQA F401
from sklearn.preprocessing import StandardScaler # NOQA F401
from sklearn.model_selection import GridSearchCV # NOQA F401
# Load dataset
df = pd.read_csv("/Users/conallnoonan/Documents/GDP_Analysis/2020-2025.csv")
# Preprocessing
df = df.dropna()  # Drop rows with missing values
# Prepare dataset
# Projected global GDP growth rate (2025) is approximately 3.3%
df["2026"] = df["2025"] * 1.03  # Assuming a 3% growth rate for 2026
features = ["2020", "2021", "2022", "2023", "2024", "2025",
            "GDP_2024_to_2025_Growth", "GDP_2023_to_2024_Growth",
            "GDP_Rolling_Avg"]
df["GDP_2024_to_2025_Growth"] = (df["2025"] - df["2024"]) / df["2024"]
df["GDP_2023_to_2024_Growth"] = (df["2024"] - df["2023"]) / df["2023"]
df["GDP_Rolling_Avg"] = df[["2023", "2024", "2025"]].mean(axis=1)
target = "2026"

# Train-test split
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2,
                                       random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter Tuning
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42),
                           param_grid, cv=3, scoring="neg_mean_squared_error")
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_

# Feature Importance
importances = model.feature_importances_
for feature, importance in zip(features, importances):
    print(f"{feature}: {importance}")

# Train Model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
importances = model.feature_importances_
for feature, importance in zip(features, importances):
    print(f"{feature}: {importance}")

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
print("Mean Absolute Error:",  mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Predict GDP for all countries
df["Predicted_2026"] = model.predict(scaler.transform(X))
print(df[["Country", "2025", "2026", "Predicted_2026"]])

# Calculate prediction metrics
df["Prediction_Error"] = abs(df["2026"] - df["Predicted_2026"])
df["Prediction_Error_Percent"] = (df["Prediction_Error"] / df["2026"]) * 100

# Save to CSV
output_columns = ["Country", "2020", "2021", "2022", "2023", "2024", "2025",
                  "2026", "Predicted_2026", "Prediction_Error",
                  "Prediction_Error_Percent"]
df[output_columns].to_csv(
    "/Users/conallnoonan/Documents/GDP_Analysis/GDP_Predictions_2026.csv",
    index=False
)
