import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
data = pd.read_csv("WineQuality/wine_quality.csv")

# Convert categorical columns to numeric
data = pd.get_dummies(data)

# Fill missing values with the column mean
data = data.fillna(data.mean())

# Features (all columns except 'quality')
X = data.drop("quality", axis=1)

# Target
y = data["quality"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Save model
joblib.dump(model, "WineQuality/wine_quality_model.pkl")
print("Model saved as WineQuality/wine_quality_model.pkl")
