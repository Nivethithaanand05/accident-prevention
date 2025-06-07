# Example model training and saving (Logistic Regression)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Example dataset
df = pd.read_csv("simulated_vehicle_behavior.csv")

# Define features and target
X = df[['speed', 'jerk_score', 'overspeed_flag', 'control_loss_flag']]
y = [1 if spd > 85 or jerk > 0.9 else 0 for spd, jerk in zip(df['speed'], df['jerk_score'])]  # Simulate a target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ model.pkl saved successfully!")
