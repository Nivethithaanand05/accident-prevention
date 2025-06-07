import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Number of rows to generate
num_rows = 5000

# Simulate start time
start_time = datetime.now()

# Generate data
data = {
    "vehicle_id": [random.randint(1000, 9999) for _ in range(num_rows)],
    "timestamp": [start_time + timedelta(seconds=i*5) for i in range(num_rows)],
    "speed": np.random.normal(loc=65, scale=15, size=num_rows).round(2),  # Mean 65 km/h
    "heading": np.random.uniform(0, 360, size=num_rows).round(2),
    "jerk_score": np.random.normal(loc=0.5, scale=0.2, size=num_rows).round(2),
}

# Create DataFrame
df_vehicle = pd.DataFrame(data)

# Add flags
df_vehicle["overspeed_flag"] = df_vehicle["speed"].apply(lambda x: 1 if x > 80 else 0)
df_vehicle["control_loss_flag"] = df_vehicle["jerk_score"].apply(lambda x: 1 if x > 0.8 else 0)

# Save to CSV
df_vehicle.to_csv("simulated_vehicle_behavior.csv", index=False)

print("✅ Dataset 'simulated_vehicle_behavior.csv' generated successfully!")
print(df_vehicle.head())
