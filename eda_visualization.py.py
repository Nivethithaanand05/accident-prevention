import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (replace with actual path or DataFrame)
# For demo, we'll simulate a simplified version of your data
df = pd.DataFrame({
    'Temperature(F)': np.random.normal(70, 10, 100),
    'Humidity(%)': np.random.normal(50, 15, 100),
    'Visibility(mi)': np.random.normal(10, 2, 100),
    'Speed': np.random.normal(60, 20, 100),
    'jerk_score': np.random.normal(0.5, 0.2, 100),
    'overspeed_flag': np.random.choice([0, 1], 100),
})

# Set style
sns.set(style="whitegrid")

# 1. Histogram: Speed Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Speed'], bins=20, kde=True, color='blue')
plt.title("Histogram of Vehicle Speed")
plt.xlabel("Speed (mph)")
plt.ylabel("Frequency")
plt.savefig("histogram_speed.png")
plt.close()

# 2. Boxplot: Jerk Score by Overspeed
plt.figure(figsize=(8, 5))
sns.boxplot(x='overspeed_flag', y='jerk_score', data=df)
plt.title("Boxplot of Jerk Score by Overspeed Flag")
plt.xlabel("Overspeed Flag (0 = No, 1 = Yes)")
plt.ylabel("Jerk Score")
plt.savefig("boxplot_jerk_overspeed.png")
plt.close()

# 3. Heatmap: Correlation Matrix
plt.figure(figsize=(10, 6))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

# 4. Pairplot (Optional if you want multiple relationships)
# sns.pairplot(df[['Speed', 'Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'jerk_score']])
# plt.savefig("pairplot_features.png")
# plt.close()

print("✅ EDA Visualizations saved:")
print("- histogram_speed.png")
print("- boxplot_jerk_overspeed.png")
print("- correlation_heatmap.png")
