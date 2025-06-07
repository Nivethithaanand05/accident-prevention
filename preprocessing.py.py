import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import table
from sklearn.preprocessing import StandardScaler

# Step 1: Create a sample dataset (you can replace this with your dataset)
data = {
    'Temperature(F)': [72, np.nan, 68, 70, np.nan, 1000],  # 1000 is outlier
    'Humidity(%)': [55, np.nan, 60, 58, 62, 200],  # 200 is outlier
    'Visibility(mi)': [10.0, 8.0, np.nan, 10.0, 9.0, -1],  # -1 is invalid
    'Speed': [45, 85, 150, 35, 300, 999],  # 300, 999 are outliers
    'Weather_Condition': ['Clear', 'Rain', 'Clear', 'Fog', 'Rain', 'Storm'],
    'overspeed_flag': [1, 0, 1, 0, 1, 1]
}
df = pd.DataFrame(data)

# Step 2: Function to save DataFrame as image
def save_df_as_image(df, filename, title):
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.axis('off')
    tbl = table(ax, df.head(), loc='center', colWidths=[0.15]*len(df.columns))
    plt.title(title, fontsize=12)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# Save BEFORE preprocessing
save_df_as_image(df, "before_preprocessing.png", "Before Preprocessing")

# Step 3: Preprocessing

## A. Handle Missing Values
df['Temperature(F)'].fillna(df['Temperature(F)'].median(), inplace=True)
df['Humidity(%)'].fillna(df['Humidity(%)'].median(), inplace=True)
df['Visibility(mi)'].fillna(df['Visibility(mi)'].median(), inplace=True)

## B. Remove Duplicates (for real datasets)
df.drop_duplicates(inplace=True)

## C. Outlier Detection & Removal using IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

for col in ['Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Speed']:
    df = remove_outliers(df, col)

## D. Feature Encoding (One-Hot)
df = pd.get_dummies(df, columns=['Weather_Condition'])

## E. Feature Scaling (Standardization)
scaler = StandardScaler()
scaled_cols = ['Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Speed']
df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

# Save AFTER preprocessing
save_df_as_image(df, "after_preprocessing.png", "After Preprocessing")

print("✅ Preprocessing complete. Screenshots saved as 'before_preprocessing.png' and 'after_preprocessing.png'")
