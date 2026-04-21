import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

# Sample Dataset
data = {
    'Price': [1200, 2500, 800, 2300, 600],
    'Performance_Tier_Encoded': [2, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# --- 1. StandardScaler ---
# Centers data around 0 with a standard deviation of 1
std_scaler = StandardScaler()
df['Price_Standard'] = std_scaler.fit_transform(df[['Price']])

# --- 2. MinMaxScaler ---
# Squashes data into a range (default is 0 to 1)
minmax_scaler = MinMaxScaler()
df['Price_MinMax'] = minmax_scaler.fit_transform(df[['Price']])

# --- 3. Normalizer ---
# Scales each row (sample) individually to have a unit norm
norm_scaler = Normalizer()
# We use two columns here because Normalizer scales across features in a row
df_normalized = norm_scaler.fit_transform(df[['Price', 'Performance_Tier_Encoded']])

print("--- Scaled Results ---")
print(df[['Price', 'Price_Standard', 'Price_MinMax']])
print("\n--- Normalized (Row-wise) Example ---")
print(df_normalized[:3]) # Showing first 3 rows
