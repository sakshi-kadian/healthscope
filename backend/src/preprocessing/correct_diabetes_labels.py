"""
Fix inverted diabetes labels
"""
import pandas as pd

# Load processed data
df = pd.read_csv('data/processed/diabetes_final.csv')

print(f"Original target distribution:")
print(df['outcome'].value_counts())

# Flip the labels: 0 -> 1, 1 -> 0
df['outcome'] = 1 - df['outcome']

print(f"\nFlipped target distribution:")
print(df['outcome'].value_counts())

# Save corrected data
df.to_csv('data/processed/diabetes_final.csv', index=False)
print(f"\n✓ Saved corrected data to: data/processed/diabetes_final.csv")
