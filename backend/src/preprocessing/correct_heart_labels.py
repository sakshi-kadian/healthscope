"""
Fix inverted heart disease labels
"""
import pandas as pd

# Load processed data
df = pd.read_csv('data/processed/heart_final.csv')

print(f"Original target distribution:")
print(df['target'].value_counts())

# Flip the labels: 0 -> 1, 1 -> 0
df['target'] = 1 - df['target']

print(f"\nFlipped target distribution:")
print(df['target'].value_counts())

# Save corrected data
df.to_csv('data/processed/heart_final_corrected.csv', index=False)
print(f"\n✓ Saved corrected data to: data/processed/heart_final_corrected.csv")
