import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib

# UCI raw data URL (comma-separated values)
URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'

# column names from UCI
COLS = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

os.makedirs('data', exist_ok=True)

# load raw data from UCI
print('Downloading raw dataset...')
df = pd.read_csv(URL, header=None, names=COLS)
print(f'Loaded {len(df)} rows')

# quick check: no missing values per UCI description but still check
if df.isnull().any().any():
    print('Warning: missing values found — dropping rows with NA')
    df = df.dropna()

# save raw data
df.to_csv('data/data_raw.csv', index=False)

# split (stratify by target to keep class distribution)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['class'], random_state=42)

# Fit encoder on train only (prevents leakage)
cat_features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

# Fit on train features
encoder.fit(train_df[cat_features])

# Transform train and test
train_encoded = encoder.transform(train_df[cat_features])
test_encoded = encoder.transform(test_df[cat_features])

# Get resulting column names
encoded_cols = encoder.get_feature_names_out(cat_features)

# Build DataFrames (keep target 'class' as last column)
train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_cols, index=train_df.index)
train_encoded_df['class'] = train_df['class'].values

test_encoded_df = pd.DataFrame(test_encoded, columns=encoded_cols, index=test_df.index)
test_encoded_df['class'] = test_df['class'].values

# Save encoded CSVs
train_encoded_df.to_csv('data/train.csv', index=False)
test_encoded_df.to_csv('data/test.csv', index=False)
print('Saved data/train.csv and data/test.csv')
