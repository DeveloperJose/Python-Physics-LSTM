import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics as metrics

# Read the data as a pandas dataframe
df = pd.read_csv('all_airfoil_data.csv', compression='gzip')

# Encode airfoil types from strings to numbers
label_encoder = LabelEncoder()
df['Airfoil_Type'] = label_encoder.fit_transform(df['Airfoil_Type'])