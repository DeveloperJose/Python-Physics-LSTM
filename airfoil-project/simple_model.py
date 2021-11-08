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

# Select the output targets, pop them so they are not inside the dataframe anymore
output_target1 = df.pop('Top_Xtr')
output_target2 = df.pop('Bot_Xtr')

# Split the inputs (features) and outputs (labels) into two arrays for easier manipulation
x_features = df.values
y_labels = np.vstack((output_target1.values, output_target2.values)).T

# First split into training (70%) and testing (30%)
x_train, x_test, y_train, y_test = train_test_split(x_features, y_labels, train_size=0.7, random_state=45, shuffle=True)


print(f"Training Shape={x_train.shape}, Testing Shape={x_test.shape}")

regr = RandomForestRegressor(n_estimators=1000, max_depth=None, random_state=42, n_jobs=4, verbose=3)
regr.fit(x_train, y_train)
joblib.dump(regr, 'rf.pkl')

y_pred = regr.predict(x_test)
# https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
print(f"Explained Variance = {metrics.explained_variance_score(y_test, y_pred)}")
print(f"MSE = {metrics.mean_squared_error(y_test, y_pred)}")
print(f"MAE = {metrics.mean_absolute_error(y_test, y_pred)}")
print(f"R2 = {metrics.r2_score(y_test, y_pred)}")
