import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import requests
import time

# load dataset
print("Loading data...")
df = pd.read_csv('https://raw.githubusercontent.com/pplonski/datasets-for-start/master/adult/data.csv', skipinitialspace=True)
x_cols = [c for c in df.columns if c != 'income']
X = df[x_cols]
y = df['income']

# data split train / test
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

print("Starting A/B test simulation...")
# Use first 100 rows of test data for A/B test
for i in range(100):
    input_data = dict(X_test.iloc[i])
    target = y_test.iloc[i]
    
    # send prediction request
    r = requests.post("http://127.0.0.1:8000/api/v1/income_classifier/predict?status=ab_testing", json=input_data)
    if r.status_code != 200:
        print(f"Error at iteration {i}: {r.text}")
        continue
        
    response = r.json()
    # provide feedback
    requests.put("http://127.0.0.1:8000/api/v1/mlrequests/{}".format(response["request_id"]), json={"feedback": target})
    
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1} requests...")

print("A/B test simulation complete.")
