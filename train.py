import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_iris

def prepare_filter_data():
    # 1. Load Iris dataset (from your filter.ipynb)
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # 2. Save as CSV for the app
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['target'] = y
    df.to_csv("iris_features.csv", index=False)
    
    # 3. Save a small sample for the missing value demo
    missing_data = pd.DataFrame({
        'Sepal Length': [5.1, 4.9, np.nan, 4.6, 5.0],
        'Sepal Width': [np.nan, 3.0, 3.2, 3.1, 3.6],
        'Petal Length': [1.4, 1.4, 1.3, np.nan, 1.4],
        'Petal Width': [0.2, 0.2, 0.2, 0.2, 0.2]
    })
    missing_data.to_csv("missing_sample.csv", index=False)
    
    print("✅ Created: iris_features.csv and missing_sample.csv")

if __name__ == "__main__":
    prepare_filter_data()
