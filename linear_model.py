import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def train_and_save_model():
    print("Loading dataset...")
    df = pd.read_csv('database.csv')

    features = [
        'Pipeline Type', 
        'Liquid Type', 
        'Accident State', 
        'Accident Latitude', 
        'Accident Longitude', 
        'Cause Category', 
        'Cause Subcategory', 
        'Liquid Recovery (Barrels)', 
        'Pipeline Shutdown'
    ]
    target = 'Unintentional Release (Barrels)'

    df = df.dropna(subset=[target])
    
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = ['Accident Latitude', 'Accident Longitude', 'Liquid Recovery (Barrels)']
    categorical_features = ['Pipeline Type', 'Liquid Type', 'Accident State', 
                            'Cause Category', 'Cause Subcategory', 'Pipeline Shutdown']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"Test MAE: {mae:.2f}")
    print(f"Test R2: {r2:.4f}")

    filename = 'linear_model.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nModel saved successfully to '{filename}'")

if __name__ == "__main__":
    train_and_save_model()