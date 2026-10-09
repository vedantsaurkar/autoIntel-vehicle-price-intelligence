from pathlib import Path
import json
import re
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'backend' / 'data' / 'car_details.csv'
MODEL_DIR = ROOT / 'backend' / 'model'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TARGET = 'selling_price'
NUMERIC = ['year', 'km_driven', 'mileage_num', 'engine_num', 'max_power_num', 'seats_num']
CATEGORICAL = ['brand', 'model_name', 'fuel', 'seller_type', 'transmission', 'owner']

def number(value):
    if pd.isna(value): return np.nan
    match = re.search(r'-?\d+(?:\.\d+)?', str(value).replace(',', ''))
    return float(match.group()) if match else np.nan

def prepare(df):
    df = df.copy()
    required = {'name','year','selling_price','km_driven','fuel','seller_type','transmission','owner'}
    missing = required - set(df.columns)
    if missing: raise ValueError(f'Dataset missing required columns: {sorted(missing)}')
    df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['km_driven'] = pd.to_numeric(df['km_driven'], errors='coerce')
    for source, dest in [('mileage','mileage_num'),('engine','engine_num'),('max_power','max_power_num'),('seats','seats_num')]:
        df[dest] = df[source].map(number) if source in df.columns else np.nan
    names = df['name'].fillna('Unknown').astype(str).str.strip()
    df['brand'] = names.str.split().str[0].replace('', 'Unknown')
    df['model_name'] = names.str.split().str[1].fillna('Unknown')
    for col in CATEGORICAL:
        df[col] = df[col].fillna('Unknown').astype(str).str.strip().replace('', 'Unknown')
    df = df[df[TARGET].notna() & (df[TARGET] > 0)]
    # Filter impossible values, while keeping the broad real-world range of this dataset.
    df = df[(df['year'].between(1980, 2026)) & (df['km_driven'].between(0, 1_000_000))]
    return df

def main():
    df = prepare(pd.read_csv(DATA))
    features = NUMERIC + CATEGORICAL
    X, y = df[features], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pre = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), NUMERIC),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), CATEGORICAL),
    ])
    model = Pipeline([('preprocessor', pre), ('regressor', RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1))])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {'rows_used': int(len(df)), 'train_rows': int(len(X_train)), 'test_rows': int(len(X_test)), 'mae_rupees': round(float(mean_absolute_error(y_test, pred)), 2), 'rmse_rupees': round(float(np.sqrt(mean_squared_error(y_test, pred))), 2), 'r2': round(float(r2_score(y_test, pred)), 4)}
    joblib.dump({'model': model, 'numeric_features': NUMERIC, 'categorical_features': CATEGORICAL, 'metrics': metrics}, MODEL_DIR / 'car_price_model.pkl')
    (MODEL_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print('Training complete.')
    print(json.dumps(metrics, indent=2))
    print(f'Model saved to: {MODEL_DIR / "car_price_model.pkl"}')

if __name__ == '__main__': main()
