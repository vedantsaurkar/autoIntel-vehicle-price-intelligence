from pathlib import Path
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / 'backend' / 'model' / 'car_price_model.pkl'
DATA_PATH = ROOT / 'backend' / 'data' / 'car_details.csv'
FRONTEND = ROOT / 'frontend'
app = FastAPI(title='AutoIntel Vehicle Price Intelligence API', version='1.0.0', description='Used-car price estimation using the supplied CarDekho dataset.')

class VehicleInput(BaseModel):
    brand: str = Field(min_length=1, max_length=50)
    model_name: str = Field(default='Unknown', max_length=80)
    year: int = Field(ge=1980, le=2026)
    km_driven: float = Field(ge=0, le=1000000)
    fuel: str
    seller_type: str = 'Individual'
    transmission: str = 'Manual'
    owner: str = 'First Owner'
    mileage: float | None = Field(default=None, ge=0, le=100)
    engine: float | None = Field(default=None, ge=0, le=10000)
    max_power: float | None = Field(default=None, ge=0, le=3000)
    seats: float | None = Field(default=5, ge=1, le=20)
    asking_price: float = Field(gt=0, le=100000000)

def get_model():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail='Model is not trained yet. Run: python train_model.py from the project root.')
    return joblib.load(MODEL_PATH)

@app.get('/')
def home(): return FileResponse(FRONTEND / 'index.html')
app.mount('/static', StaticFiles(directory=FRONTEND), name='static')

@app.get('/api/health')
def health(): return {'status':'ok', 'model_trained': MODEL_PATH.exists()}

@app.get('/api/analytics')
def analytics():
    if not DATA_PATH.exists(): raise HTTPException(status_code=404, detail='Dataset not found.')
    df = pd.read_csv(DATA_PATH)
    prices = pd.to_numeric(df['selling_price'], errors='coerce').dropna()
    brands = df.assign(brand=df['name'].fillna('Unknown').str.split().str[0])
    brand_stats = brands.groupby('brand')['selling_price'].agg(['count','median']).query('count >= 30').sort_values('count', ascending=False).head(8).reset_index()
    year_counts = pd.to_numeric(df['year'], errors='coerce').dropna().astype(int).value_counts().sort_index()
    return {'vehicles':int(len(df)), 'median_price':int(prices.median()), 'median_km':int(pd.to_numeric(df['km_driven'], errors='coerce').median()), 'brands':brand_stats.rename(columns={'median':'median_price'}).to_dict(orient='records'), 'years': [{'year':int(k),'count':int(v)} for k,v in year_counts.items() if 1990 <= k <= 2026]}

@app.post('/api/predict')
def predict(v: VehicleInput):
    bundle = get_model()
    features = {
        'year': v.year, 'km_driven': v.km_driven, 'mileage_num': v.mileage,
        'engine_num': v.engine, 'max_power_num': v.max_power, 'seats_num': v.seats,
        'brand': v.brand.strip().title(), 'model_name': v.model_name.strip().title() or 'Unknown',
        'fuel': v.fuel, 'seller_type': v.seller_type, 'transmission': v.transmission, 'owner': v.owner,
    }
    X = pd.DataFrame([features])
    price = max(0, float(bundle['model'].predict(X)[0]))
    gap = float(v.asking_price - price)
    gap_pct = gap / price * 100 if price else 0
    if gap_pct <= -8: verdict, score = 'Good price', 90
    elif gap_pct <= 5: verdict, score = 'Fair price', 78
    elif gap_pct <= 15: verdict, score = 'Negotiate', 58
    else: verdict, score = 'High asking price', 35
    return {'estimated_price':round(price), 'asking_price':round(v.asking_price), 'difference':round(gap), 'difference_percent':round(gap_pct,1), 'verdict':verdict, 'deal_score':score, 'currency':'INR', 'note':'Estimate from historical listings; inspect the vehicle and compare current local listings before buying.'}

@app.get('/api/metrics')
def metrics():
    p = ROOT / 'backend' / 'model' / 'metrics.json'
    return json.loads(p.read_text()) if p.exists() else {'status':'Model not trained yet'}
