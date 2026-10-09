# AutoIntel — Used Vehicle Price & Deal Intelligence

A starter full-stack project using the supplied CarDekho dataset, FastAPI + Uvicorn, and HTML/CSS/JavaScript.

## Features
- Trains a scikit-learn regression pipeline from `backend/data/car_details.csv`.
- Predicts estimated selling price in Indian rupees.
- Compares a seller's asking price with the model estimate.
- Gives a transparent price-based indicator: Good price / Fair price / Negotiate.
- Includes summary analytics and a responsive dashboard.

> Dataset limitation: the selected `Car details v3.csv` has 8,128 rows and fields for car name, year, selling price, kilometres, fuel, seller type, transmission, owner, mileage, engine, max power, torque, and seats. It does not contain condition labels, accident images, service history, or current live listings. The app therefore estimates price; its deal indicator is not a mechanical inspection or a guarantee of market value.

## Windows setup (VS Code terminal)
1. Extract this folder.
2. Open the `AutoIntel` folder in VS Code.
3. Create and activate a virtual environment:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
4. Install dependencies:
   ```powershell
   pip install -r backend\requirements.txt
   ```
5. Train the model:
   ```powershell
   python train_model.py
   ```
6. Start the app from the project root:
   ```powershell
   uvicorn backend.app:app --reload
   ```
7. Open http://127.0.0.1:8000

Swagger API docs: http://127.0.0.1:8000/docs

## Retraining
Run `python train_model.py` after replacing the CSV. The CSV must have a `selling_price` target column and the feature columns used by the app.
