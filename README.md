# Route Price API

FastAPI service for route price prediction. This project uses saved model artifacts and lookup data only. It does not train models at runtime.

## Project Contents

- `app.py` - API routes, preprocessing, feature engineering, lookup handling, and prediction logic.
- `models/production_model.joblib` - production route-cost model.
- `models/production_price_models.joblib` - budget, standard, and premium route-cost model bundle.
- `data/data.xlsx` - truck cost lookup table used by the pricing logic.
- `requirements.txt` - runtime Python dependencies.

## Python Dependencies

Install these packages from `requirements.txt`:

- `fastapi`
- `uvicorn`
- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`
- `openpyxl`
- `requests`

## What They Are Used For

- **FastAPI** – A modern Python web framework used to build REST APIs and serve machine learning models.
- **Uvicorn** – A lightweight ASGI server used to run FastAPI applications.
- **Pandas** – Used for data manipulation, cleaning, and analysis of structured data.
- **NumPy** – Provides support for numerical computations, arrays, and mathematical operations.
- **Scikit-learn** – A machine learning library used for training, evaluating, and making predictions with ML models.
- **Joblib** – Used to save and load trained machine learning models efficiently.
- **OpenPyXL** – Used to read, write, and modify Excel (.xlsx) files.
- **Requests** – Used to send HTTP requests and interact with external APIs and web services.

## Local Setup Guide

Follow these steps in Windows PowerShell.

### 1. Open the project folder

```powershell
cd <project-root>
```

Example:

```powershell
cd C:\Users\unbox\Route_Price_Prediction
```

### 2. Create a virtual environment

If `.venv` does not already exist, create it:

```powershell
py -3.14 -m venv .venv
```

If the venv already exists, skip this step.

### 3. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

If the saved model files fail to load, install the compatible scikit-learn version used by the artifacts:

```powershell
pip install scikit-learn==1.7.2
```

### 5. Verify the installation

```powershell
pip freeze
```

### 6. Start the API server

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 7. Test the API locally

Health check:

```powershell
curl http://localhost:8000/health
```

Swagger UI:

```text
http://localhost:8000/docs
```

## API Endpoints

### 1. Health Check

- Method: `GET`
- Path: `/health`

Returns:

```json
{"status":"ok"}
```

Example:

```powershell
curl http://localhost:8000/health
```

### 2. Single Price Prediction

- Method: `POST`
- Path: `/predict-price`

This endpoint returns the calculated driver rate for one route. It uses:

- pickup and drop coordinates
- truck lookup data from `data/data.xlsx`
- production pricing model artifacts

`pickup_city`, `destination_city`, `pickup_state`, and `drop_state` are optional, but passing them avoids reverse geocoding.

New trailer truck types from `data/data.xlsx` are supported: `20 feet trailor` / `20 feet trailer` and `24 feet trailor` / `24 feet trailer`.

Example request body:

```json
{
  "pickup_latitude": 13.0827,
  "pickup_longitude": 80.2707,
  "drop_latitude": 12.9716,
  "drop_longitude": 77.5946,
  "pickup_city": "Chennai",
  "destination_city": "Bengaluru",
  "pickup_state": "Tamil Nadu",
  "drop_state": "Karnataka",
  "truck_type": "10 feet",
  "body_type": "Open",
  "weight": "1 tons",
  "distance_km": 350,
  "diesel_price": 95
}
```

Example PowerShell request:

```powershell
$body = @{
  pickup_latitude = 13.0827
  pickup_longitude = 80.2707
  drop_latitude = 12.9716
  drop_longitude = 77.5946
  pickup_city = "Chennai"
  destination_city = "Bengaluru"
  pickup_state = "Tamil Nadu"
  drop_state = "Karnataka"
  truck_type = "10 feet"
  body_type = "Open"
  weight = "1 tons"
  distance_km = 350
  diesel_price = 95
} | ConvertTo-Json -Compress

curl.exe -X POST "http://localhost:8000/predict-price" -H "Content-Type: application/json" --data-binary $body
```

### 3. Bulk Price Prediction

- Method: `POST`
- Path: `/predict-prices`

This endpoint accepts multiple route items in one request and returns success or error details for each item.

Example request body:

```json
{
  "items": [
    {
      "pickup_latitude": 13.0827,
      "pickup_longitude": 80.2707,
      "drop_latitude": 12.9716,
      "drop_longitude": 77.5946,
      "pickup_city": "Chennai",
      "destination_city": "Bengaluru",
      "pickup_state": "Tamil Nadu",
      "drop_state": "Karnataka",
      "truck_type": "10 feet",
      "body_type": "Open",
      "weight": "1 tons",
      "distance_km": 350,
      "diesel_price": 95
    },
    {
      "pickup_latitude": 19.076,
      "pickup_longitude": 72.8777,
      "drop_latitude": 18.5204,
      "drop_longitude": 73.8567,
      "pickup_city": "Mumbai",
      "destination_city": "Pune",
      "pickup_state": "Maharashtra",
      "drop_state": "Maharashtra",
      "truck_type": "12 Feet",
      "body_type": "Open",
      "weight": "2 tons",
      "distance_km": 150,
      "diesel_price": 95
    }
  ]
}
```

Example PowerShell request:

```powershell
$body = @{
  items = @(
    @{
      pickup_latitude = 13.0827
      pickup_longitude = 80.2707
      drop_latitude = 12.9716
      drop_longitude = 77.5946
      pickup_city = "Chennai"
      destination_city = "Bengaluru"
      pickup_state = "Tamil Nadu"
      drop_state = "Karnataka"
      truck_type = "10 feet"
      body_type = "Open"
      weight = "1 tons"
      distance_km = 350
      diesel_price = 95
    }
  )
} | ConvertTo-Json -Depth 5 -Compress

curl.exe -X POST "http://localhost:8000/predict-prices" -H "Content-Type: application/json" --data-binary $body
```

## Response Notes

- `POST /predict-price` returns a single prediction object with the final `driver_rate`.
- The response also includes cost breakdown fields such as:
  - `fuel_cost_inr`
  - `emi_cost`
  - `total_wear_and_tear_cost_inr`
  - `calculated_toll_cost_inr`
  - `price_options`
- `POST /predict-prices` returns a batch wrapper with:
  - `total`
  - `success_count`
  - `error_count`
  - `results`

## Troubleshooting

- If the server returns `500`, check the terminal where `uvicorn` is running.
- If model loading fails, ensure the environment is using a compatible `scikit-learn` version.
- If reverse geocoding is needed, set `GOOGLE_MAPS_API_KEY` in your environment.
- If a truck type is rejected, check the exact values available in `data/data.xlsx`.

## Quick Start Summary

```powershell
cd <project-root>
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open:

```text
http://localhost:8000/docs
```
