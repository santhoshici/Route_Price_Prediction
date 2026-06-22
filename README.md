# Route Price API

Minimal FastAPI service for route price prediction. It uses saved model artifacts only; it does not train anything.

## Contents

- `app.py` - API, preprocessing, feature engineering, lookup handling, and prediction logic.
- `models/production_model.joblib` - standard production route-cost model.
- `models/production_price_models.joblib` - budget, standard, and premium route-cost model bundle.
- `data/data.xlsx` - truck cost lookup table.
- `requirements.txt` - runtime dependencies only.

## Setup

```powershell
cd route_price_api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```

Health check:

```powershell
curl http://localhost:8000/health
```

## Formula Price

`POST /predict-price` returns the formula-based driver rate. It uses coordinates, the truck lookup table, and the route-cost model bundle. `pickup_city`, `destination_city`, `pickup_state`, and `drop_state` are optional, but passing them avoids reverse-geocoding.

```powershell
'{"pickup_latitude":13.0827,"pickup_longitude":80.2707,"drop_latitude":12.9716,"drop_longitude":77.5946,"pickup_city":"Chennai","destination_city":"Bengaluru","pickup_state":"Tamil Nadu","drop_state":"Karnataka","truck_type":"10 feet","body_type":"Open","weight":"1 tons","distance_km":350,"diesel_price":95}' | curl.exe -X POST http://localhost:8000/predict-price -H "Content-Type: application/json" --data-binary "@-"
```

Bulk formula pricing uses `POST /predict-prices`:

```json
{"items":[{"pickup_latitude":13.0827,"pickup_longitude":80.2707,"drop_latitude":12.9716,"drop_longitude":77.5946,"pickup_city":"Chennai","destination_city":"Bengaluru","pickup_state":"Tamil Nadu","drop_state":"Karnataka","truck_type":"10 feet","body_type":"Open","weight":"1 tons","distance_km":350,"diesel_price":95}]}
```

The response keeps the top-level `driver_rate` from the standard option and also returns `price_options` for `budget`, `standard`, and `premium`.

Optional reverse-geocoding uses `GOOGLE_MAPS_API_KEY` when city/state are not supplied.
