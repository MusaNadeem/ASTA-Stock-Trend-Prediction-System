# ASTA Stock Trend Prediction System

This project is a full-stack stock trend prediction demo for a Design and Analysis of Algorithms course. It combines a FastAPI backend, a Chart.js dashboard, and a PyTorch Transformer that replaces dense attention with Adaptive Sparse Temporal Attention (ASTA).

## Features

- CSV stock selection from the bundled `Dataset/` directory
- CSV upload support from the dashboard
- Sliding window preprocessing with `T = 60`
- Z-score normalization over `Open`, `High`, `Low`, `Close`, and `Volume`
- Transformer core with:
  - Adaptive Sparse Temporal Attention (local + logarithmic + volatility-based sampling)
  - Trend-Aware Positional Encoding (momentum + volatility modulation)
  - Multi-Horizon Voting Predictor for short, mid, and long horizons
- API endpoints:
  - `POST /train`
  - `POST /predict`
  - `GET /data`

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn backend.api:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## API usage

### Train

Send a multipart form request with `symbol`, optional `epochs`, and optional CSV `file`.

### Predict

Send `symbol` or upload a CSV file. If the model is not trained yet for that symbol, the service trains it on demand.

### Data

`GET /data?symbol=ABL` returns processed series and sliding-window metadata for the selected stock.

## Notes

- The runtime comparison in the training response reports dense attention versus ASTA.
- The ASTA implementation is intentionally explicit and easy to inspect for coursework and algorithm analysis.
# ASTA-Stock-Trend-Prediction-System
