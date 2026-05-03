<<<<<<< HEAD
This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
=======
# ASTA Stock Trend Prediction System (RAG Architecture)

This project is a full-stack stock trend prediction platform for a Design and Analysis of Algorithms course. It combines a FastAPI backend, an advanced UI dashboard, and a **Retrieval-Augmented Generation (RAG)** architecture that replaces standard classification with a highly optimized, time-aware Hybrid Search algorithm.

## Core Algorithm: Hybrid ANN RAG Pipeline

The monolithic transformer classifier has been decoupled into an embedding-based search system:

1. **Transformer Feature Extractor**: Sliding windows of size `T=60` are processed by a Transformer to generate fixed-size `dim=64` high-density market embeddings.
2. **K-Means Clustering**: The embedding space is partitioned into clusters (regimes) to drastically reduce search space (from $O(N)$ to $O(N/C)$).
3. **HNSW Retrieval**: Fast approximate nearest neighbor search via Hierarchical Navigable Small World graphs retrieves the top `M=50` candidates in $O(\log(N/C))$ time.
4. **Exact Cosine Re-Ranking**: Ensures precision by computing exact cosine distance on the retrieved top `M` subset.
5. **Time-Aware Decay Scoring**: Combines similarity with chronological proximity using `Score = Similarity * exp(-λ * time_diff)`. Recent identical market states are weighted higher than old ones.
6. **Majority Voting**: A softmax probability distribution is generated over the top `k` candidates to output a robust `Uptrend`, `Downtrend`, or `Neutral` signal.

## How to Run Locally

You must have Python installed. The project uses a virtual environment for dependencies.

### 1. Set up Virtual Environment & Install Dependencies

Open a terminal in the project root directory and run:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```
*(If you are on Linux/Mac, use `source .venv/bin/activate` instead)*

### 2. Start the Backend Server

```powershell
.\.venv\Scripts\python -m uvicorn backend.api:app --reload
```
This will start the FastAPI server on port 8000.

### 3. Open the Application

Open your browser and navigate to:
[http://127.0.0.1:8000](http://127.0.0.1:8000)

## Application Features

- **ASTA Terminal UI**: A premium, hyper-saturated fintech dashboard interface.
- **Market Feed Hub**: A dedicated real-time view with simulated Order Books (Tape Analysis), Volatility Anomaly Detection, and automated Market Regime tracking.
- **PSX News & Insights**: Real-time integration of top business headlines from sources like Dawn and The Express Tribune directly within the terminal.
- **Portfolio Section**: A compact view to track your paper-trading holdings and executed trades without bloat.
- **Future Date Forecasting**: Dynamically forecast up to a specific future date using recursive prediction logic.
- **Screener & Signals**: View real-time probability distributions over the HNSW RAG algorithm.
- **Smart Alerts System**: Configure dynamic UI alerts for price crossings and volume anomalies.
>>>>>>> 7a2adeb (Fixed encoding, forecast, charts, multi-model issues and removed unnecessary sections)
