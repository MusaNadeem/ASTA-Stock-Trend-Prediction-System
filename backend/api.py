from __future__ import annotations

import io
import json
import re
import tempfile
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .attention import compare_runtime_standard_vs_asta
from .forecasting import forecast_for_date, market_regime_from_series, recursive_forecast, regime_badge_color
from .model import MultiHorizonTrendTransformer
from .preprocessing import (
    DEFAULT_HORIZONS,
    DEFAULT_TRAIN_FRACTION,
    DEFAULT_TREND_THRESHOLD,
    DEFAULT_WINDOW_SIZE,
    ProcessedBundle,
    build_processed_bundle,
    list_symbols,
    read_stock_frame,
    resolve_csv_path,
    to_processed_payload,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "Dataset"
FRONTEND_DIR = BASE_DIR / "frontend"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="ASTA Trend Lab", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


class TrainMetrics(BaseModel):
    symbol: str
    epochs: int
    train_loss: float
    val_loss: float
    accuracy: float
    runtime_standard_ms: float
    runtime_asta_ms: float
    runtime_speedup: float
    sample_count: int


class TrendModelService:
    def __init__(self) -> None:
        self.models: dict[str, dict[str, Any]] = {}

    def _key(self, symbol: str, source_name: str | None = None, use_standard_attention: bool = False) -> str:
        source = source_name or symbol
        safe_source = re.sub(r"[^A-Za-z0-9_.-]+", "_", source)
        return f"{safe_source}__{'standard' if use_standard_attention else 'asta'}"

    def _checkpoint_path(self, symbol: str, source_name: str | None = None, use_standard_attention: bool = False) -> Path:
        return MODELS_DIR / f"{self._key(symbol, source_name, use_standard_attention)}.pt"

    @staticmethod
    def _bundle_to_state(bundle: ProcessedBundle) -> dict[str, Any]:
        return {
            "symbol": bundle.symbol,
            "source_path": str(bundle.source_path),
            "feature_names": list(bundle.feature_names),
            "window_size": bundle.window_size,
            "horizons": list(bundle.horizons),
            "train_fraction": bundle.train_fraction,
            "train_x": bundle.train_x,
            "train_y": bundle.train_y,
            "val_x": bundle.val_x,
            "val_y": bundle.val_y,
            "all_x": bundle.all_x,
            "all_y": bundle.all_y,
            "raw_windows": bundle.raw_windows,
            "dates": bundle.dates,
            "closes": bundle.closes,
            "mean": bundle.mean,
            "std": bundle.std,
        }

    @staticmethod
    def _bundle_from_state(state: dict[str, Any]) -> ProcessedBundle:
        return ProcessedBundle(
            symbol=state["symbol"],
            source_path=Path(state["source_path"]),
            feature_names=tuple(state["feature_names"]),
            window_size=int(state["window_size"]),
            horizons=tuple(int(h) for h in state["horizons"]),
            train_fraction=float(state["train_fraction"]),
            train_x=state["train_x"],
            train_y=state["train_y"],
            val_x=state["val_x"],
            val_y=state["val_y"],
            all_x=state["all_x"],
            all_y=state["all_y"],
            raw_windows=state["raw_windows"],
            dates=list(state["dates"]),
            closes=state["closes"],
            mean=state["mean"],
            std=state["std"],
        )

    @staticmethod
    def _model_config(model: MultiHorizonTrendTransformer) -> dict[str, Any]:
        return {
            "feature_size": model.feature_size,
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers,
            "num_heads": model.num_heads,
            "horizons": list(model.horizons),
            "num_classes": model.num_classes,
            "use_standard_attention": model.use_standard_attention,
        }

    @staticmethod
    def _build_model(config: dict[str, Any]) -> MultiHorizonTrendTransformer:
        return MultiHorizonTrendTransformer(
            feature_size=int(config.get("feature_size", 5)),
            hidden_size=int(config.get("hidden_size", 64)),
            num_layers=int(config.get("num_layers", 2)),
            num_heads=int(config.get("num_heads", 4)),
            horizons=tuple(int(h) for h in config.get("horizons", (1, 3, 5))),
            num_classes=int(config.get("num_classes", 3)),
            use_standard_attention=bool(config.get("use_standard_attention", False)),
        )

    def _save_checkpoint(
        self,
        path: Path,
        model: MultiHorizonTrendTransformer,
        bundle: ProcessedBundle,
        metrics: dict[str, Any],
        source_name: str,
    ) -> None:
        payload = {
            "model_config": self._model_config(model),
            "model_state": model.state_dict(),
            "bundle": self._bundle_to_state(bundle),
            "metrics": metrics,
            "source_name": source_name,
        }
        torch.save(payload, path)

    def _load_checkpoint(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        payload = torch.load(path, map_location=DEVICE, weights_only=False)
        model = self._build_model(payload["model_config"]).to(DEVICE)
        model.load_state_dict(payload["model_state"])
        model.eval()
        bundle = self._bundle_from_state(payload["bundle"])
        metrics = dict(payload.get("metrics", {}))
        source_name = str(payload.get("source_name", bundle.symbol))
        return {
            "model": model,
            "bundle": bundle,
            "metrics": metrics,
            "source_name": source_name,
        }

    def _store_loaded_model(self, key: str, loaded: dict[str, Any]) -> None:
        self.models[key] = loaded

    def _bundle_from_symbol(self, symbol: str) -> ProcessedBundle:
        source_path = resolve_csv_path(DATASET_DIR, symbol)
        return build_processed_bundle(
            symbol=symbol,
            source_path=source_path,
            window_size=DEFAULT_WINDOW_SIZE,
            horizons=DEFAULT_HORIZONS,
            trend_threshold=DEFAULT_TREND_THRESHOLD,
            train_fraction=DEFAULT_TRAIN_FRACTION,
        )

    def _bundle_from_upload(self, file_name: str, frame: pd.DataFrame) -> ProcessedBundle:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
            frame.to_csv(temp_file.name, index=False)
            temp_path = Path(temp_file.name)
        try:
            return build_processed_bundle(
                symbol=Path(file_name).stem,
                source_path=temp_path,
                window_size=DEFAULT_WINDOW_SIZE,
                horizons=DEFAULT_HORIZONS,
                trend_threshold=DEFAULT_TREND_THRESHOLD,
                train_fraction=DEFAULT_TRAIN_FRACTION,
            )
        finally:
            temp_path.unlink(missing_ok=True)

    def _make_loader(self, features: np.ndarray, labels: np.ndarray, batch_size: int) -> DataLoader:
        x_tensor = torch.from_numpy(features).float()
        y_tensor = torch.from_numpy(labels).long()
        return DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=True)

    def train(
        self,
        symbol: str,
        epochs: int = 6,
        batch_size: int = 32,
        source_name: str | None = None,
        upload_frame: pd.DataFrame | None = None,
        use_standard_attention: bool = False,
    ) -> dict[str, Any]:
        key = self._key(symbol, source_name, use_standard_attention)
        checkpoint_path = self._checkpoint_path(symbol, source_name, use_standard_attention)
        if checkpoint_path.exists():
            loaded = self._load_checkpoint(checkpoint_path)
            if loaded is not None:
                self._store_loaded_model(key, loaded)
                cached_metrics = dict(loaded.get("metrics", {}))
                cached_metrics.update({"symbol": symbol, "epochs": epochs, "cached": True, "model_path": str(checkpoint_path)})
                return cached_metrics

        bundle = self._bundle_from_upload(source_name or symbol, upload_frame) if upload_frame is not None else self._bundle_from_symbol(symbol)
        model = MultiHorizonTrendTransformer(horizons=bundle.horizons, use_standard_attention=use_standard_attention).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        train_loader = self._make_loader(bundle.train_x, bundle.train_y, batch_size=batch_size)
        val_loader = self._make_loader(bundle.val_x, bundle.val_y, batch_size=batch_size) if len(bundle.val_x) else None
        vote_weights = model.vote_weights.to(DEVICE)

        model.train()
        train_loss = 0.0
        for _ in range(epochs):
            running_loss = 0.0
            for features, labels in train_loader:
                features = features.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(features)
                horizon_logits = outputs["horizon_logits"]
                per_head_loss = 0.0
                for head_index in range(horizon_logits.shape[1]):
                    per_head_loss = per_head_loss + criterion(horizon_logits[:, head_index, :], labels[:, head_index])
                combined_probs = torch.softmax(horizon_logits, dim=-1) * vote_weights.view(1, -1, 1)
                combined_logits = torch.log(combined_probs.sum(dim=1).clamp_min(1e-8))
                vote_target = MultiHorizonTrendTransformer.weighted_vote_targets(labels, vote_weights=vote_weights)
                loss = (per_head_loss / horizon_logits.shape[1]) + 0.5 * criterion(combined_logits, vote_target)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())
            train_loss = running_loss / max(len(train_loader), 1)

        val_loss = 0.0
        correct = 0
        total = 0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.to(DEVICE)
                    labels = labels.to(DEVICE)
                    outputs = model(features)
                    horizon_logits = outputs["horizon_logits"]
                    per_head_loss = 0.0
                    for head_index in range(horizon_logits.shape[1]):
                        per_head_loss = per_head_loss + criterion(horizon_logits[:, head_index, :], labels[:, head_index])
                    combined_probs = torch.softmax(horizon_logits, dim=-1) * vote_weights.view(1, -1, 1)
                    combined_logits = torch.log(combined_probs.sum(dim=1).clamp_min(1e-8))
                    vote_target = MultiHorizonTrendTransformer.weighted_vote_targets(labels, vote_weights=vote_weights)
                    loss = (per_head_loss / horizon_logits.shape[1]) + 0.5 * criterion(combined_logits, vote_target)
                    val_loss += float(loss.item())
                    prediction = combined_logits.argmax(dim=-1)
                    correct += int((prediction == vote_target).sum().item())
                    total += int(vote_target.numel())
            val_loss = val_loss / max(len(val_loader), 1)
        else:
            model.eval()

        accuracy = correct / total if total else 0.0

        benchmark = compare_runtime_standard_vs_asta(sequence_length=bundle.window_size, hidden_size=model.hidden_size, batch_size=4, runs=5, device=DEVICE)
        metrics = {
            "symbol": symbol,
            "epochs": epochs,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": accuracy,
            "runtime_standard_ms": benchmark.standard_ms,
            "runtime_asta_ms": benchmark.asta_ms,
            "runtime_speedup": benchmark.speedup,
            "sample_count": int(bundle.all_x.shape[0]),
            "cached": False,
            "model_path": str(checkpoint_path),
            "attention_mode": model.attention_mode,
        }
        self._store_loaded_model(key, {"model": model, "bundle": bundle, "metrics": metrics, "source_name": source_name or symbol})
        self._save_checkpoint(checkpoint_path, model, bundle, metrics, source_name or symbol)
        return metrics

    def predict(self, symbol: str, frame: pd.DataFrame | None = None, use_standard_attention: bool = False) -> dict[str, Any]:
        key = self._key(symbol, None, use_standard_attention)
        checkpoint_path = self._checkpoint_path(symbol, None, use_standard_attention)
        if key not in self.models:
            loaded = self._load_checkpoint(checkpoint_path)
            if loaded is not None:
                self._store_loaded_model(key, loaded)
            elif frame is not None:
                self.train(symbol=symbol, source_name=symbol, upload_frame=frame, use_standard_attention=use_standard_attention)
            else:
                self.train(symbol=symbol, use_standard_attention=use_standard_attention)
        model: MultiHorizonTrendTransformer = self.models[key]["model"]
        bundle: ProcessedBundle = self.models[key]["bundle"]
        latest_window = torch.from_numpy(bundle.all_x[-1:]).float().to(DEVICE)
        prediction = model.predict(latest_window)
        horizon_order = list(prediction.horizon_predictions.items())
        short_term_prediction = prediction.horizon_predictions.get("short_term", horizon_order[0][1] if horizon_order else {})
        mid_term_prediction = prediction.horizon_predictions.get("mid_term", horizon_order[1][1] if len(horizon_order) > 1 else short_term_prediction)
        long_term_prediction = prediction.horizon_predictions.get("long_term", horizon_order[2][1] if len(horizon_order) > 2 else mid_term_prediction)
        return {
            "symbol": symbol,
            "label": prediction.label,
            "confidence": prediction.confidence,
            "probabilities": prediction.probabilities,
            "horizon_probabilities": prediction.horizon_probabilities,
            "horizon_labels": prediction.horizon_labels,
            "horizon_confidences": prediction.horizon_confidences,
            "horizon_predictions": prediction.horizon_predictions,
            "final_prediction": {
                "label": prediction.label,
                "confidence": prediction.confidence,
                "probabilities": prediction.probabilities,
            },
            "short_term_prediction": short_term_prediction,
            "mid_term_prediction": mid_term_prediction,
            "long_term_prediction": long_term_prediction,
            "sample_count": int(bundle.all_x.shape[0]),
            "attention_mode": model.attention_mode,
        }

    def get_bundle(self, symbol: str) -> ProcessedBundle:
        if symbol in self.models:
            return self.models[symbol]["bundle"]
        return self._bundle_from_symbol(symbol)

    def ensure_model(
        self,
        symbol: str,
        use_standard_attention: bool = False,
        frame: pd.DataFrame | None = None,
    ) -> tuple[MultiHorizonTrendTransformer, ProcessedBundle, str]:
        key = self._key(symbol, None, use_standard_attention)
        checkpoint_path = self._checkpoint_path(symbol, None, use_standard_attention)
        if key in self.models:
            loaded_model = self.models[key]["model"]
            loaded_bundle = self.models[key]["bundle"]
            return loaded_model, loaded_bundle, key
        loaded = self._load_checkpoint(checkpoint_path)
        if loaded is not None:
            self._store_loaded_model(key, loaded)
            return loaded["model"], loaded["bundle"], key
        if frame is not None:
            self.train(symbol=symbol, upload_frame=frame, use_standard_attention=use_standard_attention)
        else:
            self.train(symbol=symbol, use_standard_attention=use_standard_attention)
        return self.models[key]["model"], self.models[key]["bundle"], key

    def get_model_stats(self, symbol: str | None = None, use_standard_attention: bool = False) -> dict[str, Any]:
        bundle = self.get_bundle(symbol) if symbol else self._bundle_from_symbol(list_symbols(DATASET_DIR)[0]["symbol"])
        benchmark = compare_runtime_standard_vs_asta(sequence_length=bundle.window_size, hidden_size=64, batch_size=4, runs=5, device=DEVICE)
        return {
            "symbol": symbol or bundle.symbol,
            "attention_mode": "standard" if use_standard_attention else "asta",
            "runtime": {
                "standard_ms": benchmark.standard_ms,
                "asta_ms": benchmark.asta_ms,
                "speedup": benchmark.speedup,
            },
            "complexity": {
                "standard": "O(T^2 × d)",
                "asta": "O(T log T × d) approx.",
            },
            "selection_strategy": {
                "local": "last k timesteps",
                "log_sparse": "logarithmic past points",
                "volatility": "highest-variance timesteps",
            },
        }

    def predict_future_date(
        self,
        symbol: str,
        future_date: str,
        use_standard_attention: bool = False,
        frame: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        model, bundle, _ = self.ensure_model(symbol=symbol, use_standard_attention=use_standard_attention, frame=frame)
        result = forecast_for_date(model=model, bundle=bundle, future_date=future_date, device=DEVICE)
        return {
            "symbol": symbol,
            "future_date": result.target_date,
            "step_offset": result.step_offset,
            "label": result.label,
            "confidence": result.confidence,
            "estimated_price": result.predicted_close,
            "price_range": {"low": result.low, "high": result.high},
            "explanation": result.explanation,
            "forecast_curve": result.forecast_curve,
            "focus_timesteps": result.focus_timesteps,
            "volatility_scores": result.volatility_scores,
            "market_regime": result.market_regime,
            "regime_badge": regime_badge_color(result.market_regime),
        }

    def forecast(
        self,
        symbol: str,
        steps: int = 7,
        use_standard_attention: bool = False,
        frame: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        model, bundle, _ = self.ensure_model(symbol=symbol, use_standard_attention=use_standard_attention, frame=frame)
        forecast = recursive_forecast(model=model, bundle=bundle, steps=steps, device=DEVICE)
        return {
            "symbol": symbol,
            "steps": steps,
            "label": forecast["label"],
            "confidence": forecast["confidence"],
            "predicted_price": forecast["predicted_close"],
            "price_range": {"low": forecast["low"], "high": forecast["high"]},
            "forecast_curve": forecast["forecast_curve"],
            "focus_timesteps": forecast["focus_timesteps"],
            "volatility_scores": forecast["volatility_scores"],
            "market_regime": forecast["market_regime"],
            "explanation": forecast["explanation"],
            "regime_badge": regime_badge_color(forecast["market_regime"]),
        }

    def market_regime(self, symbol: str, use_standard_attention: bool = False, frame: pd.DataFrame | None = None) -> dict[str, Any]:
        _, bundle, _ = self.ensure_model(symbol=symbol, use_standard_attention=use_standard_attention, frame=frame)
        raw_window = bundle.raw_windows[-1] if len(bundle.raw_windows) else np.zeros((bundle.window_size, 5), dtype=np.float32)
        volatility_scores = raw_window.var(axis=1).tolist()
        regime = market_regime_from_series(bundle.closes, volatility_scores)
        slope_hint = float(np.polyfit(np.arange(len(bundle.closes)), bundle.closes, 1)[0]) if len(bundle.closes) > 2 else 0.0
        return {
            "symbol": symbol,
            "market_regime": regime,
            "badge": regime_badge_color(regime),
            "volatility_scores": volatility_scores,
            "trend_slope_hint": slope_hint,
        }


service = TrendModelService()


def _fetch_pakistan_news(limit: int = 8) -> list[dict[str, Any]]:
    feeds = [
        ("Dawn", "https://www.dawn.com/feeds/business"),
        ("The Express Tribune", "https://tribune.com.pk/feed/business"),
    ]
    news_items: list[dict[str, Any]] = []

    for source_name, url in feeds:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ASTA-NewsFetcher/1.0"})
            with urllib.request.urlopen(request, timeout=6) as response:
                payload = response.read()
            root = ET.fromstring(payload)
            for item in root.findall(".//item")[:limit]:
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                pub_date = (item.findtext("pubDate") or "").strip()
                if not title or not link:
                    continue
                news_items.append(
                    {
                        "title": title,
                        "link": link,
                        "source": source_name,
                        "published_at": pub_date,
                    }
                )
                if len(news_items) >= limit:
                    return news_items
        except (urllib.error.URLError, ET.ParseError, TimeoutError, ValueError):
            continue

    if news_items:
        return news_items[:limit]

    now = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M UTC")
    return [
        {
            "title": "Pakistan market headlines are temporarily unavailable.",
            "link": "https://www.dawn.com/business",
            "source": "System",
            "published_at": now,
        }
    ]


@app.get("/")
def root() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/dashboard")
def dashboard_page() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "dashboard.html")


@app.get("/charts")
def charts_page() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "charts.html")


@app.get("/signals")
def signals_page() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "signals.html")


@app.get("/market")
def market_page() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "market.html")


@app.get("/portfolio")
def portfolio_page() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "portfolio.html")


@app.get("/insights")
def insights_page() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "insights.html")



@app.get("/stocks")
def stocks() -> dict[str, Any]:
    return {"stocks": list_symbols(DATASET_DIR)}


@app.get("/data")
def data(symbol: str) -> JSONResponse:
    bundle = service.get_bundle(symbol)
    payload = to_processed_payload(bundle)
    raw_window = bundle.raw_windows[-1] if len(bundle.raw_windows) else None
    if raw_window is not None:
        volatility_scores = raw_window.var(axis=1)
        volatility_indices = [int(index) for index in np.argsort(volatility_scores)[-6:]]
        focus_timesteps = sorted(set(volatility_indices + list(range(max(0, raw_window.shape[0] - 8), raw_window.shape[0]))))
        payload.update(
            {
                "current_price": float(raw_window[-1, 3]),
                "current_volume": float(raw_window[-1, 4]),
                "trend_direction": "Uptrend" if raw_window[-1, 3] >= raw_window[0, 3] else "Downtrend",
                "focus_timesteps": focus_timesteps,
                "volatility_spikes": [int(index) for index in np.argsort(volatility_scores)[-6:]],
                "volatility_scores": volatility_scores.tolist(),
                "recent_open_series": raw_window[:, 0].tolist(),
                "recent_high_series": raw_window[:, 1].tolist(),
                "recent_low_series": raw_window[:, 2].tolist(),
                "recent_close_series": raw_window[:, 3].tolist(),
                "recent_volume_series": raw_window[:, 4].tolist(),
            }
        )
    return JSONResponse(payload)


async def _extract_payload(request: Request) -> tuple[dict[str, Any], pd.DataFrame | None]:
    payload: dict[str, Any] = {}
    upload_frame: pd.DataFrame | None = None
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        for key in ("symbol", "stock", "epochs", "batch_size", "future_date", "steps", "horizon"):
            if key in form and form.get(key) is not None:
                payload[key] = form.get(key)
        if form.get("use_standard_attention") is not None:
            payload["use_standard_attention"] = form.get("use_standard_attention")
        form_file = form.get("file")
        if isinstance(form_file, UploadFile) or getattr(form_file, "filename", None):
            file_content = await form_file.read()
            upload_frame = read_stock_frame(io.BytesIO(file_content))
            payload["symbol"] = Path(form_file.filename or "uploaded").stem
    elif "application/json" in content_type:
        payload = await request.json()
    else:
        raw_body = await request.body()
        if raw_body:
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except json.JSONDecodeError:
                payload = {}

    return payload, upload_frame


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@app.post("/train")
async def train(request: Request) -> JSONResponse:
    payload, upload_frame = await _extract_payload(request)
    symbol = str(payload.get("symbol") or payload.get("stock") or "").strip()
    if not symbol:
        return JSONResponse({"detail": "symbol is required"}, status_code=400)
    epochs = int(payload.get("epochs", 6))
    batch_size = int(payload.get("batch_size", 32))
    use_standard_attention = _as_bool(payload.get("use_standard_attention"), default=False)
    result = service.train(
        symbol=symbol,
        epochs=epochs,
        batch_size=batch_size,
        source_name=symbol,
        upload_frame=upload_frame,
        use_standard_attention=use_standard_attention,
    )
    return JSONResponse(result)


@app.post("/predict")
async def predict(request: Request) -> JSONResponse:
    payload, upload_frame = await _extract_payload(request)
    symbol = str(payload.get("symbol") or payload.get("stock") or "").strip()
    if not symbol:
        return JSONResponse({"detail": "symbol is required"}, status_code=400)
    use_standard_attention = _as_bool(payload.get("use_standard_attention"), default=False)
    result = service.predict(symbol=symbol, frame=upload_frame, use_standard_attention=use_standard_attention)
    return JSONResponse(result)


@app.get("/model-stats")
def model_stats(symbol: str | None = None, use_standard_attention: bool = False) -> JSONResponse:
    return JSONResponse(service.get_model_stats(symbol=symbol, use_standard_attention=use_standard_attention))


@app.post("/predict-date")
async def predict_date(request: Request) -> JSONResponse:
    payload, upload_frame = await _extract_payload(request)
    symbol = str(payload.get("symbol") or payload.get("stock") or "").strip()
    future_date = str(payload.get("future_date") or payload.get("date") or "").strip()
    if not symbol:
        return JSONResponse({"detail": "symbol is required"}, status_code=400)
    if not future_date:
        return JSONResponse({"detail": "future_date is required"}, status_code=400)
    use_standard_attention = _as_bool(payload.get("use_standard_attention"), default=False)
    result = service.predict_future_date(symbol=symbol, future_date=future_date, use_standard_attention=use_standard_attention, frame=upload_frame)
    return JSONResponse(result)


@app.post("/forecast")
async def forecast(request: Request) -> JSONResponse:
    payload, upload_frame = await _extract_payload(request)
    symbol = str(payload.get("symbol") or payload.get("stock") or "").strip()
    if not symbol:
        return JSONResponse({"detail": "symbol is required"}, status_code=400)
    steps = int(payload.get("steps") or payload.get("horizon") or 7)
    use_standard_attention = _as_bool(payload.get("use_standard_attention"), default=False)
    result = service.forecast(symbol=symbol, steps=steps, use_standard_attention=use_standard_attention, frame=upload_frame)
    return JSONResponse(result)


@app.get("/market-regime")
def market_regime(symbol: str, use_standard_attention: bool = False) -> JSONResponse:
    return JSONResponse(service.market_regime(symbol=symbol, use_standard_attention=use_standard_attention))


@app.get("/news")
def news(limit: int = 8) -> JSONResponse:
    safe_limit = max(1, min(int(limit), 20))
    return JSONResponse({"news": _fetch_pakistan_news(limit=safe_limit)})
