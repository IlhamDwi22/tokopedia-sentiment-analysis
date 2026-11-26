"""FastAPI monolith server serving REST API endpoints and static SPA assets."""

import base64
from contextlib import asynccontextmanager
import io
import logging
from typing import Any, Dict, List, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
from pydantic import BaseModel, Field

from src.config import MAX_BATCH_ROWS, STATIC_DIR, SUPPORTED_MODELS
from src.inference import SentimentEngine
from src.scraper import extract_top_keywords, scrape_playstore_reviews

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

engine = SentimentEngine()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load trained ML/DL artifacts once when the application starts."""
    try:
        engine.load_artifacts()
    except Exception as exc:
        logger.exception("Gagal memuat artefak model saat startup: %s", exc)
    yield


app = FastAPI(
    title="Tokopedia Play Store Sentiment Analysis API",
    description="Full-Stack AI Sentiment Analysis (CNN+Bi-LSTM, SVM, Logistic Regression)",
    version="1.0.0",
    lifespan=lifespan,
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Pydantic Request Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Teks ulasan yang akan dianalisis")
    model_type: Literal["bilstm", "svm", "logreg"] = Field(
        default="bilstm", description="Pilihan model inferensi"
    )


class ScrapeAnalyzeRequest(BaseModel):
    app_id: str = Field(
        default="com.tokopedia.tkpd",
        min_length=3,
        description="Google Play Store App ID",
    )
    count: int = Field(default=50, ge=1, le=200, description="Jumlah ulasan terbaru")
    model_type: Literal["bilstm", "svm", "logreg"] = Field(
        default="bilstm", description="Pilihan model inferensi"
    )


def _compute_distribution(predictions: List[Dict[str, Any]]) -> Dict[str, int]:
    dist = {"Positif": 0, "Netral": 0, "Negatif": 0}
    for item in predictions:
        label = item.get("sentiment", "Netral")
        if label in dist:
            dist[label] += 1
    return dist


def _read_csv_bytes(raw_bytes: bytes) -> pd.DataFrame:
    """Decode uploaded CSV bytes with UTF-8 / UTF-8-SIG / Latin-1 fallback."""
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"File CSV tidak valid atau rusak: {exc}"
            ) from exc
    raise HTTPException(
        status_code=400,
        detail="Gagal membaca encoding file CSV. Pastikan file berformat UTF-8.",
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def serve_index() -> FileResponse:
    """Serve the Single Page Application frontend."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="File static/index.html tidak ditemukan.")
    return FileResponse(index_path)


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """Return server health status and loaded model list."""
    return {
        "status": "ok" if engine.models_loaded else "degraded",
        "models_loaded": engine.models_loaded,
    }


@app.post("/api/predict")
async def predict_single(payload: PredictRequest) -> Dict[str, Any]:
    """Run sentiment inference on a single review text."""
    if not engine.models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model belum dimuat di server. Pastikan folder artifacts/ sudah tersedia.",
        )

    cleaned_input = payload.text.strip()
    if not cleaned_input:
        raise HTTPException(status_code=400, detail="Teks ulasan tidak boleh kosong.")

    try:
        return engine.predict_one(cleaned_input, model_type=payload.model_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Error saat inferensi tunggal: %s", exc)
        raise HTTPException(status_code=500, detail=f"Kesalahan inferensi: {exc}") from exc


@app.post("/api/predict-batch")
async def predict_batch_csv(
    file: UploadFile = File(...),
    text_column: str = Form("content"),
    model_type: str = Form("bilstm"),
) -> Dict[str, Any]:
    """Run batch sentiment inference on an uploaded CSV file (max 2,000 rows)."""
    if not engine.models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model belum dimuat di server. Pastikan folder artifacts/ sudah tersedia.",
        )

    filename = (file.filename or "").lower()
    if not filename.endswith(".csv"):
        raise HTTPException(
            status_code=400, detail="Format file harus .csv (Comma-Separated Values)."
        )

    normalized_model = (model_type or "bilstm").strip().lower()
    if normalized_model not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_type}' tidak valid. Pilihan: {', '.join(SUPPORTED_MODELS)}",
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="File CSV yang diunggah kosong.")

    df = _read_csv_bytes(raw_bytes)
    if df.empty:
        raise HTTPException(status_code=400, detail="File CSV tidak memiliki baris data.")

    # Resolve target text column (with helpful fallback from 'content' -> 'review')
    col_name = (text_column or "content").strip()
    if col_name not in df.columns:
        if col_name == "content" and "review" in df.columns:
            col_name = "review"
        else:
            available = ", ".join(map(str, df.columns.tolist()[:10]))
            raise HTTPException(
                status_code=400,
                detail=f"Kolom '{col_name}' tidak ditemukan di CSV. Kolom tersedia: {available}",
            )

    truncated = False
    if len(df) > MAX_BATCH_ROWS:
        df = df.head(MAX_BATCH_ROWS).copy()
        truncated = True
    else:
        df = df.copy()

    texts = df[col_name].fillna("").astype(str).tolist()
    predictions = engine.predict_batch(texts, model_type=normalized_model)

    df["clean_text"] = [p["clean_text"] for p in predictions]
    df["predicted_sentiment"] = [p["sentiment"] for p in predictions]
    df["confidence"] = [p["confidence"] for p in predictions]

    distribution = _compute_distribution(predictions)
    preview = [
        {
            "content": p["original_text"],
            "clean_text": p["clean_text"],
            "sentiment": p["sentiment"],
            "confidence": p["confidence"],
        }
        for p in predictions[:10]
    ]

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, encoding="utf-8")
    csv_base64 = base64.b64encode(csv_buffer.getvalue().encode("utf-8")).decode("ascii")

    return {
        "total_rows": len(df),
        "truncated": truncated,
        "text_column_used": col_name,
        "model_used": normalized_model,
        "distribution": distribution,
        "preview": preview,
        "csv_base64": csv_base64,
    }


@app.post("/api/scrape-analyze")
async def scrape_and_analyze(payload: ScrapeAnalyzeRequest) -> Dict[str, Any]:
    """Scrape live reviews from Google Play Store and run sentiment + keyword analysis."""
    if not engine.models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model belum dimuat di server. Pastikan folder artifacts/ sudah tersedia.",
        )

    try:
        scraped_reviews = scrape_playstore_reviews(
            app_id=payload.app_id, count=payload.count
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Gagal melakukan scraping Play Store: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Gagal mengambil ulasan dari Google Play Store: {exc}",
        ) from exc

    texts = [item["content"] for item in scraped_reviews]
    predictions = engine.predict_batch(texts, model_type=payload.model_type)

    combined_reviews: List[Dict[str, Any]] = []
    for raw, pred in zip(scraped_reviews, predictions):
        combined_reviews.append(
            {
                "userName": raw["userName"],
                "score": raw["score"],
                "at": raw["at"],
                "content": raw["content"],
                "clean_text": pred["clean_text"],
                "sentiment": pred["sentiment"],
                "confidence": pred["confidence"],
            }
        )

    distribution = _compute_distribution(predictions)
    top_keywords = extract_top_keywords(combined_reviews, top_n=10)

    # Strip clean_text from response reviews payload to keep JSON concise per PRD
    public_reviews = [
        {
            "userName": r["userName"],
            "score": r["score"],
            "at": r["at"],
            "content": r["content"],
            "sentiment": r["sentiment"],
            "confidence": r["confidence"],
        }
        for r in combined_reviews
    ]

    return {
        "app_id": payload.app_id.strip(),
        "model_used": payload.model_type,
        "total_scraped": len(public_reviews),
        "distribution": distribution,
        "top_keywords": top_keywords,
        "reviews": public_reviews,
    }
