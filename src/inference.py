"""Model artifact loader and multi-model sentiment inference router (ONNX Runtime + Scikit-Learn)."""

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
import onnxruntime as ort

from src.config import (
    ARTIFACT_PATHS,
    MAX_LEN,
    MAX_WORDS,
    NEUTRAL_FALLBACK_CONFIDENCE,
    SUPPORTED_MODELS,
)
from src.preprocessing import clean_and_normalize_text, get_stopwords

logger = logging.getLogger(__name__)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Compute numerically stable softmax probabilities across the last axis."""
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    shifted = arr - np.max(arr, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


def _texts_to_padded_sequences(
    texts: Sequence[str], tokenizer_cfg: Dict[str, Any]
) -> np.ndarray:
    """Convert cleaned texts into padded INT64 token ID sequences without Keras."""
    word_index: Dict[str, int] = tokenizer_cfg["word_index"]
    num_words: int = int(tokenizer_cfg.get("num_words", MAX_WORDS))
    oov_token: str = str(tokenizer_cfg.get("oov_token", "<OOV>"))
    oov_idx: int = int(word_index.get(oov_token, 1))
    max_len: int = int(tokenizer_cfg.get("max_len", MAX_LEN))

    sequences = np.zeros((len(texts), max_len), dtype=np.int64)
    for row_idx, text in enumerate(texts):
        pos = 0
        for token in str(text).lower().split():
            if pos >= max_len:
                break
            idx = word_index.get(token)
            if idx is None or idx >= num_words:
                idx = oov_idx
            sequences[row_idx, pos] = idx
            pos += 1
    return sequences


class SentimentEngine:
    """Loads all trained NLP artifacts and routes single or batch sentiment inference."""

    def __init__(self) -> None:
        self.bilstm_session: Optional[ort.InferenceSession] = None
        self.bilstm_input_name: str = "input_ids"
        self.bilstm_output_name: str = "probabilities"
        self.tokenizer_cfg: Dict[str, Any] = {}
        self.model_svm: Any = None
        self.model_logreg: Any = None
        self.tfidf: Any = None
        self.bow: Any = None
        self.label_encoder: Any = None
        self.unified_lexicon: Dict[str, int] = {}
        self.models_loaded: List[str] = []

    def load_artifacts(self) -> None:
        """Load ONNX Bi-LSTM, Scikit-Learn models, tokenizer JSON, encoder, and lexicon."""
        # Warm up NLTK stopwords cache at startup
        get_stopwords()

        missing = [
            f"{key} ({path.name})"
            for key, path in ARTIFACT_PATHS.items()
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Artefak model belum lengkap di '{ARTIFACT_PATHS['bilstm'].parent}': "
                + ", ".join(missing)
            )

        logger.info("Loading label encoder & unified lexicon...")
        self.label_encoder = joblib.load(ARTIFACT_PATHS["label_encoder"])
        self.unified_lexicon = joblib.load(ARTIFACT_PATHS["unified_lexicon"])

        loaded: List[str] = []

        logger.info("Loading CNN + Bi-LSTM ONNX session and JSON tokenizer...")
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 2
        sess_opts.inter_op_num_threads = 1
        self.bilstm_session = ort.InferenceSession(
            str(ARTIFACT_PATHS["bilstm"]),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.bilstm_input_name = self.bilstm_session.get_inputs()[0].name
        self.bilstm_output_name = self.bilstm_session.get_outputs()[0].name
        self.tokenizer_cfg = json.loads(
            ARTIFACT_PATHS["tokenizer"].read_text(encoding="utf-8")
        )
        loaded.append("bilstm")

        logger.info("Loading SVM (LinearSVC) model and TF-IDF vectorizer...")
        self.model_svm = joblib.load(ARTIFACT_PATHS["svm"])
        self.tfidf = joblib.load(ARTIFACT_PATHS["tfidf"])
        loaded.append("svm")

        logger.info("Loading Logistic Regression model and BoW vectorizer...")
        self.model_logreg = joblib.load(ARTIFACT_PATHS["logreg"])
        self.bow = joblib.load(ARTIFACT_PATHS["bow"])
        loaded.append("logreg")

        self.models_loaded = loaded
        logger.info("All sentiment models loaded successfully: %s", self.models_loaded)

    def _validate_model_type(self, model_type: str) -> str:
        normalized = (model_type or "bilstm").strip().lower()
        if normalized not in SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{model_type}' tidak didukung. Pilih salah satu dari: {', '.join(SUPPORTED_MODELS)}"
            )
        if normalized not in self.models_loaded:
            raise RuntimeError(f"Model '{normalized}' belum dimuat ke memori.")
        return normalized

    def _has_lexicon_signal(self, clean_text: str) -> bool:
        """Return True if clean_text contains at least one polarity keyword."""
        if not clean_text:
            return False
        return any(word in self.unified_lexicon for word in clean_text.split())

    def _predict_probabilities(
        self, clean_texts: Sequence[str], model_type: str
    ) -> np.ndarray:
        """Compute class probability matrix of shape (N, num_classes) for the chosen model."""
        if model_type == "bilstm":
            padded = _texts_to_padded_sequences(clean_texts, self.tokenizer_cfg)
            probs = self.bilstm_session.run(
                [self.bilstm_output_name], {self.bilstm_input_name: padded}
            )[0]
            return np.asarray(probs, dtype=np.float64)

        if model_type == "svm":
            features = self.tfidf.transform(clean_texts)
            decision_scores = self.model_svm.decision_function(features)
            return _softmax(decision_scores)

        if model_type == "logreg":
            features = self.bow.transform(clean_texts)
            probs = self.model_logreg.predict_proba(features)
            return np.asarray(probs, dtype=np.float64)

        raise ValueError(f"Unsupported model_type: {model_type}")

    def predict_one(self, text: str, model_type: str = "bilstm") -> Dict[str, Any]:
        """Predict sentiment for a single review string."""
        selected_model = self._validate_model_type(model_type)
        clean_text = clean_and_normalize_text(text)

        if not self._has_lexicon_signal(clean_text):
            return {
                "original_text": text,
                "clean_text": clean_text,
                "sentiment": "Netral",
                "confidence": NEUTRAL_FALLBACK_CONFIDENCE,
                "model_used": selected_model,
            }

        probs = self._predict_probabilities([clean_text], selected_model)[0]
        pred_idx = int(np.argmax(probs))
        label = str(self.label_encoder.inverse_transform([pred_idx])[0])
        confidence = round(float(probs[pred_idx] * 100.0), 2)

        return {
            "original_text": text,
            "clean_text": clean_text,
            "sentiment": label,
            "confidence": confidence,
            "model_used": selected_model,
        }

    def predict_batch(
        self, texts: Sequence[str], model_type: str = "bilstm"
    ) -> List[Dict[str, Any]]:
        """Run vectorized batch sentiment prediction with Neutral Fallback."""
        selected_model = self._validate_model_type(model_type)
        cleaned_list = [clean_and_normalize_text(t) for t in texts]

        active_indices: List[int] = []
        active_texts: List[str] = []
        for idx, cleaned in enumerate(cleaned_list):
            if self._has_lexicon_signal(cleaned):
                active_indices.append(idx)
                active_texts.append(cleaned)

        results: List[Dict[str, Any]] = [
            {
                "original_text": str(orig) if orig is not None else "",
                "clean_text": cleaned,
                "sentiment": "Netral",
                "confidence": NEUTRAL_FALLBACK_CONFIDENCE,
                "model_used": selected_model,
            }
            for orig, cleaned in zip(texts, cleaned_list)
        ]

        if active_texts:
            prob_matrix = self._predict_probabilities(active_texts, selected_model)
            pred_indices = np.argmax(prob_matrix, axis=1)
            pred_labels = self.label_encoder.inverse_transform(pred_indices)

            for row_pos, orig_idx in enumerate(active_indices):
                conf = round(
                    float(prob_matrix[row_pos, pred_indices[row_pos]] * 100.0), 2
                )
                results[orig_idx]["sentiment"] = str(pred_labels[row_pos])
                results[orig_idx]["confidence"] = conf

        return results
