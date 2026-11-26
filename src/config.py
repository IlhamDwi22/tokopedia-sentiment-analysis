"""Configuration constants, artifact paths, and NLP dictionaries."""

from pathlib import Path
from typing import Dict, Set

# Base directory paths
BASE_DIR: Path = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
STATIC_DIR: Path = BASE_DIR / "static"

# Model & processing constants
MAX_LEN: int = 50
MAX_WORDS: int = 8000
MAX_BATCH_ROWS: int = 2000
NEUTRAL_FALLBACK_CONFIDENCE: float = 90.0
SUPPORTED_MODELS: tuple[str, ...] = ("bilstm", "svm", "logreg")

# Artifact file paths
ARTIFACT_PATHS: Dict[str, Path] = {
    "bilstm": ARTIFACTS_DIR / "model_bilstm.keras",
    "svm": ARTIFACTS_DIR / "model_svm.joblib",
    "logreg": ARTIFACTS_DIR / "model_logreg.joblib",
    "tfidf": ARTIFACTS_DIR / "tfidf_vectorizer.joblib",
    "bow": ARTIFACTS_DIR / "bow_vectorizer.joblib",
    "tokenizer": ARTIFACTS_DIR / "tokenizer.joblib",
    "label_encoder": ARTIFACTS_DIR / "label_encoder.joblib",
    "unified_lexicon": ARTIFACTS_DIR / "unified_lexicon.joblib",
}

# Emoji-to-sentiment-word mapping (applied before removing non-ASCII characters)
EMOJI_DICT: Dict[str, str] = {
    "👎": " buruk ",
    "😡": " buruk ",
    "😠": " buruk ",
    "🤬": " buruk ",
    "😢": " buruk ",
    "😭": " buruk ",
    "😤": " buruk ",
    "🤮": " buruk ",
    "💩": " buruk ",
    "👍": " bagus ",
    "❤️": " bagus ",
    "😍": " bagus ",
    "🥰": " bagus ",
    "😊": " bagus ",
    "😁": " bagus ",
    "🔥": " bagus ",
    "⭐": " bagus ",
}

# Indonesian slang & code-mixing normalization dictionary
SLANG_DICT: Dict[str, str] = {
    "gk": "tidak",
    "gak": "tidak",
    "ga": "tidak",
    "nggak": "tidak",
    "enggak": "tidak",
    "tdk": "tidak",
    "g": "tidak",
    "bgt": "banget",
    "bgtt": "banget",
    "bget": "banget",
    "yg": "yang",
    "utk": "untuk",
    "dgn": "dengan",
    "krn": "karena",
    "karna": "karena",
    "klo": "kalau",
    "kalo": "kalau",
    "udh": "sudah",
    "udah": "sudah",
    "sdh": "sudah",
    "blm": "belum",
    "blom": "belum",
    "bs": "bisa",
    "bisa": "dapat",
    "d": "di",
    "pdhl": "padahal",
    "pdahal": "padahal",
    "sampek": "sampai",
    "apk": "aplikasi",
    "app": "aplikasi",
    "aplikasinya": "aplikasi",
    "tokped": "tokopedia",
    "mantap": "bagus",
    "mantul": "bagus",
    "jos": "bagus",
    "oke": "bagus",
    "ok": "bagus",
    "good": "bagus",
    "best": "bagus",
    "nice": "bagus",
    "great": "bagus",
    "love": "bagus",
    "lemot": "lambat",
    "lelet": "lambat",
    "ngelag": "lambat",
    "lag": "lambat",
    "slow": "lambat",
    "eror": "error",
    "bug": "error",
    "force": "error",
    "close": "error",
    "fc": "error",
    "gone": "hilang",
    "lost": "hilang",
    "bad": "buruk",
    "worst": "buruk",
    "poor": "buruk",
    "mahal": "buruk",
    "kecewa": "buruk",
    "parah": "buruk",
    "jelek": "buruk",
    "ampas": "buruk",
    "bantu": "membantu",
    "mudah": "memudahkan",
    "cepat": "bagus",
    "cepet": "bagus",
    "ongkir": "ongkos kirim",
    "ongkirnya": "ongkos kirim",
    "free": "gratis",
    "promo": "diskon",
    "voucher": "diskon",
    "cs": "layanan",
    "admin": "layanan",
    "respon": "layanan",
    "kurir": "pengiriman",
    "packing": "kemasan",
    "seller": "penjual",
    "buyer": "pembeli",
    "cod": "bayar ditempat",
    "sy": "saya",
    "aku": "saya",
    "gw": "saya",
    "gua": "saya",
    "aja": "saja",
    "doang": "saja",
    "tp": "tapi",
    "tpi": "tapi",
    "sm": "sama",
    "dr": "dari",
    "dlm": "dalam",
}

# Additional basic English stopwords frequently seen in Play Store reviews
ENGLISH_STOPWORDS: Set[str] = {"is", "the", "to", "and", "in", "of", "for"}

# Negation and sentiment-critical words retained during stopword removal
IMPORTANT_WORDS: Set[str] = {
    "tidak",
    "bukan",
    "jangan",
    "kurang",
    "belum",
    "tapi",
    "tetapi",
    "sangat",
    "banget",
    "lebih",
    "baik",
    "buruk",
    "bagus",
    "lama",
    "hilang",
    "bisa",
    "dapat",
    "masalah",
}
