"""Google Play Store live review scraper and top-keyword extractor."""

from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Sequence

from google_play_scraper import Sort, reviews


def _format_review_date(raw_date: Any) -> str:
    """Format google_play_scraper datetime value into YYYY-MM-DD string."""
    if isinstance(raw_date, datetime):
        return raw_date.strftime("%Y-%m-%d")
    if raw_date is None:
        return ""
    return str(raw_date)[:10]


def scrape_playstore_reviews(app_id: str, count: int = 50) -> List[Dict[str, Any]]:
    """Fetch the latest Indonesian Play Store reviews for a given application ID.

    Args:
        app_id: Google Play package name (e.g., 'com.tokopedia.tkpd').
        count: Number of newest reviews to retrieve (1..200).

    Returns:
        List of review dicts containing userName, score, at, and content.
    """
    cleaned_app_id = (app_id or "").strip()
    if not cleaned_app_id:
        raise ValueError("App ID tidak boleh kosong.")

    safe_count = max(1, min(int(count), 200))

    raw_reviews, _ = reviews(
        cleaned_app_id,
        lang="id",
        country="id",
        sort=Sort.NEWEST,
        count=safe_count,
        filter_score_with=None,
    )

    if not raw_reviews:
        raise LookupError(
            f"Tidak ada ulasan yang ditemukan untuk App ID '{cleaned_app_id}'. "
            "Pastikan App ID Play Store valid."
        )

    formatted: List[Dict[str, Any]] = []
    for item in raw_reviews:
        content = (item.get("content") or "").strip()
        if not content:
            continue
        formatted.append(
            {
                "userName": item.get("userName") or "Pengguna Play Store",
                "score": int(item.get("score") or 0),
                "at": _format_review_date(item.get("at")),
                "content": content,
            }
        )

    if not formatted:
        raise LookupError(
            f"Ulasan untuk '{cleaned_app_id}' tidak mengandung teks yang dapat dianalisis."
        )

    return formatted


def extract_top_keywords(
    analyzed_items: Sequence[Dict[str, Any]], top_n: int = 10
) -> Dict[str, List[Dict[str, Any]]]:
    """Compute top-N most frequent tokens in Positive and Negative reviews."""
    pos_counter: Counter[str] = Counter()
    neg_counter: Counter[str] = Counter()

    for item in analyzed_items:
        sentiment = item.get("sentiment")
        clean_text = item.get("clean_text") or ""
        tokens = [tok for tok in clean_text.split() if len(tok) > 1]

        if sentiment == "Positif":
            pos_counter.update(tokens)
        elif sentiment == "Negatif":
            neg_counter.update(tokens)

    return {
        "Positif": [
            {"word": word, "count": count}
            for word, count in pos_counter.most_common(top_n)
        ],
        "Negatif": [
            {"word": word, "count": count}
            for word, count in neg_counter.most_common(top_n)
        ],
    }
