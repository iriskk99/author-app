# src/data_loader.py

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import requests


OPENALEX_AUTHORS_URL = "https://api.openalex.org/authors"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"


def _request_json(url: str, params: Optional[Dict] = None) -> Dict:
    """Helper for GET requests to OpenAlex API."""
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def find_author_by_name(
    author_name: str,
    institution_hint: Optional[str] = None,
    mailto: Optional[str] = None,
    max_results: int = 25,
) -> Optional[Dict]:
    """
    Search OpenAlex authors by name and optionally institution hint.
    Returns the best-matching author dict, or None if not found.
    """
    params = {
        "search": author_name,
        "per-page": max_results,
    }
    if mailto:
        params["mailto"] = mailto

    data = _request_json(OPENALEX_AUTHORS_URL, params=params)
    results = data.get("results", [])

    if not results:
        return None

    # Prefer matches with institution hint in last_known_institution display_name
    if institution_hint:
        institution_hint_lower = institution_hint.lower()
        filtered = []
        for a in results:
            inst = a.get("last_known_institution") or {}
            inst_name = (inst.get("display_name") or "").lower()
            if institution_hint_lower in inst_name:
                filtered.append(a)
        if filtered:
            return filtered[0]

    # Fallback: just take the first result
    return results[0]


def fetch_works_for_author(
    author_id: str,
    mailto: Optional[str] = None,
    per_page: int = 200,
    max_pages: Optional[int] = None,
) -> List[Dict]:
    """
    Fetch all works for an author using OpenAlex works endpoint.

    Uses filter: authorships.author.id:<author_id>
    """
    page = 1
    all_results: List[Dict] = []

    while True:
        params = {
            "filter": f"authorships.author.id:{author_id}",
            "per-page": per_page,
            "page": page,
            "sort": "publication_year:asc",
        }
        if mailto:
            params["mailto"] = mailto

        data = _request_json(OPENALEX_WORKS_URL, params=params)
        results = data.get("results", [])
        if not results:
            break

        all_results.extend(results)

        # pagination
        meta = data.get("meta", {})
        total_pages = meta.get("last_page")
        if total_pages is not None and page >= total_pages:
            break

        if max_pages is not None and page >= max_pages:
            break

        page += 1

    return all_results


def save_raw_works_to_file(
    works: List[Dict],
    path: str,
) -> None:
    """Save raw OpenAlex works list to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(works, f, ensure_ascii=False, indent=2)


def load_raw_works_from_file(path: str) -> Optional[List[Dict]]:
    """Load raw works JSON if it exists, otherwise return None."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_author_and_works(
    author_name: str,
    institution_hint: Optional[str],
    data_dir: str = "data",
    mailto: Optional[str] = None,
    refresh: bool = False,
) -> Tuple[Dict, List[Dict]]:
    """
    Main helper:
      - Find author by name + institution hint
      - Fetch or load works
      - Persist raw JSON into data/author_works_raw.json
    """
    os.makedirs(data_dir, exist_ok=True)
    raw_path = os.path.join(data_dir, "author_works_raw.json")

    # Find author
    author = find_author_by_name(
        author_name=author_name,
        institution_hint=institution_hint,
        mailto=mailto,
    )
    if not author:
        raise RuntimeError(f"No OpenAlex author found for name '{author_name}'.")

    author_id = author["id"]

    # Raw works
    if not refresh:
        local_works = load_raw_works_from_file(raw_path)
        if local_works:
            return author, local_works

    works = fetch_works_for_author(
        author_id=author_id,
        mailto=mailto,
    )
    save_raw_works_to_file(works, raw_path)
    return author, works
