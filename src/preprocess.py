import ast
import re
import pandas as pd
from typing import Iterable


def normalize_whitespace(text: str) -> str:
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_categories(cat_value) -> list[str]:
    if pd.isna(cat_value):
        return []

    if isinstance(cat_value, list):
        tokens = []
        for x in cat_value:
            sx = str(x).strip()
            if not sx:
                continue
            tokens.extend([c for c in sx.split() if c.strip()])
        return tokens

    s = str(cat_value).strip()
    if not s:
        return []

    # stringa serializzata tipo "['cs.IR cs.LG']" o "['cs.IR', 'cs.LG']"
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                tokens = []
                for x in parsed:
                    sx = str(x).strip()
                    if not sx:
                        continue
                    tokens.extend([c for c in sx.split() if c.strip()])
                return tokens
        except Exception:
            pass

    return [c.strip() for c in s.split() if c.strip()]

def build_url(arxiv_id: str) -> str:
    arxiv_id = str(arxiv_id).strip()
    return f"https://arxiv.org/abs/{arxiv_id}"

def preprocess_arxiv_df(
    df: pd.DataFrame,
    target_categories: Iterable[str],
    min_title_chars: int = 5,
    min_abstract_chars: int = 40,
    max_abstract_chars: int = 10000,
) -> pd.DataFrame:
    target_categories = set(target_categories)
    out = df.copy()

    # Basic cleanup
    out["id"] = out["id"].astype(str).str.strip()
    out["title"] = out["title"].fillna("").map(normalize_whitespace)
    out["abstract"] = out["abstract"].fillna("").map(normalize_whitespace)
    out["authors"] = out["authors"].fillna("").astype(str).map(normalize_whitespace)

    # Categories parsing
    out["category_list"] = out["categories"].apply(parse_categories)
    out["primary_category"] = out["category_list"].apply(lambda x: x[0] if x else None)

    # Keep docs that contain at least one target category
    out["has_target_category"] = out["category_list"].apply(
        lambda cats: any(c in target_categories for c in cats)
    )

    # Date parsing
    out["published"] = pd.to_datetime(out["published"], errors="coerce")

    # year from published
    out["year"] = out["published"].dt.year

    # fallback from versions if year is missing
    if "versions" in out.columns:
        versions_str = out["versions"].astype(str)
        extracted_year = versions_str.str.extract(r"(19\d{2}|20\d{2})", expand=False)
        extracted_year = pd.to_numeric(extracted_year, errors="coerce")
        out.loc[out["year"].isna(), "year"] = extracted_year

    # URL fallback
    if "url" not in out.columns:
        out["url"] = None
    out["url"] = out["url"].where(out["url"].notna(), out["id"].map(build_url))
    out["url"] = out["url"].astype(str).map(lambda x: x if x.startswith("http") else build_url(x))

    # Text quality filters
    out["title_len"] = out["title"].str.len()
    out["abstract_len"] = out["abstract"].str.len()
    out["has_valid_text"] = (
        (out["title_len"] >= min_title_chars)
        & (out["abstract_len"] >= min_abstract_chars)
        & (out["abstract_len"] <= max_abstract_chars)
    )

    # Combine text for indexing (MVP)
    out["doc_text"] = (out["title"] + "\n\n" + out["abstract"]).str.strip()
    out["doc_len"] = out["doc_text"].str.len()

    # Remove exact duplicate IDs (keep latest row if duplicates exist)
    # If published exists, sorting helps keep most complete recent version.
    out = out.sort_values(by=["id", "published"], ascending=[True, False], na_position="last")
    out = out.drop_duplicates(subset=["id"], keep="first")

    # Final filter
    out = out[out["has_target_category"] & out["has_valid_text"]].copy()

    # Useful column order
    cols = [
        "id", "title", "abstract", "authors", "categories", "category_list",
        "primary_category", "published", "year", "url", "doc_text",
        "title_len", "abstract_len", "doc_len"
    ]
    existing_cols = [c for c in cols if c in out.columns]
    out = out[existing_cols]

    return out.reset_index(drop=True)