"""Extract structured features from Polish court rulings (txt files).

Usage
-----
$ python extract_features.py input_dir/*.txt -o results.jsonl

Each line of the output file is a JSON dict with the following keys:
    Typ_sprawy,
    Wiek_poszkodowanego,
    Czas_trwania_naruszenia_dni,
    Kwota_zadana,
    Procent_uszczerbku,
    Przyczynienie_procent,
    Typ_obrazenia_kategoria,
    Czy_cierpienie_znaczne,
    Czy_skutki_trwale,
    Czy_potrzeba_opieki,
    Czy_wczesniejsze_swiadczenia,
    Czy_diagnoza_psych,
    Kwota_zasadzona_ostatecznie

Dependencies
------------
https://spacy.io/usage -- manual for installation on ARM
python -m pip install spacy dateparser
python -m spacy download pl_core_news_lg
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dateparser  # type: ignore
import spacy  # type: ignore
from dateparser.search import search_dates

# Load the small Polish model only once
try:
    _NLP = spacy.load("pl_core_news_lg", disable=["ner", "parser"])
except OSError as exc:  # pragma: no cover
    raise SystemExit(
        "❗ Brak modelu spaCy dla języka polskiego. Wykonaj: "
        "python -m spacy download pl_core_news_lg"
    ) from exc

# ---------------------------------------------------------------------------
# Regex patterns & lookup dictionaries
# ---------------------------------------------------------------------------

_CURRENCY = r"(?:zł|PLN)"

AMOUNT_PATTERNS: Dict[str, List[str]] = {
    "Kwota_zadana": [
        rf"kwot[eyai]?\s+(\d[\d\s\.]*)\s*{_CURRENCY}",
        rf"żąda[ł|ła]?\s+(?:zapłaty|zasądzenia)\s+(\d[\d\s\.]*)\s*{_CURRENCY}",
    ],
    "Kwota_zasadzona_ostatecznie": [
        rf"zasądza.*?(\d[\d\s\.]*)\s*{_CURRENCY}",
        rf"przyznaje\s+(\d[\d\s\.]*)\s*{_CURRENCY}",
    ],
}

PERCENT_PATTERNS: Dict[str, List[str]] = {
    "Procent_uszczerbku": [r"(\d{1,3})\s*%\s*(?:uszczerbku|t.?zw\.?)"],
    "Przyczynienie_procent": [r"(\d{1,3})\s*%\s*(?:przyczynienia|odpowiedzialno[śćsc])"],
}

WIEK_PATTERNS: List[str] = [
    r"w\s+wieku\s+(\d{1,3})\s*lat",
    r"(\d{1,3})-letni[ea]?",
    r"lat\s+(\d{1,3})\s*(?:.|,| )",
]

# Map keywords -> high-level categories for Typ_sprawy
CASE_TYPE_MAP: Dict[str, str] = {
    "zadośćuczynienie": "Zadośćuczynienie",
    "odszkodowanie": "Odszkodowanie",
    "renta": "Renta",
    "naruszenie dóbr osobistych": "DobraOsobiste",
    "błąd medyczny": "BłądMedyczny",
}

# Map injury keywords -> category label (expand/adapt as needed)
INJURY_MAP: Dict[str, str] = {
    "kręgosłup": "UrazKręgosłupa",
    "głow": "UrazGłowy",
    "złaman": "Złamania",
    "poparzeni": "Poparzenia",
    "amputac": "Amputacje",
}

BOOLEAN_PATTERNS: Dict[str, List[str]] = {
    "Czy_cierpienie_znaczne": [r"cierpieni[ae]?\s+znaczne", r"znaczn[ea]\s+intensywność\s+cierpien"],
    "Czy_skutki_trwale": [r"skutki\s+trwałe", r"trwał[eay]?\s+uszkodzeni"],
    "Czy_potrzeba_opieki": [r"wymag[ał]?[ea]?\s+stałej?\s+opieki", r"potrzeb[aey]?\s+opieki"],
    "Czy_wczesniejsze_swiadczenia": [r"wcześniejsz[ae]?\s+świadczen", r"uzysk[ał]?[ae]?\s+już\s+.*?świadczen"],
    "Czy_diagnoza_psych": [r"diagnoz[ae]?\s+psychiatr", r"stwierdzon[ae]?\s+zaburzenia\s+psychicz"],
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _to_int(num_str: str) -> Optional[int]:
    """Convert a string with spaces/thousands separators to int."""
    cleaned = re.sub(r"[\s\.]", "", num_str)
    try:
        return int(cleaned)
    except ValueError:
        return None


def _search_patterns(text: str, patterns: List[str], flags: int = re.I) -> Optional[str]:
    for pat in patterns:
        m = re.search(pat, text, flags)
        if m:
            return m.group(1)
    return None


def extract_amount(text: str, key: str) -> Optional[int]:
    raw = _search_patterns(text, AMOUNT_PATTERNS[key])
    return _to_int(raw) if raw else None


def extract_percentage(text: str, key: str) -> Optional[int]:
    raw = _search_patterns(text, PERCENT_PATTERNS[key])
    return int(raw) if raw else None


def extract_wiek(text: str) -> Optional[int]:
    raw = _search_patterns(text, WIEK_PATTERNS)
    return int(raw) if raw else None


def extract_typ_sprawy(doc: spacy.tokens.Doc) -> Optional[str]:
    lowered = doc.text.lower()
    for kw, label in CASE_TYPE_MAP.items():
        if kw in lowered:
            return label
    return None


def extract_injury_category(doc: spacy.tokens.Doc) -> Optional[str]:
    lowered = doc.text.lower()
    for kw, label in INJURY_MAP.items():
        if kw in lowered:
            return label
    return None


def detect_boolean(text: str, key: str) -> int:
    patterns = BOOLEAN_PATTERNS[key]
    return int(any(re.search(pat, text, re.I) for pat in patterns))


def extract_duration_days(text: str) -> Optional[int]:
    """Try to compute duration between two dates mentioned in the text."""
    dates = dateparser.search.search_dates(text, languages=["pl"])
    if dates and len(dates) >= 2:
        # Take first two dates found for a rough estimate
        d1, d2 = dates[0][1], dates[1][1]
        delta = abs((d2 - d1).days)
        return delta
    return None


def extract_features(text: str) -> Dict[str, Any]:
    doc = _NLP(text)

    features: Dict[str, Any] = {
        "Typ_sprawy": extract_typ_sprawy(doc),
        "Wiek_poszkodowanego": extract_wiek(text),
        "Czas_trwania_naruszenia_dni": extract_duration_days(text),
        "Kwota_zadana": extract_amount(text, "Kwota_zadana"),
        "Procent_uszczerbku": extract_percentage(text, "Procent_uszczerbku"),
        "Przyczynienie_procent": extract_percentage(text, "Przyczynienie_procent"),
        "Typ_obrazenia_kategoria": extract_injury_category(doc),
        "Czy_cierpienie_znaczne": detect_boolean(text, "Czy_cierpienie_znaczne"),
        "Czy_skutki_trwale": detect_boolean(text, "Czy_skutki_trwale"),
        "Czy_potrzeba_opieki": detect_boolean(text, "Czy_potrzeba_opieki"),
        "Czy_wczesniejsze_swiadczenia": detect_boolean(text, "Czy_wczesniejsze_swiadczenia"),
        "Czy_diagnoza_psych": detect_boolean(text, "Czy_diagnoza_psych"),
        "Kwota_zasadzona_ostatecznie": extract_amount(text, "Kwota_zasadzona_ostatecznie"),
    }
    return features


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract features from Polish court ruling txt files.")
    ap.add_argument("inputs", nargs="+", type=Path, help="Input .txt files or glob pattern.")
    ap.add_argument("-o", "--output", type=Path, default=Path("results.jsonl"), help="Output JSONL file path.")
    return ap.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv or sys.argv[1:])

    with args.output.open("w", encoding="utf-8") as fout:
        for path in args.inputs:
            for txt_file in sorted(Path().glob(str(path))):
                text = txt_file.read_text("utf-8", errors="ignore")
                features = extract_features(text)
                features["filename"] = txt_file.name
                json.dump(features, fout, ensure_ascii=False)
                fout.write("\n")
                print(f"✔ Przetworzono {txt_file}")


if __name__ == "__main__":
    main()
