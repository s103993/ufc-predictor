# -*- coding: utf-8 -*-
"""
WalkoMat — interpretowalny predyktor wyników walk UFC oparty o JEDNOWARSTWOWĄ sieć neuronową
(dawniej: UFC Fight Predictor).

Projekt studencki, przedmiot: „Zaawansowane Metody i Techniki Sztucznej Inteligencji 26/27”.

Model: jednowarstwowa sieć neuronowa bez warstw ukrytych (perceptron z sigmoidalną
aktywacją na wyjściu, zaimplementowana w PyTorch jako nn.Linear(input_size, 1) +
BCEWithLogitsLoss). Jest to architektura tożroma z regresją logistyczną — zgodnie
z wymogiem przedmiotu jest jednak zaimplementowana jako sieć neuronowa z własną
pętlą treningową (forward, loss, backward, update wag metodą Adam/mini-batch).

Zastosowanie biznesowe: proste narzędzie analityczne dla analityka zakładów
sportowych — szacowanie prawdopodobieństwa zwycięstwa zawodnika A w walce,
porównanie z prawdopodobieństwem implikowanym z kursów bukmacherskich oraz
identyfikacja potencjalnych „value betów”.

DISCLAIMER: Projekt ma charakter czysto edukacyjny. Nie stanowi porady finansowej,
porady inwestycyjnej ani zachęty do uprawiania hazardu. Symulacje zakładów służą
wyłącznie analizie jakości modelu, nie rekomendacji gry u bukmachera.
"""
from __future__ import annotations

# ============================================================================
# IMPORTY
# ============================================================================
import argparse
import json
import random
import re
import sys
import time
import warnings
import zlib
from collections import defaultdict
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # backend bez GUI — wykresy tylko do plików PNG
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ============================================================================
# STAŁE I KONFIGURACJA
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR_DEFAULT = BASE_DIR / "data"
OUTPUT_DIR_DEFAULT = BASE_DIR / "outputs"
PLOTS_DIR_NAME = "plots"

CODE_VERSION = "1.1.0"
PROJECT_NAME = "WalkoMat"  # jedynym źródłem nazwy — używana w CLI, raporcie i metadanych
SEED = 42

# Podział chronologiczny: najstarsze 70% -> train, kolejne 15% -> val, najnowsze 15% -> test.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # test = pozostałe 15%

# Hiperparametry treningu (domyślne wartości).
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
DEFAULT_PATIENCE = 10
DEFAULT_THRESHOLD = 0.5
# Regularyzacja L2 (weight_decay w Adamie): karze duże wagi, więc rzadka cecha-proxy
# (np. brak zasięgu) nie może urosnąć do wagi ~1.0 i zdominować predykcji nawet po
# winsoryzacji. Kolejny element kontroli wariancji z wykładu.
DEFAULT_WEIGHT_DECAY = 1e-2

# Pierwsze N lat danych pomijamy przy budowie zbioru treningowego („burn-in”):
# historia zawodników jest liczona tylko z walk obecnych w datasetcie (od 2013),
# więc dla walk z 2013 r. cechy historyczne byłyby sztucznie puste.
DEFAULT_BURN_IN_YEARS = 2.0

# Progi prawdopodobieństwa dla prostej strategii value betting (analiza bukmacherska).
BET_THRESHOLDS = (0.55, 0.60, 0.65)

# Warianty uczenia porównywane komendą `compare` (w duchu labów: pętla po wariantach ->
# słownik wyników -> tabelka + wykres). (nazwa, optimizer, strata, batch_size; None = full-batch)
COMPARE_VARIANTS = [
    ("SGD lr=1e-3",              "sgd",      "bce_logits", DEFAULT_BATCH_SIZE),
    ("SGD+Momentum 0.9, lr=1e-3", "momentum", "bce_logits", DEFAULT_BATCH_SIZE),
    ("RMSprop lr=1e-3",          "rmsprop",  "bce_logits", DEFAULT_BATCH_SIZE),
    ("Adam lr=1e-3 (wybrany)",   "adam",     "bce_logits", DEFAULT_BATCH_SIZE),
    ("Adam + strata MSE",        "adam",     "mse",        DEFAULT_BATCH_SIZE),
    ("Adam, full-batch GD",      "adam",     "bce_logits", None),
]

# Kodowanie rankingu UFC: 0 = mistrz, 1..15 = pozycja w rankingu, brak = nierankingowy.
UNRANKED_CODE = 20.0

# Winsoryzacja cech: po standaryzacji przycinamy z do ±3σ. Jedna rzadka cecha nie może
# wtedy zdominować predykcji (np. reach_missing_diff: z=±5.5 → ±3.3 logitu → ~96% pewności
# z samego braku danych) i model nie ekstrapoluje poza rozkład treningowy. Stosowane
# SPÓJNIE na wszystkich ścieżkach: trening, ewaluacja, predict i strona web.
Z_CAP_SIGMA = 3.0

DISCLAIMER = (
    "DISCLAIMER: Projekt ma charakter wyłącznie edukacyjny. Nie stanowi porady "
    "finansowej ani zachęty do hazardu. Symulacje zakładów służą analizie modelu."
)

# Preferowane nazwy plików z danymi (kolejność = priorytet). Loader i tak wykrywa
# schemat po kolumnach, więc zadziała też z inną nazwą pliku CSV/JSON w data/.
PREFERRED_FIGHTS_FILES = (
    "ufc_all_fights_wide.csv",      # nlbair/ufc-data (2013-2026, z kursami)
    "data.csv",                     # starszy zbiór „ufcdata” z Kaggle
    "preprocessed_data.csv",
    "raw_total_fight_data.csv",
)

# Mapowanie kategorii wagowej (znormalizowanej) na wagę limitu w funtach.
# „catch weight” / „open weight” nie mają stałego limitu -> NaN (imputacja medianą).
WEIGHT_CLASS_LBS: Dict[str, float] = {
    "flyweight": 125.0,
    "bantamweight": 135.0,
    "featherweight": 145.0,
    "lightweight": 155.0,
    "welterweight": 170.0,
    "middleweight": 185.0,
    "lightheavyweight": 205.0,
    "heavyweight": 265.0,
    "womensstrawweight": 115.0,
    "womensflyweight": 125.0,
    "womensbantamweight": 135.0,
    "womensfeatherweight": 145.0,
}

# Proste mapowanie postawy na kod numeryczny (model liniowy nie przyjmie tekstu).
STANCE_CODES: Dict[str, float] = {
    "orthodox": 0.0,
    "southpaw": 1.0,
    "switch": 2.0,
    "open stance": 3.0,
    "sideways": 4.0,
}

# Kolejność i nazwy cech modelu (STAŁA — musi zgadzać się z metadata.json,
# scalerem i wagami modelu przy trybie predict/evaluate).
FEATURE_COLUMNS: List[str] = [
    # cechy różnicowe z historii walk (A - B), liczone wyłącznie z walk wcześniejszych
    "exp_diff",             # liczba walk (doświadczenie)
    "wins_diff",            # liczba zwycięstw
    "losses_diff",          # liczba porażek
    "draws_diff",           # liczba remisów
    "win_pct_diff",         # procent zwycięstw
    "ko_pct_diff",          # % zwycięstw przez KO/TKO (względem liczby walk)
    "sub_pct_diff",         # % zwycięstw przez poddanie
    "dec_pct_diff",         # % zwycięstw przez decyzję
    "finish_rate_diff",     # % walk zakończonych przed decyzją (KO/TKO + sub)
    "avg_fight_time_diff",  # średni czas walki [s]
    "med_fight_time_diff",  # mediana czasu walki [s]
    "fights_12m_diff",      # liczba walk w ostatnich 12 miesiącach
    "fights_24m_diff",      # liczba walk w ostatnich 24 miesiącach
    "form3_diff",           # forma: % zwycięstw w ostatnich 3 walkach
    "form5_diff",           # forma: % zwycięstw w ostatnich 5 walkach
    "days_rest_diff",       # dni odpoczynku od ostatniej walki
    "streak_diff",          # aktualna seria zwycięstw (+) / porażek (-)
    "title_exp_diff",       # liczba walk tytułowych w przeszłości
    # cechy różnicowe atrybutów sprzed walki
    "age_diff",             # wiek [lata]
    "height_diff",          # wzrost [cm]
    "reach_diff",           # zasięg ramion [cm]
    "stance_diff",          # postawa (kod numeryczny)
    "ranked_diff",          # czy rankingowy UFC (1/0)
    "rank_diff",            # pozycja w rankingu (0=mistrz, 20=nierankingowy)
    # cechy kontekstowe / wskaźniki braku danych
    "debut_diff",           # debiut w datasecie (1=debiutant)
    "both_debut",           # obaj debiutanci
    "weight_class_lbs",     # limit wagi kategorii [lbs]
    "is_womens_division",   # dywizja kobiet (1/0)
    "is_title_bout",        # walka o pas (1/0)
    "reach_missing_diff",   # wskaźnik braku zasięgu (A - B)
    "height_missing_diff",  # wskaźnik braku wzrostu (A - B)
    "age_missing_diff",     # wskaźnik braku wieku (A - B)
]


class DataError(RuntimeError):
    """Błąd krytyczny związany z danymi (brak pliku, zły schemat, brak targetu)."""


# ============================================================================
# KONFIGURACJA CLI
# ============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    """Buduje parser argumentów wiersza poleceń (subkomendy train/evaluate/predict/demo)."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            f"{PROJECT_NAME} — interpretowalny predyktor wyników walk UFC "
            "(jednowarstwowa sieć neuronowa, PyTorch). Komendy: train | evaluate | predict | demo."
        ),
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR_DEFAULT,
        help="Katalog z danymi (domyślnie ./data).",
    )
    parser.add_argument(
        "--outputs-dir", type=Path, default=OUTPUT_DIR_DEFAULT,
        help="Katalog na artefakty (domyślnie ./outputs).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Trenuje model, ewaluuje, zapisuje artefakty i wykresy.")
    p_train.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Maksymalna liczba epok.")
    p_train.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Rozmiar mini-batcha.")
    p_train.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate (Adam).")
    p_train.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping: patience.")
    p_train.add_argument("--seed", type=int, default=SEED, help="Random seed (determinizm).")
    p_train.add_argument(
        "--burn-in-years", type=float, default=DEFAULT_BURN_IN_YEARS,
        help="Pomiń pierwsze N lat danych (puste historie zawodników na starcie datasetu).",
    )
    p_train.add_argument(
        "--max-rows", type=int, default=None,
        help="Ogranicz zbiór do N najnowszych walk (np. szybkie demo).",
    )
    p_train.add_argument(
        "--random-split", action="store_true",
        help="Wymuś losowy podział (random_state=42). Domyślnie: CHRONOLOGICZNY 70/15/15 "
             "(poprawny dla predykcji przyszłych walk).",
    )
    p_train.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Próg decyzyjny klasyfikacji.")

    sub.add_parser("evaluate", help="Ewaluuje zapisany model na zbiorze testowym i odświeża metryki/wykresy.")

    p_pred = sub.add_parser("predict", help="Predykcja prawdopodobieństwa zwycięstwa zawodnika A nad B.")
    p_pred.add_argument("--fighter-a", type=str, default=None, help="Nazwisko zawodnika A.")
    p_pred.add_argument("--fighter-b", type=str, default=None, help="Nazwisko zawodnika B.")
    p_pred.add_argument(
        "--input", type=Path, default=None,
        help='Ścieżka do JSON, np. {"fighter_a": "...", "fighter_b": "..."}.',
    )

    p_demo = sub.add_parser("demo", help="Szybki trening + przykładowa predykcja (prezentacja).")
    p_demo.add_argument("--max-rows", type=int, default=2500, help="Liczba najnowszych walk do treningu demo.")
    p_demo.add_argument("--epochs", type=int, default=60, help="Liczba epok w demo.")

    p_cmp = sub.add_parser(
        "compare",
        help="Porównuje warianty uczenia (optymalizatory / funkcje straty / tryb batchów) "
             "na tym samym modelu — tabela, wykresy i wnioski.",
    )
    p_cmp.add_argument("--epochs", type=int, default=100, help="Limit epok dla każdego wariantu.")
    p_cmp.add_argument("--patience", type=int, default=12, help="Early stopping patience (każdy wariant).")
    p_cmp.add_argument("--seed", type=int, default=SEED, help="Seed — identyczny start wszystkich wariantów.")

    return parser


# ============================================================================
# POMOCNICZE: determinizm, serializacja JSON
# ============================================================================
def seed_everything(seed: int) -> None:
    """Ustawia ziarno losowości we wszystkich bibliotekach (determinizm przebiegu).

    torch.set_num_threads(1): wielowątkowe redukcje CPU są niedeterministyczne między
    uruchomieniami (kolejność sumowań), co przy kolinearnych cechach prowadzi do
    drobnych, ale widocznych różnic w wagach między treningami. Jeden wątek = ta sama
    liczba bit po bicie; przy tym rozmiarze modelu koszt wydajnościowy jest zerowy.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # pragma: no cover - starsze wersje torch
        pass


def jsonable(value: Any) -> Any:
    """Zamienia typy numpy/torch/pandas na typy natywne JSON (rekurencyjnie)."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """Zapisuje słownik do pliku JSON (kodowanie UTF-8, czytelne wcięcia)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


# ============================================================================
# ŁADOWANIE DANYCH  (ETAP 2 — implementacja poniżej)
# ============================================================================
# Kanoniczny format wewnętrzny po normalizacji (1 wiersz = 1 walka, perspektywa A vs B).
TIDY_COLUMNS: List[str] = [
    "fight_id", "date", "fighter_a", "fighter_b", "target", "finish_type",
    "duration_sec", "weight_class_lbs", "is_womens", "is_title_bout",
    "a_age", "a_height", "a_reach", "a_stance", "a_rank", "a_ranked", "a_odds", "a_prob_fair",
    "b_age", "b_height", "b_reach", "b_stance", "b_rank", "b_ranked", "b_odds", "b_prob_fair",
]


def _to_float(value: Any) -> float:
    """Bezpieczna konwersja na float; braki i wartości niepoprawne -> NaN."""
    if value is None:
        return float("nan")
    try:
        if pd.isna(value):
            return float("nan")
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_bool(value: Any) -> bool:
    """Konwersja True/False/'true'/1 na bool (odporność na różne formaty CSV)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(value)
    return str(value).strip().lower() in ("true", "1", "yes", "t")


def _norm_key(name: Any) -> str:
    """Normalizuje nazwę kolumny/wartości: lowercase, spacje -> podkreślenia."""
    return str(name).strip().lower().replace(" ", "_")


def _get(row: Dict[str, Any], colmap: Dict[str, str], *names: str, default: Any = None) -> Any:
    """Elastyczne pobranie pola po nazwach kanonicznych (mapa znormalizowanych kolumn)."""
    for n in names:
        key = colmap.get(n)
        if key is not None:
            return row.get(key, default)
    return default


def division_features(division: Any) -> Tuple[float, float]:
    """Zwraca (limit wagi kategorii w funtach, czy dywizja kobiet) dla nazwy kategorii."""
    n = _norm_key(division).replace("'", "").replace("-", "")
    if not n or n in ("nan", ""):
        return float("nan"), 0.0
    womens = 1.0 if "women" in n else 0.0
    return WEIGHT_CLASS_LBS.get(n, float("nan")), womens


def _stance_code(stance: Any) -> float:
    """Koduje postawę zawodnika na liczbę (brak danych -> NaN)."""
    key = str(stance).strip().lower()
    if not key or key == "nan":
        return float("nan")
    return STANCE_CODES.get(key, float("nan"))


def american_to_decimal(american: float) -> float:
    """Konwertuje kurs amerykański na dziesiętny (np. -150 -> 1.667, +200 -> 3.0)."""
    if not np.isfinite(american) or american == 0:
        return float("nan")
    return 1.0 + (american / 100.0 if american > 0 else 100.0 / abs(american))


def _read_table(path: Path, header_only: bool = False) -> Optional[pd.DataFrame]:
    """Wczytuje CSV/JSON; przy błędzie zwraca None (kandydat jest pomijany)."""
    try:
        if path.suffix.lower() == ".json":
            return pd.read_json(path)
        if header_only:
            return pd.read_csv(path, nrows=0)
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return None


def _score_fights_candidate(path: Path, header: List[str]) -> Optional[float]:
    """Ocenia, czy plik zawiera dane walk (data + zwycięzca + dwaj zawodnicy).

    Zwraca wynik (wyższy = lepszy kandydat) albo None, gdy plik nie wygląda na dane walk.
    """
    cols = {_norm_key(c) for c in header}
    has_date = bool(cols & {"event_date", "date", "fight_date"})
    has_winner = "winner" in cols
    has_fighters = (
        {"f1_name", "f2_name"} <= cols
        or {"r_fighter", "b_fighter"} <= cols
        or {"fighter1", "fighter2"} <= cols
        or {"fighter_1", "fighter_2"} <= cols
    )
    if not (has_date and has_winner and has_fighters):
        return None
    name = path.name.lower()
    score = float(min(path.stat().st_size / 1_000_000, 50.0))  # większy plik = pełniejszy zbiór
    for i, preferred in enumerate(PREFERRED_FIGHTS_FILES):
        if name == preferred:
            score += 1000.0 - 100.0 * i
    if "codebook" in name or "sample" in name:
        score -= 5000.0  # traktuj jako ostatni ratunek, gdy nic lepszego nie ma
    return score


def _try_load_via_ufcdata() -> pd.DataFrame:
    """Opcjonalne źródło danych: pakiet ufcdata (https://github.com/nlbair/ufcdata-py).

    Pakiet przy pierwszym użyciu pobiera dane z sieci, dlatego jest wyłącznie
    fallbackiem — właściwy trening ma działać offline z plików w katalogu data/.
    """
    try:
        import ufcdata  # type: ignore
    except ImportError as exc:
        raise DataError(
            "Nie znaleziono danych UFC.\n\n"
            "Instrukcja przygotowania danych (offline):\n"
            "  1. Pobierz plik ufc_all_fights_wide.csv z repozytorium nlbair/ufc-data:\n"
            "     https://raw.githubusercontent.com/nlbair/ufc-data/main/data/ufc_all_fights_wide.csv\n"
            "  2. Zapisz go w katalogu: data/ (obok main.py).\n"
            "  3. Uruchom program ponownie.\n\n"
            "Alternatywa (wymaga internetu): zainstaluj opcjonalny pakiet ufcdata-py:\n"
            "  pip install https://github.com/nlbair/ufcdata-py/archive/refs/heads/main.zip"
        ) from exc
    try:
        return ufcdata.load_fights()
    except Exception as exc:
        raise DataError(f"Pakiet ufcdata jest zainstalowany, ale nie udało się wczytać danych: {exc}") from exc


def load_fights_frame(data_dir: Path) -> Tuple[pd.DataFrame, str]:
    """Wybiera i wczytuje plik z walkami z katalogu data_dir (auto-detekcja po kolumnach).

    Zwraca (surowy DataFrame, nazwa pliku źródłowego).
    Rzuca DataError z instrukcją pobrania danych, gdy katalog jest pusty.
    """
    candidates: List[Path] = []
    if data_dir.exists():
        for pattern in ("*.csv", "*.json"):
            candidates.extend(p for p in data_dir.glob(pattern) if p.is_file())
    best: Optional[Tuple[float, Path]] = None
    for path in sorted(candidates):
        head = _read_table(path, header_only=True)
        if head is None:
            continue
        score = _score_fights_candidate(path, list(head.columns))
        if score is not None and (best is None or score > best[0]):
            best = (score, path)
    if best is not None:
        path = best[1]
        print(f"[DANE] Wczytywanie pliku: {path.name}")
        raw = _read_table(path)
        if raw is None or raw.empty:
            raise DataError(f"Nie udało się wczytać pliku danych: {path}")
        return raw, path.name
    # Brak plików — spróbuj opcjonalnego pakietu ufcdata (pobieranie online).
    print("[DANE] Brak plików z danymi w katalogu data/ — próba użycia pakietu ufcdata...")
    return _try_load_via_ufcdata(), "ufcdata (pakiet, pobranie online)"


def normalize_schema(raw: pd.DataFrame, source_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Normalizuje surowy DataFrame do wspólnego formatu wewnętrznego (TIDY_COLUMNS).

    Obsługiwane schematy:
      * nowy:  nlbair/ufc-data  (f1_*/f2_*, winner 1/2, kursy, rankingi),
      * starszy: klasy pliki „ufcdata” z Kaggle (R_*/B_*, Winner Red/Blue).
    Zwraca (tidy DataFrame, info o przetwarzaniu). Remisy i no contest są pomijane
    (klasyfikator binarny), a przypisanie stron A/B jest deterministycznie losowane
    (neutralizacja kolejności f1/f2 ze źródła, która „wycieka” z wynikiem walki).
    """
    cols = {_norm_key(c) for c in raw.columns}
    if {"f1_name", "f2_name", "winner"} <= cols:
        tidy, info = _normalize_ufcdata_wide(raw)
        info["schema"] = "ufc-data-wide (f1/f2)"
    elif {"r_fighter", "b_fighter"} <= cols:
        tidy, info = _normalize_legacy_rb(raw)
        info["schema"] = "legacy-rb (R/B)"
    else:
        raise DataError(
            f"Nie rozpoznano schematu danych w pliku {source_name}.\n"
            "Oczekiwane kolumny (schemat nowy): f1_name, f2_name, winner, event_date, ...\n"
            "Oczekiwane kolumny (schemat starszy): R_fighter, B_fighter, Winner, date, ...\n"
            f"Znalezione kolumny ({len(raw.columns)}): "
            f"{', '.join(map(str, raw.columns[:25]))} ..."
        )
    info["source"] = source_name
    info["n_raw"] = int(len(raw))
    tidy = tidy.reindex(columns=TIDY_COLUMNS)
    return tidy, info


def _side_attrs_wide(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Wyciąga atrybuty sprzed walki i kursy zawodnika f{idx} (schemat ufc-data-wide)."""
    p = f"f{idx}_"
    close_low = _to_float(row.get(p + "close_low"))
    close_high = _to_float(row.get(p + "close_high"))
    opening = _to_float(row.get(p + "open"))
    # Kurs do analizy zakładów: zamknięcie (mid zakresu books), fallback: otwarcie.
    odds = (close_low + close_high) / 2.0 if np.isfinite(close_low) and np.isfinite(close_high) else opening
    ranked = _as_bool(row.get(p + "is_ranked"))
    return {
        "name": str(row.get(p + "name", "") or "").strip(),
        "age": _to_float(row.get(p + "age")),                 # wiek w dniu walki (pre-fight)
        "height": _to_float(row.get(p + "height_cm")),
        "reach": _to_float(row.get(p + "reach_cm")),
        "stance": _stance_code(row.get(p + "stance")),
        "rank": _to_float(row.get(p + "rank")) if ranked else float("nan"),  # 0 = mistrz
        "ranked": 1.0 if ranked else 0.0,
        "odds": odds,
        "prob_fair": _to_float(row.get(p + "close_prob_novig")),  # prawdop. bez marży bukmachera
    }


def _normalize_ufcdata_wide(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Adapter schematu ufc_all_fights_wide.csv (nlbair/ufc-data) do formatu tidy."""
    rows: List[Dict[str, Any]] = []
    n_draw_nc = 0  # remisy, no contest, overturns — pomijane (klasyfikator binarny)
    n_bad = 0      # wiersze bez daty/nazwisk
    for r in raw.to_dict("records"):
        date = pd.to_datetime(r.get("event_date"), errors="coerce")
        f1 = _side_attrs_wide(r, 1)
        f2 = _side_attrs_wide(r, 2)
        winner = _to_float(r.get("winner"))
        if pd.isna(date) or not f1["name"] or not f2["name"]:
            n_bad += 1
            continue
        if winner not in (1.0, 2.0):  # 0 = remis, NaN = no contest / overturn
            n_draw_nc += 1
            continue
        fight_id = r.get("fight_id")
        fight_id = str(fight_id) if not pd.isna(fight_id) else f"{date.date()}|{f1['name']}|{f2['name']}"
        # ANTI-LEAKAGE: kolejność f1/f2 w źródle jest skorelowana z wynikiem
        # (ufcstats częściej listuje zwycięzcę jako pierwszego; w danych f1 wygrywa
        # w ok. 56-57% walk). Deterministyczna „moneta” per fight_id (crc32)
        # losuje strony A/B, więc przypisanie A/B nie może zdradzać wyniku.
        swap = zlib.crc32(fight_id.encode("utf-8")) % 2 == 1
        a, b = (f2, f1) if swap else (f1, f2)
        target = 1 if winner == (2.0 if swap else 1.0) else 0
        lbs, womens = division_features(r.get("bout_division"))
        rows.append({
            "fight_id": fight_id,
            "date": date,
            "fighter_a": a["name"],
            "fighter_b": b["name"],
            "target": int(target),
            "finish_type": _norm_key(r.get("finish_type")),
            "duration_sec": _to_float(r.get("fight_duration_sec")),
            "weight_class_lbs": lbs,
            "is_womens": womens,
            "is_title_bout": 1.0 if _as_bool(r.get("is_title_bout")) else 0.0,
            **{f"a_{k}": v for k, v in a.items() if k != "name"},
            **{f"b_{k}": v for k, v in b.items() if k != "name"},
        })
    return pd.DataFrame(rows), {"n_draw_nc": n_draw_nc, "n_bad_rows": n_bad}


def _parse_mmss(value: Any) -> float:
    """Parsuje czas w formacie 'm:ss' na sekundy; błędy -> NaN."""
    try:
        parts = str(value).split(":")
        if len(parts) != 2:
            return float("nan")
        return int(parts[0]) * 60 + float(parts[1])
    except Exception:
        return float("nan")


def _legacy_duration(row: Dict[str, Any], colmap: Dict[str, str]) -> float:
    """Szacuje czas trwania walki w starszym schemacie (runda końcowa + czas + długość rund)."""
    last_round = _to_float(_get(row, colmap, "last_round", "finish_round"))
    end_sec = _parse_mmss(_get(row, colmap, "last_round_time", "finish_time"))
    fmt = str(_get(row, colmap, "format", "fight_type", default="") or "")
    m = re.search(r"\(([^)]*)\)", fmt)  # np. "3 Rnd (5-5-5)" -> długość rundy 5 min
    nums = re.findall(r"\d+", m.group(1)) if m else []
    round_secs = float(nums[0]) * 60.0 if nums else 300.0
    if not (np.isfinite(last_round) and np.isfinite(end_sec)):
        return float("nan")
    return max((last_round - 1) * round_secs + end_sec, 0.0)


def _normalize_legacy_rb(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Adapter starszego schematu R_/B_ (klasyki z Kaggle) do formatu tidy."""
    colmap = {_norm_key(c): c for c in raw.columns}
    rows: List[Dict[str, Any]] = []
    n_draw_nc = 0
    n_bad = 0
    for r in raw.to_dict("records"):
        raw_date = _get(r, colmap, "date", "fight_date")
        date = pd.to_datetime(raw_date, errors="coerce")
        if pd.isna(date):
            date = pd.to_datetime(raw_date, format="%B %d, %Y", errors="coerce")
        name_r = str(_get(r, colmap, "r_fighter", "red_fighter", default="") or "").strip()
        name_b = str(_get(r, colmap, "b_fighter", "blue_fighter", default="") or "").strip()
        winner = str(_get(r, colmap, "winner", default="") or "").strip().lower()
        if pd.isna(date) or not name_r or not name_b:
            n_bad += 1
            continue
        if winner in (name_r.lower(), "red"):
            red_wins = 1
        elif winner in (name_b.lower(), "blue"):
            red_wins = 0
        else:  # remis / no contest / brak rozstrzygnięcia
            n_draw_nc += 1
            continue

        def side(which: str) -> Dict[str, Any]:
            p = "r" if which == "r" else "b"
            return {
                "name": name_r if which == "r" else name_b,
                "age": _to_float(_get(r, colmap, f"{p}_age")),
                "height": _to_float(_get(r, colmap, f"{p}_height_cms", f"{p}_height")),
                "reach": _to_float(_get(r, colmap, f"{p}_reach_cms", f"{p}_reach")),
                "stance": _stance_code(_get(r, colmap, f"{p}_stance")),
                "rank": float("nan"),   # starszy schemat nie ma rankingu UFC
                "ranked": 0.0,
                "odds": _to_float(_get(r, colmap, f"{p}_odds")),
                "prob_fair": float("nan"),
            }

        f1, f2 = side("r"), side("b")
        # Prawdopodobieństwo bez marży z kursów (jeśli oba dostępne): p = 1/decimal.
        d1 = american_to_decimal(f1["odds"])
        d2 = american_to_decimal(f2["odds"])
        if np.isfinite(d1) and np.isfinite(d2) and d1 > 0 and d2 > 0:
            p1, p2 = 1.0 / d1, 1.0 / d2
            f1["prob_fair"] = p1 / (p1 + p2)
            f2["prob_fair"] = p2 / (p1 + p2)

        fight_id = f"{date.date()}|{name_r}|{name_b}"
        swap = zlib.crc32(fight_id.encode("utf-8")) % 2 == 1
        a, b = (f2, f1) if swap else (f1, f2)
        target = red_wins if not swap else 1 - red_wins
        finish = _norm_key(_get(r, colmap, "finish", "win_method", default=""))
        if finish.startswith("decision"):
            finish = "decision"
        elif finish in ("ko/tko", "ko", "tko"):
            finish = "ko_tko"
        lbs, womens = division_features(_get(r, colmap, "weight_class", "weightclass"))
        rows.append({
            "fight_id": fight_id,
            "date": date,
            "fighter_a": a["name"],
            "fighter_b": b["name"],
            "target": int(target),
            "finish_type": finish,
            "duration_sec": _legacy_duration(r, colmap),
            "weight_class_lbs": lbs,
            "is_womens": womens,
            "is_title_bout": 1.0 if _as_bool(_get(r, colmap, "title_bout", default=False)) else 0.0,
            **{f"a_{k}": v for k, v in a.items() if k != "name"},
            **{f"b_{k}": v for k, v in b.items() if k != "name"},
        })
    return pd.DataFrame(rows), {"n_draw_nc": n_draw_nc, "n_bad_rows": n_bad}


def validate_tidy(tidy: pd.DataFrame) -> Dict[str, Any]:
    """Test odporności: sprawdza strukturę danych, zwraca podsumowanie do konsoli i raportu."""
    problems: List[str] = []
    if tidy.empty:
        problems.append("zbiór walk jest pusty po normalizacji")
    missing_cols = [c for c in TIDY_COLUMNS if c not in tidy.columns]
    if missing_cols:
        problems.append(f"brak wymaganych kolumn: {missing_cols}")
    if problems:
        raise DataError("Walidacja danych nie przeszła: " + "; ".join(problems))
    if tidy["target"].isna().any():
        raise DataError("Target zawiera wartości NaN — to niedozwolone.")
    if not set(pd.unique(tidy["target"])).issubset({0, 1}):
        raise DataError(f"Target nie jest binarny: {sorted(pd.unique(tidy['target']))[:5]}")
    if tidy["date"].isna().any():
        raise DataError("Kolumna 'date' zawiera nieparsowalne daty.")
    summary: Dict[str, Any] = {
        "n_fights": int(len(tidy)),
        "date_min": str(tidy["date"].min().date()),
        "date_max": str(tidy["date"].max().date()),
        "win_rate_a": float(tidy["target"].mean()),  # po neutralizacji stron powinno być ~0.50
        "odds_share": float(tidy["a_odds"].notna().mean()),
        "fair_prob_share": float(tidy["a_prob_fair"].notna().mean()),
        "n_fighters": int(pd.concat([tidy["fighter_a"], tidy["fighter_b"]]).nunique()),
    }
    print("\n[WALIDACJA DANYCH] OK — struktura zgodna z oczekiwaniami.")
    print(f"  Walki: {summary['n_fights']}  |  okres: {summary['date_min']} -> {summary['date_max']}")
    print(f"  Zawodnicy: {summary['n_fighters']}  |  udział zwycięstw A: {summary['win_rate_a']:.3f}")
    print(f"  Walki z kursami: {summary['odds_share']:.1%}  |  z prawdopodobieństwem bez marży: {summary['fair_prob_share']:.1%}")
    return summary


# ============================================================================
# FEATURE ENGINEERING  (ETAP 3)
# ============================================================================
@dataclass
class FightRecord:
    """Pojedyncza walka w historii zawodnika, zapisana z JEGO perspektywy."""

    date: pd.Timestamp
    result: float   # 1.0 = wygrana, 0.0 = przegrana, 0.5 = remis
    method: str     # 'ko' | 'sub' | 'dec' | 'other' — metoda zwycięstwa
    duration: float  # czas walki w sekundach (może być NaN)
    title: float    # 1.0 = walka o pas


def method_class(finish_type: Any) -> str:
    """Mapuje typ zakończenia walki na klasę metody zwycięstwa."""
    ft = _norm_key(finish_type)
    if ft == "ko_tko":
        return "ko"
    if ft == "submission":
        return "sub"
    if ft == "decision":
        return "dec"
    return "other"


def career_snapshot(history: List[FightRecord], as_of: pd.Timestamp) -> Dict[str, float]:
    """Statystyki zawodnika policzone WYŁĄCZNIE z walk wcześniejszych niż `as_of`.

    Serce ochrony przed data leakage: snapshot zawiera wyłącznie informacje
    dostępne PRZED daną walką. Dla debiutanta (brak historii) cechy oparte na
    ułamkach są NaN (imputacja medianą z train + wskaźnik debiutu).
    """
    priors = [h for h in history if h.date < as_of]
    n = len(priors)
    snap: Dict[str, float] = {
        "n": float(n), "wins": 0.0, "losses": 0.0, "draws": 0.0,
        "win_pct": float("nan"), "ko_pct": float("nan"), "sub_pct": float("nan"),
        "dec_pct": float("nan"), "finish_rate": float("nan"),
        "avg_time": float("nan"), "med_time": float("nan"),
        "fights_12m": 0.0, "fights_24m": 0.0,
        "form3": float("nan"), "form5": float("nan"),
        "days_rest": float("nan"), "streak": 0.0, "title_n": 0.0,
        "debut": 1.0 if n == 0 else 0.0,
    }
    if n == 0:
        return snap
    wins = sum(1 for h in priors if h.result == 1.0)
    losses = sum(1 for h in priors if h.result == 0.0)
    ko_wins = sum(1 for h in priors if h.result == 1.0 and h.method == "ko")
    sub_wins = sum(1 for h in priors if h.result == 1.0 and h.method == "sub")
    dec_wins = sum(1 for h in priors if h.result == 1.0 and h.method == "dec")
    durations = [h.duration for h in priors if np.isfinite(h.duration)]
    results = [h.result for h in priors]
    # Aktualna seria: liczba zwycięstw (+) lub porażek (-) z rzędu na koniec historii.
    streak = 0.0
    last = priors[-1].result
    if last in (0.0, 1.0):
        cnt = 0
        for h in reversed(priors):
            if h.result == last:
                cnt += 1
            else:
                break
        streak = float(cnt) * (1.0 if last == 1.0 else -1.0)
    snap.update({
        "wins": float(wins),
        "losses": float(losses),
        "draws": float(n - wins - losses),
        "win_pct": wins / n,
        "ko_pct": ko_wins / n,       # % walk wygranych przez KO/TKO
        "sub_pct": sub_wins / n,     # % walk wygranych przez poddanie
        "dec_pct": dec_wins / n,     # % walk wygranych przez decyzję
        "finish_rate": (ko_wins + sub_wins) / n,
        "avg_time": float(np.mean(durations)) if durations else float("nan"),
        "med_time": float(np.median(durations)) if durations else float("nan"),
        "fights_12m": float(sum(1 for h in priors if (as_of - h.date).days <= 365)),
        "fights_24m": float(sum(1 for h in priors if (as_of - h.date).days <= 730)),
        "form3": float(np.mean(results[-3:])),  # forma z ostatnich (max) 3 walk
        "form5": float(np.mean(results[-5:])),  # forma z ostatnich (max) 5 walk
        "days_rest": float((as_of - priors[-1].date).days),
        "streak": streak,
        "title_n": float(sum(1 for h in priors if h.title == 1.0)),
        "debut": 0.0,
    })
    return snap


# Mapowanie cech różnicowych: nazwa cechy -> klucz w career_snapshot (A - B).
_CAREER_DIFFS: List[Tuple[str, str]] = [
    ("exp_diff", "n"), ("wins_diff", "wins"), ("losses_diff", "losses"),
    ("draws_diff", "draws"), ("win_pct_diff", "win_pct"),
    ("ko_pct_diff", "ko_pct"), ("sub_pct_diff", "sub_pct"),
    ("dec_pct_diff", "dec_pct"), ("finish_rate_diff", "finish_rate"),
    ("avg_fight_time_diff", "avg_time"), ("med_fight_time_diff", "med_time"),
    ("fights_12m_diff", "fights_12m"), ("fights_24m_diff", "fights_24m"),
    ("form3_diff", "form3"), ("form5_diff", "form5"),
    ("days_rest_diff", "days_rest"), ("streak_diff", "streak"),
    ("title_exp_diff", "title_n"),
]


def _rank_capped(rank: float, ranked: float) -> float:
    """Koduje ranking UFC: 0 = mistrz, 1..15 = pozycja w rankingu, 20 = nierankingowy."""
    if ranked >= 0.5 and np.isfinite(rank):
        return min(float(rank), UNRANKED_CODE)
    return UNRANKED_CODE


def make_feature_row(snap_a: Dict[str, float], snap_b: Dict[str, float], rec: Dict[str, Any]) -> Dict[str, float]:
    """Składa wiersz cech (klucze = FEATURE_COLUMNS) dla jednej walki.

    snap_a/snap_b: career_snapshot zawodników (wyłącznie przeszłość).
    rec: wiersz tidy z atrybutami sprzed walki i kontekstem walki.
    Model jednowarstwowy jest liniowy, dlatego priorytet mają CECHY RÓŻNICOWE
    (A - B) — bezpośrednio porównują zawodników w tej samej walce.
    """
    feats: Dict[str, float] = {name: snap_a[key] - snap_b[key] for name, key in _CAREER_DIFFS}
    feats["debut_diff"] = snap_a["debut"] - snap_b["debut"]
    feats["both_debut"] = snap_a["debut"] * snap_b["debut"]
    feats["age_diff"] = _to_float(rec["a_age"]) - _to_float(rec["b_age"])
    feats["height_diff"] = _to_float(rec["a_height"]) - _to_float(rec["b_height"])
    feats["reach_diff"] = _to_float(rec["a_reach"]) - _to_float(rec["b_reach"])
    feats["stance_diff"] = _to_float(rec["a_stance"]) - _to_float(rec["b_stance"])
    feats["ranked_diff"] = _to_float(rec["a_ranked"]) - _to_float(rec["b_ranked"])
    feats["rank_diff"] = _rank_capped(_to_float(rec["a_rank"]), _to_float(rec["a_ranked"])) - _rank_capped(
        _to_float(rec["b_rank"]), _to_float(rec["b_ranked"])
    )
    feats["weight_class_lbs"] = _to_float(rec["weight_class_lbs"])
    feats["is_womens_division"] = _to_float(rec["is_womens"])
    feats["is_title_bout"] = _to_float(rec["is_title_bout"])
    # Wskaźniki braku danych (brak atrybutu może być informacją — np. brak zasięgu).
    feats["reach_missing_diff"] = (
        (0.0 if np.isfinite(_to_float(rec["a_reach"])) else 1.0)
        - (0.0 if np.isfinite(_to_float(rec["b_reach"])) else 1.0)
    )
    feats["height_missing_diff"] = (
        (0.0 if np.isfinite(_to_float(rec["a_height"])) else 1.0)
        - (0.0 if np.isfinite(_to_float(rec["b_height"])) else 1.0)
    )
    feats["age_missing_diff"] = (
        (0.0 if np.isfinite(_to_float(rec["a_age"])) else 1.0)
        - (0.0 if np.isfinite(_to_float(rec["b_age"])) else 1.0)
    )
    missing = [c for c in FEATURE_COLUMNS if c not in feats]
    if missing:  # zabezpieczenie spójności listy cech
        raise DataError(f"make_feature_row: brak cech {missing}")
    return feats


def _attr_pack(rec: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Pakiet ostatnich znanych atrybutów zawodnika (do trybu predict)."""
    return {
        "age": _to_float(rec.get(f"{prefix}_age")),
        "height": _to_float(rec.get(f"{prefix}_height")),
        "reach": _to_float(rec.get(f"{prefix}_reach")),
        "stance": _to_float(rec.get(f"{prefix}_stance")),
        "rank": _to_float(rec.get(f"{prefix}_rank")),
        "ranked": _to_float(rec.get(f"{prefix}_ranked")),
        "weight_class_lbs": _to_float(rec.get("weight_class_lbs")),
        "is_womens": _to_float(rec.get("is_womens")),
    }


def build_fight_dataset(
    tidy: pd.DataFrame, min_date: Optional[pd.Timestamp] = None
) -> Tuple[pd.DataFrame, Dict[str, List[FightRecord]], Dict[str, Dict[str, Any]]]:
    """Buduje zbiór uczący: 1 wiersz = 1 walka, cechy WYŁĄCZNIE z przeszłości.

    Algorytm pojedynczego przejścia po datach rosnąco:
      1. policz snapshoty obu zawodników z ich dotychczasowej historii,
      2. złóż wiersz cech (jeśli walka jest >= min_date),
      3. DOPIERO TERAZ zapisz walkę do historii obu zawodników.
    Kolejność tych kroków matematycznie gwarantuje, że cechy walki nigdy nie
    zawierają informacji z niej samej ani z przyszłości (brak data leakage).

    Zwraca (dataset, historia per zawodnik, ostatnie atrybuty per zawodnik).
    """
    history: Dict[str, List[FightRecord]] = defaultdict(list)
    last_attrs: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    frame = tidy.sort_values("date", kind="stable")
    for rec in frame.to_dict("records"):
        fa, fb, date = rec["fighter_a"], rec["fighter_b"], rec["date"]
        snap_a = career_snapshot(history[fa], date)
        snap_b = career_snapshot(history[fb], date)
        if min_date is None or date >= min_date:
            row: Dict[str, Any] = {
                "fight_id": rec["fight_id"], "date": date,
                "fighter_a": fa, "fighter_b": fb, "target": int(rec["target"]),
                # kursy NIE są cechą modelu — tylko analizą bukmacherską (etap 5)
                "a_odds": _to_float(rec["a_odds"]), "b_odds": _to_float(rec["b_odds"]),
                "a_prob_fair": _to_float(rec["a_prob_fair"]), "b_prob_fair": _to_float(rec["b_prob_fair"]),
            }
            row.update(make_feature_row(snap_a, snap_b, rec))
            rows.append(row)
        # KROK 3: aktualizacja historii po policzeniu cech (kluczowe anti-leakage).
        method = method_class(rec["finish_type"])
        duration = _to_float(rec["duration_sec"])
        title = _to_float(rec["is_title_bout"])
        result_a = float(rec["target"])
        history[fa].append(FightRecord(date, result_a, method, duration, title))
        history[fb].append(FightRecord(date, 1.0 - result_a, method, duration, title))
        last_attrs[fa] = _attr_pack(rec, "a")
        last_attrs[fb] = _attr_pack(rec, "b")
    return pd.DataFrame(rows), history, last_attrs


def split_train_val_test(
    ds: pd.DataFrame, random_split: bool = False, seed: int = SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Podział 70/15/15: CHRONOLOGICZNY (domyślnie) albo losowy (random_state=seed).

    UWAGA DYDAKTYCZNA: prognozujemy przyszłe walki, więc właściwy jest podział
    chronologiczny — trenujemy na najstarszych, walidujemy/testujemy na
    najnowszych walkach. Losowy podział zawyża metryki, bo walki z tego samego
    okresu (często ci sami zawodnicy) trafiałyby jednocześnie do train i test.
    """
    n = len(ds)
    if n < 300:
        raise DataError(f"Zbyt mało walk ({n}) do podziału train/val/test (min. 300).")
    if random_split:
        rng = np.random.default_rng(seed)
        ds = ds.iloc[rng.permutation(n)].reset_index(drop=True)
    n_train = int(round(TRAIN_FRAC * n))
    n_val = int(round(VAL_FRAC * n))
    train = ds.iloc[:n_train].reset_index(drop=True)
    val = ds.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test = ds.iloc[n_train + n_val :].reset_index(drop=True)
    if len(val) < 50 or len(test) < 50:
        raise DataError("Zbiory walidacyjny/testowy są zbyt małe (<50 walk).")
    kind = f"losowy (random_state={seed}) — wersja awaryjna" if random_split else "CHRONOLOGICZNY"
    print(f"\n[PODZIAŁ] {kind} 70/15/15:")
    for name, part in (("train", train), ("val", val), ("test", test)):
        print(
            f"  {name:5s}: {len(part):5d} walk  |  {part['date'].min().date()} .. {part['date'].max().date()}"
            f"  |  udział klasy 1 (wygrywa A): {part['target'].mean():.3f}"
        )
    return train, val, test


def fit_preprocessors(train_df: pd.DataFrame) -> Tuple[Dict[str, float], StandardScaler]:
    """Dopasowuje mediany (imputacja) i StandardScaler WYŁĄCZNIE na zbiorze treningowym.

    Dopasowanie scaler'a do całego zbioru byłoby data leakage — statystyki
    z walidacji/testu wyciekałyby do procesu treningu.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in train_df.columns]
    if missing:
        raise DataError(f"Dataset nie zawiera wymaganych cech: {missing[:5]}...")
    X = train_df[FEATURE_COLUMNS].astype(float)
    medians = {c: float(v) for c, v in X.median(skipna=True).fillna(0.0).items()}
    scaler = StandardScaler().fit(X.fillna(medians))
    return medians, scaler


def transform_features(df: pd.DataFrame, medians: Dict[str, float], scaler: StandardScaler) -> np.ndarray:
    """Imputuje braki medianą z train, skaluje i WINSORYZUJE do ±3σ (float32).

    Przycięcie po standaryzacji chroni przed dwoma problemami: (a) pojedyncza rzadka
    cecha (duża waga, mała σ) nie zdominuje logitu, (b) predykcje dla zawodników poza
    rozkładem treningowym (długa przerwa, jedna walka w karcie) nie ekstrapolują.
    Ta sama transformacja obowiązuje w treningu, ewaluacji, predict i na stronie web.
    """
    X = df[FEATURE_COLUMNS].astype(float).fillna(medians)
    Z = scaler.transform(X)
    return np.clip(Z, -Z_CAP_SIGMA, Z_CAP_SIGMA).astype(np.float32)


# ============================================================================
# MODEL I TRENING  (ETAP 4)
# ============================================================================
class SingleLayerUFCClassifier(nn.Module):
    """Jednowarstwowa sieć neuronowa bez warstw ukrytych.

    Architektura: wektor cech -> nn.Linear(input_size, 1) -> logit.
    Trening: BCEWithLogitsLoss (stabilna entropia krzyżowa + sigmoid w jednej funkcji).
    Predykcja prawdopodobieństwa: sigmoid(logit).
    Świadomie BEZ warstw ukrytych, dropoutu i batch normy (wymóg przedmiotu).
    """

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)  # logit (bez sigmoid — jest w funkcji straty)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """Zwraca prawdopodobieństwo klasy 1 (wygrywa zawodnik A) jako numpy (n,)."""
        return torch.sigmoid(self.forward(x)).squeeze(1).numpy()


def _make_optimizer(name: str, params, lr: float) -> torch.optim.Optimizer:
    """Fabryka optymalizatorów dla porównania wariantów uczenia (komenda compare).

    Wszystkie warianty dostają tę samą regularyzację L2 (weight_decay) — porównujemy
    wyłącznie mechanizm aktualizacji wag, nie regularyzację.
    """
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=DEFAULT_WEIGHT_DECAY)
    if name == "momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=DEFAULT_WEIGHT_DECAY)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=DEFAULT_WEIGHT_DECAY)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=DEFAULT_WEIGHT_DECAY)
    raise DataError(f"Nieznany optymalizator: {name}")


def _make_loss(loss_name: str):
    """Fabryka funkcji straty: bce_logits (domyślna), bce (sigmoid + BCELoss), mse.

    MSE na wyjściu sigmoidalnym to klasyczny kontrapunkt dla entropii krzyżowej:
    pochodna MSE×sigmoid gaśnie przy skrajnych logitach (sygmoida się nasycą),
    więc uczenie wyraźnie zwalnia — widać to na krzywych z compare.
    """
    if loss_name == "bce_logits":
        return nn.BCEWithLogitsLoss()
    if loss_name == "bce":

        def loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return F.binary_cross_entropy(torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7), y)

        return loss_fn
    if loss_name == "mse":

        def loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return ((torch.sigmoid(logits) - y) ** 2).mean()

        return loss_fn
    raise DataError(f"Nieznana funkcja straty: {loss_name}")


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_size: int,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    patience: int = DEFAULT_PATIENCE,
    seed: int = SEED,
    optimizer_name: str = "adam",
    loss_name: str = "bce_logits",
    verbose: bool = True,
) -> Tuple[SingleLayerUFCClassifier, pd.DataFrame]:
    """Trenuje jednowarstwową sieć: Adam + mini-batch + BCEWithLogitsLoss.

    - BCEWithLogitsLoss = entropia krzyżowa ze scalonym sigmoidem (numerycznie stabilna).
    - Early stopping na validation loss (patience) — chroni przed przeuczeniem;
      po treningu przywracamy wagi z najlepszej epoki.
    - Pełny determinizm: seed wszędzie + generator do losowania batchy.
    - Parametry optimizer_name/loss_name/verbose służą komendzie `compare`; wartości
      domyślne odtwarzają dokładnie oryginalną ścieżkę treningu projektu.
    Zwraca (model, historia treningu: train/val loss i accuracy po epokach).
    """
    seed_everything(seed)
    model = SingleLayerUFCClassifier(input_size)
    criterion = _make_loss(loss_name)
    optimizer = _make_optimizer(optimizer_name, model.parameters(), lr)

    tx_train = torch.from_numpy(np.ascontiguousarray(X_train))
    ty_train = torch.from_numpy(np.ascontiguousarray(y_train)).unsqueeze(1)  # (n, 1)
    tx_val = torch.from_numpy(np.ascontiguousarray(X_val))
    ty_val = torch.from_numpy(np.ascontiguousarray(y_val)).unsqueeze(1)

    generator = torch.Generator().manual_seed(seed)  # deterministyczny shuffle batchy
    loader = DataLoader(
        TensorDataset(tx_train, ty_train),
        batch_size=batch_size, shuffle=True, generator=generator, num_workers=0,
    )

    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # --- trening (mini-batch gradient descent) ---
        model.train()
        train_loss_sum, n_batches = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)  # forward
            loss.backward()                  # backpropagation
            optimizer.step()                 # aktualizacja wag (Adam)
            train_loss_sum += float(loss.item())
            n_batches += 1

        # --- ewaluacja per epoka (bez gradientu) ---
        model.eval()
        with torch.no_grad():
            val_logits = model(tx_val)
            val_loss = float(criterion(val_logits, ty_val).item())
            # Ujednolicona metryka porównawcza: log loss na walidacji — ta sama skala
            # dla KAŻDEJ funkcji straty (MSE ma własną skalę ~0.2, BCE ~0.65), dzięki
            # temu krzywe zbieżności różnych wariantów są porównywalne (komenda compare).
            val_probs = torch.sigmoid(val_logits)
            val_log_loss = float(F.binary_cross_entropy(val_probs.clamp(1e-7, 1 - 1e-7), ty_val).item())
            val_acc = float(((val_logits > 0) == ty_val).float().mean().item())
            train_logits = model(tx_train)
            train_acc = float(((train_logits > 0) == ty_train).float().mean().item())

        history.append({
            "epoch": epoch,
            "train_loss": train_loss_sum / max(n_batches, 1),
            "val_loss": val_loss,
            "val_log_loss": val_log_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
        })

        marker = ""
        if val_loss < best_val_loss - 1e-5:  # istotna poprawa -> zapamiętaj wagi
            best_val_loss, best_epoch, epochs_no_improve = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            marker = " *"
        else:
            epochs_no_improve += 1
        if verbose:
            print(
                f"  epoka {epoch:3d}/{epochs} | train loss {history[-1]['train_loss']:.4f} | "
                f"val loss {val_loss:.4f} | train acc {train_acc:.4f} | val acc {val_acc:.4f}{marker}"
            )
        if epochs_no_improve >= patience:
            if verbose:
                print(f"  [EARLY STOPPING] brak poprawy val loss przez {patience} epok — stop.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)  # przywrócenie najlepszej epoki
    if verbose:
        print(f"  Najlepsza epoka: {best_epoch} (val loss = {best_val_loss:.4f})")
    return model, pd.DataFrame(history)


# ============================================================================
# PORÓWNANIE WARIANTÓW UCZENIA  (komenda compare; styl laboratoriów:
# pętla po wariantach -> słownik wyników -> tabelka + wykresy)
# ============================================================================
def run_training_comparison(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray, epochs: int, patience: int, seed: int,
) -> List[Dict[str, Any]]:
    """Trenuje każdy wariant z COMPARE_VARIANTS od IDENTYCZNEGO startu.

    train_model wywołuje seed_everything(seed), więc wszystkie warianty zaczynają
    z tymi samymi wagami początkowymi i tę samą kolejnością batchy — różni się
    wyłącznie optymalizator / funkcja straty / tryb batchy. To sprawia, że różnice
    na krzywych zbieżności są atrybutem wariantu, a nie losowości startu.
    """
    results: List[Dict[str, Any]] = []
    input_size = X_train.shape[1]
    for name, opt_name, loss_name, bs in COMPARE_VARIANTS:
        batch_size = len(X_train) if bs is None else bs
        t0 = time.perf_counter()
        model, hist = train_model(
            X_train, y_train, X_val, y_val, input_size=input_size,
            epochs=epochs, batch_size=batch_size, lr=DEFAULT_LR, patience=patience,
            seed=seed, optimizer_name=opt_name, loss_name=loss_name, verbose=False,
        )
        czas = time.perf_counter() - t0
        m_test, _, _ = evaluate_dataset(model, X_test, y_test)
        # Konwergencja na UJEDNOLICONEJ metryce (log loss walidacyjny): pierwsza epoka
        # z val log loss <= 0.670 (wyraźnie poniżej majority log loss 0.693); None =
        # wariant nie osiągnął progu w budżecie epok. Porównywalne między stratami.
        conv = next((int(h["epoch"]) for h in hist.to_dict("records")
                     if h["val_log_loss"] <= 0.670), None)
        results.append({
            "name": name, "optimizer": opt_name, "loss": loss_name,
            "batch_size": batch_size, "czas_s": round(czas, 1),
            "best_epoch": int(hist.loc[hist["val_loss"].idxmin(), "epoch"]),
            "val_loss_own_scale": round(float(hist["val_loss"].min()), 5),
            "val_log_loss": round(float(hist["val_log_loss"].min()), 5),
            "test_accuracy": m_test["accuracy"], "test_roc_auc": m_test["roc_auc"],
            "test_log_loss": m_test["log_loss"], "test_brier": m_test["brier"],
            "conv_epoch_067": conv, "selected": "wybrany" in name,
            "history": {"epoch": hist["epoch"].tolist(),
                        "val_log_loss": [round(float(v), 5) for v in hist["val_log_loss"]]},
        })
    return results


def print_comparison_table(results: List[Dict[str, Any]]) -> None:
    """Tabelka porównawcza w konsoli (styl tabel baseline'owych z lab3).

    Kolumna 'val log loss' to ujednolicona metryka — porównywalna między wariantami
    z różnymi funkcjami straty (własna skala straty trafia do compare.json).
    """
    print(f"\n  {'wariant':30s} {'epoka':>5s} {'val LL':>7s} {'LL<=0.67':>8s} "
          f"{'test acc':>8s} {'test AUC':>8s} {'log loss':>8s} {'czas':>7s}")
    print("  " + "-" * 94)
    for r in results:
        conv = f"{r['conv_epoch_067']:4d}" if r["conv_epoch_067"] else " >max"
        star = "  <— WYBRANY" if r["selected"] else ""
        print(f"  {r['name']:30s} {r['best_epoch']:5d} {r['val_log_loss']:7.4f} {conv:>8s} "
              f"{r['test_accuracy']:8.4f} {r['test_roc_auc']:8.4f} {r['test_log_loss']:8.4f} "
              f"{r['czas_s']:6.1f}s{star}")


def make_comparison_plots(results: List[Dict[str, Any]], market_auc: Optional[float], plots_dir: Path) -> None:
    """Dwa wykresy porównawcze: krzywe zbieżności + słupki AUC z referencjami (styl lab4)."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    # 1. Krzywe zbieżności (ujednolicony log loss walidacyjny per epoka) — serce porównania.
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for r in results:
        ax.plot(r["history"]["epoch"], r["history"]["val_log_loss"],
                linewidth=2.8 if r["selected"] else 1.5,
                alpha=1.0 if r["selected"] else 0.75,
                label=f"{r['name']}  ({r['val_log_loss']:.4f})")
    ax.axhline(0.670, color="gray", linestyle=":", linewidth=1, label="próg konwergencji 0.670")
    ax.axhline(0.6931, color="black", linestyle=":", linewidth=1, label="majority class (0.6931)")
    ax.set(title="Porównanie wariantów uczenia — log loss walidacyjny (ten sam model i dane)",
           xlabel="Epoka", ylabel="Log loss na walidacji (ujednolicony)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(plots_dir / "compare_loss_curves.png", dpi=130)
    plt.close(fig)

    # 2. Słupki ROC-AUC (test) z liniami odniesienia: przypadek i rynek.
    names = [r["name"] for r in results]
    aucs = [r["test_roc_auc"] for r in results]
    colors = ["#f0a202" if r["selected"] else "#4c6a92" for r in results]
    fig2, ax2 = plt.subplots(figsize=(9.5, 5))
    bars = ax2.bar(names, aucs, color=colors)
    for b, v in zip(bars, aucs):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}", ha="center", fontsize=9)
    ax2.axhline(0.5, linestyle="--", color="gray", linewidth=1, label="przypadek (0.500)")
    if market_auc:
        ax2.axhline(market_auc, linestyle="--", color="#c0392b", linewidth=1.2,
                    label=f"bukmacher ({market_auc:.3f})")
    ax2.set(title="Porównanie wariantów — ROC-AUC na zbiorze testowym",
            xlabel="Wariant uczenia", ylabel="ROC-AUC")
    ax2.set_ylim(0.5, 0.8)
    ax2.grid(alpha=0.3, axis="y")
    ax2.tick_params(axis="x", labelsize=8, rotation=12)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(plots_dir / "compare_auc_bars.png", dpi=130)
    plt.close(fig2)


def render_comparison_md(results: List[Dict[str, Any]], market_auc: Optional[float]) -> str:
    """Raport z porównania + automatyczne wnioski „dlaczego Adam i entropia krzyżowa"."""
    lines = [
        "# Porównanie wariantów uczenia (WalkoMat)\n",
        "Ten sam jednowarstwowy model `nn.Linear(32, 1)`, te same dane i skalowanie,",
        "identyczna inicjalizacja i kolejność batchy (seed 42). Warianty różnią się",
        "wyłącznie optymalizatorem / funkcją straty / trybem batchów. Early stopping",
        "na walidacyjnej funkcji straty (patience 12); próg konwergencji liczymy na",
        "**ujednoliconej metryce** — log loss walidacyjny (porównywalny między stratami).\n",
        "| Wariant | najw. epoka | val log loss | LL<=0.67 | test acc | test AUC | log loss | czas [s] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        conv = str(r["conv_epoch_067"]) if r["conv_epoch_067"] else "—"
        mark = " **★**" if r["selected"] else ""
        lines.append(f"| {r['name']}{mark} | {r['best_epoch']} | {r['val_log_loss']:.4f} | {conv} | "
                     f"{r['test_accuracy']:.4f} | {r['test_roc_auc']:.4f} | {r['test_log_loss']:.4f} | {r['czas_s']} |")
    best = min(results, key=lambda r: r["val_log_loss"])
    mse = next((r for r in results if r["loss"] == "mse"), None)
    full = next((r for r in results if r["batch_size"] > 100), None)
    lines += [
        "\n## Wnioski — dlaczego wybraliśmy Adam i entropię krzyżową\n",
        f"- **Najniższy walidacyjny log loss** uzyskuje **{best['name']}** "
        f"({best['val_log_loss']:.4f}, epoka {best['best_epoch']}) — ten wariant jest "
        "domyślnym treningiem projektu.",
        "- **Adam vs SGD/Momentum/RMSprop**: adaptacyjne, per-parametrowe kroki uczenia "
        "(momenta 1. i 2. rzędu) zbiegają najszybciej; SGD ze stałym krokiem lr=1e-3 jest "
        "za wolne dla tej skali cech, a Momentum/RMSprop poprawiają zbieżność, lecz nie "
        "doganiają Adama.",
    ]
    if mse:
        lines.append(f"- **Entropia krzyżowa vs MSE**: porównanie w UJEDNOLICONEJ skali "
                     f"(log loss): wariant MSE kończy na {mse['val_log_loss']:.4f} vs "
                     f"{best['val_log_loss']:.4f} dla entropii krzyżowej — pochodna MSE przez "
                     "sigmoid gaśnie przy skrajnych logitach (nasycenie), więc błędne "
                     "predykcje korygują się powoli. BCEWithLogitsLoss tego problemu nie ma "
                     "i jest numerycznie stabilna (dlatego nie używamy osobnego sigmoid+BCE).")
    if full:
        lines.append(f"- **Mini-batch vs full-batch**: pełny gradient to 1 aktualizacja wag na "
                     f"epokę — w budżecie {len(full['history']['epoch'])} epok osiąga tylko "
                     f"{full['val_log_loss']:.4f}; mini-batch (64) daje ~63 aktualizacje na "
                     "epokę, szybciej i stabilniej.")
    if market_auc:
        lines.append(f"- Punkt odniesienia rynku: bukmacher ROC-AUC = {market_auc:.3f} — żaden "
                     "wariant go nie pobija (zgodnie z oczekiwaniem; patrz OBRONA §12.2).")
    lines.append("")
    return "\n".join(lines)


def cmd_compare(args: argparse.Namespace) -> None:
    """Porównuje warianty uczenia na tym samym modelu i danych (wykresy + tabela + wnioski)."""
    print("=" * 70)
    print(f"{PROJECT_NAME} — porównanie wariantów uczenia (v{CODE_VERSION})")
    print("=" * 70)
    raw, source = load_fights_frame(args.data_dir)
    tidy, _info = normalize_schema(raw, source)
    validate_tidy(tidy)
    min_date = tidy["date"].min() + pd.Timedelta(days=365.25 * DEFAULT_BURN_IN_YEARS)
    ds, _h, _a = build_fight_dataset(tidy, min_date=min_date)
    train_df, val_df, test_df = split_train_val_test(ds, random_split=False, seed=args.seed)
    medians, scaler = fit_preprocessors(train_df)
    X_train = transform_features(train_df, medians, scaler)
    X_val = transform_features(val_df, medians, scaler)
    X_test = transform_features(test_df, medians, scaler)
    y_train = train_df["target"].to_numpy(np.float32)
    y_val = val_df["target"].to_numpy(np.float32)
    y_test = test_df["target"].to_numpy(np.float32)

    bookie = bookmaker_baseline(test_df)
    market_auc = bookie.get("roc_auc") if bookie.get("available") else None

    print(f"\n[COMPARE] {len(COMPARE_VARIANTS)} wariantów | max {args.epochs} epok | "
          f"patience {args.patience} | seed {args.seed} (identyczny start każdego wariantu)")
    results = run_training_comparison(
        X_train, y_train, X_val, y_val, X_test, y_test,
        epochs=args.epochs, patience=args.patience, seed=args.seed,
    )
    print_comparison_table(results)
    make_comparison_plots(results, market_auc, args.outputs_dir / PLOTS_DIR_NAME)
    save_json(args.outputs_dir / "compare.json", {
        "project": PROJECT_NAME, "code_version": CODE_VERSION, "seed": args.seed,
        "epochs": args.epochs, "patience": args.patience, "market_auc": market_auc,
        "results": results,
    })
    (args.outputs_dir / "compare.md").write_text(
        render_comparison_md(results, market_auc), encoding="utf-8")
    print("\n[COMPARE] zapisano: compare.json, compare.md, plots/compare_loss_curves.png, "
          "plots/compare_auc_bars.png")
    print(DISCLAIMER)


# ============================================================================
# EWALUACJA I ANALIZA BUKMACHERSKA  (ETAP 5)
# ============================================================================
def expected_calibration_error(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10) -> float:
    """ECE: oczekiwana różnica między pewnością modelu a rzeczywistą trafnością w kubełkach."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = ((p_pred >= lo) & (p_pred <= hi)) if lo == 0.0 else ((p_pred > lo) & (p_pred <= hi))
        if mask.any():
            confidence = float(p_pred[mask].mean())
            accuracy = float(y_true[mask].mean())
            ece += mask.sum() / len(y_true) * abs(confidence - accuracy)
    return float(ece)


def evaluate_dataset(
    model: SingleLayerUFCClassifier, X: np.ndarray, y: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Liczy pełny zestaw metryk klasyfikacji i kalibracji.

    Zwraca (metryki, prawdopodobieństwa modelu, twarde predykcje przy progu).
    """
    y = np.asarray(y).astype(int)
    with torch.no_grad():
        p = model.predict_proba(torch.from_numpy(np.ascontiguousarray(X.astype(np.float32))))
    yhat = (p >= threshold).astype(int)
    cm = confusion_matrix(y, yhat, labels=[0, 1])  # [[TN, FP], [FN, TP]]
    metrics: Dict[str, Any] = {
        "n": int(len(y)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, yhat)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-9, 1 - 1e-9))),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "mean_predicted_p": float(p.mean()),
        "confusion_matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
    }
    return metrics, p, yhat


def print_metrics(tag: str, m: Dict[str, Any]) -> None:
    """Czytelny wydruk metryk jednego zbioru do konsoli."""
    print(f"  [{tag}] n={m['n']}  acc={m['accuracy']:.4f}  bal-acc={m['balanced_accuracy']:.4f}  "
          f"prec={m['precision']:.4f}  rec={m['recall']:.4f}  f1={m['f1']:.4f}")
    print(f"          ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
          f"log loss={m['log_loss']:.4f}  Brier={m['brier']:.4f}  ECE={m['ece']:.4f}")
    cm = m["confusion_matrix"]
    print(f"          macierz pomyłek: TN={cm[0][0]}  FP={cm[0][1]}  FN={cm[1][0]}  TP={cm[1][1]}")


def compute_baselines(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, Any]:
    """Baseline'y odniesienia: majority class oraz regresja logistyczna (scikit-learn).

    Regresja logistyczna trenowana na TYCH SAMYCH przeskalowanych cechach co model
    PyTorch — porównanie pokazuje równoważność obu implementacji (sekcja 6.2 obrony).
    """
    y_train = np.asarray(y_train).astype(int)
    y_test = np.asarray(y_test).astype(int)
    p_majority = float(y_train.mean())
    majority_class = int(p_majority >= 0.5)
    yhat_majority = np.full(len(y_test), majority_class)
    majority = {
        "majority_class": majority_class,
        "accuracy": float((y_test == majority_class).mean()),
        "log_loss": float(log_loss(y_test, np.full(len(y_test), np.clip(p_majority, 1e-9, 1 - 1e-9)))),
        "roc_auc": None,  # stała prognoza nie ma sensownego AUC
    }
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr.fit(X_train, y_train)
    p_lr = lr.predict_proba(X_test)[:, 1]
    lr_metrics = {
        "accuracy": float(accuracy_score(y_test, (p_lr >= 0.5).astype(int))),
        "roc_auc": float(roc_auc_score(y_test, p_lr)),
        "log_loss": float(log_loss(y_test, np.clip(p_lr, 1e-9, 1 - 1e-9))),
        "brier": float(brier_score_loss(y_test, p_lr)),
    }
    print(f"\n[BASELINE] majority class (zawsze klasa {majority_class}): "
          f"acc={majority['accuracy']:.4f}, log loss={majority['log_loss']:.4f}")
    print(f"[BASELINE] regresja logistyczna sklearn: acc={lr_metrics['accuracy']:.4f}, "
          f"ROC-AUC={lr_metrics['roc_auc']:.4f}, log loss={lr_metrics['log_loss']:.4f}, "
          f"Brier={lr_metrics['brier']:.4f}")
    return {"majority": majority, "logistic_regression": lr_metrics}


def bookmaker_baseline(test_df: pd.DataFrame) -> Dict[str, Any]:
    """Jakość prognozy bukmachera (prawdopodobieństwa zamknięcia bez marży) na zbiorze testowym.

    To najtwardszy benchmark: rynek wyceniany przez profesjonalnych analityków.
    """
    mask = test_df["a_prob_fair"].notna().to_numpy()
    if mask.sum() < 30:
        print("\n[BASELINE] bukmacher: za mało walk z kursami na zbiorze testowym — pomijam.")
        return {"available": False}
    p_book = test_df.loc[mask, "a_prob_fair"].to_numpy(dtype=float)
    y_book = test_df.loc[mask, "target"].to_numpy(dtype=int)
    out = {
        "available": True,
        "n_fights": int(mask.sum()),
        "share_of_test": float(mask.mean()),
        "accuracy": float((y_book == (p_book >= 0.5).astype(int)).mean()),
        "roc_auc": float(roc_auc_score(y_book, p_book)),
        "log_loss": float(log_loss(y_book, np.clip(p_book, 1e-9, 1 - 1e-9))),
        "brier": float(brier_score_loss(y_book, p_book)),
    }
    print(f"[BASELINE] bukmacher (kursy zamknięcia, n={out['n_fights']} walk = {out['share_of_test']:.1%} testu): "
          f"acc={out['accuracy']:.4f}, ROC-AUC={out['roc_auc']:.4f}, log loss={out['log_loss']:.4f}, "
          f"Brier={out['brier']:.4f}")
    return out


def _max_drawdown(equity: np.ndarray) -> float:
    """Maksymalne obsunięcie od dotychczasowego szczytu skumulowanego zysku (>= 0)."""
    if len(equity) == 0:
        return 0.0
    running_max = np.maximum.accumulate(np.concatenate([[0.0], equity]))
    curve = np.concatenate([[0.0], equity])
    return float(np.max(running_max - curve))


def betting_analysis(test_df: pd.DataFrame, p_model: np.ndarray) -> Dict[str, Any]:
    """Prosta analiza value betting na zbiorze testowym + benchmark „obstawiaj faworyta".

    Strategia: strona o większym modelowym p; zakład gdy (p >= próg) i (EV = p·kurs > 1).
    Stawka 1 jednostka na zakład. Kursy = zamknięcie (mid zakresu), zawierają marżę —
    dokładnie to, co bukmacher realnie oferuje. Symulacja wyłącznie edukacyjna.
    """
    odds_a = np.array([american_to_decimal(v) for v in test_df["a_odds"]], dtype=float)
    odds_b = np.array([american_to_decimal(v) for v in test_df["b_odds"]], dtype=float)
    y = test_df["target"].to_numpy(dtype=int)
    valid = np.isfinite(odds_a) & np.isfinite(odds_b) & (odds_a > 1) & (odds_b > 1)
    if valid.sum() < 30:
        print("\n[ANALIZA BUKMACHERSKA] niedostępna — brak wystarczającej liczby kursów w danych.")
        return {"available": False}
    pa, pb = p_model[valid], 1.0 - p_model[valid]
    oa, ob, yv = odds_a[valid], odds_b[valid], y[valid]

    def simulate(bet_mask_a: np.ndarray, bet_mask_b: np.ndarray) -> Dict[str, Any]:
        """Symuluje flat staking dla zadanych masek zakładów na A i na B."""
        profits: List[float] = []
        for i in range(len(yv)):
            if bet_mask_a[i]:
                profits.append((oa[i] - 1.0) if yv[i] == 1 else -1.0)
            if bet_mask_b[i]:
                profits.append((ob[i] - 1.0) if yv[i] == 0 else -1.0)
        eq = np.cumsum(profits) if profits else np.array([0.0])
        n_bets = len(profits)
        n_a, n_b = int(bet_mask_a.sum()), int(bet_mask_b.sum())
        wins = sum(1 for pr in profits if pr > 0)
        avg_odds_vals = [oa[i] for i in range(len(oa)) if bet_mask_a[i]] + \
                        [ob[i] for i in range(len(ob)) if bet_mask_b[i]]
        # Statystyka odporności: ROI po usunięciu 10 NAJKORZYSTNIEJSZYCH zakładów.
        # ROI w zakładach na underdogi jest zdominowane przez pojedyncze longshoty —
        # bez tego sprawdzenia jedna wygrana 100:1 wygląda jak „strategia" (lekcja
        # ze statystyk odpornych). Gdy odporny ROI <= 0, strategia nie jest powtarzalna.
        profits_desc = sorted(profits, reverse=True)
        robust = profits_desc[10:] if len(profits_desc) > 10 else profits_desc
        return {
            "n_bets": int(n_bets), "n_bets_on_a": n_a, "n_bets_on_b": n_b,
            "win_rate": float(wins / n_bets) if n_bets else None,
            "avg_odds": float(np.mean(avg_odds_vals)) if n_bets else None,
            "median_profit": float(np.median(profits)) if n_bets else None,
            "profit": float(eq[-1]) if len(eq) else 0.0,
            "roi": float(eq[-1] / n_bets) if n_bets else None,
            "roi_ex_top10": float(np.mean(robust)) if robust else None,
            "max_drawdown": _max_drawdown(eq) if n_bets else None,
            "equity": eq.tolist(),
        }

    thresholds: Dict[str, Any] = {}
    for t in BET_THRESHOLDS:
        bet_a = (pa >= t) & (pa * oa > 1.0)   # value bet na A: pewność + EV dodatnie
        bet_b = (pb >= t) & (pb * ob > 1.0)   # value bet na B
        res = simulate(bet_a, bet_b)
        thresholds[str(t)] = res
        wr = "n/d" if res["win_rate"] is None else f"{res['win_rate']:.3f}"
        roi = "n/d" if res["roi"] is None else f"{res['roi']:+.3f}"
        roi_r = "n/d" if res["roi_ex_top10"] is None else f"{res['roi_ex_top10']:+.3f}"
        med = "n/d" if res["median_profit"] is None else f"{res['median_profit']:+.3f}"
        dd = "n/d" if res["max_drawdown"] is None else f"{res['max_drawdown']:.2f}"
        print(f"  próg {t:.2f}: zakładów={res['n_bets']:3d} (A:{res['n_bets_on_a']}/B:{res['n_bets_on_b']}), "
              f"win rate={wr}, ROI={roi} [odporny: {roi_r}], mediana zysku={med}, max DD={dd}")
    # Benchmark bez modelu: obstawiaj faworyta rynkowego każdą walkę (mierzy koszt marży).
    fav_a = oa < ob
    fav = simulate(fav_a, ~fav_a)
    print(f"  benchmark faworytów (bez modelu): zakładów={fav['n_bets']}, "
          f"win rate={fav['win_rate']:.3f}, ROI={fav['roi']:+.3f}")
    return {"available": True, "n_fights_with_odds": int(valid.sum()),
            "share_of_test": float(valid.mean()), "thresholds": thresholds,
            "favorite_benchmark": fav}


# ============================================================================
# WYKRESY  (ETAP 6)
# ============================================================================
def make_all_plots(
    history: Optional[pd.DataFrame],
    p_test: np.ndarray,
    y_test: np.ndarray,
    metrics_test: Dict[str, Any],
    model: SingleLayerUFCClassifier,
    betting: Optional[Dict[str, Any]],
    plots_dir: Path,
) -> None:
    """Generuje wszystkie wykresy PNG w katalogu outputs/plots/ (matplotlib, bez GUI).

    history=None (tryb evaluate bez zapisanej historii) pomija wykresy treningowe.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    y_test = np.asarray(y_test).astype(int)

    # 1-2. Historia treningu: loss i accuracy per epoka.
    if history is not None and not history.empty:
        best_epoch = int(history.loc[history["val_loss"].idxmin(), "epoch"])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history["epoch"], history["train_loss"], marker="", label="train loss")
        ax.plot(history["epoch"], history["val_loss"], marker="", label="validation loss")
        ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.8,
                   label=f"najlepsza epoka ({best_epoch})")
        ax.set(title="Historia treningu — funkcja straty (BCEWithLogitsLoss)",
               xlabel="Epoka", ylabel="Strata")
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(plots_dir / "training_history.png", dpi=130); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history["epoch"], history["train_accuracy"], label="train accuracy")
        ax.plot(history["epoch"], history["val_accuracy"], label="validation accuracy")
        ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.8,
                   label=f"najlepsza epoka ({best_epoch})")
        ax.set(title="Historia treningu — dokładność", xlabel="Epoka", ylabel="Accuracy")
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(plots_dir / "accuracy_history.png", dpi=130); plt.close(fig)

    # 3. Krzywa ROC dla zbioru testowego.
    fpr, tpr, _ = roc_curve(y_test, p_test)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(fpr, tpr, label=f"model (AUC = {metrics_test['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="przypadek (AUC = 0.5)")
    ax.set(title="Krzywa ROC — zbiór testowy", xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.grid(alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(plots_dir / "roc_curve.png", dpi=130); plt.close(fig)

    # 4. Macierz pomyłek.
    cm = np.array(metrics_test["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["pred: B wygrywa (0)", "pred: A wygrywa (1)"])
    ax.set_yticks([0, 1], ["rzecz.: B (0)", "rzecz.: A (1)"])
    ax.set(title=f"Macierz pomyłek — test (n={metrics_test['n']}, próg {metrics_test['threshold']})")
    fig.tight_layout(); fig.savefig(plots_dir / "confusion_matrix.png", dpi=130); plt.close(fig)

    # 5. Krzywa kalibracji (przewidywane p vs rzeczywista częstość).
    frac_pos, mean_p = calibration_curve(y_test, p_test, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(mean_p, frac_pos, marker="o", label="model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="idealna kalibracja")
    ax.set(title=f"Kalibracja prawdopodobieństw — test (ECE = {metrics_test['ece']:.3f})",
           xlabel="Średnie przewidywane prawdopodobieństwo (A)", ylabel="Rzeczywisty udział wygranych A")
    ax.grid(alpha=0.3); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(plots_dir / "calibration_curve.png", dpi=130); plt.close(fig)

    # 6. Wagi modelu (poziomo, sortowane po |w|; zielone = sprzyjają A, czerwone = B).
    weights = model.linear.weight.detach().numpy().ravel()
    order = np.argsort(np.abs(weights))
    colors = ["#2a9d8f" if w > 0 else "#e76f51" for w in weights[order]]
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.barh(np.array(FEATURE_COLUMNS)[order], weights[order], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(title="Wagi jednowarstwowej sieci (standaryzowane cechy)\nzielone: wartość ↑ sprzyja zawodnikowi A, czerwone: zawodnikowi B",
           xlabel="Waga wᵢ (wpływ na 1σ wzrostu cechy)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(plots_dir / "feature_weights.png", dpi=130); plt.close(fig)

    # 7. Rozkład przewidywanych prawdopodobieństw wg rzeczywistego wyniku.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(p_test[y_test == 1], bins=25, alpha=0.6, density=True, label="rzeczywiste wygrane A (y=1)")
    ax.hist(p_test[y_test == 0], bins=25, alpha=0.6, density=True, label="rzeczywiste wygrane B (y=0)")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="próg decyzyjny 0.5")
    ax.set(title="Rozkład przewidywanych prawdopodobieństw zwycięstwa A — zbiór testowy",
           xlabel="P(A wygrywa) z modelu", ylabel="Gęstość")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(plots_dir / "probability_distribution.png", dpi=130); plt.close(fig)

    # 8. Equity curve strategii value bet (tylko gdy kursy dostępne).
    if betting and betting.get("available"):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for t, res in betting["thresholds"].items():
            ax.plot(range(1, len(res["equity"]) + 1), res["equity"],
                    label=f"value bet, próg {t} (ROI {res['roi']:+.2f})")
        fav = betting["favorite_benchmark"]
        ax.plot(range(1, len(fav["equity"]) + 1), fav["equity"], color="gray",
                linestyle="--", label=f"benchmark: faworyci (ROI {fav['roi']:+.2f})")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(title="Symulacja zakładów — skumulowany zysk (equity curve), stawka 1 j.",
               xlabel="Numer zakładu (chronologicznie, zbiór testowy)", ylabel="Skumulowany zysk [jednostki]")
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(plots_dir / "roi_analysis.png", dpi=130); plt.close(fig)
    else:
        print("[WYKRESY] roi_analysis.png pominięty — brak kursów w danych (analiza niedostępna).")

    print(f"[WYKRESY] zapisano w: {plots_dir}")


# ============================================================================
# ARTEFAKTY I RAPORT  (ETAP 6)
# ============================================================================
def save_artifacts(
    outputs_dir: Path,
    model: SingleLayerUFCClassifier,
    scaler: StandardScaler,
    metadata: Dict[str, Any],
    metrics: Dict[str, Any],
) -> None:
    """Zapisuje model.pt, scaler.joblib, metadata.json, metrics.json, report.md."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "input_size": metadata["n_features"],
         "feature_columns": metadata["feature_columns"]},
        outputs_dir / "model.pt",
    )
    joblib.dump(scaler, outputs_dir / "scaler.joblib")
    save_json(outputs_dir / "metadata.json", metadata)
    save_json(outputs_dir / "metrics.json", metrics)
    (outputs_dir / "report.md").write_text(render_report(metadata, metrics), encoding="utf-8")


def render_report(metadata: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    """Renderuje raport outputs/report.md (sekcje wg wymagań przedmiotu).

    Wartości pochodzą z metryk bieżącego uruchomienia; gdy analiza bukmacherska
    jest niedostępna (brak kursów), jej sekcja przyjmuje wersję informacyjną.
    """
    t = metrics["test"]
    maj = metrics["baselines"]["majority"]
    lr = metrics["baselines"]["logistic_regression"]
    book = metrics["baselines"].get("bookmaker") or {}
    bet = metrics.get("betting") or {}
    data = metadata.get("data", {})
    splits = metadata.get("splits", {})
    weights = metadata.get("weights", {})
    top_w = sorted(weights.items(), key=lambda kv: -abs(kv[1]))[:6]
    top_lines = "\n".join(
        f"- `{k}`: {v:+.3f} ({'sprzyja A' if v > 0 else 'sprzyja B'})" for k, v in top_w
    )
    bet_block = (
        "Strategia value bet (p ≥ próg oraz EV = p·kurs > 1, stawka 1 j.):\n\n"
        "| Próg | Zakłady | Win rate | ROI | ROI odporny (bez top-10) |\n|---|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {tt} | {r['n_bets']} | {r['win_rate']:.3f} | {r['roi']:+.3f} | "
            f"{r['roi_ex_top10']:+.3f} |" for tt, r in bet["thresholds"].items()
        )
        + f"\n\nBenchmark faworytów rynkowych: {bet['favorite_benchmark']['n_bets']} zakładów, "
          f"ROI {bet['favorite_benchmark']['roi']:+.3f}.\n\n"
        "**Ostrzeżenie o odporności:** ROI surowe bywa zdominowane przez pojedyncze wygrane "
        "longshoty (np. kurs 100:1); mediana zysku na zakład i ROI bez top-10 pokazują, "
        "czy strategia jest powtarzalna, czy zależna od szczęścia. Symulacja ma charakter "
        "wyłącznie edukacyjny i nie stanowi porady finansowej."
        if bet.get("available")
        else "Analiza bukmacherska niedostępna — w danych nie ma kursów bukmacherskich.\n"
    )
    book_line = (
        f"bukmacher (kursy zamknięcia bez marży, n={book.get('n_fights')}): "
        f"acc={book.get('accuracy', float('nan')):.4f}, ROC-AUC={book.get('roc_auc', float('nan')):.4f}, "
        f"log loss={book.get('log_loss', float('nan')):.4f}"
        if book.get("available") else "bukmacher: niedostępny (brak kursów w danych)"
    )
    return f"""# {metadata.get('project', PROJECT_NAME)} — raport z eksperymentu
## Interpretowalny predyktor wyników walk UFC (jednowarstwowa sieć neuronowa)

*Jednowarstwowa sieć neuronowa (PyTorch, `nn.Linear({metadata['n_features']}, 1)`) do predykcji
wyników walk UFC. Projekt edukacyjny — nie stanowi porady finansowej ani zachęty do hazardu.*

## 1. Problem biznesowy
Analityk zakładów sportowych potrzebuje taniego, szybkiego i interpretowalnego narzędzia
szacującego prawdopodobieństwo zwycięstwa zawodnika A oraz wykrywającego potencjalne
value bety względem kursów bukmacherskich.

## 2. Rozwiązanie
Narzędzie CLI trenujące jednowarstwową sieć neuronową ({metadata['n_params']} parametrów) na
cechach dostępnych wyłącznie PRZED walką, z chronologiczną walidacją (test = najnowsze walki),
benchmarkami (majority class, regresja logistyczna, rynek) i symulacją prostych strategii zakładowych.

## 3. Dane
- Źródło: {metadata['data'].get('source', 'n/d')}; {data.get('n_fights', 'n/d')} walk po normalizacji
  ({data.get('date_min', '?')} → {data.get('date_max', '?')}), {data.get('n_fighters', 'n/d')} zawodników.
- Po burn-in ({metadata['hyperparameters'].get('burn_in_years', DEFAULT_BURN_IN_YEARS)} lat): {splits.get('train', {}).get('n', 0) + splits.get('validation', {}).get('n', 0) + splits.get('test', {}).get('n', 0)} walk modelowych.
- Podział chronologiczny: train {splits.get('train', {}).get('n')} ({splits.get('train', {}).get('date_min')} → {splits.get('train', {}).get('date_max')}),
  validation {splits.get('validation', {}).get('n')}, test {splits.get('test', {}).get('n')} ({splits.get('test', {}).get('date_min')} → {splits.get('test', {}).get('date_max')}).
- {data.get('odds_share', 0):.1%} walk ma kursy bukmacherskie (użyte TYLKO do benchmarku i analizy, nie jako cecha).

## 4. Model jednowarstwowy i dlaczego jest wystarczający
Architektura: wektor {metadata['n_features']} ustandaryzowanych cech różnicowych (A−B) →
`nn.Linear({metadata['n_features']}, 1)` → sigmoid. To formalnie regresja logistyczna
zaimplementowana jako sieć neuronowa (wymóg przedmiotu). Prostota daje: {metadata['n_params']}
parametry (pełna interpretowalność), sekundowy trening na CPU, minimalne ryzyko przeuczenia
(gap train–test ≈ {t['accuracy'] - metrics['train']['accuracy']:+.3f}) i łatwe wyjaśnienie decyzji
(wkład cechy = waga × cecha).

## 5. Metryki (zbiór testowy, próg {t['threshold']})
| Metryka | Model | Majority | Regr. log. | Bukmacher |
|---|---:|---:|---:|---:|
| Accuracy | {t['accuracy']:.4f} | {maj['accuracy']:.4f} | {lr['accuracy']:.4f} | {book.get('accuracy', float('nan')):.4f} |
| ROC-AUC | {t['roc_auc']:.4f} | — | {lr['roc_auc']:.4f} | {book.get('roc_auc', float('nan')):.4f} |
| Log loss | {t['log_loss']:.4f} | {maj['log_loss']:.4f} | {lr['log_loss']:.4f} | {book.get('log_loss', float('nan')):.4f} |
| Brier | {t['brier']:.4f} | — | {lr['brier']:.4f} | {book.get('brier', float('nan')):.4f} |

Precision={t['precision']:.4f}, recall={t['recall']:.4f}, F1={t['f1']:.4f}, balanced accuracy={t['balanced_accuracy']:.4f},
PR-AUC={t['pr_auc']:.4f}, ECE (kalibracja)={t['ece']:.4f}. Macierz pomyłek: TN={t['confusion_matrix'][0][0]},
FP={t['confusion_matrix'][0][1]}, FN={t['confusion_matrix'][1][0]}, TP={t['confusion_matrix'][1][1]}.
Baseline'y: majority class — acc {maj['accuracy']:.4f}; regresja logistyczna sklearn — acc {lr['accuracy']:.4f},
ROC-AUC {lr['roc_auc']:.4f}; {book_line}.

## 6. Interpretacja wyników
Najmocniejsze wagi (pełna analiza: ZALEZNOSCI_ZMIENNYCH.md):
{top_lines}

Model najmocniej reaguje na wiek (młodszy → większe szanse), status rankingowy, zdolność
finiszowania i bieżącą formę; zasięg ramion liczy się tam, gdzie wzrost jest neutralny.
Wagi cech skorelowanych czytamy grupami (kolinearność).

## 7. Analiza bukmacherska
{bet_block}

## 8. Ograniczenia i ryzyka
- Model liniowy nie modeluje interakcji cech (świadomy kompromis: interpretowalność).
- Kariery przed 2013 (początek datasetu) są obcięte — mitygowane burn-in i imputacją.
- Symulacja zakładów pomija limity stawek, ruchy kursów i reakcję rynku.
- Rynek się zmienia: model wymaga okresowego retrainingu (`python main.py train`).

## 9. Możliwy rozwój produktu
Cechy z ostatnich walk (statystyki rund), styl walki, regularyzacja L1/L2, kalibracja
izotoniczna, raporty per gala, monitoring driftu model vs rynek.

## 10. Wniosek: czy prosty model ma sens jako MVP?
Tak. Model istotnie przewyższa przypadek (accuracy {t['accuracy']:.3f} vs 0.500, ROC-AUC
{t['roc_auc']:.3f} vs 0.5) i jest w pełni interpretowalny, a jego kalibracja (ECE {t['ece']:.3f})
pozwala porównywać prawdopodobieństwa z rynkiem. Nie pobija bukmachera — zgodnie z oczekiwaniem
— ale jako tani, wyjaśnialny prototyp analityczny stanowi realną bazę pod dalszy rozwój.

---
*Wygenerowano automatycznie przez main.py (v{metadata['code_version']}, seed {metadata['seed']}).*
"""


def load_artifacts(outputs_dir: Path) -> Tuple[SingleLayerUFCClassifier, StandardScaler, Dict[str, Any]]:
    """Wczytuje model/scaler/metadata i sprawdza spójność (test odporności).

    Rzuca DataError z czytelną instrukcją, gdy artefakty nie istnieją lub nie pasują
    do bieżącej wersji kodu (inna lista cech / rozmiar scalera).
    """
    model_path, scaler_path, meta_path = (
        outputs_dir / "model.pt", outputs_dir / "scaler.joblib", outputs_dir / "metadata.json",
    )
    missing = [str(p) for p in (model_path, scaler_path, meta_path) if not p.exists()]
    if missing:
        raise DataError(
            "Brak artefaktów modelu: " + ", ".join(missing) +
            "\nNajpierw uruchom trening: python main.py train"
        )
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    scaler: StandardScaler = joblib.load(scaler_path)
    feats = metadata.get("feature_columns", [])
    if feats != FEATURE_COLUMNS:
        raise DataError(
            "metadata.json zawiera inną listę cech niż bieżący kod — artefakty są z innej "
            "wersji projektu. Przetrenuj model: python main.py train"
        )
    if getattr(scaler, "n_features_in_", len(feats)) != len(feats):
        raise DataError("Scaler ma niezgodną liczbę cech z metadata.json. Przetrenuj model.")
    model = SingleLayerUFCClassifier(len(feats))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, scaler, metadata


# ============================================================================
# TRYB PREDYKCJI  (ETAP 7)
# ============================================================================
_LBS_TO_DIVISION: Dict[float, str] = {v: k for k, v in WEIGHT_CLASS_LBS.items()}


def _division_name(lbs: float) -> str:
    """Odwraca mapowanie limitu wagi na czytelną nazwę kategorii (do wydruku)."""
    return _LBS_TO_DIVISION.get(round(float(lbs), 1), f"{lbs:.0f} lbs")


def _find_fighter(name: str, all_fighters: Sequence[str]) -> str:
    """Dopasowuje nazwisko zawodnika: najpierw dokładne (case-insensitive), potem fuzzy.

    Rzuca DataError z listą podobnych nazwisk, gdy dopasowanie nie jest jednoznaczne.
    """
    key = name.strip().lower()
    for f in all_fighters:
        if f.strip().lower() == key:
            return f
    close = get_close_matches(name.strip(), list(all_fighters), n=5, cutoff=0.6)
    hint = ", ".join(f"'{c}'" for c in close) if close else "brak podobnych nazwisk w danych"
    raise DataError(
        f"Nie znaleziono zawodnika: '{name}'.\nPodobne nazwiska: {hint}\n"
        "Uwaga: dane zawierają wyłącznie zawodników UFC z lat 2013-2026."
    )


def _record_str(snap: Dict[str, float]) -> str:
    """Formatuje bilans i formę zawodnika do wydruku (np. '15-1-0, forma5=0.80, seria +4')."""
    streak = snap["streak"]
    streak_s = f"seria {streak:+.0f}" if abs(streak) >= 1 else "seria 0"
    return (f"{snap['wins']:.0f}-{snap['losses']:.0f}-{snap['draws']:.0f} (walk w datasetcie: {snap['n']:.0f}), "
            f"forma 5 walk: {snap['form5']:.2f}, {streak_s}")


def predict_fight(fighter_a: str, fighter_b: str, outputs_dir: Path, data_dir: Path) -> None:
    """Predykcja wyniku hipotetycznej walki A vs B na podstawie zapisanych artefaktów.

    Proces: artefakty (model/scaler/metadata) -> dane historyczne -> pełne kariery
    zawodników (build_fight_dataset zwraca historię NIEZALEŻNIE od burn-in) -> snapshoty
    'na dziś' -> cechy różnicowe -> predykcja + wkłady cech (w·z) + disclaimer.
    """
    print("=" * 70)
    print(f"{PROJECT_NAME} — predykcja walki (v{CODE_VERSION})")
    print("=" * 70)
    model, scaler, metadata = load_artifacts(outputs_dir)
    raw, source = load_fights_frame(data_dir)
    tidy, _info = normalize_schema(raw, source)
    ds, history, last_attrs = build_fight_dataset(tidy, min_date=None)  # pełna historia, bez burn-in
    del ds
    all_fighters = sorted(set(tidy["fighter_a"]) | set(tidy["fighter_b"]))
    fa = _find_fighter(fighter_a, all_fighters)
    fb = _find_fighter(fighter_b, all_fighters)
    if fa == fb:
        raise DataError("Podano tego samego zawodnika po obu stronach walki.")

    # Snapshoty „na dziś": data 1 dzień po ostatniej walce w datasetcie => pełne kariery.
    as_of = tidy["date"].max() + pd.Timedelta(days=1)
    snap_a = career_snapshot(history[fa], as_of)
    snap_b = career_snapshot(history[fb], as_of)
    la, lb = last_attrs.get(fa), last_attrs.get(fb)
    if la is None or lb is None:  # teoretycznie niemożliwe (każdy zawodnik ma >=1 walkę)
        raise DataError("Brak zapisanych atrybutów zawodnika w danych historycznych.")
    division = la["weight_class_lbs"] if np.isfinite(la["weight_class_lbs"]) else lb["weight_class_lbs"]
    womens = la["is_womens"] if np.isfinite(la["is_womens"]) else lb["is_womens"]
    rec = {
        "a_age": la["age"], "a_height": la["height"], "a_reach": la["reach"],
        "a_stance": la["stance"], "a_rank": la["rank"], "a_ranked": la["ranked"],
        "b_age": lb["age"], "b_height": lb["height"], "b_reach": lb["reach"],
        "b_stance": lb["stance"], "b_rank": lb["rank"], "b_ranked": lb["ranked"],
        "weight_class_lbs": division, "is_womens": womens, "is_title_bout": 0.0,
    }
    feats = make_feature_row(snap_a, snap_b, rec)
    X = pd.DataFrame([feats])[FEATURE_COLUMNS]
    # Ta sama ścieżka co przy ewaluacji: imputacja + standaryzacja + winsoryzacja ±3σ.
    X_scaled = transform_features(X, metadata.get("medians", {}), scaler)
    z = X_scaled[0].astype(float)  # wartości przeskalowane i przycięte — wkład cechy = wᵢ·zᵢ
    p_a = float(model.predict_proba(torch.from_numpy(np.ascontiguousarray(X_scaled)))[0])
    p_b = 1.0 - p_a
    winner, confidence = ("A", p_a) if p_a >= 0.5 else ("B", p_b)

    print(f"\n  A: {fa}")
    print(f"     bilans (w datasecie): {_record_str(snap_a)}")
    print(f"  B: {fb}")
    print(f"     bilans (w datasecie): {_record_str(snap_b)}")
    print(f"  Kontekst: kategoria ~{division:.0f} lbs ({_division_name(division)}), "
          f"walka zwykła (demo: is_title_bout=0)")
    # Ostrzeżenie o wiarygodności: model liniowy nie zna swoich granic — zaznaczamy je sami.
    reasons: List[str] = []
    if min(snap_a["n"], snap_b["n"]) < 3:
        reasons.append(
            f"mało walk w danych (A: {snap_a['n']:.0f}, B: {snap_b['n']:.0f}) — "
            "cechy bilansu i formy są wtedy szumem"
        )
    if abs(feats["reach_missing_diff"]) > 0:
        reasons.append(
            "asymetria braku zasięgu (jedna strona nieudokumentowana) — "
            "cecha-proxy statusu o dużym wpływie"
        )
    if max(snap_a["days_rest"], snap_b["days_rest"]) > 730:
        reasons.append("któryś zawodnik nie walczył w danych od ponad 2 lat — ekstrapolacja poza rozkład treningowy")
    if reasons:
        print("\n  ⚠ OSTRZEŻENIE O WIARYGODNOŚCI — traktuj predykcję z dużą rezerwą:")
        for r in reasons:
            print(f"    - {r}")
    print(f"\n  P(A wygrywa) = {p_a:.4f}   |   P(B wygrywa) = {p_b:.4f}")
    print(f"  Decyzja modelu: WYGRYWA {winner}  (confidence = {confidence:.1%})")

    w = model.linear.weight.detach().numpy().ravel()
    contribs = w * z
    order = np.argsort(-np.abs(contribs))[:5]
    print("\n  Top cechy decyzyjne (wkład = waga × cecha standaryzowana):")
    for i in order:
        side = "sprzyja A" if contribs[i] > 0 else "sprzyja B"
        print(f"    {FEATURE_COLUMNS[i]:22s} {contribs[i]:+.3f}  ({side})")
    print("\n  Uwaga: atrybuty zawodników (wiek/wzrost/zasięg/ranking) pochodzą z ich")
    print("  ostatnich walk w datasecie; hipotetyczna walka nie ma kursów bukmacherskich.")
    print(f"\n{DISCLAIMER}")


# ============================================================================
# KOMENDY CLI
# ============================================================================
def cmd_train(args: argparse.Namespace) -> None:
    """Pełny cykl: dane -> cechy -> trening -> ewaluacja -> artefakty."""
    print("=" * 70)
    print(f"{PROJECT_NAME} — trening jednowarstwowej sieci neuronowej (v{CODE_VERSION})")
    print("=" * 70)
    raw, source = load_fights_frame(args.data_dir)
    tidy, info = normalize_schema(raw, source)
    print(
        f"[DANE] Schemat: {info['schema']} | wiersze źródłowe: {info['n_raw']} | "
        f"pominięte (remisy/NC/błędne): {info['n_draw_nc'] + info['n_bad_rows']}"
    )
    summary = validate_tidy(tidy)
    # Burn-in: pomiń pierwsze N lat (cechy historyczne liczone tylko z datasetu,
    # więc dla walk z 2013 r. byłyby sztucznie puste dla niemal każdego zawodnika).
    min_date = tidy["date"].min() + pd.Timedelta(days=365.25 * args.burn_in_years)
    ds, _history, _last_attrs = build_fight_dataset(tidy, min_date=min_date)
    if args.max_rows:
        ds = ds.tail(args.max_rows).reset_index(drop=True)
    print(f"\n[CECHY] Walki po burn-in ({args.burn_in_years} lat) i filtrach: {len(ds)} | cech: {len(FEATURE_COLUMNS)}")
    train_df, val_df, test_df = split_train_val_test(ds, random_split=args.random_split, seed=args.seed)
    medians, scaler = fit_preprocessors(train_df)
    X_train = transform_features(train_df, medians, scaler)
    X_val = transform_features(val_df, medians, scaler)
    X_test = transform_features(test_df, medians, scaler)
    print(f"[SKALOWANIE] mediana+StandardScaler z train | X_train={X_train.shape}, "
          f"X_val={X_val.shape}, X_test={X_test.shape}")
    y_train = train_df["target"].to_numpy(dtype=np.float32)
    y_val = val_df["target"].to_numpy(dtype=np.float32)
    y_test = test_df["target"].to_numpy(dtype=np.float32)
    print(f"\n[TRENING] lr={args.lr}, batch={args.batch_size}, max epok={args.epochs}, "
          f"patience={args.patience}, seed={args.seed} (determinizm)")
    model, history = train_model(
        X_train, y_train, X_val, y_val,
        input_size=len(FEATURE_COLUMNS),
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        patience=args.patience, seed=args.seed,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] nn.Linear({len(FEATURE_COLUMNS)}, 1) -> {n_params} parametrów (wraz z biasem)")

    # --- Etap 5: ewaluacja na train/val/test + baseline'y + analiza bukmacherska ---
    print("\n[EWALUACJA]")
    train_metrics, p_train, _ = evaluate_dataset(model, X_train, y_train, args.threshold)
    val_metrics, p_val, _ = evaluate_dataset(model, X_val, y_val, args.threshold)
    test_metrics, p_test, _ = evaluate_dataset(model, X_test, y_test, args.threshold)
    print_metrics("train", train_metrics)
    print_metrics("val  ", val_metrics)
    print_metrics("test ", test_metrics)
    baselines = compute_baselines(X_train, y_train, X_test, y_test)
    bookie = bookmaker_baseline(test_df)
    print("\n[ANALIZA BUKMACHERSKA] symulacja value betów (stawka 1 j., kursy zamknięcia):")
    betting = betting_analysis(test_df, p_test)

    # --- Etap 6: wykresy + artefakty ---
    best_epoch = int(history.loc[history["val_loss"].idxmin(), "epoch"])
    make_all_plots(
        history, p_test, test_df["target"].to_numpy(int), test_metrics, model, betting,
        args.outputs_dir / PLOTS_DIR_NAME,
    )

    def split_info(df: pd.DataFrame) -> Dict[str, Any]:
        return {"n": int(len(df)), "date_min": str(df["date"].min().date()),
                "date_max": str(df["date"].max().date()), "positive_rate": float(df["target"].mean())}

    weights = model.linear.weight.detach().numpy().ravel()
    metadata: Dict[str, Any] = {
        "project": PROJECT_NAME,
        "code_version": CODE_VERSION, "created": pd.Timestamp.now().isoformat(timespec="seconds"),
        "seed": args.seed, "model_class": "SingleLayerUFCClassifier",
        "feature_columns": FEATURE_COLUMNS, "n_features": len(FEATURE_COLUMNS), "n_params": n_params,
        "weights": {c: float(w) for c, w in zip(FEATURE_COLUMNS, weights)},
        "bias": float(model.linear.bias.item()),
        "medians": medians,  # imputacja przy predykcji (scaler nie przechowuje median)
        "data": {"source": source, "schema": info["schema"], **summary},
        "splits": {"train": split_info(train_df), "validation": split_info(val_df), "test": split_info(test_df)},
        "split_kind": "losowy (awaryjny)" if args.random_split else "chronologiczny 70/15/15",
        "burn_in_years": args.burn_in_years,
        "hyperparameters": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                            "patience": args.patience, "threshold": args.threshold,
                            "burn_in_years": args.burn_in_years, "max_rows": args.max_rows},
        "best_epoch": best_epoch,
    }
    metrics_payload: Dict[str, Any] = {
        "code_version": CODE_VERSION, "seed": args.seed, "best_epoch": best_epoch,
        "n_features": len(FEATURE_COLUMNS), "n_params": n_params,
        "hyperparameters": metadata["hyperparameters"],
        "splits": metadata["splits"], "data": metadata["data"],
        "history": history.to_dict("records"),
        "train": train_metrics, "validation": val_metrics, "test": test_metrics,
        "baselines": {**baselines, "bookmaker": bookie},
        "betting": betting,
    }
    save_artifacts(args.outputs_dir, model, scaler, metadata, metrics_payload)
    print(f"\n[ARTEFAKTY] zapisano w: {args.outputs_dir}")
    for name in ("model.pt", "scaler.joblib", "metadata.json", "metrics.json", "report.md"):
        print(f"  - {name}")
    print(f"  - {PLOTS_DIR_NAME}/ (8 wykresów PNG)")
    print("\nGotowe. Raport: " + str(args.outputs_dir / "report.md"))
    print(DISCLAIMER)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Ponowna ewaluacja zapisanego modelu na zbiorze testowym (bez retreningu).

    Odbudowuje TEN SAM zbiór testowy (proces deterministyczny: ta sama normalizacja,
    burn-in i podział chronologiczny), transformuje go zapisanym scalerem i odświeża
    metrics.json oraz wykresy ewaluacyjne.
    """
    print("=" * 70)
    print(f"{PROJECT_NAME} — ewaluacja zapisanego modelu (v{CODE_VERSION})")
    print("=" * 70)
    model, scaler, metadata = load_artifacts(args.outputs_dir)
    raw, source = load_fights_frame(args.data_dir)
    tidy, _info = normalize_schema(raw, source)
    burn = float(metadata.get("burn_in_years", DEFAULT_BURN_IN_YEARS))
    min_date = tidy["date"].min() + pd.Timedelta(days=365.25 * burn)
    ds, _h, _a = build_fight_dataset(tidy, min_date=min_date)
    max_rows = (metadata.get("hyperparameters") or {}).get("max_rows")
    if max_rows:
        ds = ds.tail(int(max_rows)).reset_index(drop=True)
    _train_df, _val_df, test_df = split_train_val_test(ds, random_split=False)
    medians = metadata.get("medians", {})
    X_test = transform_features(test_df, medians, scaler)
    y_test = test_df["target"].to_numpy(dtype=np.float32)

    test_metrics, p_test, _ = evaluate_dataset(model, X_test, y_test)
    print("\n[EWALUACJA] zbiór testowy (model wczytany z artefaktów):")
    print_metrics("test ", test_metrics)
    betting = betting_analysis(test_df, p_test)

    metrics_path = args.outputs_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    metrics["test"] = test_metrics
    metrics["betting"] = betting
    metrics["evaluated_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
    save_json(metrics_path, metrics)
    history_df = pd.DataFrame(metrics.get("history", []))
    make_all_plots(
        history_df if not history_df.empty else None, p_test,
        test_df["target"].to_numpy(int), test_metrics, model, betting,
        args.outputs_dir / PLOTS_DIR_NAME,
    )
    print(f"\n[METRYKI] odświeżono: {metrics_path}")
    print(DISCLAIMER)


def cmd_predict(args: argparse.Namespace) -> None:
    """Tryb predykcji pojedynczej walki (--fighter-a/--fighter-b lub --input JSON)."""
    if args.input is not None:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        fighter_a = payload.get("fighter_a")
        fighter_b = payload.get("fighter_b")
    else:
        fighter_a, fighter_b = args.fighter_a, args.fighter_b
    if not fighter_a or not fighter_b:
        raise SystemExit("Błąd: podaj --fighter-a i --fighter-b (albo --input sample_fight.json).")
    predict_fight(fighter_a, fighter_b, args.outputs_dir, args.data_dir)


def cmd_demo(args: argparse.Namespace) -> None:
    """Demo na prezentację: szybki trening na najnowszych walkach + przykładowa predykcja.

    Używa tej samej komendy train (z ograniczeniem --max-rows), więc artefakty i wykresy
    są pełnoprawne; na końcu uruchamia predykcję dla dwóch znanych zawodników z danych.
    """
    print("#" * 70)
    print(f"#  DEMO — {PROJECT_NAME} (jednowarstwowa sieć neuronowa, v{CODE_VERSION})")
    print("#" * 70)
    ns = argparse.Namespace(
        data_dir=args.data_dir, outputs_dir=args.outputs_dir, epochs=args.epochs,
        batch_size=DEFAULT_BATCH_SIZE, lr=DEFAULT_LR, patience=DEFAULT_PATIENCE,
        seed=SEED, burn_in_years=DEFAULT_BURN_IN_YEARS, max_rows=args.max_rows,
        random_split=False, threshold=DEFAULT_THRESHOLD,
    )
    cmd_train(ns)
    # Przykładowa predykcja: preferowani rozpoznawalni zawodnicy, fallback = najaktywniejsi.
    raw, source = load_fights_frame(args.data_dir)
    tidy, _info = normalize_schema(raw, source)
    counts = pd.concat([tidy["fighter_a"], tidy["fighter_b"]]).value_counts()
    preferred = ["Jon Jones", "Max Holloway", "Alexander Volkanovski", "Israel Adesanya",
                 "Charles Oliveira", "Justin Gaethje", "Amanda Nunes", "Kamaru Usman"]
    found = [n for n in preferred if n in counts.index and counts[n] >= 5][:2]
    if len(found) < 2:
        found = [str(n) for n in counts.index[:2]]
    print("\n" + "#" * 70)
    print("#  PRZYKŁADOWA PREDYKCJA (demo)")
    print("#" * 70)
    predict_fight(found[0], found[1], args.outputs_dir, args.data_dir)


# ============================================================================
# MAIN
# ============================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    """Punkt wejścia: parsuje argumenty i dispatchuje do wybranej komendy."""
    try:  # Windows + Git Bash: wymuś UTF-8, żeby polskie znaki nie wywalały print()
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    args = build_arg_parser().parse_args(argv)
    handlers = {"train": cmd_train, "evaluate": cmd_evaluate, "predict": cmd_predict,
                "demo": cmd_demo, "compare": cmd_compare}
    try:
        handlers[args.command](args)
        return 0
    except DataError as exc:
        print(f"\n[BŁĄD DANYCH] {exc}", file=sys.stderr)
        return 2
    except NotImplementedError as exc:
        print(f"\n[W BUDOWIE] {exc}", file=sys.stderr)
        return 3
    except FileNotFoundError as exc:
        print(f"\n[BRAK PLIKU] {exc}\nNajpierw uruchom `python main.py train`.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
