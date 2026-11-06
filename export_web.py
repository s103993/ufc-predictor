# -*- coding: utf-8 -*-
"""Eksport modelu i snapshotów zawodników do strony demonstracyjnej (web/index.html).

Generuje web/data.js (zmienna globalna UFC_MODEL_DATA), dzięki której strona liczy
predykcje w przeglądarce BEZ serwera i BEZ PyTorcha: model jest liniowy, więc to
tylko 33 liczby (32 wagi + bias) + parametry scalera + snapshoty kariery zawodników.
Logika cech w JS jest wiernym lustrem make_feature_row() z main.py.

Uruchomienie:  python export_web.py
Deploy (darmowy): przeciągnij folder web/ na https://app.netlify.com/drop
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main as m  # ponowne użycie pipeline'u: ładowanie, normalizacja, snapshoty

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

# Polskie etykiety cech do wyświetlenia na stronie (wartość dodatnia = sprzyja A).
FEATURE_LABELS_PL = {
    "exp_diff": "Doświadczenie (liczba walk)",
    "wins_diff": "Zwycięstwa w karierze",
    "losses_diff": "Porażki w karierze",
    "draws_diff": "Remisy w karierie",
    "win_pct_diff": "Procent zwycięstw",
    "ko_pct_diff": "% zwycięstw przez KO/TKO",
    "sub_pct_diff": "% zwycięstw przez poddanie",
    "dec_pct_diff": "% zwycięstw przez decyzję",
    "finish_rate_diff": "Finish rate (KO + SUB)",
    "avg_fight_time_diff": "Średni czas walki",
    "med_fight_time_diff": "Mediana czasu walki",
    "fights_12m_diff": "Walki w ostatnich 12 mies.",
    "fights_24m_diff": "Walki w ostatnich 24 mies.",
    "form3_diff": "Forma z 3 walk",
    "form5_diff": "Forma z 5 walk",
    "days_rest_diff": "Dni odpoczynku od ostatniej walki",
    "streak_diff": "Seria zwycięstw/porażek",
    "title_exp_diff": "Walki tytułowe w przeszłości",
    "age_diff": "Wiek",
    "height_diff": "Wzrost",
    "reach_diff": "Zasięg ramion",
    "stance_diff": "Postawa",
    "ranked_diff": "Status rankingowy",
    "rank_diff": "Pozycja w rankingu UFC",
    "debut_diff": "Debiutant w datasecie",
    "both_debut": "Obaj debiutanci",
    "weight_class_lbs": "Kategoria wagowa",
    "is_womens_division": "Dywizja kobiet",
    "is_title_bout": "Walka o pas",
    "reach_missing_diff": "Brak danych o zasięgu",
    "height_missing_diff": "Brak danych o wzroście",
    "age_missing_diff": "Brak danych o wieku",
}


def _division_label(lbs: float, is_womens: float) -> str:
    """Czytelna nazwa kategorii wagowej dla wydruku (125 lbs występuje u obu płci)."""
    if not pd.notna(lbs):
        return "catchweight"
    lbs = round(float(lbs), 1)
    if is_womens == 1.0:
        womens = {v: k.removeprefix("womens") for k, v in m.WEIGHT_CLASS_LBS.items() if k.startswith("womens")}
        return womens.get(lbs, "catchweight")
    mens = {v: k for k, v in m.WEIGHT_CLASS_LBS.items() if not k.startswith("womens")}
    return mens.get(lbs, "catchweight")


def main() -> int:
    """Buduje web/data.js z wagami, parametrami preprocessingu i snapshotami zawodników."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    model, scaler, metadata = m.load_artifacts(BASE_DIR / "outputs")
    raw, source = m.load_fights_frame(BASE_DIR / "data")
    tidy, _info = m.normalize_schema(raw, source)
    # min_date=None => pełne kariery (history) dla wszystkich zawodników, tak jak w predict.
    _ds, history, last_attrs = m.build_fight_dataset(tidy, min_date=None)
    as_of = tidy["date"].max() + pd.Timedelta(days=1)

    fighters = {}
    for name, attrs in last_attrs.items():
        snap = m.career_snapshot(history[name], as_of)
        last_date = max(h.date for h in history[name]).date().isoformat()
        division = _division_label(attrs["weight_class_lbs"], attrs["is_womens"])
        fighters[name] = {"snap": snap, "attrs": attrs, "last_fight": last_date, "division": division}

    # Metryki testowe z ostatniego pełnego treningu — do stopki strony (uczciwa informacja).
    metrics_path = BASE_DIR / "outputs" / "metrics.json"
    test_metrics = {}
    if metrics_path.exists():
        mt = json.loads(metrics_path.read_text(encoding="utf-8")).get("test", {})
        test_metrics = {k: mt.get(k) for k in ("accuracy", "roc_auc", "log_loss", "brier", "ece")}

    payload = {
        "meta": {
            "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
            "code_version": m.CODE_VERSION,
            "as_of": str(as_of.date()),
            "date_min": str(tidy["date"].min().date()),
            "date_max": str(tidy["date"].max().date()),
            "n_fighters": len(fighters),
            "n_features": len(m.FEATURE_COLUMNS),
            "model": "nn.Linear(32, 1) + sigmoid (jednowarstwowa sieć neuronowa, PyTorch)",
            "test_metrics": test_metrics,
            "disclaimer": m.DISCLAIMER,
        },
        "feature_columns": m.FEATURE_COLUMNS,
        "feature_labels_pl": FEATURE_LABELS_PL,
        "weights": metadata["weights"],  # {cecha: waga} — bias osobno
        "bias": metadata["bias"],
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "medians": metadata["medians"],
        "fighters": fighters,
    }
    WEB_DIR.mkdir(exist_ok=True)
    out = WEB_DIR / "data.js"
    body = json.dumps(m.jsonable(payload), ensure_ascii=False, separators=(",", ":"))
    out.write_text("const UFC_MODEL_DATA = " + body + ";\n", encoding="utf-8")
    print(f"[EXPORT] zawodnicy: {len(fighters)}  ->  {out.name} ({out.stat().st_size / 1024:.0f} KB)")
    print(f"[EXPORT] snapshoty na dzień: {as_of.date()}  |  dane: {payload['meta']['date_min']} .. {payload['meta']['date_max']}")
    print("[EXPORT] gotowe. Lokalnie: otwórz web/index.html. Hosting: przeciągnij web/ na app.netlify.com/drop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
