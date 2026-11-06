# HANDOVER — przekazanie projektu WalkoMat

**Dla:** studenta przejmującego projekt (przedmiot ZMTSI 26/27)
**Od:** poprzedni autor · **Data przekazania:** 2026-11-06 · **Wersja projektu:** 1.1.0
**Stan ogólny:** ✅ KOMPLETNY I PRZETESTOWANY END-TO-END — gotowy do obrony; poniżej lista tego,
co można jeszcze dopełnić (backlog), oraz wszystkie procedury potrzebne do dalszej pracy.

---

## 1. TL;DR — czym jest ten projekt (30 sekund)

**WalkoMat** (dawniej: UFC Fight Predictor) to interpretowalny predyktor wyników walk UFC.
Model: **jednowarstwowa sieć neuronowa** `nn.Linear(32, 1)` + sigmoid (formalnie regresja
logistyczna, per wymóg przedmiotu zaimplementowana jako sieć w PyTorch z własną pętlą
treningową: entropia krzyżowa, mini-batch, backprop, early stopping). Model szacuje
prawdopodobieństwo zwycięstwa zawodnika A na podstawie 32 cech liczonych **wyłącznie
z informacji sprzed walki** (reżim anti-leakage). Dostarczamy: CLI (`train / evaluate /
predict / demo`), pełną ewaluację z baseline'ami (w tym rynek bukmacherski), 8 wykresów,
automatyczny raport, statyczną stronę demo z predykcją w przeglądarce oraz komplet
dokumentacji obronowej. Wynik testowy: **accuracy 0.637, ROC-AUC 0.687** przy 33 parametrach.

## 2. Pierwsze 30 minut przejmującego (onboarding)

1. Przeczytaj `README.md` (10 min) — instalacja, dane, komendy.
2. `python main.py demo` — zobacz cały cykl na żywo (trening na 2500 walk + predykcja).
3. Przeczytaj `OBRONA_PROJEKTU.md` §1–8 (metodyka) — to najlepsze wprowadzenie do „dlaczego tak".
4. Uruchom `python main.py train` (pełny, ~1–2 min) i **porównaj wyniki z tabelą referencyjną
   w sekcji 5 tego dokumentu** — dzięki determinizmowi powinieneś dostać IDENTYCZNE liczby.
   Jeśli nie są identyczne — coś się zmieniło w kodzie/danych; zanim cokolwiek zmienisz,
   sprawdź sekcję 8 (zasady krytyczne).
5. Otwórz `web/index.html` w przeglądarce (lokalnie, bez serwera) — strona demo.
6. Przejrzyj mapę kodu (sekcja 6) i wybierz pierwszy punkt z backlogu (sekcja 7).

## 3. Struktura repozytorium

| Plik/katalog | Rola | Status |
|---|---|---|
| `main.py` (~1900 linii) | **Cała logika** w sekcjach: dane → cechy → model → trening → ewaluacja → wykresy → artefakty → predict/demo | ✅ kompletny |
| `export_web.py` | Eksport wag + snapshotów zawodników do `web/data.js` (uruchamiać po każdym retreningu) | ✅ |
| `requirements.txt` | numpy, pandas, matplotlib, scikit-learn, torch, joblib (+ opcjonalny `ufcdata` zakomentowany) | ✅ |
| `README.md` | Instrukcja + streszczenie akademickie + cytowanie | ✅ |
| `OBRONA_PROJEKTU.md` | Kompendium na obronę: słowniki zmiennych, mapowanie na wykłady, wyniki, Q&A | ✅ |
| `ZALEZNOSCI_ZMIENNYCH.md` | Analiza 32 wag, korelacji, kolinearności + historia „eksplozji pewności" i naprawy | ✅ |
| `HANDOVER.md` | Ten dokument | ✅ |
| `sample_fight.json` | Przykład wejścia dla `predict --input` | ✅ |
| `data/ufc_all_fights_wide.csv` (~11 MB) | Dane: nlbair/ufc-data, 6743 walki 2013–2026, 496 kolumn | ✅ w miejscu |
| `web/index.html` (~16 KB) | Strona demo (ciemny motyw, animacje, glify ◈ ⇶ ⌁) | ✅ |
| `web/data.js` (~1 MB) | Wygenerowany: wagi, scaler, mediany, snapshoty 2060 zawodników | ✅ generowany |
| `outputs/` | Artefakty: `model.pt`, `scaler.joblib`, `metadata.json`, `metrics.json`, `report.md`, `plots/*.png` (8) | ✅ generowane |

**Generowane vs. w repo:** do repo trzymasz `data/` i wszystkie `.md`; `outputs/` i `web/data.js`
są regenerowalne (train + export_web.py). Jeśli robisz gita — dodaj `.gitignore` dla
`outputs/`, `__pycache__/`, `web/data.js` (opcjonalnie) i `data/` (11 MB).

## 4. Środowisko i uruchomienie od zera

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows (testowane: Python 3.11.9)
pip install -r requirements.txt                     # torch 2.15.0+cpu wystarcza
# dane: pobierz ufc_all_fights_wide.csv z https://github.com/nlbair/ufc-data (link w README §3)
python main.py train                                # pełny cykl (~1–2 min na CPU)
python main.py evaluate                             # re-ewaluacja z artefaktów
python main.py predict --fighter-a "Jon Jones" --fighter-b "Tom Aspinall"
python main.py predict --input sample_fight.json
python main.py demo                                 # szybki cykl na 2500 walkach
python export_web.py                                # odświeża web/data.js
```

**Znane pułapki:**
- Windows/Git Bash: program sam wymusza UTF-8 na stdout; przy „krzakach" `chcp 65001`.
- Brak pliku danych → czytelny `DataError` z instrukcją pobrania (nie śmiertelny crash).
- Brak artefaktów przy `predict/evaluate` → komunikat „najpierw uruchom train".
- Inna lista cech w `metadata.json` vs kod → `load_artifacts` przerywa z komunikatem
  „przetrenuj model" (kontrakt wersji; patrz sekcja 8).

## 5. Wyniki referencyjne (do porównania po własnym treningu)

**Determinizm jest bitowy** (seed 42 + `torch.set_num_threads(1)`): twój trening powinien
odtworzyć te liczby co do cyfry. Jeśli odtwarza — środowisko OK.

**Pipeline danych:** 6743 surowe → 6618 po czyszczeniu (125 remisy/NC/błędne) → 5726 po
burn-in 2 lata → **train 4008** (2015-02→2023-06) / **val 859** (→2025-03) / **test 859**
(→2026-10), podział CHRONOLOGICZNY, udział klasy 1 ≈ 0.50 (po losowaniu stron A/B z crc32).

**Model:** epoka 39 (early stopping, patience 10), val loss 0.6491, 33 parametry,
hiperparametry: Adam lr=1e-3, weight_decay=0.01 (L2), batch=64, winsoryzacja ±3σ, seed 42.

| Metryka (test) | Model | Majority | Regr. log. sklearn | Bukmacher |
|---|---:|---:|---:|---:|
| Accuracy | **0.6368** | 0.4948 | 0.6298 | 0.7052 |
| ROC-AUC | **0.6865** | — | 0.6767 | 0.7647 |
| Log loss | **0.6434** | 0.6932 | 0.6458 | 0.5799 |
| Brier | **0.2260** | — | 0.2273 | 0.1977 |
| ECE | **0.0421** | — | — | — |

Macierz pomyłek (test): TN=287, FP=147, FN=165, TP=260. Model osiąga ~70% przewagi rynku
nad przypadkiem; **nie pobija bukmachera** (i nie miał — patrz OBRONA §12.2).

**Analiza bukmacherska (edukacyjna, 842/859 walk z kursami):** próg 0.55 → 179 zakładów,
surowy ROI +1.225, ale **odporny (bez top-10) −0.036** i mediana zysku −1.00; progi 0.60/0.65
odpowiednio +0.783/−0.056 i +0.594/−0.219. Benchmark faworytów: +0.003 (odporny −0.029).
**Wniosek obowiązujący: strategia NIE jest powtarzalna** — nie zmieniaj tego wniosku bez
bardzo dobrego powodu.

**Top wagi (z `metadata.json`):** `reach_missing_diff` −0.700 (proxy statusu, wpływ ograniczony
capem do 2.10 logitu), `age_diff` −0.348, `rank_diff` −0.146, `wins_diff` +0.142,
`finish_rate_diff` +0.134, `losses_diff` −0.129; trzy cechy o zerowej wariancji mają wagi
dokładnie 0.000 (wyzerowane przez L2).

## 6. Mapa kodu `main.py` (sekcje w kolejności występowania)

| Sekcja | Kluczowe funkcje | Co robi / kiedy dotykać |
|---|---|---|
| Stałe i konfiguracja | `FEATURE_COLUMNS`, `Z_CAP_SIGMA`, `DEFAULT_WEIGHT_DECAY` | **Kontrakt cech** — lista 32 nazw; kolejność = identyfikacja artefaktów |
| CLI | `build_arg_parser` | subkomendy train/evaluate/predict/demo + flagi |
| Pomocnicze | `seed_everything`, `save_json`, `jsonable` | determinizm (nie usuwaj `set_num_threads(1)`) |
| Ładowanie danych | `load_fights_frame`, `normalize_schema`, `_normalize_ufcdata_wide`, `_normalize_legacy_rb`, `validate_tidy` | auto-detekcja pliku, 2 adaptery schematów (wide f1/f2 + legacy R/B), losowanie stron A/B (crc32), walidacja |
| Feature engineering | `career_snapshot`, `build_fight_dataset`, `make_feature_row`, `split_train_val_test`, `fit_preprocessors`, `transform_features` | **Serce anti-leakage**: snapshot → cechy → dopiero update historii; split 70/15/15; mediana+scaler+cap z train |
| Model i trening | `SingleLayerUFCClassifier`, `train_model` | `nn.Linear(32,1)`, Adam+L2, mini-batch, early stopping, przywracanie najlepszych wag |
| Ewaluacja i biznes | `evaluate_dataset`, `expected_calibration_error`, `compute_baselines`, `bookmaker_baseline`, `betting_analysis` | pełne metryki + ECE, 3 baseline'y, symulacja value betów ze statystykami odpornymi |
| Wykresy | `make_all_plots` | 8 PNG do `outputs/plots/` |
| Artefakty i raport | `save_artifacts`, `render_report`, `load_artifacts` | zapis/zapis modelu+scaler+metadata+metrics+report; spójność przy wczytaniu |
| Tryb predykcji | `predict_fight`, `_find_fighter` | fuzzy matching nazwisk, ostrzeżenie o wiarygodności, wkłady wᵢ·zᵢ |
| Porównanie uczenia | `COMPARE_VARIANTS`, `_make_optimizer`, `_make_loss`, `run_training_comparison`, `cmd_compare` | komenda `compare`: 6 wariantów (optymalizatory/straty/batch) od identycznego startu; wyniki: compare.md/json + 2 wykresy |
| Komendy | `cmd_train`, `cmd_evaluate`, `cmd_predict`, `cmd_demo`, `cmd_compare` | `cmd_demo` woła `cmd_train` z `--max-rows` |

**Strona (`web/index.html`):** JS to lustro `make_feature_row` + `transform_features`
(imputacja → standaryzacja → clip ±3) — **każda zmiana cech musi trafić w OBA miejsca**.
Test parzystości: 5 par walk, web vs CLI zgodne do 4. miejsca (patrz sekcja 8).

## 7. Backlog — co można dopełnić (priorytetyzowane)

**P1 — wysoka wartość, mały koszt (polecam na start):**
1. **Regresja z kursami jako cechą (wariant „rynkowy")** — dodać `prob_fair_diff` do cech,
   porównać metryki vs obecne; odpowiada na naturalne pytanie obronowe „ile daje rynek?".
   Koszt: ~20 linii + retrening. Uwaga: `a_prob_fair` już jest w tidy, tylko nie wchodzi do cech.
2. **Automatyczny test parzystości web↔CLI** — skrypt z sekcji testów (był uruchamiany
   ręcznie w konsoli) zamienić w `test_parity.py` + dodać `pytest` dla `make_feature_row`
   (kilka przypadków: debiutant, brak zasięgu, remis w historii). Chroni przed rozjazdami
   main.py ↔ index.html.
3. **Mini-grid hiperparametrów** — ✅ CZĘŚCIOWO ZROBIONE: komenda `compare` porównuje już
   optymalizatory (SGD/Momentum/RMSprop/Adam), funkcje straty (BCE vs MSE) i tryb batchów
   (mini vs full) na ujednoliconej metryce (val log loss). Do dopełnienia: siatka po
   learning rate (np. 1e-4/1e-3/1e-2) — jeden dodatkowy wymiar w `COMPARE_VARIANTS`.

**P2 — wartościowe, średni koszt:**
4. **Bootstrap CI dla ROI** (np. 1000 resamplingów equity curve) — zamiast tylko
   „ROI bez top-10"; daje przedział ufności do sekcji bukmacherskiej.
5. **Cache datasetu** — tidy jako parquet po pierwszym wczytaniu (predict/export uruchamiają
   się wtedy w sekundy zamiast ~40 s).
6. **Nowe dane 2027** — gdy repo ufc-data wyjdzie z aktualizacją: podmiana CSV →
   `train` → `export_web.py` → aktualizacja liczb w 3 dokumentach; pipeline jest gotowy.
7. **Kalibracja izotoniczna/Platta** na val (sklearn) jako warstwa post-procesingu —
   ECE 0.042 da zejść niżej; pokazać krzywą przed/po.

**P3 — porządkowe, opcjonalne:**
8. **Odchudzenie kodu** (była decyzja odłożona): usunąć adapter legacy R/B, fallback
   `ufcdata`, auto-detekcję plików, `--input JSON` (~1870 → ~1100 linii, styl „bardziej
   studencki"). Wymaga aktualizacji nazw funkcji w dokumentach, jeśli się zmienią.
9. **GitHub Actions**: workflow „train + porównaj metryki z referencją + test parzystości".
10. **CLI `--json`** dla predict (maszynowalne API dla innych narzędzi).

**Czego celowo NIE robić** (decyzje projektowe, nie braki): warstw ukrytych/deep learningu
(wymóg przedmiotu + interpretowalność), kursów jako cechy bez wyraźnego oznaczenia wariantu,
losowego podziału jako domyślnego, statystyk bieżącej walki jako cech (leakage).

## 8. Zasady krytyczne (nie łamać — tu boimy się regresji)

1. **Anti-leakage:** cechy tylko z walk wcześniejszych; kolejność w `build_fight_dataset`:
   snapshot → cechy → update historii. Statystyki bieżącej walki i wynik nigdy nie są cechą.
2. **Kursy** służą wyłącznie benchmarkowi/analizie — nie cechom (patrz backlog #1, jeśli
   chcesz to zmienić świadomie).
3. **Kolejność `FEATURE_COLUMNS` to kontrakt** — `load_artifacts` porównuje ją z
   `metadata.json`; zmiana listy = konieczny retrening + re-eksport web + aktualizacja
   tabel w 2 dokumentach.
4. **Jedna ścieżka preprocessingu:** `transform_features` (imputacja → scaler → clip ±3σ)
   obowiązuje w train/eval/predict **i** w lustrze JS na stronie. Zmiana po jednej stronie
   = rozjazd predykcji.
5. **Determinizm:** `torch.set_num_threads(1)` w `seed_everything` jest obowiązkowy
   (wielowątkowe redukcje CPU dawały różne wagi między uruchomieniami przy tym samym
   seedzie — naprawione, potwierdzone hashem wag).
6. **Neutralizacja f1/f2** (crc32 z fight_id) — bez niej kolejność ze źródła wycieka
   z wynikiem (f1 wygrywało 56.6% walk; po losowaniu 49.9%).
7. **Strona czyta statyczny `web/data.js`** — po każdym retreningu uruchom `export_web.py`,
   inaczej strona pokazuje stary model (zawsze zgodny z metadanymi tylko po eksporcie).

## 9. Procedury operacyjne

**Jak dodać nową cechę (checklist):**
1. `FEATURE_COLUMNS` — dopisz nazwę (na końcu listy, żeby nie przesunąć istniejących).
2. Zasil ją w `make_feature_row` (dane wejściowe: snapshoty z `career_snapshot` lub `rec`).
3. Jeśli cecha potrzebuje nowego pola z tidy — dodaj je w `normalize_schema` + `TIDY_COLUMNS`.
4. `python main.py train` → sprawdź metryki i wagę nowej cechy w `metadata.json`.
5. `python export_web.py` → jeśli cecha liczona inaczej w JS, zaktualizuj `buildFeatures`
   w `web/index.html` → test parzystości (5 par, tolerancja 5e-4).
6. Zaktualizuj: tabelę w `ZALEZNOSCI_ZMIENNYCH.md`, sekcję 4 `OBRONA_PROJEKTU.md`
   (słownik cech) i sekcję wyników.

**Jak odświeżyć dane (nowa wersja CSV):**
podmień plik w `data/` → `python main.py train` → `python export_web.py` → porównaj
metryki z poprzednimi (metrics.json jest nadpisywany — zrób kopię przed) → zaktualizuj
liczby w OBRONA §12 i README §5.

**Jak zregenerować wszystko od zera:** `train` → `evaluate` (sanity: identyczne metryki)
→ `predict` (smoke) → `export_web.py` → otwórz `web/index.html`.

## 10. Kwestie formalne

- **Dane:** nlbair/ufc-data, licencja CC BY 4.0 (attribution wymagane — jest w README §12
  i stopce strony). Źródła wtórne: ufcstats.com, BestFightOdds, MMA Decisions.
- **Disclaimer** obowiązkowo przy każdej prezentacji: projekt edukacyjny, nie porada
  finansowa, symulacje zakładów nie są rekomendacją gry (stała `DISCLAIMER` w main.py,
  stopka strony, report.md).
- **Prowadzący:** najmocniejsze karty na obronę to (1) reżim anti-leakage z dowodem
  (49.9% vs 56.6%), (2) benchmark rynkowy z uczciwą konstatacją, (3) historia winsoryzacji+L2
  jako lekcja z realnej awarii (ZALEZNOSCI §7), (4) determinizm z hashem wag, (5) statystyki
  odporne w analizie zakładów. Wszystko opisane w `OBRONA_PROJEKTU.md` z Q&A.

**Powodzenia!** Projekt jest w stanie, w którym każdy następny krok to czysty zysk.
