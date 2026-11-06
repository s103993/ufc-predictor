# WalkoMat 🥊
### Interpretowalny predyktor wyników walk UFC — jednowarstwowa sieć neuronowa (PyTorch)

**Projekt na przedmiot:** Zaawansowane Metody i Techniki Sztucznej Inteligencji 26/27 ·
**Wersja:** 1.1.0 · *(dawniej: UFC Fight Predictor)*

> **DISCLAIMER:** Projekt ma charakter wyłącznie **edukacyjny**. Nie stanowi porady
> finansowej, inwestycyjnej ani zachęty do hazardu. Symulacje zakładów służą analizie
> jakości modelu, nie rekomendacji gry u bukmachera.

## Streszczenie

Praca prezentuje **WalkoMat** — interpretowalny system predykcji wyników walk organizacji
UFC, zbudowany w oparciu o jednowarstwową sieć neuronową (`nn.Linear(32, 1)`, aktywacja
sigmoidalna), zaimplementowaną w PyTorch z własną pętlą treningową (entropia krzyżowa,
mini-batch gradient descent, early stopping). Model uczy się wyłącznie na informacjach
dostępnych przed walką (reżim anti-leakage) na zbiorze 6618 walk z lat 2013–2026, z
chronologicznym podziałem train/validation/test (70/15/15), winsoryzacją cech ±3σ oraz
regularyzacją L2. Na najnowszych 15% walk (zbiór testowy, 2025–2026) model osiąga
accuracy 0.637 i ROC-AUC 0.687 — ok. 70% przewagi rynku bukmacherskiego nad przypadkiem —
przy zaledwie 33 parametrach i pełnej interpretowalności decyzji (wkłady wᵢ·zᵢ). System
zawiera ponadto benchmarki (majority class, regresja logistyczna, kursy rynkowe), symulację
strategii value betów ze statystykami odpornymi, ostrzeżenia o wiarygodności dla danych
niskiej jakości oraz statyczną stronę demo z predykcją liczona w przeglądarce.

**Słowa kluczowe:** perceptron; regresja logistyczna; entropia krzyżowa; mini-batch gradient
descent; backpropagation; data leakage; podział chronologiczny; bias-variance trade-off;
kalibracja probabilistyczna; winsoryzacja; regularyzacja L2; value bet

---

## 1. Co to jest

Narzędzie CLI, które:

1. **trenuje** jednowarstwową sieć neuronową (`nn.Linear(32, 1)` — **bez warstw ukrytych**)
   przewidującą prawdopodobieństwo zwycięstwa zawodnika A w walce UFC,
2. **ewaluuje** model na najnowszych walkach (chronologiczny test = symulacja przyszłości),
3. **zapisuje metryki** (accuracy, ROC-AUC, log loss, Brier, kalibracja ECE i in.) oraz **8 wykresów**,
4. porównuje model z **benchmarkami** (majority class, regresja logistyczna, kursy bukmacherskie),
5. symuluje prostą **strategię value betów** (ROI, equity curve, statystyki odporne),
6. pozwala zrobić **przykładową predykcję** dowolnej pary zawodników z danych.

**Zastosowanie biznesowe:** proste narzędzie analityczne dla analityka zakładów sportowych /
fanów MMA — MVP do porównywania własnych oszacowań z rynkiem i wyszukiwania potencjalnych
value betów. Tani w utrzymaniu (trening na CPU w ~1 min), w pełni interpretowalny (33 parametry).

## 2. Model (wymóg przedmiotu)

Jednowarstwowa sieć neuronowa **bez warstw ukrytych** — czyli formalnie regresja logistyczna,
ale zgodnie z zakresem przedmiotu zaimplementowana jako sieć neuronowa w PyTorch z **własną
pętlą treningową**:

```python
class SingleLayerUFCClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)   # JEDYNA warstwa
    def forward(self, x):
        return self.linear(x)                     # logit; sigmoid w funkcji straty / przy predykcji
```

- strata: **BCEWithLogitsLoss** (entropia krzyżowa, numerycznie stabilna),
- optymalizator: **Adam** (lr=1e-3, **weight_decay=0.01** — regularyzacja L2),
  **mini-batch** learning (batch=64, deterministyczny shuffle),
- **winsoryzacja cech do ±3σ** — jedna rzadka cecha nie może zdominować predykcji,
- **early stopping** na validation loss (patience=10) z przywracaniem najlepszych wag,
- seed=42 + `torch.set_num_threads(1)` (determinizm bitowy — identyczne wagi przy każdym treningu),
- **brak** dropoutu, batch normy i warstw ukrytych — świadomie.

Dlaczego prosty model: niski koszt, szybki trening, mniejsze ryzyko przeuczenia,
pełna interpretowalność (wkład cechy = waga × cecha), łatwe wyjaśnienie decyzji użytkownikowi.
Szczegółowa analiza wag i zależności: [`ZALEZNOSCI_ZMIENNYCH.md`](ZALEZNOSCI_ZMIENNYCH.md),
kompendium na obronę: [`OBRONA_PROJEKTU.md`](OBRONA_PROJEKTU.md).

## 3. Dane

- Źródło: **[nlbair/ufc-data](https://github.com/nlbair/ufc-data)** (CC BY 4.0) —
  6743 walki UFC (2013–2026), 496 kolumn, 99.4% walk z kursami bukmacherskimi.
- Loader **auto-detekcją** szuka pliku w `data/` (CSV/JSON; obsługuje też starsze zbiory
  Kaggle ze schematem R/B). Trenowanie działa **offline** — internet potrzebny tylko do pobrania danych.

### Przygotowanie danych (jednorazowo)

Pobierz plik i zapisz go jako `data/ufc_all_fights_wide.csv` (obok `main.py`):

```
https://raw.githubusercontent.com/nlbair/ufc-data/main/data/ufc_all_fights_wide.csv
```

lub sklonuj repozytorium i skopiuj plik:

```bash
git clone https://github.com/nlbair/ufc-data.git
mkdir -p data && cp ufc-data/data/ufc_all_fights_wide.csv data/
```

Bez danych program przerwie działanie z czytelną instrukcją. Opcjonalnie można zainstalować
pakiet `ufcdata` (patrz `requirements.txt`) — wtedy loader spróbuje pobrać dane przez niego
(wymaga internetu przy pierwszym użyciu).

## 4. Instalacja i szybki start

Wymagania: Python 3.10+ (testowane na 3.11).

```bash
pip install -r requirements.txt

python main.py train       # pełny cykl: trening -> ewaluacja -> wykresy -> artefakty
python main.py evaluate    # re-ewaluacja zapisanego modelu (bez retreningu)
python main.py demo        # szybka prezentacja: trening na 2500 walkach + predykcja
python main.py compare     # porównanie wariantów uczenia (SGD/Momentum/RMSprop/Adam, MSE, batch)
```

Przykładowa predykcja:

```bash
python main.py predict --fighter-a "Jon Jones" --fighter-b "Tom Aspinall"
python main.py predict --input sample_fight.json
```

Wybrane parametry treningu:

```bash
python main.py train --epochs 200 --lr 5e-4 --batch-size 32 --patience 15
python main.py train --burn-in-years 3        # pomiń więcej wczesnych walk (puste historie)
python main.py train --random-split           # awaryjny podział losowy (domyślnie CHRONOLOGICZNY)
```

## 5. Wyniki (realne, seed 42)

Podział chronologiczny: train 4008 walk (2015-02→2023-06), validation 859 (do 2025-03),
test **859 najnowszych** (2025-03→2026-10).

| Metryka (test) | Model | Majority | Regr. log. | Bukmacher |
|---|---:|---:|---:|---:|
| Accuracy | **0.637** | 0.495 | 0.630 | 0.705 |
| ROC-AUC | **0.687** | — | 0.677 | 0.765 |
| Log loss | **0.643** | 0.693 | 0.646 | 0.580 |
| Brier | **0.226** | — | 0.227 | 0.198 |
| ECE (kalibracja) | **0.042** | — | — | — |

Czytanie: model wyraźnie bije przypadek i równa regresji logistycznej sklearn (kontrola
poprawności własnej pętli treningowej); **nie pobija rynku** (i nie miał pobijać — rynek
agreguje informacje spoza danych), osiąga ~70% jego przewagi nad przypadkiem. Pełne wyniki
i interpretacja: `outputs/report.md` (po treningu) oraz `OBRONA_PROJEKTU.md` §12.

Analiza zakładów (edukacyjna): surowy ROI progu 0.55 jest zdominowany przez pojedyncze
longshoty; **statystyki odporne** (mediana zysku na zakład, ROI bez top-10) pokazują, że
strategia nie jest powtarzalna — uczciwy wniosek, zgodny z disclaimerem.

## 6. Struktura projektu

```
├── main.py                 # CAŁA logika (sekcje: dane -> cechy -> model -> trening -> ewaluacja -> raport)
├── export_web.py           # eksport modelu + snapshotów do strony (web/data.js)
├── requirements.txt
├── README.md               # ten plik
├── ZALEZNOSCI_ZMIENNYCH.md # analiza wag, korelacji i zależności między cechami
├── OBRONA_PROJEKTU.md      # kompendium na obronę (słowniki zmiennych, pojęcia z wykładów, Q&A)
├── HANDOVER.md             # dokument przekazania: stan, mapa kodu, procedury, backlog
├── sample_fight.json       # przykład wejścia dla `predict --input`
├── data/                   # TU zapisz ufc_all_fights_wide.csv
├── web/                    # statyczna strona demo (index.html + data.js) — hosting Netlify
└── outputs/                # artefakty (powstają po treningu)
    ├── model.pt, scaler.joblib, metadata.json, metrics.json, report.md
    └── plots/*.png         # 8 wykresów
```

## 7. Co powstaje po treningu

| Artefakt | Zawartość |
|---|---|
| `outputs/model.pt` | wytrenowany model (state_dict + lista cech) |
| `outputs/scaler.joblib` | StandardScaler dopasowany **tylko na train** |
| `outputs/metadata.json` | lista cech (kolejność!), wagi, mediany imputacji, dane, split, hiperparametry |
| `outputs/metrics.json` | metryki train/val/test, baseline'y, historia epok, analiza zakładów |
| `outputs/report.md` | raport z 10 sekcjami (problem biznesowy → wniosek MVP) |
| `outputs/compare.md` + `compare.json` | porównanie 6 wariantów uczenia (tabela + wnioski „dlaczego Adam i BCE") |
| `outputs/plots/*.png` | training_history, accuracy_history, roc_curve, confusion_matrix, calibration_curve, feature_weights, probability_distribution, roi_analysis, compare_loss_curves, compare_auc_bars |

## 8. Jak zapewniona jest poprawność (anti-leakage)

1. Cechy historyczne liczone **wyłącznie z walk wcześniejszych** (snapshot przed zapisem walki do historii).
2. Statystyki bieżącej walki (uderzenia, kontrole, nokdauny) **niewykorzystane**.
3. Kursy bukmacherskie **nie są cechą** — tylko benchmark/analiza.
4. Kolejność f1/f2 ze źródła zneutralizowana losową zamianą stron (f1 wygrywało 56.6% walk —
   po zamianie: 49.9%).
5. StandardScaler i mediany **tylko z train**; val/test transformowane tym samym „miernikiem".
6. **Podział chronologiczny** — test to walki najnowsze (losowy dostępny jako awaryjny `--random-split`).
7. **Winsoryzacja cech do ±3σ** na wszystkich ścieżkach (trening/ewaluacja/predict/strona)
   + **ostrzeżenie o wiarygodności** w `predict` i na stronie, gdy zawodnik ma <3 walki
   w danych, jest asymetria braku zasięgu albo przerwa >2 lata — model nie udaje pewności
   poza rozkładem treningowym.

## 9. Sekcja dydaktyczna (powiązanie z wykładem)

| Pojęcie z wykładu | Gdzie w projekcie |
|---|---|
| Perceptron, sieć jednowarstwowa, brak warstw ukrytych | `SingleLayerUFCClassifier` = `nn.Linear(32, 1)` |
| Warstwa wejściowa / wyjściowa | 32 cechy → 1 neuron → logit → sigmoid |
| Regresja logistyczna jako sieć jednowarstwowa | ta sama architektura; sklearn jako baseline |
| Entropia krzyżowa | `BCEWithLogitsLoss` |
| Gradient descent / SGD / mini-batch | pętla treningowa: batch=64, `loss.backward()` + `optimizer.step()` |
| Backpropagation | autograd PyTorch (reguła łańcuchowa) |
| Generalizacja, train/val/test | split chronologiczny 70/15/15 + early stopping |
| Bias-variance trade-off | 33 parametry, gap train–test ≈ 0 → niska wariancja |

## 10. Rozwiązywanie problemów

- **„Nie znaleziono danych"** — zapisz `ufc_all_fights_wide.csv` w `data/` (sekcja 3).
- **„Brak artefaktów modelu"** — najpierw `python main.py train`.
- **„metadata zawiera inną listę cech"** — artefakty z innej wersji kodu; uruchom `train` ponownie.
- **Kraki w terminalu Windows** — program wymusza UTF-8; ewentualnie `chcp 65001` przed uruchomieniem.
- Predykcja „nie zna" zawodnika — dane obejmują UFC 2013–2026; program podsunie podobne nazwiska.

## 11. Strona demonstracyjna (web + darmowy hosting Netlify)

Model jest liniowy (33 liczby), więc predykcję można liczyć **w przeglądarce** — bez serwera
i bez PyTorcha. Skrypt eksportowy buduje `web/data.js` (wagi, scaler, mediany i snapshoty
kariery wszystkich zawodników), a `web/index.html` liczy te same 32 cechy co `main.py`
(wyniki zgodne z CLI do 4. miejsca po przecinku).

```bash
python export_web.py        # generuje web/data.js z aktualnego modelu (outputs/)
```

Podgląd lokalnie: otwórz `web/index.html` w przeglądarce (działa też z podwójnego kliknięcia).

**Deploy na Netlify (darmowy):**
1. Wejdź na https://app.netlify.com/drop,
2. przeciągnij folder `web/` na stronę — gotowe (URL typu `https://twoja-nazwa.netlify.app`).

Alternatywnie: wrzuć projekt na GitHub i w Netlify podłącz repo (Publish directory: `web`).
Po każdym retreningu uruchom `python export_web.py` ponownie, aby odświeżyć `data.js`.
Uwaga: strona prognozuje modelem zamrożonym na dane do `meta.date_max` (2026-10-10);
dla zawodników nieaktywnych od dawna prognoza jest obarczona niepewnością.

## 12. Cytowanie

**Projekt:** *WalkoMat v1.1.0 — interpretowalny predyktor wyników walk UFC oparty o
jednowarstwową sieć neuronową.* Projekt studencki, przedmiot ZMTSI 26/27, 2026.

**Dane:** nlbair/ufc-data (licencja CC BY 4.0); źródła: ufcstats.com, BestFightOdds,
MMA Decisions. Projekt edukacyjny, bez wartości komercyjnej.
