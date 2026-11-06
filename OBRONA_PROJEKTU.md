# WalkoMat — kompendium na obronę projektu

**Przedmiot:** Zaawansowane Metody i Techniki Sztucznej Inteligencji 26/27
**Projekt:** WalkoMat (dawniej: UFC Fight Predictor) — interpretowalny predyktor wyników walk UFC oparty o jednowarstwową sieć neuronową
**Pliki projektu:** `main.py` (cała logika), `export_web.py` + `web/` (statyczna strona demo), `requirements.txt`, `README.md`, `ZALEZNOSCI_ZMIENNYCH.md` (analiza wag), `data/ufc_all_fights_wide.csv` (dane), `outputs/` (artefakty)
**Dokument jest aktualizowany po każdym etapie prac; wyniki z placeholdery `{{...}}` wypełniają się realnymi liczbami z uruchomień.**

> DISCLAIMER (powtarzany w całym projekcie): projekt ma charakter wyłącznie edukacyjny,
> nie stanowi porady finansowej ani zachęty do hazardu.

## 0. Ściąga komend (live demo na obronie)

```bash
pip install -r requirements.txt            # zależności
python main.py train                       # pełny cykl: trening+ewaluacja+wykresy+artefakty
python main.py evaluate                    # re-ewaluacja zapisanego modelu (bez retreningu)
python main.py predict --fighter-a "Jon Jones" --fighter-b "Tom Aspinall"
python main.py predict --input sample_fight.json
python main.py demo                        # szybki trening (2500 walk) + przykładowa predykcja
python main.py train --epochs 200 --lr 5e-4 --batch-size 32   # własne hiperparametry
python main.py train --random-split        # awaryjny podział losowy (random_state=42)
python export_web.py                       # odświeża web/data.js (strona demo) po retreningu
```

Artefakty po treningu: `outputs/model.pt`, `scaler.joblib`, `metadata.json`, `metrics.json`,
`report.md`, `plots/*.png` (8 wykresów). Dane: `data/ufc_all_fights_wide.csv` (instrukcja
pobrania w README; loader wykrywa plik automatycznie).

Przykładowy wynik predykcji (Jon Jones vs Tom Aspinall, model po pełnym treningu):
P(A)=0.659; predykcja w pełni rozważalna na wkłady cech (wᵢ·zᵢ). Dla par z mało danych
(strona/CLI pokazują ⚠ ostrzeżenie o wiarygodności): Sergio Moraes vs Keith Berish →
P(A)=0.889 z listą powodów (mało walk, asymetria braku zasięgu, długa przerwa).

**Strona demo (bonus):** `web/index.html` liczy te same predykcje w przeglądarce bez serwera
(wagi + snapshoty w `web/data.js`, generuje `export_web.py`). Testy: zgodność z CLI do 4.
miejsca po przecinku (4 pary walk), obsługa nieznanych nazwisk, determinizm treningu
potwierdzony identycznym hashem wag z dwóch kolejnych uruchomień (torch.set_num_threads(1)).

---

## 1. Problem biznesowy i produkt

**Problem:** analityk zakładów sportowych chce szybko oszacować prawdopodobieństwo zwycięstwa
zawodnika A w walce UFC i porównać swoje oszacowanie z rynkiem bukmacherskim, aby wykrywać
potencjalne value bety (zakłady z dodatnią wartością oczekiwaną).

**Rozwiązanie (MVP):** narzędzie CLI, które trenuje interpretowalny model liniowy na
historycznych danych walk (cechy dostępne wyłącznie PRZED walką), ewaluuje go na
najnowszych walkach (chronologiczny test = symulacja „przyszłości"), porównuje z
benchmarkami (majority class, regresja logistyczna sklearn, kursy bukmacherskie) oraz
symuluje prostą strategię zakładową.

**Dlaczego prosty model to świadoma decyzja produktowa:** niski koszt wdrożenia i utrzymania,
trening w sekundy na CPU, pełna interpretowalność (33 parametry), małe ryzyko przeuczenia
na ograniczonych danych (5726 walk), łatwe wyjaśnienie decyzji klientowi/użytkownikowi.

## 2. Dane

### 2.1. Źródło i zakres

- Repozytorium: **nlbair/ufc-data** (dane CC BY 4.0, zbierane z ufcstats.com i BestFightOdds).
- Plik: `data/ufc_all_fights_wide.csv` — 6743 walki, 496 kolumn, okres 2013-02 → 2026-10.
- 99.4% walk ma kursy bukmacherskie (otwarcie i zamknięcie, z prawdopodobieństwami bez marży).
- Loader w `main.py` wykrywa plik automatycznie (auto-detekcja po kolumnach); alternatywnie
  obsługuje starsze zbiory Kaggle (schemat R/B) oraz opcjonalny pakiet `ufcdata-py`.

### 2.2. Słownik zmiennych surowych (wykorzystywane kolumny źródłowe)

| Nazwa w zbiorze | Polska nazwa | Znaczenie / zakres |
|---|---|---|
| `fight_id` | identyfikator walki | 16-znakowy klucz z ufcstats.com; unikalny |
| `event_date` | data wydarzenia | YYYY-MM-DD; baza podziału chronologicznego |
| `bout_division` | kategoria wagowa | 12 wartości (flyweight…heavyweight, dywizje kobiet) |
| `is_title_bout` | walka o pas | true/false; mistrzowskie + tymczasowe |
| `finish_type` | sposób zakończenia | decision / ko_tko / submission / dq / overturned / … |
| `fight_duration_sec` | czas trwania walki | sekundy; podstawa średniej/mediany czasu walk |
| `winner` | zwycięzca | 1 lub 2; 0 = remis; puste = no contest |
| `f1_name`, `f2_name` | nazwiska zawodników | kolejność wg ufcstats (BIASED: zwycięzca częściej f1!) |
| `f1_age`, `f2_age` | wiek w dniu walki | lata; atrybut PRZED walką |
| `f1_height_cm`, `f2_height_cm` | wzrost | cm; atrybut stały zawodnika |
| `f1_reach_cm`, `f2_reach_cm` | zasięg ramion | cm; atrybut stały zawodnika |
| `f1_stance`, `f2_stance` | postawa | orthodox / southpaw / switch |
| `f1_rank`, `f2_rank` | ranking UFC | 0 = mistrz, 1–15 = pozycja; brak = nierankingowy (z ostatniego cotygodniowego rankingu przed galą) |
| `f1_is_ranked`, `f2_is_ranked` | czy rankingowy | true/false |
| `f1_is_champion`, `f2_is_champion` | czy mistrz | true/false |
| `f1_open`, `f2_open` | kurs otwarcia | kurs amerykański (np. −150, +200) |
| `f1_close_low/high`, `f2_close_low/high` | kurs zamknięcia | zakres linii zamknięcia w księgarniach |
| `f1_close_prob_novig`, `f2_close_prob_novig` | prawd. zamknięcia bez marży | prawdopodobieństwo rynkowe po usunięciu marży bukmachera |
| `has_odds` | czy są kursy | true/false (99.4% walk) |

**Celowo NIEWYKORZYSTYWANE grupy kolumn** (data leakage): statystyki bieżącej walki
(`sig_str_landed_diff`, `ctrl_sec_diff`, `kd_diff`, grupa „performance: by round" — 220 kolumn),
sędziowskie (`judges_for_f1`), wynikowe (`result_f1/f2`, `is_split`).

### 2.3. Format wewnętrzny (tidy) po normalizacji

Adapter `normalize_schema()` sprowadza dane do perspektywy **A vs B**:

| Nazwa wewnętrzna | Polska nazwa | Uwagi |
|---|---|---|
| `fight_id` | identyfikator walki | źródłowy lub złożony „data\|A\|B" |
| `date` | data walki | parsowana, walidowana |
| `fighter_a`, `fighter_b` | zawodnik A / B | **przypisanie losowane monetą crc32(fight_id)** |
| `target` | target | 1 = wygrywa A, 0 = wygrywa B |
| `finish_type` | sposób zakończenia | do cech historycznych (metoda zwycięstw) |
| `duration_sec` | czas walki | do cech historycznych |
| `weight_class_lbs` | limit wagi kategorii | funty (flyweight=125 … heavyweight=265) |
| `is_womens` | dywizja kobiet | 1/0 |
| `is_title_bout` | walka o pas | 1/0 |
| `a_age`…`a_prob_fair` | atrybuty i kursy A | wiek, wzrost, zasięg, postawa (kod), ranking, czy rankingowy, kurs, prawd. rynkowe |
| `b_age`…`b_prob_fair` | atrybuty i kursy B | jw. |

**Neutralizacja biasu kolejności:** w źródle f1 wygrywa 56.6% walk (ufcstats listuje zwycięzcę
jako pierwszego). Deterministyczna „moneta" (crc32 z fight_id) losuje strony A/B — po niej
udział zwycięstw A wynosi **49.9%**. Bez tego model nauczyłby się „obstawiaj A" jako artefaktu.

### 2.4. Czyszczenie

- Remisy (winner=0) i no contest/overturned (winner puste): **usunięte** (125 walk) —
  model jest klasyfikatorem binarnym (decyzja dydaktyczna opisana w raporcie).
- Wiersze bez daty lub nazwisk: usunięte (walidacja `validate_tidy` przerywa działanie
  z czytelnym błędem, gdy danych jest za mało lub target nie jest binarny).

## 3. Anti-leakage (najważniejsza zasada projektu)

Model może używać **wyłącznie informacji dostępnych przed walką**. Mechanizmy ochrony:

1. **Snapshot kariery przed walką** (`career_snapshot`): statystyki liczone z walk o dacie
   WCZEŚNIEJSZEJ niż bieżąca (`h.date < as_of`).
2. **Kolejność operacji** (`build_fight_dataset`): snapshot → zbudowanie cech → **dopiero
   potem** dopisanie walki do historii obu zawodników. Matematycznie wyklucza przeciek.
3. **Statystyki bieżącej walki wykluczone** (są w CSV — nie są czytane).
4. **Kursy wykluczone z cech** — używane tylko do benchmarku i analizy zakładów
   (motywacja: niezależność od rynku + niezasłanianie sygnału modelu; kursy przewidują
   wynik niemal idealnie i zdominowałyby wagi).
5. **Skalowanie tylko na train** (`fit_preprocessors` na zbiorze treningowym).
6. **Podział chronologiczny** — test = walki najnowsze (nigdy nie widziane w train).
7. **Neutralizacja kolejności f1/f2** (sekcja 2.3).

## 4. Słownik cech modelu (FEATURE_COLUMNS — 32 cechy)

Cechy różnicowe (A − B) — bo model liniowy porównuje zawodników bezpośrednio. Historia =
wyłącznie walki wcześniejsze. „Polska nazwa" = nazwa używana na obronie.

### Grupa I — doświadczenie i aktywność (z historii)

| Nazwa cechy | Polska nazwa | Definicja |
|---|---|---|
| `exp_diff` | różnica doświadczenia | liczba walk w datasecie: A − B |
| `wins_diff` | różnica zwycięstw | liczba zwycięstw w karierze (z datasetu) |
| `losses_diff` | różnica porażek | liczba porażek |
| `draws_diff` | różnica remisów | liczba remisów |
| `fights_12m_diff` | różnica aktywności 12 m | liczba walk w ostatnich 365 dniach |
| `fights_24m_diff` | różnica aktywności 24 m | liczba walk w ostatnich 730 dniach |
| `days_rest_diff` | różnica dni odpoczynku | dni od ostatniej walki |
| `title_exp_diff` | różnica doświadczenia tytułowego | liczba walk o pas w przeszłości |
| `debut_diff` | różnica statusu debiutanta | 1 = zawodnik bez historii w datasecie |
| `both_debut` | obaj debiutanci | 1 gdy obaj bez historii |

### Grupa II — jakość i forma (z historii)

| Nazwa cechy | Polska nazwa | Definicja |
|---|---|---|
| `win_pct_diff` | różnica procenta zwycięstw | wins/n |
| `ko_pct_diff` | różnica % zwycięstw przez KO/TKO | KO/n |
| `sub_pct_diff` | różnica % zwycięstw przez poddanie | SUB/n |
| `dec_pct_diff` | różnica % zwycięstw przez decyzję | DEC/n |
| `finish_rate_diff` | różnica finish rate | (KO+SUB)/n — udział walk zakończonych przed decyzją |
| `form3_diff` | różnica formy (3 walki) | win rate z ostatnich (max) 3 walk |
| `form5_diff` | różnica formy (5 walk) | win rate z ostatnich (max) 5 walk |
| `streak_diff` | różnica serii | +k = k zwycięstw z rzędu; −k = k porażek |
| `avg_fight_time_diff` | różnica średniego czasu walki | średnia duration_sec z historii |
| `med_fight_time_diff` | różnica mediany czasu walki | mediana duration_sec |

### Grupa III — atrybuty fizyczne i status (sprzed walki)

| Nazwa cechy | Polska nazwa | Definicja |
|---|---|---|
| `age_diff` | różnica wieku | wiek A − B w dniu walki (lata) |
| `height_diff` | różnica wzrostu | cm |
| `reach_diff` | różnica zasięgu | cm |
| `stance_diff` | różnica postawy | kod: orthodox=0, southpaw=1, switch=2, … |
| `ranked_diff` | różnica statusu rankingowego | 1/0 − 1/0 |
| `rank_diff` | różnica pozycji rankingowej | 0=mistrz … 15; nierankingowy=20 (UNRANKED_CODE) |

### Grupa IV — kontekst walki i braki danych

| Nazwa cechy | Polska nazwa | Definicja |
|---|---|---|
| `weight_class_lbs` | limit wagi kategorii | funty (kontekst, nie różnica) |
| `is_womens_division` | dywizja kobiet | 1/0 |
| `is_title_bout` | walka o pas | 1/0 |
| `reach_missing_diff` | różnica braku zasięgu | wskaźnik 1/0: A bez danych − B bez danych |
| `height_missing_diff` | różnica braku wzrostu | jw. |
| `age_missing_diff` | różnica braku wieku | jw. |

**Braki danych:** cechy ułamkowe debiutanta = NaN → imputacja medianą z train + cechy
flagowe (`debut_diff`); braki atrybutów → mediana + wskaźniki missing (brak może być
informacją o statusie zawodnika — patrz `ZALEZNOSCI_ZMIENNYCH.md` §4.1).

## 5. Podział danych

- **Chronologiczny 70/15/15** (domyślny): najstarsze 70% → train, kolejne 15% → validation,
  najnowsze 15% → test. Uzasadnienie: prognozujemy przyszłe walki; test symuluje
  rzeczywiste użycie produktu („jutrzejsze gale"). Losowy podział zawyżałby metryki
  (walki z tej samej ery i ci sami zawodnicy w train i test) — dostępny jako awaryjny
  `--random-split` z random_state=42.
- **Burn-in 2 lata** (`--burn-in-years`): pomijamy walki 2013–2015 w zbiorze modelowym,
  bo historia zawodników liczona jest tylko z datasetu (dla walk z 2013 r. byłaby pusta
  sztucznie).
- Realne wymiary: train 4008 (2015-02→2023-06) / val 859 (2023-06→2025-03) / test 859
  (2025-03→2026-10); szczegóły w sekcji 12.1.

## 6. Model

### 6.1. Architektura (wymóg przedmiotu)

```python
class SingleLayerUFCClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)   # JEDYNA warstwa — bez ukrytych
    def forward(self, x):
        return self.linear(x)                     # logit
```

- Wejście: wektor 32 ustandaryzowanych cech (warstwa wejściowa).
- Wyjście: 1 neuron liniowy → logit (warstwa wyjściowa).
- Prawdopodobieństwo: P(A wygrywa) = sigmoid(logit).
- **Brak** warstw ukrytych, dropoutu, batch normy — świadomie.
- Liczba parametrów: **33** (32 wagi + 1 bias).

### 6.2. Związek z regresją logistyczną (klasyczne pytanie na obronie)

Jednowarstwowa sieć z sigmoidą na wyjściu i entropią krzyżową to **matematycznie ta sama
rodzina modeli co regresja logistyczna** — ten sam zbiór funkcji decyzyjnych
(logit = w·x + b) i ten sam estymator (MLE z entropią krzyżową). Różnice są
implementacyjne: (a) implementacja w PyTorch jako sieć neuronowa z własną pętlą
treningową (wymóg przedmiotu), (b) optymalizacja Adam z mini-batchami zamiast pełnego
zbioru/LBFGS, (c) early stopping jako regularyzacja iteracyjna. W projekcie regresja
logistyczna sklearn jest dodatkowo baseline'em — porównanie pokazuje, że obie drogi
zbiegają do podobnej jakości (patrz sekcja wyników).

### 6.3. Funkcja straty

**BCEWithLogitsLoss** = binarna entropia krzyżowa z wbudowaną, numerycznie stabilną sigmoidą:
L(y, logit) = max(logit, 0) − logit·y + log(1 + e^(−|logit|)). To odpowiednik entropii
krzyżowej z wykładu dla klasyfikacji binarnej. Dlaczego nie osobny sigmoid + BCE:
oddzielne operacje są podatne na niedomiar/nadmiar zmiennoprzecinkowe przy skrajnych logitach.

### 6.4. Winsoryzacja ±3σ i regularyzacja L2 (stabilność predykcji)

- **Winsoryzacja** (`transform_features`): po standaryzacji z przycinane do ±3 — SPÓJNIE
  w treningu, ewaluacji, predict i na stronie. Jedna rzadka cecha (duża waga, mała σ) nie
  może zdominować logitu, a model nie ekstrapoluje poza rozkład treningowy (np. dni
  odpoczynku 7 lat → z=−21 przycinane do −3).
- **L2 (weight_decay=0.01 w Adamie)** — kara za duże wagi; bez niej optymalizator kompensuje
  przycięte wejścia pompowaniem wagi cechy-proxy, a martwe cechy zostają z losowym szumem.
  Z L2: martwe cechy mają wagi dokładnie 0.000, waga proxy spadła 0.994 → 0.700.
- **Kontekst (lekcja projektowa):** para dwóch zawodników z 1 walką w danych dawała 97.7%
  pewności z samego braku zasięgu u jednej strony (wkład ±3.34 logitu > cały logit). Po
  winsoryzacji + L2: 91.3%/88.9% + **ostrzeżenie o wiarygodności** (CLI i strona), a metryki
  testowe wzrosły (acc 0.629 → 0.637, AUC 0.683 → 0.686). Szczegóły: `ZALEZNOSCI_ZMIENNYCH.md` §7.

## 7. Trening

| Parametr | Wartość | Uzasadnienie |
|---|---|---|
| Optymalizator | Adam | adaptacyjne kroki, szybka zbieżność na małym modelu |
| Learning rate | 1e-3 | standardowy punkt startowy Adama |
| Weight decay (L2) | 0.01 | kara za duże wagi; zeruje martwe cechy, stabilizuje proxy (sekcja 6.4) |
| Batch size | 64 | mini-batch gradient descent: kompromis szum/efektywność |
| Epoki (max) | 100 | z early stopping zwykle kończy się ~40–60 |
| Early stopping | na val loss, patience 10 | regularyzacja; przywracane wagi najlepszej epoki |
| Winsoryzacja cech | clip z do ±3σ | ogranicza wpływ pojedynczej cechy do 3·\|wᵢ\| (sekcja 6.4) |
| Seed | 42 wszędzie + `torch.set_num_threads(1)` | pełny determinizm bitowy (identyczny hash wag) |
| Strata | BCEWithLogitsLoss | sekcja 6.3 |

**Mini-batch learning** (z wykładu): każdy epoch dzieli train na batche po 64 walki;
wagi aktualizowane po każdym batchu (forward → loss → backward → step). To wariant między
pełnym gradient descent (1 update/epoch) a czystym SGD (1 update/próbka).

**Backpropagation:** w PyTorch realizuje go autograd (`loss.backward()`) — mechanizm
reguły łańcuchowej propagujący gradienty ∂L/∂wᵢ i ∂L/∂b przez wykres obliczeń. Dla naszego
modelu gradienty mają formę analityczną ∂L/∂w = (sigmoid(logit) − y)·x — autograd daje
dokładnie to samo.

## 8. Powiązanie z pojęciami z wykładów (mapowanie)

| Pojęcie z wykładu | Gdzie w projekcie |
|---|---|
| Perceptron / sieć jednowarstwowa | `SingleLayerUFCClassifier` — nn.Linear(32, 1) |
| Warstwa wejściowa | 32 ustandaryzowane cechy różnicowe/kontekstowe |
| Warstwa wyjściowa | 1 neuron → logit → sigmoid |
| Brak warstw ukrytych | architektura celowo minimalna (wymóg) |
| Regresja logistyczna jako sieć jednowarstwowa | sekcja 6.2 + baseline sklearn (sekcja 12) |
| Entropia krzyżowa | `BCEWithLogitsLoss` |
| Gradient descent | `optimizer.step()` po gradientach z `loss.backward()` |
| SGD / mini-batch | `DataLoader(batch_size=64, shuffle=True)` |
| Backpropagation | autograd w pętli treningowej (sekcja 7) |
| Generalizacja | podział chronologiczny + early stopping + val loss |
| Train/validation/test | 70/15/15 chronologicznie (sekcja 5) |
| Bias-variance trade-off | 33 parametry; luka train–val ≈ 0 → niska wariancja (sekcja 15) |

## 9. Metryki — słownik (co mierzymy i po co)

| Metryka | Definicja | Dlaczego ważna dla tego produktu |
|---|---|---|
| Accuracy | udział trafień | podstawowa czytelność dla klienta |
| Balanced accuracy | średnia czułość klas | odporna na niezbalansowane klasy |
| Precision (klasa 1) | TP/(TP+FP) | jak często „typ A" jest trafny |
| Recall (klasa 1) | TP/(TP+FN) | jaki udział wygranych A wykrywamy |
| F1 | średnia harmoniczna P i R | balans precision/recall |
| ROC-AUC | pole pod krzywą ROC | jakość rangowania walk niezależnie od progu |
| PR-AUC | pole pod krzywą precision-recall | wrażliwsza na klasę pozytywną |
| Log loss | −mean(y·log p + (1−y)·log(1−p)) | kara za pewne błędne prognozy — kluczowa dla zakładów |
| Brier score | mean((p−y)²) | dokładność probabilistyczna (0 = idealnie) |
| ECE | Σ |acc_b − conf_b|·n_b/N (10 kubełków) | kalibracja: czy „70%" znaczy 70% |
| Macierz pomyłek | TN/FP/FN/TP | pełny obraz błędów klasyfikacji |

Dla produktu zakładowego **metryki probabilistyczne (log loss, Brier, ECE) są ważniejsze
niż accuracy** — obstawiamy kwoty na podstawie prawdopodobieństw, więc liczy się ich jakość.

## 10. Baseline'y (punkty odniesienia)

1. **Majority class** — zawsze typuj częstszą klasę z train. Dolna granica sensowności.
2. **Regresja logistyczna sklearn** — ta sama architektura matematyczna, inna implementacja;
   kontrola, że własna pętla PyTorch osiąga to samo.
3. **Bukmacher (kursy zamknięcia bez marży)** — rynek jako twardy benchmark: jak blisko
   rynku potrafi być model z 33 parametrami zbudowany z publicznych danych?

## 11. Analiza bukmacherska (metodologia)

- **Prawdopodobieństwo implikowane z kursu:** kurs amerykański → dziesiętny
  (−150 → 1.667; +200 → 3.0) → p_impl = 1/kurs_dziesiętny (zawiera marżę bukmachera).
- **Value bet (strategia):** obstawiamy stronę (A lub B) o większym modelowym
  prawdopodobieństwie p, gdy: (a) p ≥ próg (0.55 / 0.60 / 0.65) ORAZ (b) EV = p · kurs > 1
  (dodatnia wartość oczekiwana względem realnie oferowanego kursu). Stawka 1 jednostka.
- **Mierzymy:** liczbę zakładów, win rate, średni kurs, ROI = zysk/liczbę zakładów,
  equity curve (skumulowany zysk), maksymalne obsunięcie kapitału (max drawdown).
- **Benchmark:** strategia „obstawiaj faworyta rynkowego" (bez modelu) — pokazuje rolę marży.
- Gdy kursów brak w danych: analiza elegancko pominięta, oznaczona jako niedostępna.
- **DISCLAIMER:** symulacja na danych historycznych; przeszłe ROI nie gwarantują przyszłych
  wyników; projekt nie jest poradą finansową.

## 12. Wyniki (wypełniane realnymi liczbami po każdym etapie)

*Stan: PROJEKT UKOŃCZONY (Etap 8/8 + web + winsoryzacja/L2). Seed 42, najlepsza epoka 39
(val loss 0.6491). Wszystkie komendy zweryfikowane end-to-end: `train` (exit 0, pełne
artefakty), `evaluate` (odtwarza metryki testowe identycznie), `predict`
(`--fighter-a/--fighter-b` i `--input sample_fight.json`; fuzzy matching; błąd nieznanego
zawodnika → czytelny komunikat, exit 2; ⚠ ostrzeżenie o wiarygodności dla par z mało danych),
`demo`, strona `web/` (zgodność z CLI do 4. miejsca po przecinku na 5 parach testowych).
Determinizm: identyczny hash wag z dwóch uruchomień. Dokumentacja: README.md, requirements.txt,
report.md (auto), ZALEZNOSCI_ZMIENNYCH.md.*

### 12.1. Podział danych (chronologiczny)

| Zbiór | Walki | Zakres dat | Udział klasy 1 |
|---|---:|---|---:|
| train | 4008 | 2015-02-22 → 2023-06-24 | 0.502 |
| validation | 859 | 2023-06-24 → 2025-03-08 | 0.501 |
| test | 859 | 2025-03-08 → 2026-10-10 | 0.495 |

(Po normalizacji: 6618 walk; po burn-in 2 lata: 5726; cech: 32; parametrów modelu: 33.)

### 12.2. Metryki — model vs baseline'y (zbiór testowy)

| Metryka | **Model (PyTorch)** | Majority class | Regr. log. sklearn | Bukmacher (kursy) |
|---|---:|---:|---:|---:|
| Accuracy | **0.6368** | 0.4948 | 0.6298 | 0.7052 |
| ROC-AUC | **0.6865** | — | 0.6767 | 0.7647 |
| Log loss | **0.6434** | 0.6932 | 0.6458 | 0.5799 |
| Brier | **0.2260** | — | 0.2273 | 0.1977 |
| PR-AUC | **0.6602** | — | — | — |
| Balanced acc. | **0.6365** | — | — | — |

Metryki modelu na wszystkich zbiorach (train/val/test): acc 0.6118 / 0.6123 / **0.6368**;
ROC-AUC 0.6657 / 0.6714 / **0.6865**; log loss 0.6434 / 0.6491 / **0.6434**; ECE 0.012 / 0.031 / **0.042**.

Macierz pomyłek (test, próg 0.5): TN=287, FP=147, FN=165, TP=260.

**Czytanie wyników (na obronę):**
- Model wyraźnie bije majority class (+14.2 pp acc) i **równa/lekko bije regresję
  logistyczną sklearn** (AUC 0.686 vs 0.677) — dwie implementacje tego samego modelu
  zbiegają do podobnej jakości (kontrola poprawności własnej pętli treningowej).
- Luka train–test jest ujemna (test lepszy od train) — brak przeuczenia; niska wariancja.
- **Bukmacher jest lepszy od modelu we wszystkim** (AUC 0.765, log loss 0.580): rynek
  agreguje informacje, których model nie ma (statystyki, kontuzje, kampy, pieniądze
  analityków). Uczciwa konstatacja: 33 parametry z publicznych danych nie pobiją rynku,
  ale osiągają ~70% jego przewagi nad przypadkiem ((0.686−0.5)/(0.765−0.5) ≈ 0.71).
- Kalibracja dobra (ECE 0.042 na teście) — prawdopodobieństwa modelu nadają się do
  porównań z rynkiem.
- Winsoryzacja ±3σ + L2 **poprawiły** metryki względem wersji bez nich (acc 0.629 → 0.637):
  przycięcie ekstremalnych cech usunęło szum także z treningu (sekcja 6.4).

### 12.3. Analiza bukmacherska (symulacja na teście, 842/859 walk z kursami)

| Próg | Zakłady | Win rate | ROI (surowy) | ROI odporny (bez top-10) | Mediana zysku | Max DD |
|---|---:|---:|---:|---:|---:|---:|
| 0.55 | 179 | 0.497 | **+1.225** | **−0.036** | −1.000 | 15.9 |
| 0.60 | 103 | 0.573 | +0.783 | **−0.056** | +0.476 | 5.7 |
| 0.65 | 51 | 0.608 | +0.594 | **−0.219** | +0.460 | 6.7 |
| benchmark: faworyci | 842 | 0.704 | +0.003 | −0.029 | — | 19.3 |

**Kluczowy wniosek (statystyki odporne):** surowy ROI +122% przy progu 0.55 jest
**efektem pojedynczych wygranych longshotów** (najwyższy kurs w teście: 201.0). Mediana
zysku na zakład przy progu 0.55 jest UJEMNA (−1.00), a ROI po usunięciu 10 najbardziej
fortunnych zakładów jest ujemne przy WSZYSTKICH progach (−0.04 / −0.06 / −0.22). Wniosek
uczciwy i ważny: **strategia value bet z tym modelem NIE jest statystycznie powtarzalna**
— symulacja pokazuje mechanikę value betów i rolę wariancji, a nie działający „printing
machine". To świadoma lekcja na obronę (statystyki odporne, ryzyko zmienności), spójna
z disclaimerem edukacyjnym. Benchmark faworytów (+0.3% surowe ROI, ale −2.9% odporne
na 842 zakłady) pokazuje, że sama marża bukmachera zjada cały zysk z typowania
faworytów — bez realnej przewagi nie ma ROI.

### 12.4. Najważniejsze wagi modelu (top-6 po |w|)

| Cecha | Waga | Kierunek wpływu |
|---|---:|---|
| `reach_missing_diff` | −0.700 | brak danych o zasięgu A → niższe P(A) (proxy statusu; wpływ ograniczony winsoryzacją do 2.10 logitu) |
| `age_diff` | −0.348 | starszy A → niższe P(A); najsilniejsza cecha fizyczna |
| `rank_diff` | −0.146 | gorszy ranking A → niższe P(A) |
| `wins_diff` / `losses_diff` | +0.142 / −0.129 | lepszy bilans karierowy |
| `finish_rate_diff` | +0.134 | finisher ma przewagę nad punktowcem |
| `streak_diff` / `form5_diff` | +0.124 / +0.114 | bieżąca forma i seria zwycięstw |

Wagi trzech „martwych" cech (zerowa wariancja w train) po L2 = dokładnie 0.000.

Pełna analiza 32 cech, korelacji i kolinearności: `ZALEZNOSCI_ZMIENNYCH.md`.

### 12.5. Porównanie wariantów uczenia (komenda `compare`)

Świadomy wybór optymalizatora i funkcji straty poparty **kontrolowanym eksperymentem**:
ten sam jednowarstwowy model, te same dane i skalowanie, **identyczna inicjalizacja
i kolejność batchy** (seed 42) — różnią się wyłącznie optymalizator / strata / tryb batchów
(6 wariantów, max 100 epok, early stopping patience 12). Metryka porównawcza: **ujednolicony
log loss walidacyjny** — ta sama skala niezależnie od funkcji straty wariantu.

| Wariant | konwergencja (LL≤0.670) | val log loss | test AUC | test log loss |
|---|---:|---:|---:|---:|
| SGD lr=1e-3 | ep. 23 | 0.6533 | 0.6791 | 0.6494 |
| SGD+Momentum 0.9, lr=1e-3 | ep. 3 | 0.6500 | 0.6865 | 0.6439 |
| RMSprop lr=1e-3 | ep. 1 | 0.6493 | 0.6860 | 0.6435 |
| **Adam lr=1e-3 (WYBRANY)** | ep. 2 | **0.6491** | **0.6865** | **0.6434** |
| Adam + strata MSE | ep. 2 | 0.6496 | 0.6857 | 0.6451 |
| Adam, full-batch GD | ep. 54 | 0.6603 | 0.6553 | 0.6599 |

Artefakty: `outputs/compare.md` (tabela + wnioski), `compare.json`, wykresy
`plots/compare_loss_curves.png` (krzywe zbieżności) i `plots/compare_auc_bars.png`
(słupki z liniami referencyjnymi: przypadek 0.500, bukmacher 0.765).

**Wnioski (na obronę):**
- **Adam**: najniższy walidacyjny log loss (0.6491) i najlepszy testowy (0.6434);
  adaptacyjne, per-parametrowe kroki (momenta 1. i 2. rzędu) zbiegają najszybciej.
- **SGD ze stałym krokiem lr=1e-3** jest zbyt wolne dla skali naszych cech: próg 0.670
  przekracza dopiero w epoce 23 (adaptacyjne: 1–3) i wyczerpuje budżet 100 epok.
- **Full-batch GD** (1 aktualizacja wag na epokę vs ~63 przy mini-batch 64) po 100 epokach
  osiąga tylko 0.6603 i AUC 0.655 — ilustracja, dlaczego mini-batch learning jest standardem.
- **MSE zamiast entropii krzyżowej**: porównanie w ujednoliconej skali pokazuje wyższy
  test log loss (0.6451 vs 0.6434) i niższe AUC (0.6857 vs 0.6865) — pochodna MSE przez
  sigmoid gaśnie przy skrajnych logitach (nasycenie), więc błędne predykcje korygują się
  wolniej. Przy Adamie różnica jest umiarkowana (adaptacyjne kroki częściowo ją łatają),
  ale BCEWithLogitsLoss jest też numerycznie stabilna — dlatego ją wybieramy.
- Żaden wariant nie pobija bukmachera (AUC 0.765) — spójne z sekcją 12.2.

## 13. Interpretacja i ograniczenia

**Interpretacja** (szczegóły w `ZALEZNOSCI_ZMIENNYCH.md`): model najmocniej reaguje na wiek
(młodszy zawodnik wygrywa częściej), status rankingowy, zdolność finiszowania i bieżącą formę;
zasięg ramion działa tam, gdzie wzrost jest neutralny; aktywność/odpoczynek ≈ neutralne.
Wagi czytamy GRUPAMI (kolinearność) — pojedyncza waga to udział w konsensusie grupy.

**Ograniczenia i ryzyka:**
1. Model liniowy — nie modeluje interakcji (np. zasięg × styl walki); warstwa ukryta mogłaby,
   ale kosztem interpretowalności (i wbrew wymogowi przedmiotu).
2. Cechy historyczne liczne tylko z datasetu (od 2013) — kariery wcześniejsze są obcięte
   (mitygowane burn-in i imputacją debiutanta).
3. Kolejność walk tego samego dnia — remisowy problem sortowania; wpływ pomijalny.
4. Rynek zmienia się w czasie; model testowany na walkach do 2026 — w produkcji wymagał
   retrainingu (CLI już to umożliwia: `python main.py train`).
5. Symulacje zakładowe nie uwzględniają limitów stawek, zmienności kursów w czasie i
   reakcji rynku — to uproszczenie edukacyjne.

## 14. Możliwy rozwój produktu

- Dodanie kursów jako cechy (wariant „rynkowy") i pomiar, ile sygnału dodaje model nad rynek.
- Cechy z rund (statystyki z ostatnich walk), styl walki (striker/grappler z danych).
- Regularyzacja L1/L2 (selectivność cech), kalibracja izotoniczna/Platta.
- Interfejs dla analityka: raport PDF per gala; monitoring driftu wyników vs rynek.

## 15. Pytania na obronę (Q&A)

**Q: Dlaczego jednowarstwowa sieć, a nie głęboka sieć?**
A: Trzy powody: (1) wymóg przedmiotu — zakres materiału obejmuje sieci bez warstw ukrytych;
(2) rozmiar danych (5726 walk, 33 parametry) — głęboka sieć przeuczyłaby; (3) produkt
bukmacherski wymaga wyjaśnialności: decyzja to suma 32 iloczynów waga×cecha, którą mogę
pokazać klientowi. Bias-variance: prosty model ma niską wariancję, a metryki pokazują, że
sygnał w danych jest głównie liniowy (accuracy ≈ baseline z regresją logistyczną).

**Q: Czym to się różni od zwykłej regresji logistycznej?**
A: Matematycznie niczym (ta sama rodzina modeli — sekcja 6.2); implementacyjnie: własna
pętla treningowa w PyTorch (Adam, mini-batch, early stopping), zgodnie z zakresem przedmiotu.
Baseline sklearn potwierdza równoważność wyników.

**Q: Jak działa backpropagation w Państwa modelu?**
A: `loss.backward()` liczy gradienty ∂L/∂wᵢ regułą łańcuchową przez wykres obliczeń. Dla
warstwy liniowej z BCE gradient ma postać analityczną (sigmoid(logit) − y)·xᵢ — autograd
zwraca dokładnie te wartości; `optimizer.step()` (Adam) aktualizuje wagi o −lr·znormalizowany
gradient (z momentami pierwszego i drugiego rzędu).

**Q: Czym różni się SGD od mini-batch w Państwa implementacji?**
A: Czysty SGD aktualizuje wagi po każdej pojedynczej próbce (szum duży, ale częste updates);
pełny batch gradient descent raz na epokę. Nasz wariant: mini-batch = 64 walki per update —
kompromis między szumem gradientu a kosztami obliczeń; shuffle batchy jest deterministyczny
(generator z seedem 42).

**Q: Skąd wiadomo, że nie ma data leakage?**
A: Sześć warstw ochrony (sekcja 3): snapshot kariery tylko z przeszłości, kolejność
operacji snapshot→cechy→update historii, wykluczenie statystyk bieżącej walki, kursów i
kolejności f1/f2, skalowanie tylko na train, podział chronologiczny. Dodatkowo sanity check:
udział zwycięstw A po losowaniu stron = 49.9% (bez losowania: 56.6% — wyciek).

**Q: Dlaczego podział chronologiczny, a nie losowy?**
A: Produkt prognozuje PRZYSZŁE walki. Losowy podział dopuszcza walki nowsze w train niż
w test (nieosiągalne w realnym użyciu) i duplikuje „erę" MMA między zbiorami — metryki
byłyby zawyżone. Chronologia symuluje wdrożenie: trenujemy na przeszłości, testujemy
na przyszłości.

**Q: Dlaczego StandardScaler tylko na train?**
A: Dopasowanie średniej/odchylenia na całości wyciekałoby statystyki val/test do treningu
(subtelna forma leakage). W produkcji scaler z train jest artefaktem (scaler.joblib) —
nowe walki transformujemy „tym samym miernikiem".

**Q: Co jest ważniejsze: accuracy czy log loss?**
A: Dla zakładów log loss/Brier/kalibracja, bo gra probabilistyczna (kwoty na
prawdopodobieństwach). Accuracy traktujemy jak komunikację biznesową („6 na 10 trafień").

**Q: Co to kalibracja i ECE?**
A: Kalibracja = zgodność przewidywanych prawdopodobieństw z częstością rzeczywistą
(wśród walk z prognozą „70%" powinno wygrywać A ~70% czasu). ECE sumuje ważone różnice
w 10 kubełkach prawdopodobieństw. Model źle skalibrowany daje błędne EV zakładów nawet
przy dobrym rangowaniu.

**Q: Czym jest value bet w Państwa analizie?**
A: Zakład, dla którego modelowe prawdopodobieństwo p razy kurs bukmachera daje EV > 1
(przy progu pewności 0.55/0.60/0.65). ROI symulacji mierzy, czy przewaga modelu nad
rynkiem pokrywa marżę bukmachera — wyniki w sekcji 12.3, z zastrzeżeniem edukacyjnym.

**Q: Model ma 33 parametry — to za mało na „zaawansowane metody AI"?**
A: Zaawansowanie tego projektu jest w metodyce, nie w architekturze: anti-leakage
feature engineering na danych prawdziwego rynku, chronologiczna walidacja, benchmark
rynkowy, analiza kalibracji i ROI — to standard praktyki ML produktowej, częsty brak
w projektach „przyczepiających głębokie sieci do małych danych".

**Q: Co to winsoryzacja ±3σ i po co jest?**
A: Po standaryzacji przycinamy wartości cech do przedziału [−3, +3]. Bez tego pojedyncza
rzadka cecha (duża waga, mała σ — np. wskaźnik braku zasięgu) potrafi dać wkład ±3.3 logitu
i „sama" generować 96% pewności, a cechy typu dni-odpoczyynku ekstrapolują (z=−21 przy
7-letniej przerwie). Cap ogranicza wpływ jednej cechy do 3·|wᵢ| i jest stosowany SPÓJNIE
w treningu, ewaluacji i produkcji. Uzupełnieniem jest L2 (weight_decay), która karze duże
wagi (bez niej optymalizator kompensuje cap pompowaniem wag), oraz ostrzeżenie UX dla
zawodników z mało danych. Efekt: lepsze metryki testowe i brak absurdalnych pewności.

**Q: Dlaczego Adam, a nie SGD albo RMSprop? I czemu entropia krzyżowa, a nie MSE?**
A: Nie musieliśmy zgadywać — zrobiliśmy kontrolowany eksperyment (`python main.py compare`):
6 wariantów uczenia, ten sam model, dane i identyczna inicjalizacja (seed 42). Adam osiągnął
najniższy walidacyjny log loss (0.6491) i najlepszy testowy (0.6434); SGD ze stałym krokiem
potrzebowało 23 epok na próg konwergencji (Adam: 2) i wyczerpało budżet 100 epok; full-batch
GD pokazał w praktyce, dlaczego mini-batch jest standardem (AUC 0.655 vs 0.687). MSE przez
sigmoid dało wyższy test log loss (0.6451) — gradient MSE×sigmoid gaśnie przy skrajnych
logitach — a BCEWithLogitsLoss jest dodatkowo numerycznie stabilna. Pełna tabela i wykresy:
OBRONA §12.5 / `outputs/compare.md`.

**Q: Jakie są największe słabości?**
A: Sekcja 13 — brak interakcji (liniowość), obcięte kariery przed 2013, uproszczenia
symulacji zakładowej, konieczność retrainingu w czasie.

## 16. Słowniczek (EN → PL)

| Termin | Polski odpowiednik |
|---|---|
| logit | logarytm szans (log-odds) wejścia do sigmoidy |
| sigmoid | funkcja logistyczna σ(z) = 1/(1+e^(−z)) |
| cross-entropy loss | entropia krzyżowa (strata) |
| mini-batch | podzbiór próbki uczącej do jednej aktualizacji wag |
| epoch | jedno przejście przez cały zbiór treningowy |
| early stopping | zatrzymanie treningu przy braku poprawy na walidacji |
| overfitting / underfitting | przeuczenie / niedouczenie |
| bias-variance trade-off | kompromis obciążenie–wariancja |
| data leakage | przeciek informacji z przyszłości do cech |
| value bet | zakład z dodatnią wartością oczekiwaną |
| ROI | zwrot z inwestycji (zysk/stawka) |
| equity curve | krzywa skumulowanego zysku |
| max drawdown | maksymalne obsunięcie kapitału od szczytu |
| vig/margin | marża bukmachera |
