# Zależności zmiennych w WalkoMat (predyktor wyników walk UFC)

**Jak zmienne wpływają na siebie i na przewidywany wynik walki.**

Dokument oparty na **rzeczywistych wagach wytrenowanego modelu** (seed 42, najlepsza epoka 39,
val loss 0.6491, val accuracy 0.612, test accuracy 0.637, trening na 4008 walk z lat
2015-02 → 2023-06) oraz na korelacjach Pearsona policzonych na zbiorze treningowym. Model
obejmuje **winsoryzację cech do ±3σ** oraz **regularyzację L2** (weight_decay=0.01) — obie
dodane po zaobserwowaniu eksplozji pewności na parach z mało danych (sekcja 7). Żadna liczba
nie jest przypisana — pochodzi z uruchomienia `main.py` na danych `data/ufc_all_fights_wide.csv`.

---

## 1. Przepływ informacji: od surowych danych do prawdopodobieństwa

```
surowy CSV (496 kolumn, schemat f1/f2)
   │  normalize_schema(): strony A/B losowane monetą crc32(fight_id)  ← ANTI-LEAKAGE
   ▼
tidy: 1 wiersz = 1 walka (target: 1 = A wygrywa, 0 = B wygrywa)
   │  career_snapshot(): statystyki liczone WYŁĄCZNIE z walk WCZEŚNIEJSZYCH niż bieżąca
   ▼
32 cechy (różnice A−B + kontekst)          [build_fight_dataset → make_feature_row]
   │  imputacja medianą z train + StandardScaler z train + WINSORYZACJA ±3σ
   ▼                                          [fit_preprocessors + transform_features]
z = clip((x − μ_train) / σ_train, −3, +3)   (porównywalne, przycięte)
   │  model: logit = w₀ + Σᵢ wᵢ·zᵢ                            [SingleLayerUFCClassifier]
   ▼
P(A wygrywa) = sigmoid(logit) = 1 / (1 + e^(−logit))
```

**Kluczowa zasada**: historia zawodnika jest dopisywana **PO** policzeniu cech jego walki
(`build_fight_dataset`: snapshot → cechy → dopiero potem update historii). Dlatego żadna
cecha nie „widzi" wyniku bieżącej walki ani przyszłości.

## 2. Mechanika wpływu: jak czytać wagi modelu

Model ma **33 parametry**: 32 wagi `wᵢ` + bias `w₀ = 0.021`. Bo cechy są ustandaryzowane:

- **`wᵢ > 0`** → wzrost cechy o 1 odchylenie standardowe (1σ) **zwiększa** szanse zawodnika A;
- **`wᵢ < 0`** → zmniejsza; **`|wᵢ|`** → siła wpływu;
- dzięki winsoryzacji wkład pojedynczej cechy jest ograniczony: **max |wᵢ·zᵢ| = 3·|wᵢ|**.

Przykłady z realnymi wagami (pełna tabela w sekcji 3; punkt wyjścia sigmoid(0.021) = **0.505**):

| Scenariusz (zᵢ = +1) | Zmiana P(A) |
|---|---|
| A starszy o 1σ ≈ **5.3 roku** | 0.505 → 0.419 (**−8.6 pp**) |
| A na serii dłuższej o 1σ ≈ **2.4 walki** | 0.505 → 0.536 (+3.1 pp) |
| A ma wyższy finish rate o 1σ ≈ **+31 pp** | 0.505 → 0.538 (+3.3 pp) |
| A gorzej rankingowy o 1σ ≈ **3.9 pozycji** | 0.505 → 0.469 (−3.7 pp) |
| A dłuższy zasięg o 1σ ≈ **8.0 cm** | 0.505 → 0.523 (+1.8 pp) |

Przykład złożony: A starszy (z=+1), ale w formie (form5 z=+1) i na serii (streak z=+1):
logit = 0.021 − 0.348 + 0.114 + 0.124 = −0.089 → **P(A) = 0.478**. Model ważuje czynniki
przeciw sobie liniowo — dokładnie tak, jak sugeruje architektura jednowarstwowa.

## 3. Pełna tabela wag (posortowana po |w|)

| # | Cecha | Waga wᵢ | σ (1 odch. std.) | korelacja z targetem | Interpretacja |
|---|-------|--------:|-----------------:|---------------------:|---------------|
| 1 | `reach_missing_diff` | **−0.700** | 0.181 | −0.170 | Proxy statusu (brak zasięgu = mało znany); L2 + cap ograniczają wpływ do ≤2.10 logitu |
| 2 | `age_diff` | **−0.348** | 5.32 lata | −0.167 | Wiek — najmocniejsza pojedyncza cecha fizyczna |
| 3 | `rank_diff` | −0.146 | 3.91 poz. | −0.100 | Lepszy ranking (mniejszy numer) → większe szanse |
| 4 | `wins_diff` | +0.142 | 3.07 | +0.069 | Więcej zwycięstw → lepiej |
| 5 | `finish_rate_diff` | +0.134 | 0.312 | +0.067 | Finisher ma przewagę nad „decyzyjnym" punktowcem |
| 6 | `losses_diff` | −0.129 | 2.47 | −0.054 | Więcej porażek w karierze → gorzej |
| 7 | `streak_diff` | +0.124 | 2.38 | +0.105 | Seria zwycięstw pomaga |
| 8 | `avg_fight_time_diff` | +0.120 | 255 s | +0.029 | Dłuższe walki (styl decyzyjny/wytrzymały) |
| 9 | `form5_diff` | +0.114 | 0.298 | +0.117 | Forma z 5 walk pomaga |
| 10 | `debut_diff` | −0.112 | 0.424 | −0.056 | Debiutant przegrywa statystycznie częściej |
| 11 | `form3_diff` | −0.082 | 0.331 | +0.092 | ⚠️ ujemna MIMO dodatniej korelacji — artefakt kolinearności (sekcja 5) |
| 12 | `reach_diff` | +0.073 | 7.98 cm | +0.056 | Zasięg pomaga; **wzrost — nie** (nr 29) |
| 13 | `exp_diff` | +0.059 | 5.04 walki | +0.015 | Doświadczenie liczbowo słabe po uwzględnieniu reszty |
| 14 | `ranked_diff` | +0.053 | 0.344 | +0.094 | Status rankingowy (dubel z rank_diff — korelacja −0.81) |
| 15 | `win_pct_diff` | −0.045 | 0.277 | +0.112 | ⚠️ jw. — informacja „przeniesiona" do form5/streak |
| 16 | `stance_diff` | +0.043 | 0.818 | +0.026 | Postawa (southpaw itd.) niemal bez znaczenia |
| 17 | `is_title_bout` | +0.041 | 0.195 | +0.011 | Walka o pas nie zmienia „losowości" wyniku |
| 18 | `weight_class_lbs` | −0.041 | 38 lbs | −0.017 | Cięższe kategorie ≈ minimalnie mniej przewidywalne |
| 19 | `is_womens_division` | −0.038 | 0.362 | −0.008 | Bez istotnego wpływu |
| 20 | `both_debut` | +0.031 | 0.180 | +0.012 | Obaj debiutanci ≈ losowo |
| 21 | `title_exp_diff` | +0.031 | 1.17 | +0.009 | Doświadczenie tytułowe — słabe |
| 22 | `days_rest_diff` | −0.030 | 210 dni | −0.045 | Dni odpoczynku ≈ neutralne (po winsoryzacji wpływ ≤0.09 logitu) |
| 23 | `dec_pct_diff` | +0.020 | 0.308 | +0.033 | Słaby sygnał |
| 24 | `ko_pct_diff` | −0.011 | 0.271 | +0.037 | Sygnał przeniesiony do finish_rate_diff (sekcja 5.1) |
| 25 | `med_fight_time_diff` | +0.010 | 317 s | +0.014 | Dubel z avg_fight_time_diff (r=0.91) |
| 26 | `sub_pct_diff` | +0.010 | 0.240 | +0.047 | Sygnał przeniesiony do finish_rate_diff |
| 27 | `fights_12m_diff` | −0.009 | 1.16 | +0.063 | Aktywność ≈ neutralna po kontroli reszty |
| 28 | `fights_24m_diff` | +0.004 | 1.97 | +0.065 | Aktywność ≈ neutralna |
| 29 | `height_diff` | +0.003 | 6.27 cm | +0.047 | Wzrost BEZ znaczenia (zasięg pochłania sygnał, r=0.59) |
| 30 | `draws_diff` | **0.000** | — (stała) | — | Martwa cecha — **L2 wyzerowała** szum inicjalizacji |
| 31 | `height_missing_diff` | **0.000** | — (stała) | — | Martwa cecha — wyzerowana przez L2 |
| 32 | `age_missing_diff` | **0.000** | — (stała) | — | Martwa cecha — wyzerowana przez L2 |

„Cechy martwe" (`draws_diff`, `height_missing_diff`, `age_missing_diff`) mają w train
wariancję zero — ich wagi to szum inicjalizacji PyTorch. L2 (weight_decay) ściągnęła je
**dokładnie do zera** — ładna ilustracja działania regularyzacji.

## 4. Grupy czynników — jak ja to widzę

### 4.1 Kto wygrywa według modelu (sygnały realne)

1. **Młodszy zawodnik wygrywa częściej** — wiek to najsilniejsza pojedyncza cecha fizyczna
   (−8.6 pp na 1σ ≈ 5 lat różnicy). Zgodne z literaturą MMA: starzejący się zawodnicy tracą
   szybkość i regenerację.
2. **Kompletność profilu = status** — `reach_missing_diff` ma największą wagę, ale to **proxy**,
   nie przyczyna: brak zasięgu w danych ufcstats dotyczy zawodników mniej znanych, z krótszą
   karierą w UFC (niska ekspozycja). Model „przeczytał" z braku danych informację o statusie
   zawodnika. Ryzyko tej cechy i mechanizm kontroli opisane w sekcji 7.
3. **Umiejętność finiszowania** — finish rate (+3.3 pp na 1σ) bije procent samego wygrywania.
4. **Status rankingowy** — bycie w rankingu i wyższa pozycja (−3.7 pp na 1σ ≈ 4 pozycje).
5. **Bieżąca forma i seria** — form5 (+) i streak (+) mówią o bieżącej dyspozycji dnia.

### 4.2 Co jest ZASKAKUJĄCO neutralne

- **Wzrost** (w≈0) przy jednoczesnym **działającym zasięgu** (+0.073): wzrost i zasięg są
  skorelowane (r=0.59), a model przeniósł cały sygnał na zasięg. Sensowne — w MMA liczy się
  długość ramion (jab, kontrola dystansu), nie sam wzrost.
- **Aktywność (walki 12/24 m) i dni odpoczynku** ≈ 0. Hipoteza „rdzewienia" od przerw nie
  znajduje potwierdzenia w tym zbiorze po uwzględnieniu wieku i formy.
- **Doświadczenie łączne** (exp_diff +0.059) słabsze niż bilans (wins/losses ±0.14):
  liczy się jakość kariery, nie jej długość.
- **Styl walki** (stance_diff +0.043): southpaw/switch nie daje mierzalnej przewagi liniowej.
- **Kontekst** (tytuł, dywizja kobiet, kategoria wagowa): wynik walki jest w przybliżeniu
  „equal opportunity" — te zmienne nie poprawiają predykcji.

### 4.3 Bias-variance w liczbach

33 parametry na 4008 próbek treningowych. Train acc 0.612 ≈ val acc 0.612 < test acc **0.637** —
**brak luki generalizacyjnej** (test najlepszy, bo to najnowsza era danych). Model prosty nie
przeucza (niska wariancja), a ~64% trafień przy bazie 50% pokazuje, że w danych JEST sygnał
(bias umiarkowany). Regularyzacja L2 i winsoryzacja dodatkowo zawężają przestrzeń wag —
świadomy wybór po stronie niskiej wariancji, uzasadniony małym zbiorem i kolinearnością.

## 5. Zależności MIĘDZY cechami (kolinearność)

### 5.1 Tożsamości algebraiczne (zależności dokładne, nie statystyczne)

- `finish_rate = ko_pct + sub_pct` (dokładnie: (KO+SUB)/n = ko/n + sub/n).
  Wagi: ko −0.011 + sub +0.010 + finish_rate **+0.134** → sygnał finiszowania trafił
  całościowo do `finish_rate_diff`. Model wybrał najprostszą formę.
- `dec_pct = win_pct − ko_pct − sub_pct` (reszta zwycięstw to decyzje).
- `form3` to średnia z PODZBIORU okna `form5` (ostatnie 3 ⊂ ostatnie 5) → korelacja 0.87.
- `ranked_diff` jest zawarty w `rank_diff` (kodowanie 20 = nierankingowy) → korelacja −0.81.
- `debut_diff`, `both_debut` są funkcją `exp_diff = 0` (debiut ⇔ zero historii).

### 5.2 Najsilniejsze korelacje empiryczne (|r| ≥ 0.45, zbiór treningowy, po winsoryzacji)

| Para cech | r | Czytanie biznesowe |
|---|---:|--------------------|
| exp_diff ↔ wins_diff | **0.93** | Im dłuższa kariera, tym więcej zwycięstw (i porażek: wins↔losses 0.65) |
| win_pct_diff ↔ form5_diff | **0.92** | Forma z 5 walk to przybliżenie karierowego win rate |
| avg ↔ med fight time | **0.91** | Te same stylowe info — model wziął średnią, medianę zostawił z ~0 |
| exp_diff ↔ losses_diff | **0.89** | Weteran ma też najwięcej porażek — sama liczba walk nie chwałą |
| form3_diff ↔ form5_diff | **0.87** | Krótsza i dłuższa forma mierzą to samo |
| ranked_diff ↔ rank_diff | **−0.81** | Dubel statusu rankingowego |
| fights_12m ↔ fights_24m | **0.71** | Aktywność krótko- i długoterminowa zbieżna |
| ko_pct ↔ finish_rate | **0.67** | KO to główny składnik finish rate (sub 0.54) |
| wins_diff ↔ losses_diff | **0.65** | Więcej walk = więcej i wygranych, i przegranych |
| form3_diff ↔ streak_diff | **0.63** | Forma krótkoterminowa spójna z serią |
| finish_rate ↔ avg_fight_time | **−0.59** | Finishery kończą SZYBCIEJ — realna zależność stylowa |
| height_diff ↔ reach_diff | **0.59** | Wyżsi mają zwykle dłuższe ramiona |
| dec_pct ↔ avg_fight_time | **+0.55** | Decyzyjni punktowcy walczą dłużej |
| fights_12m ↔ days_rest | **−0.54** | Aktywni odpoczywają krócej |
| fights_24m ↔ debut_diff | **−0.51** | Debiutant z definicji nie ma aktywności |
| form5_diff ↔ streak_diff | **+0.51** | Forma i seria to bliscy kuzyni |

### 5.3 Konsekwencja: jak czytać wagi przy kolinearności

**Wagi pojedynczych skorelowanych cech NIE są przyczynowe.** Model liniowy rozdziela sygnał
między dublujące zmienne w sposób zależny od losowości optymalizacji. Dowód w tabeli 3:
`form3_diff` ma ujemną wagę (−0.08) MIMO dodatniej korelacji z targetem (+0.09), bo jego
informacja została „przejęta" przez form5/streak/win_pct. Poprawna interpretacja jest
**grupowa**: „grupa formy (win_pct + form3 + form5 + streak) ma zbiorczo dodatni wpływ",
a nie „forma z 3 walk szkodzi". Na wykresie `feature_weights.png` czytać grupami:

- **grupa doświadczenia**: exp, wins, losses, fights_12m/24m, debut, title_exp → łącznie lekko (+),
- **grupa formy/jakości**: win_pct, form3, form5, streak, finish_rate, ko/sub/dec_pct → łącznie wyraźnie (+),
- **grupa fizyczna**: age (−), reach (+), height (≈0), stance (≈0),
- **grupa statusu**: rank, ranked (− dla gorszego rankingu = + dla lepszego),
- **grupa braków danych**: reach_missing (−) jako proxy statusu.

## 6. Czego model ŚWIADOMIE NIE widzi (anti-leakage)

| Wykluczone | Dlaczego to leakage |
|---|---|
| Wynik bieżącej walki (winner, result_f1/f2) | To target — nie cecha |
| Statystyki BIEŻĄCEJ walki (sig. strikes, knockdowny, kontrola, rundy) | Liczone PO/w trakcie walki; w surowym CSV są (grupa „performance"), celowo pominięte |
| Kursy bukmacherskie (a_odds, prob_fair) | Ustawiane przed walką, więc formalnie legalne, ale: (a) model miał być niezależny od rynku, (b) kursy prawie doskonale przewidują wynik → zdominowałyby wagi i zniweczyły analizę value bet. Kursy służą jako BENCHMARK i do analizy ROI, nie jako cecha |
| Kolejność f1/f2 ze źródła | Koreluje z wynikiem (f1 wygrywa 56.6% walk) — neutralizowana losową zamianą A/B (po niej: 49.9%) |
| Statystyki przyszłych walk | Struktura „snapshot → cechy → update historii" wyklucza przeciek |

## 7. Winsoryzacja ±3σ i L2 — lekcja z „eksplozji pewności"

**Problem, który zaobserwowaliśmy na produkcji (strona demo):** para dwóch zawodników
z 1 walką w danych dawała P(A)=97.7%. Mechanika: `reach_missing_diff` ma największą wagę
i malutkie σ (0.181), więc pojedynczy brak danych o zasięgu u jednej strony to z=±5.5σ
→ wkład ±3.34 logitu — **więcej niż cały logit predykcji**. Bez tej jednej cechy model
mówiłby ~60%, z nią — 97.7%. Do tego cechy z 1 walki (forma „100% vs 0%") to czysty szum.

**Rozwiązanie (dwie warstwy):**
1. **Winsoryzacja ±3σ** w `transform_features` — spójnie w treningu, ewaluacji, predict
   i na stronie: maksymalny wkład jednej cechy = 3·|wᵢ| (dla reach_missing: 3×0.700 = **2.10
   logitu** zamiast 3.34+; dla days_rest: 3×0.030 = 0.09 zamiast ekstrapolacji z=−21).
2. **Regularyzacja L2** (weight_decay=0.01) — karze duże wagi; bez niej optymalizator
   „kompensuje" przycięte wejścia przez pompowanie wagi proxy cechy (0.700 → 0.994 w teście
   bez L2). Z L2 waga wróciła do 0.700, a martwe cechy zostały wyzerowane.
3. **Ostrzeżenie o wiarygodności** (strona + CLI): gdy któryś zawodnik ma <3 walki w danych,
   jest asymetria braku zasięgu albo przerwa >2 lata — predykcja oznaczana jako
   niewiarygodna (model ekstrapoluje poza rozkład treningowy).

**Efekt**: para Moraes vs Berish spadła z 96.3% → **88.9%** (z ostrzeżeniem), metryki testowe
**poprawiły się** (acc 0.629 → 0.637, AUC 0.683 → 0.686, log loss 0.644 → 0.643) — przycięcie
outlierów usunęło szum także z treningu. Wniosek dydaktyczny: cecha-proxy z małej liczby
obserwacji potrafi zdominować model liniowy; kontrola = ograniczenie zakresu cech (cap),
kara za duże wagi (L2) i uczciwa komunikacja ograniczeń użytkownikowi.

## 8. Podsumowanie jednym akapitem

Model uczy się, że walkę UFC (w tym zbiorze, 2015–2023) najlepiej tłumaczą: **wiek
(młodszy lepiej), status rankingowy, zdolność finiszowania i bieżąca forma**, uzupełnione
subtelnie przez **zasięg ramion** (nie wzrost!) i **bilans karierowy**. Aktywność, odpoczynek,
styl i kontekst walki są w przybliżeniu neutralne. Zależności między cechami są silne
(kolinearność), więc wagi czytamy grupowo — pojedyncza waga to „udział w konsensusie
grupy", nie przyczyna. Całość jest liniowa względem standaryzowanych i przyciętych cech,
więc decyzja modelu jest w pełni rozkładalna na wkłady: wkład cechy i = wᵢ·zᵢ (pokazywany
w trybie `predict` i na stronie jako „top cechy decyzyjne").

---

*Proweniencja liczb: `main.py` (v1.0.0) z winsoryzacją ±3σ i L2 (weight_decay=0.01), seed 42,
dane `ufc_all_fights_wide.csv` (6618 walk 2013–2026; po burn-in 2 lata: 5726; train 4008 /
val 859 / test 859), model nn.Linear(32, 1) = 33 parametry, najlepsza epoka 39, val loss
0.6491, val acc 0.612, test acc 0.637, test ROC-AUC 0.686. Determinizm potwierdzony
identycznym hashem wag z dwóch uruchomień (torch.set_num_threads(1)).*
