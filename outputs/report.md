# WalkoMat — raport z eksperymentu
## Interpretowalny predyktor wyników walk UFC (jednowarstwowa sieć neuronowa)

*Jednowarstwowa sieć neuronowa (PyTorch, `nn.Linear(32, 1)`) do predykcji
wyników walk UFC. Projekt edukacyjny — nie stanowi porady finansowej ani zachęty do hazardu.*

## 1. Problem biznesowy
Analityk zakładów sportowych potrzebuje taniego, szybkiego i interpretowalnego narzędzia
szacującego prawdopodobieństwo zwycięstwa zawodnika A oraz wykrywającego potencjalne
value bety względem kursów bukmacherskich.

## 2. Rozwiązanie
Narzędzie CLI trenujące jednowarstwową sieć neuronową (33 parametrów) na
cechach dostępnych wyłącznie PRZED walką, z chronologiczną walidacją (test = najnowsze walki),
benchmarkami (majority class, regresja logistyczna, rynek) i symulacją prostych strategii zakładowych.

## 3. Dane
- Źródło: ufc_all_fights_wide.csv; 6618 walk po normalizacji
  (2013-02-16 → 2026-10-10), 2060 zawodników.
- Po burn-in (2.0 lat): 5726 walk modelowych.
- Podział chronologiczny: train 4008 (2015-02-22 → 2023-06-24),
  validation 859, test 859 (2025-03-08 → 2026-10-10).
- 99.4% walk ma kursy bukmacherskie (użyte TYLKO do benchmarku i analizy, nie jako cecha).

## 4. Model jednowarstwowy i dlaczego jest wystarczający
Architektura: wektor 32 ustandaryzowanych cech różnicowych (A−B) →
`nn.Linear(32, 1)` → sigmoid. To formalnie regresja logistyczna
zaimplementowana jako sieć neuronowa (wymóg przedmiotu). Prostota daje: 33
parametry (pełna interpretowalność), sekundowy trening na CPU, minimalne ryzyko przeuczenia
(gap train–test ≈ +0.025) i łatwe wyjaśnienie decyzji
(wkład cechy = waga × cecha).

## 5. Metryki (zbiór testowy, próg 0.5)
| Metryka | Model | Majority | Regr. log. | Bukmacher |
|---|---:|---:|---:|---:|
| Accuracy | 0.6368 | 0.4948 | 0.6298 | 0.7052 |
| ROC-AUC | 0.6865 | — | 0.6767 | 0.7647 |
| Log loss | 0.6434 | 0.6932 | 0.6458 | 0.5799 |
| Brier | 0.2260 | — | 0.2273 | 0.1977 |

Precision=0.6388, recall=0.6118, F1=0.6250, balanced accuracy=0.6365,
PR-AUC=0.6602, ECE (kalibracja)=0.0421. Macierz pomyłek: TN=287,
FP=147, FN=165, TP=260.
Baseline'y: majority class — acc 0.4948; regresja logistyczna sklearn — acc 0.6298,
ROC-AUC 0.6767; bukmacher (kursy zamknięcia bez marży, n=848): acc=0.7052, ROC-AUC=0.7647, log loss=0.5799.

## 6. Interpretacja wyników
Najmocniejsze wagi (pełna analiza: ZALEZNOSCI_ZMIENNYCH.md):
- `reach_missing_diff`: -0.700 (sprzyja B)
- `age_diff`: -0.348 (sprzyja B)
- `rank_diff`: -0.146 (sprzyja B)
- `wins_diff`: +0.142 (sprzyja A)
- `finish_rate_diff`: +0.133 (sprzyja A)
- `losses_diff`: -0.129 (sprzyja B)

Model najmocniej reaguje na wiek (młodszy → większe szanse), status rankingowy, zdolność
finiszowania i bieżącą formę; zasięg ramion liczy się tam, gdzie wzrost jest neutralny.
Wagi cech skorelowanych czytamy grupami (kolinearność).

## 7. Analiza bukmacherska
Strategia value bet (p ≥ próg oraz EV = p·kurs > 1, stawka 1 j.):

| Próg | Zakłady | Win rate | ROI | ROI odporny (bez top-10) |
|---|---:|---:|---:|---:|
| 0.55 | 179 | 0.497 | +1.225 | -0.036 |
| 0.6 | 103 | 0.573 | +0.783 | -0.056 |
| 0.65 | 51 | 0.608 | +0.594 | -0.219 |

Benchmark faworytów rynkowych: 842 zakładów, ROI +0.003.

**Ostrzeżenie o odporności:** ROI surowe bywa zdominowane przez pojedyncze wygrane longshoty (np. kurs 100:1); mediana zysku na zakład i ROI bez top-10 pokazują, czy strategia jest powtarzalna, czy zależna od szczęścia. Symulacja ma charakter wyłącznie edukacyjny i nie stanowi porady finansowej.

## 8. Ograniczenia i ryzyka
- Model liniowy nie modeluje interakcji cech (świadomy kompromis: interpretowalność).
- Kariery przed 2013 (początek datasetu) są obcięte — mitygowane burn-in i imputacją.
- Symulacja zakładów pomija limity stawek, ruchy kursów i reakcję rynku.
- Rynek się zmienia: model wymaga okresowego retrainingu (`python main.py train`).

## 9. Możliwy rozwój produktu
Cechy z ostatnich walk (statystyki rund), styl walki, regularyzacja L1/L2, kalibracja
izotoniczna, raporty per gala, monitoring driftu model vs rynek.

## 10. Wniosek: czy prosty model ma sens jako MVP?
Tak. Model istotnie przewyższa przypadek (accuracy 0.637 vs 0.500, ROC-AUC
0.686 vs 0.5) i jest w pełni interpretowalny, a jego kalibracja (ECE 0.042)
pozwala porównywać prawdopodobieństwa z rynkiem. Nie pobija bukmachera — zgodnie z oczekiwaniem
— ale jako tani, wyjaśnialny prototyp analityczny stanowi realną bazę pod dalszy rozwój.

---
*Wygenerowano automatycznie przez main.py (v1.1.0, seed 42).*
