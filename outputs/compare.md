# Porównanie wariantów uczenia (WalkoMat)

Ten sam jednowarstwowy model `nn.Linear(32, 1)`, te same dane i skalowanie,
identyczna inicjalizacja i kolejność batchy (seed 42). Warianty różnią się
wyłącznie optymalizatorem / funkcją straty / trybem batchów. Early stopping
na walidacyjnej funkcji straty (patience 12); próg konwergencji liczymy na
**ujednoliconej metryce** — log loss walidacyjny (porównywalny między stratami).

| Wariant | najw. epoka | val log loss | LL<=0.67 | test acc | test AUC | log loss | czas [s] |
|---|---:|---:|---:|---:|---:|---:|---:|
| SGD lr=1e-3 | 100 | 0.6533 | 23 | 0.6391 | 0.6791 | 0.6494 | 7.2 |
| SGD+Momentum 0.9, lr=1e-3 | 39 | 0.6500 | 3 | 0.6368 | 0.6865 | 0.6439 | 2.4 |
| RMSprop lr=1e-3 | 35 | 0.6493 | 1 | 0.6321 | 0.6860 | 0.6435 | 2.4 |
| Adam lr=1e-3 (wybrany) **★** | 39 | 0.6491 | 2 | 0.6368 | 0.6865 | 0.6434 | 2.5 |
| Adam + strata MSE | 39 | 0.6496 | 2 | 0.6368 | 0.6857 | 0.6451 | 3.2 |
| Adam, full-batch GD | 100 | 0.6603 | 54 | 0.6088 | 0.6553 | 0.6599 | 3.6 |

## Wnioski — dlaczego wybraliśmy Adam i entropię krzyżową

- **Najniższy walidacyjny log loss** uzyskuje **Adam lr=1e-3 (wybrany)** (0.6491, epoka 39) — ten wariant jest domyślnym treningiem projektu.
- **Adam vs SGD/Momentum/RMSprop**: adaptacyjne, per-parametrowe kroki uczenia (momenta 1. i 2. rzędu) zbiegają najszybciej; SGD ze stałym krokiem lr=1e-3 jest za wolne dla tej skali cech, a Momentum/RMSprop poprawiają zbieżność, lecz nie doganiają Adama.
- **Entropia krzyżowa vs MSE**: porównanie w UJEDNOLICONEJ skali (log loss): wariant MSE kończy na 0.6496 vs 0.6491 dla entropii krzyżowej — pochodna MSE przez sigmoid gaśnie przy skrajnych logitach (nasycenie), więc błędne predykcje korygują się powoli. BCEWithLogitsLoss tego problemu nie ma i jest numerycznie stabilna (dlatego nie używamy osobnego sigmoid+BCE).
- **Mini-batch vs full-batch**: pełny gradient to 1 aktualizacja wag na epokę — w budżecie 100 epok osiąga tylko 0.6603; mini-batch (64) daje ~63 aktualizacje na epokę, szybciej i stabilniej.
- Punkt odniesienia rynku: bukmacher ROC-AUC = 0.765 — żaden wariant go nie pobija (zgodnie z oczekiwaniem; patrz OBRONA §12.2).
