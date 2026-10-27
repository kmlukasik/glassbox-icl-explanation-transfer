
Porównanie działania modeli typu glassbox i ICL na podstawie zmian 

wejścia sterowanego wyjaśnialnością

Petrashevich Ulyana, Łukasik Karol, Chabowski Kacper

## Cel projektu
Modele typu glassbox, takie jak Explainable Boosting Machine (EBM), osiągają dobre wyniki na danych tabelarycznych i są w pełni interpretowalne. Modele typu ICL, takie jak TabPFN, bywają dokładniejsze, ale ich działanie jest trudne do wyjaśnienia. Celem projektu jest sprawdzenie czy wyjaśnienia modelu glassbox przenoszą się na model ICL. Wytrenujemy EBM na trzech zbiorach danych, a następnie będziemy zmieniać wejście zgodnie z jego wyjaśnieniami, czyli modyfikować cechy tak, jak wskazują ich funkcje kształtu. Porównamy zmianę wyjścia TabPFN ze zmianą przewidzianą przez EBM. 
## Bibliografia
Nori, Jenkins, Koch, Caruana: InterpretML: A Unified Framework for Machine Learning Interpretability [2019]

Lou, Caruana, Gehrke: Intelligible Models for Classification and Regression [2012]

Hollmann, Müller, Eggensperger, Hutter: TABPFN: A Transformer That Solves Small Tabular Classification Problems In A Second [2022]

Prior Labs Team: TabPFN-3.5: Technical Report [2026]

Rundel, Kobialka, von Crailsheim, Feurer, Nagler, Rugamer: Interpretable Machine Learning for TabPFN [2024]

Chen, Guestrin: XGBoost: A Scalable Tree Boosting System [2016]

Grinsztajn, Oyallon, Varoquaux: Why do tree-based models still outperform deep learning on typical tabular data? [2022]

Wachter, Mittelstadt, Russell: Counterfactual Explanations without Opening the Black Box: Automated Decisions and the GDPR [2017]
## Zbiory danych
W projekcie wykorzystamy trzy binarne zbiory klasyfikacyjne z repozytorium OpenML: [German Credit Data](http://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data)), [Adult Income](https://archive.ics.uci.edu/ml/datasets/adult) i [MAGIC Gamma Telescope](https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope). Wybraliśmy je tak, aby różniły się rozmiarem i typami cech, a jednocześnie miały umiarkowany poziom trudności. 

German Credit (ok. 1 000 próbek, 20 cech, głównie kategorycznych) to mały zbiór, w którym modele ICL są szczególnie skuteczne. Adult Income (ok. 49 tys. próbek, 14 cech mieszanych) to duży zbiór zawierający zarówno liczby, jak i tekst, który wymaga uzupełnienia brakujących wartości. MagicTelescope (ok. 19 tys. próbek, 10 cech numerycznych) jest zbiorem czysto numerycznym.
## Planowany zakres eksperymentów
1. Sprawdzenie początkowej jakości: Uczymy EBM oraz XGBoost na części treningowej, a także sprawdzamy gotowy model TabPFN. Liczymy podstawowe metryki (ROC-AUC), żeby upewnić się, że zbiory są dobrze przygotowane, a modele działają poprawnie.

2. Wyciągnięcie reguł z EBM: Odczytujemy z wytrenowanego EBM informację, które cechy są najważniejsze i w którą stronę zmiana wartości cechy zwiększa szansę na pozytywny wynik.

3. Modyfikacja pojedynczych cech (1D): Bierzemy próbki ze zbioru testowego i zmieniamy w nich wartość najważniejszej cechy. Sprawdzamy, czy prawdopodobieństwo zwrócone przez TabPFN rośnie lub maleje zgodnie z przewidywaniem EBM.

4. Próba kontrolna – cecha nieistotna: Zmieniamy cechę, która według EBM nie ma żadnego znaczenia. Wynik w TabPFN również nie powinien się zauważalnie zmienić.

5. Modyfikacja kilku cech naraz (kontrfaktyczna): Bierzemy przypadki, w których model dał odmowę, i zmieniamy zestaw cech według EBM tak, aby wymusić decyzję pozytywną. Sprawdzamy, czy TabPFN po tych samych zmianach również zmieni zdanie.
## Planowana funkcjonalność programu
Program będzie pakietem Pythona z interfejsem wiersza poleceń, który pozwala powtórzyć wszystkie eksperymenty z projektu na podstawie plików konfiguracyjnych. Program:

- pobiera zbiór i przygotowuje go do eksperymentów

- trenuje model EBM i udostępnia jego wyjaśnienia lokalne i globalne w formacie niezależnym od modelu

- generuje perturbacje wejścia sterowane wyjaśnieniami oraz kontrolne. Wysyła je do modelu TabPFN

- oblicza metryki zgodności wraz z przedziałami ufności i testami statystycznymi, zapisuje wyniki wraz z konfiguracją

- generuje wykresy i tabele. 

Wszystkie parametry eksperymentów (zbiór danych, ziarno, parametry EBM, strategia i wielkość perturbacji, wybór metryk) są zadawane w plikach konfiguracyjnych YAML. Każdy eksperyment będzie można powtórzyć lub zmodyfikować bez zmiany kodu, a rozszerzenie programu o nowy zbiór lub strategię perturbacji będzie wymagało jedynie dodania modułu o ustalonym interfejsie.
## Planowany stack technologiczny
Projekt powstanie w Pythonie 3.11. Środowiskiem wirtualnym i zależnościami (z plikiem blokującym wersje) zarządza `uv`, a struktura repozytorium opiera się na szablonie `cookiecutter``-data-science`.

**Modele.** Model glassbox to Explainable Boosting Machine z biblioteki `interpret`. Jako model ICL użyjemy TabPFN 3.5 (pakiet `tabpfn`, PyTorch, wstępnie wytrenowane wagi). Modelem odniesienia będzie XGBoost z pakietu `xgboo``st`.

**Dane i analiza.** Zbiory pobieramy z OpenML (`openml` lub `scikit-learn`), przetwarzamy je w `pandas` i `numpy`, a wyniki zapisujemy w formacie Parquet (`pyarrow`). Statystykę (testy, bootstrap) wykonujemy w `scipy` i `statsmodels`. Wykresy powstają w `matplotlib` i `seaborn`.

**Konfiguracja i śledzenie eksperymentów.** Konfiguracja jest zapisana w plikach YAML (`pyyaml`), odseparowana od kodu wykonawczego. Przebiegi śledzimy w Weights & Biases, każdy wynik zapisujemy razem z konfiguracją.

**Jakość kodu.** Formatowanie i linting zapewnia `ruff`, styl jest zgodny z PEP8 ze zwiększonym limitem długości linii. Testy jednostkowe uruchamiamy w `pytest`, a hooki `pre-commit` i GitHub Actions pilnują, żeby kod przechodził lint i testy przy każdej zmianie. Commity są pisane zgodnie z konwencją conventional commits.

**Uruchamianie i dokumentacja.** Budowanie, testowanie i uruchamianie eksperymentów będzie odbywało się w `just`, a interfejs wiersza poleceń będzie zbudowany w `typer`. Eksploracja danych będzie odbywała się w notatnikach Jupyter, a gotowy kod eksportowany do biblioteki w `src/`. Dokumentację tworzymy w `mkdocs`, a bieżące postępy w notatkach w `docs/standup/`.
## Zasoby obliczeniowe
Trenowanie EBM i modelu odniesienia odbywa się na CPU. Dla inferencji modelu TabPFN zalecane jest zastosowanie GPU, wykorzystamy do tego RTX5070.

## Harmonogram


| Daty | Deadline | Opis |
| :---: | --- | --- |
| 21.10 - 28.10 | 28.10 (środa) – Design Proposal | Wstępne zapoznanie z tematem, utworzenie design proposal. |
| 30.10 - 06.11 | 30.10 – feedback do design proposal 06.11 – prototyp – deadline na spotkania | Uwzględnienie uwag od prowadzącego, konfiguracja środowiska, początek udokumentowania tabeli z przeglądem literatury, utworzenie funkcjonalnego prototypu - wytrenowanie EBM, podejrzenie jednej cechy, zmiana jej wartości i sprawdzenie odpowiedzi w TabPFN. |
| 07.11 - 15.11 |   | Przygotowanie zbiorów danych, eksperyment początkowej jakości, implementacja modułu ekstrakcji funkcji kształtu EBM |
| 16.11 - 22.11 |   | Implementacja generatora perturbacji jednowymiarowych wraz ze strategiami kontrolnymi; nauka do kolokwium |
| 23.11 - 29.11 |   | Przeprowadzenie eksperymentu z modyfikacją pojedynczych cech |
| 30.11 - 06.12 |   | Implementacja generatora perturbacji wielowymiarowych |
| 07.12 - 13.12 |   | Przeprowadzenie eksperymentu z modyfikacją kilku cech na raz, początek pisania artykułu |
| 14.12 - 23.12 |   | Wnioski z przeprowadzonych testów |
| 24.12 - 10.01 |   | Dokończenie artykułu, nagranie filmu, stworzenie instrukcji użytkowania |
| 11.01 - 14.01 | 14.01 – termin oddania projektu (zwolnienie z K2) | Ostatnie poprawki |

