"""
Autorzy:
Prętki, Mikołaj (s122982)
Hołdakowski, Mikołaj (s23739)
System planowania zmiany biegów przy użyciu fuzzy logic
na podstawie danych prędkości, obrotów silnika, stopnia wciśnięcia gazu i trybu jazdy

Użyte moduły: numpy, skfuzzy, matplotlib
Do otrzymania symulacji, należy jedynie uruchomić program
Python 3.10
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

speed = ctrl.Antecedent(np.arange(0, 101, 1), 'speed')
engine_rpm = ctrl.Antecedent(np.arange(0, 8001, 1), 'engine_rpm')
throttle = ctrl.Antecedent(np.arange(0, 101, 1), 'throttle')
driving_mode = ctrl.Antecedent(np.arange(0, 11, 1), 'driving_mode')
shift = ctrl.Consequent(np.arange(1, 7, 1), 'shift')
"""
Zmienne lingwistyczne:
- speed: Reprezentuje prędkość pojazdu. Jest używana jako jedna z głównych zmiennych wejściowych w systemie logiki rozmytej, aby określić optymalną zmianę biegu w zależności od prędkości. Zmienna ta jest podzielona na trzy kategorie: 'low' (niska prędkość), 'medium' (średnia prędkość) i 'high' (wysoka prędkość).

- engine_rpm: Reprezentuje obroty silnika pojazdu. Obroty silnika są ważnym czynnikiem wpływającym na moment zmiany biegu. Zmienna ta jest podzielona na trzy kategorie: 'low' (niskie obroty), 'medium' (średnie obroty) i 'high' (wysokie obroty).

- throttle: Reprezentuje stopień wciskania pedału gazu. Jest uwzględniany w systemie logiki rozmytej jako ważny czynnik wpływający na moment zmiany biegu. Zmienna ta jest podzielona na trzy kategorie: 'low' (niski stopień wciskania gazu), 'medium' (średni stopień wciskania gazu) i 'high' (wysoki stopień wciskania gazu).

- driving_mode: Reprezentuje tryb jazdy pojazdu, takie jak "eco", "normal" i "sport". Tryb jazdy wpływa na strategię zmiany biegu w systemie logiki rozmytej. Zmienna ta jest podzielona na trzy kategorie: 'eco', 'normal' i 'sport'.

- shift: Reprezentuje zmianę biegu skrzyni biegów. Jest to zmienna wyjściowa systemu logiki rozmytej, która określa sugerowany bieg w zależności od wartości wejściowych. Zmienna ta jest podzielona na trzy kategorie: 'low' (niski bieg), 'medium' (średni bieg) i 'high' (wysoki bieg).

Te zmienne lingwistyczne są kluczowe dla określenia strategii zmiany biegu w automatycznej skrzyni biegów w zależności od różnych czynników, takich jak prędkość pojazdu, obroty silnika, stopień wciskania gazu i wybrany tryb jazdy.
"""
speed['low'] = fuzz.trimf(speed.universe, [0, 0, 50])
speed['medium'] = fuzz.trimf(speed.universe, [0, 50, 100])
speed['high'] = fuzz.trimf(speed.universe, [50, 100, 100])

engine_rpm['low'] = fuzz.trimf(engine_rpm.universe, [0, 0, 2500])
engine_rpm['medium'] = fuzz.trimf(engine_rpm.universe, [1500, 2500, 3500])
engine_rpm['high'] = fuzz.trimf(engine_rpm.universe, [3000, 8000, 8000])

throttle['low'] = fuzz.trimf(throttle.universe, [0, 0, 50])
throttle['medium'] = fuzz.trimf(throttle.universe, [0, 50, 100])
throttle['high'] = fuzz.trimf(throttle.universe, [50, 100, 100])

driving_mode['eco'] = fuzz.trimf(driving_mode.universe, [0, 0, 3])
driving_mode['normal'] = fuzz.trimf(driving_mode.universe, [2, 5, 8])
driving_mode['sport'] = fuzz.trimf(driving_mode.universe, [7, 10, 10])

shift['low'] = fuzz.trimf(shift.universe, [1, 1, 3])
shift['medium'] = fuzz.trimf(shift.universe, [2, 3, 4])
shift['high'] = fuzz.trimf(shift.universe, [3, 6, 6])
"""
Definicje funkcji przynależności:
- Dla zmiennej 'speed':
    - 'low': Przynależność niskiej prędkości. Przyjęto trójkątny kształt funkcji przynależności w zakresie od 0 do 50 jednostek prędkości.
    - 'medium': Przynależność średniej prędkości. Przyjęto trójkątny kształt funkcji przynależności w zakresie od 0 do 100 jednostek prędkości.
    - 'high': Przynależność wysokiej prędkości. Przyjęto trójkątny kształt funkcji przynależności w zakresie od 50 do 100 jednostek prędkości.

- Dla zmiennej 'engine_rpm':
    - 'low': Przynależność niskich obrotów silnika. Przyjęto trójkątny kształt funkcji przynależności w zakresie od 0 do 2500 obrotów na minutę.
    - 'medium': Przynależność średnich obrotów silnika. Przyjęto trójkątny kształt funkcji przynależności w zakresie od 1500 do 3500 obrotów na minutę.
    - 'high': Przynależność wysokich obrotów silnika. Przyjęto trójkątny kształt funkcji przynależności w zakresie od 3000 do 8000 obrotów na minutę.

- Dla zmiennej 'throttle':
    - 'low': Przynależność niskiego stopnia wciskania gazu. Przyjęto trójkątny kształt funkcji przynależności w zakresie od 0 do 50 procent.
    - 'medium': Przynależność średniego stopnia wciskania gazu. Przyjęto trójkątny kształt funkcji przynależności w zakresie od 0 do 100 procent.
    - 'high': Przynależność wysokiego stopnia wciskania gazu. Przyjęto trójkątny kształt funkcji przynależności w zakresie od 50 do 100 procent.

- Dla zmiennej 'driving_mode':
    - 'eco': Przynależność trybu jazdy "eco". Przyjęto trójkątny kształt funkcji przynależności w zakresie od 0 do 3 jednostek.
    - 'normal': Przynależność trybu jazdy "normal". Przyjęto trójkątny kształt funkcji przynależności w zakresie od 2 do 8 jednostek.
    - 'sport': Przynależność trybu jazdy "sport". Przyjęto trójkątny kształt funkcji przynależności w zakresie od 7 do 10 jednostek.

- Dla zmiennej 'shift':
    - 'low': Przynależność niskiego biegu. Przyjęto trójkątny kształt funkcji przynależności w zakresie od 1 do 3 jednostek.
    - 'medium': Przynależność średniego biegu. Przyjęto trójkątny kształt funkcji przynależności w zakresie od 2 do 4 jednostek.
    - 'high': Przynależność wysokiego biegu. Przyjęto trójkątny kształt funkcji przynależności w zakresie od 3 do 6 jednostek.

Te funkcje przynależności określają, w jakim stopniu dana wartość przynależy do danej kategorii lingwistycznej w kontekście każdej zmiennej lingwistycznej.
"""

rule_eco_1 = ctrl.Rule(speed['low'] & engine_rpm['low'] & throttle['low'] & driving_mode['eco'], shift['low'])
rule_eco_2 = ctrl.Rule(speed['medium'] & engine_rpm['low'] & throttle['medium'] & driving_mode['eco'], shift['medium'])
rule_eco_3 = ctrl.Rule(speed['medium'] & engine_rpm['medium'] & throttle['medium'] & driving_mode['eco'], shift['high'])

rule_normal_1 = ctrl.Rule(speed['low'] & engine_rpm['low'] & throttle['low'] & driving_mode['normal'], shift['low'])
rule_normal_2 = ctrl.Rule(speed['medium'] & engine_rpm['medium'] & throttle['medium'] & driving_mode['normal'], shift['medium'])
rule_normal_3 = ctrl.Rule(speed['high'] & engine_rpm['high'] & throttle['high'] & driving_mode['normal'], shift['high'])

rule_sport_1 = ctrl.Rule(speed['medium'] & engine_rpm['medium'] & throttle['high'] & driving_mode['sport'], shift['medium'])
rule_sport_2 = ctrl.Rule(speed['high'] & engine_rpm['high'] & throttle['high'] & driving_mode['sport'], shift['high'])

rule_sport_dynamic = ctrl.Rule(speed['low'] & engine_rpm['high'] & throttle['high'] & driving_mode['sport'], shift['low'])

rule_kickdown = ctrl.Rule(throttle['high'], shift['low'])

"""
Reguły logiki rozmytej w zależności od trybu jazdy:
- Reguły dla trybu "eco":
    Reguła 1:
        Warunki:
        - Niska prędkość ('speed' = 'low')
        - Niskie obroty silnika ('engine_rpm' = 'low')
        - Niski stopień wciskania gazu ('throttle' = 'low')
        - Tryb jazdy "eco" ('driving_mode' = 'eco')
        Działanie:
        - Sugerowany niski bieg ('shift' = 'low')

    Reguła 2:
        Warunki:
        - Średnia prędkość ('speed' = 'medium')
        - Niskie obroty silnika ('engine_rpm' = 'low')
        - Średni stopień wciskania gazu ('throttle' = 'medium')
        - Tryb jazdy "eco" ('driving_mode' = 'eco')
        Działanie:
        - Sugerowany średni bieg ('shift' = 'medium')

    Reguła 3:
        Warunki:
        - Średnia prędkość ('speed' = 'medium')
        - Średnie obroty silnika ('engine_rpm' = 'medium')
        - Średni stopień wciskania gazu ('throttle' = 'medium')
        - Tryb jazdy "eco" ('driving_mode' = 'eco')
        Działanie:
        - Sugerowany wysoki bieg ('shift' = 'high')
        
        
        - Reguły dla trybu "normal":
    Reguła 1:
        Warunki:
        - Niska prędkość ('speed' = 'low')
        - Niskie obroty silnika ('engine_rpm' = 'low')
        - Niski stopień wciskania gazu ('throttle' = 'low')
        - Tryb jazdy "normal" ('driving_mode' = 'normal')
        Działanie:
        - Sugerowany niski bieg ('shift' = 'low')

    Reguła 2:
        Warunki:
        - Średnia prędkość ('speed' = 'medium')
        - Średnie obroty silnika ('engine_rpm' = 'medium')
        - Średni stopień wciskania gazu ('throttle' = 'medium')
        - Tryb jazdy "normal" ('driving_mode' = 'normal')
        Działanie:
        - Sugerowany średni bieg ('shift' = 'medium')

    Reguła 3:
        Warunki:
        - Wysoka prędkość ('speed' = 'high')
        - Wysokie obroty silnika ('engine_rpm' = 'high')
        - Wysoki stopień wciskania gazu ('throttle' = 'high')
        - Tryb jazdy "normal" ('driving_mode' = 'normal')
        Działanie:
        - Sugerowany wysoki bieg ('shift' = 'high')

- Reguły dla trybu "sport":
    Reguła 1:
        Warunki:
        - Średnia prędkość ('speed' = 'medium')
        - Średnie obroty silnika ('engine_rpm' = 'medium')
        - Wysoki stopień wciskania gazu ('throttle' = 'high')
        - Tryb jazdy "sport" ('driving_mode' = 'sport')
        Działanie:
        - Sugerowany średni bieg ('shift' = 'medium')

    Reguła 2:
        Warunki:
        - Wysoka prędkość ('speed' = 'high')
        - Wysokie obroty silnika ('engine_rpm' = 'high')
        - Wysoki stopień wciskania gazu ('throttle' = 'high')
        - Tryb jazdy "sport" ('driving_mode' = 'sport')
        Działanie:
        - Sugerowany wysoki bieg ('shift' = 'high')
        
        Reguła 3:
        Warunki:
        - Niska prędkość ('speed' = 'low' lub 'medium')
        - Wysokie obroty silnika ('engine_rpm' = 'high')
        - Wysoki stopień wciskania gazu ('throttle' = 'high')
        - Tryb jazdy "sport" ('driving_mode' = 'sport')
        Działanie:
        - Sugerowany niski bieg ('shift' = 'low')

Te reguły logiki rozmytej określają zachowanie systemu w zależności od wybranego trybu jazdy. Działanie systemu jest dostosowywane do strategii zmiany biegów, które najlepiej pasują do danego trybu jazdy, uwzględniając czynniki takie jak prędkość pojazdu, obroty silnika, stopień wciskania gazu i wybrany tryb jazdy.
"""


# Dodatkowe reguły dla trybu "eco", "normal" i "sport" mogą być dodane w podobny sposób

shift_ctrl = ctrl.ControlSystem([rule_eco_1, rule_eco_2, rule_eco_3, rule_normal_1, rule_normal_2, rule_normal_3, rule_sport_1, rule_sport_2, rule_sport_dynamic, rule_kickdown])
"""
Utworzenie systemu logiki rozmytej `shift_ctrl` dla zarządzania zmianą biegów w zależności od różnych warunków i trybów jazdy.

Argumenty:
- rule_eco_1: Reguła dla trybu "eco".
- rule_eco_2: Reguła dla trybu "eco".
- rule_eco_3: Reguła dla trybu "eco".
- rule_normal_1: Reguła dla trybu "normal".
- rule_normal_2: Reguła dla trybu "normal".
- rule_normal_3: Reguła dla trybu "normal".
- rule_sport_1: Reguła dla trybu "sport".
- rule_sport_2: Reguła dla trybu "sport".
- rule_sport_dynamic: Reguła dla dynamicznego przyspieszania w trybie "sport".
- rule_kickdown: Reguła kickdownu dla szybkiej redukcji biegu w odpowiedzi na głębokie wciśnięcie gazu.

Zwraca:
- shift_ctrl: System kontroli logiki rozmytej dla zarządzania zmianą biegów.

Opis:
Ten system logiki rozmytej `shift_ctrl` skupia się na określaniu sugerowanej zmiany biegu na podstawie wielu czynników, takich jak prędkość pojazdu, obroty silnika, stopień wciskania gazu i wybrany tryb jazdy. Reguła kickdownu została dodana, aby umożliwić szybką redukcję biegu w odpowiedzi na głębokie wciśnięcie pedału gazu, co pozwala uzyskać maksymalne przyspieszenie.

To jest system logiki rozmytej używany do rekomendacji biegu dla automatycznej skrzyni biegów w zależności od różnych warunków jazdy.
"""

shift_simulation = ctrl.ControlSystemSimulation(shift_ctrl)
shift_simulation.input['speed'] = 30  # Prędkość pojazdu (np. 30 km/h)
shift_simulation.input['engine_rpm'] = 2000  # Obroty silnika (np. 2000 RPM)
shift_simulation.input['throttle'] = 40  # Stopień wciskania gazu (np. 40%)
shift_simulation.input['driving_mode'] = 2  # Tryb jazdy (np. "eco")
"""
Symulacja systemu logiki rozmytej `shift_simulation` dla zarządzania zmianą biegów w zależności od określonych warunków i trybu jazdy.

Argumenty:
- shift_ctrl: System kontroli logiki rozmytej dla zarządzania zmianą biegów.
- Prędkość pojazdu: Parametr wejściowy reprezentujący prędkość pojazdu (np. 30 km/h).
- Obroty silnika: Parametr wejściowy reprezentujący obroty silnika (np. 2000 RPM).
- Stopień wciskania gazu: Parametr wejściowy reprezentujący stopień wciskania pedału gazu (np. 40%).
- Tryb jazdy: Parametr wejściowy reprezentujący wybrany tryb jazdy (np. "eco").

Opis:
Ta symulacja systemu logiki rozmytej `shift_simulation` służy do przetestowania i uzyskania rekomendacji zmiany biegu w zależności od określonych parametrów, takich jak prędkość pojazdu, obroty silnika, stopień wciskania gazu i wybrany tryb jazdy. Symulacja pozwala na zobaczenie, jaki bieg zostanie zaproponowany przez system logiki rozmytej w konkretnych warunkach.

Przykładowe ustawienia parametrów:
- Prędkość pojazdu: 30 km/h
- Obroty silnika: 2000 RPM
- Stopień wciskania gazu: 40%
- Tryb jazdy: "eco"

Po uruchomieniu symulacji, można odczytać sugerowany bieg na podstawie określonych warunków i parametrów wejściowych. Symulacja pozwala zrozumieć, jak system logiki rozmytej reaguje na zmienne warunki jazdy i trybów.
"""

shift_simulation.compute()
"""
Wykonywanie symulacji i wyświetlanie wyników sugerowanej zmiany biegu w zależności od określonych parametrów wejściowych.

Argumenty:
- shift_simulation: Symulacja systemu logiki rozmytej dla zarządzania zmianą biegów.
- Prędkość pojazdu: Parametr wejściowy reprezentujący prędkość pojazdu.
- Obroty silnika: Parametr wejściowy reprezentujący obroty silnika.
- Stopień wciskania gazu: Parametr wejściowy reprezentujący stopień wciskania pedału gazu.
- Tryb jazdy: Parametr wejściowy reprezentujący wybrany tryb jazdy.

Opis:
W tej części kodu przeprowadzana jest symulacja systemu logiki rozmytej w celu uzyskania sugerowanej zmiany biegu w zależności od określonych parametrów wejściowych. Parametry te obejmują prędkość pojazdu, obroty silnika, stopień wciskania gazu i wybrany tryb jazdy.

Przykładowe ustawienia parametrów wejściowych:
- Prędkość pojazdu: 30 km/h
- Obroty silnika: 2000 RPM
- Stopień wciskania gazu: 40%
- Tryb jazdy: "eco"

Po przeliczeniach w systemie logiki rozmytej, uzyskana jest sugerowana zmiana biegu, która jest wyświetlana na konsoli. Dodatkowo wyświetlane są wykresy funkcji przynależności dla każdej zmiennej lingwistycznej oraz sugerowana zmiana biegu na wykresie.

Ta część kodu pozwala na wizualizację wyników i zrozumienie, jakie zmiany biegu są sugerowane w konkretnych warunkach jazdy.
"""

# Wyświetlanie wyniku
print("Suggested Shift:", shift_simulation.output['shift'])
shift.view(sim=shift_simulation)

# Wyświetlenie wykresu funkcji przynależności
speed.view()
engine_rpm.view()
throttle.view()
driving_mode.view()
shift.view()
plt.show()