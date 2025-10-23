#!/usr/bin/env python3
"""
Tutorial 1: Podstawowe Sygnały (Basic Signals)

TEORIA:
-------
Sygnały są funkcjami matematycznymi reprezentującymi fizyczne wielkości zmieniające się w czasie.
W analizie sygnałów najczęściej spotykane są sygnały okresowe, które powtarzają się w regularnych
odstępach czasu.

1. SYGNAŁ SINUSOIDALNY (Sine Wave):
   s(t) = A * sin(2πft + φ)
   gdzie:
   - A: amplituda (maksymalna wartość sygnału)
   - f: częstotliwość [Hz] (liczba cykli na sekundę)
   - t: czas [s]
   - φ: przesunięcie fazowe [radiany]
   
   Sygnały sinusoidalne są podstawą analizy Fouriera - każdy sygnał okresowy może być
   przedstawiony jako suma sygnałów sinusoidalnych o różnych częstotliwościach.

2. SYGNAŁ PROSTOKĄTNY (Square Wave):
   Sygnał przełączający się między dwiema wartościami. W praktyce używany w systemach
   cyfrowych. Zawiera składowe harmoniczne (nieparzyste wielokrotności częstotliwości podstawowej).
   
3. SYGNAŁ TRÓJKĄTNY (Triangle Wave):
   Liniowo narastający i opadający sygnał. Zawiera harmoniczne nieparzyste, ale o mniejszej
   amplitudzie niż sygnał prostokątny.
   
4. SYGNAŁ PIŁOKSZTAŁTNY (Sawtooth Wave):
   Liniowo narastający, z nagłym spadkiem. Zawiera wszystkie harmoniczne (parzyste i nieparzyste).

5. SZUM BIAŁY (White Noise):
   Sygnał losowy o stałej gęstości widmowej mocy we wszystkich częstotliwościach.
   Ważny w testowaniu systemów i symulacjach.

PARAMETRY SYGNAŁÓW:
-------------------
- Częstotliwość (f): Określa jak szybko sygnał się zmienia [Hz]
- Okres (T): Czas jednego pełnego cyklu, T = 1/f [s]
- Amplituda (A): Maksymalna wartość sygnału
- Częstotliwość próbkowania (fs): Liczba próbek na sekundę [Hz]

Zgodnie z twierdzeniem Nyquista-Shannona, aby prawidłowo odtworzyć sygnał,
częstotliwość próbkowania musi być co najmniej dwukrotnie większa od najwyższej
częstotliwości w sygnale: fs ≥ 2 * fmax
"""

import numpy as np
import matplotlib.pyplot as plt

# Parametry sygnałów
fs = 1000  # Częstotliwość próbkowania [Hz]
duration = 2.0  # Czas trwania [s]
f = 5  # Częstotliwość sygnału [Hz]
amplitude = 1.0  # Amplituda

# Generowanie osi czasu
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# 1. Sygnał sinusoidalny
sine_wave = amplitude * np.sin(2 * np.pi * f * t)

# 2. Sygnał cosinusoidalny (przesunięcie fazowe o 90°)
cosine_wave = amplitude * np.cos(2 * np.pi * f * t)

# 3. Sygnał prostokątny
# square() generuje sygnał prostokątny o wartościach -1 i 1
from scipy import signal
square_wave = amplitude * signal.square(2 * np.pi * f * t)

# 4. Sygnał trójkątny
# sawtooth() z szerokością 0.5 daje sygnał trójkątny
triangle_wave = amplitude * signal.sawtooth(2 * np.pi * f * t, width=0.5)

# 5. Sygnał piłokształtny
# sawtooth() z szerokością 1.0 daje sygnał piłokształtny
sawtooth_wave = amplitude * signal.sawtooth(2 * np.pi * f * t, width=1.0)

# 6. Szum biały
# Szum gaussowski o średniej 0 i odchyleniu standardowym 0.3
white_noise = np.random.normal(0, 0.3, size=t.shape)

# Wizualizacja
fig, axes = plt.subplots(6, 1, figsize=(12, 14))
fig.suptitle('Podstawowe Typy Sygnałów', fontsize=16, fontweight='bold')

# Wykres 1: Sygnał sinusoidalny
axes[0].plot(t, sine_wave, 'b-', linewidth=2)
axes[0].set_title('1. Sygnał Sinusoidalny: A·sin(2πft)', fontsize=12)
axes[0].set_ylabel('Amplituda')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, 1)  # Pokazujemy tylko pierwszą sekundę dla lepszej czytelności
axes[0].text(0.02, 0.7, f'f = {f} Hz\nA = {amplitude}\nT = {1/f} s', 
             transform=axes[0].transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Wykres 2: Sygnał cosinusoidalny
axes[1].plot(t, cosine_wave, 'r-', linewidth=2)
axes[1].set_title('2. Sygnał Cosinusoidalny: A·cos(2πft) [przesunięcie fazowe 90°]', fontsize=12)
axes[1].set_ylabel('Amplituda')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, 1)
axes[1].text(0.02, 0.7, 'Cosinus to sinus\nprzesunięty o π/2', 
             transform=axes[1].transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Wykres 3: Sygnał prostokątny
axes[2].plot(t, square_wave, 'g-', linewidth=2)
axes[2].set_title('3. Sygnał Prostokątny (Square Wave)', fontsize=12)
axes[2].set_ylabel('Amplituda')
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim(0, 1)
axes[2].text(0.02, 0.7, 'Zawiera nieparzyste\nharmoniczne:\nf, 3f, 5f, 7f...', 
             transform=axes[2].transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Wykres 4: Sygnał trójkątny
axes[3].plot(t, triangle_wave, 'm-', linewidth=2)
axes[3].set_title('4. Sygnał Trójkątny (Triangle Wave)', fontsize=12)
axes[3].set_ylabel('Amplituda')
axes[3].grid(True, alpha=0.3)
axes[3].set_xlim(0, 1)
axes[3].text(0.02, 0.7, 'Liniowo narastający\ni opadający', 
             transform=axes[3].transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Wykres 5: Sygnał piłokształtny
axes[4].plot(t, sawtooth_wave, 'c-', linewidth=2)
axes[4].set_title('5. Sygnał Piłokształtny (Sawtooth Wave)', fontsize=12)
axes[4].set_ylabel('Amplituda')
axes[4].grid(True, alpha=0.3)
axes[4].set_xlim(0, 1)
axes[4].text(0.02, 0.7, 'Zawiera wszystkie\nharmoniczne:\nf, 2f, 3f, 4f...', 
             transform=axes[4].transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Wykres 6: Szum biały
axes[5].plot(t, white_noise, 'k-', linewidth=0.5, alpha=0.7)
axes[5].set_title('6. Szum Biały (White Noise)', fontsize=12)
axes[5].set_ylabel('Amplituda')
axes[5].set_xlabel('Czas [s]')
axes[5].grid(True, alpha=0.3)
axes[5].set_xlim(0, 1)
axes[5].text(0.02, 0.7, 'Losowy sygnał\no równomiernym\nwidmie mocy', 
             transform=axes[5].transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('tutorial_01_basic_signals.png', dpi=150, bbox_inches='tight')
print("✓ Wykres zapisany jako 'tutorial_01_basic_signals.png'")
plt.show()

# Dodatkowa demonstracja: porównanie sygnałów o różnych częstotliwościach
fig2, axes2 = plt.subplots(3, 1, figsize=(12, 8))
fig2.suptitle('Wpływ Częstotliwości na Sygnał Sinusoidalny', fontsize=16, fontweight='bold')

frequencies = [2, 5, 10]  # Hz
colors = ['b', 'r', 'g']

for i, (freq, color) in enumerate(zip(frequencies, colors)):
    signal_data = amplitude * np.sin(2 * np.pi * freq * t)
    axes2[i].plot(t, signal_data, color=color, linewidth=2, label=f'f = {freq} Hz')
    axes2[i].set_ylabel('Amplituda')
    axes2[i].grid(True, alpha=0.3)
    axes2[i].legend(loc='upper right')
    axes2[i].set_xlim(0, 1)
    axes2[i].text(0.02, 0.75, f'Okres T = {1/freq:.3f} s\nW 1 sekundę: {freq} cykli', 
                  transform=axes2[i].transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

axes2[2].set_xlabel('Czas [s]')
plt.tight_layout()
plt.savefig('tutorial_01_frequency_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Wykres zapisany jako 'tutorial_01_frequency_comparison.png'")
plt.show()

# Podsumowanie
print("\n" + "="*70)
print("PODSUMOWANIE TUTORIAL 1: PODSTAWOWE SYGNAŁY")
print("="*70)
print(f"\nParametry użyte w demonstracji:")
print(f"  • Częstotliwość próbkowania (fs): {fs} Hz")
print(f"  • Czas trwania: {duration} s")
print(f"  • Częstotliwość podstawowa (f): {f} Hz")
print(f"  • Amplituda: {amplitude}")
print(f"  • Liczba próbek: {len(t)}")
print(f"\nWażne wzory:")
print(f"  • Sygnał sinusoidalny: s(t) = A·sin(2πft)")
print(f"  • Okres sygnału: T = 1/f = {1/f} s")
print(f"  • Częstotliwość kołowa: ω = 2πf = {2*np.pi*f:.2f} rad/s")
print(f"\nZasada Nyquista-Shannona:")
print(f"  • Minimalna częstotliwość próbkowania: fs ≥ 2·f")
print(f"  • W tym przykładzie: fs = {fs} Hz >> 2·f = {2*f} Hz ✓")
print("\n" + "="*70)
