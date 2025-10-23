#!/usr/bin/env python3
"""
Tutorial 2: Transformata Fouriera i Analiza Częstotliwościowa

TEORIA:
-------
Transformata Fouriera jest jednym z najważniejszych narzędzi w analizie sygnałów.
Pozwala przekształcić sygnał z dziedziny czasu do dziedziny częstotliwości.

1. TRANSFORMATA FOURIERA:
   X(f) = ∫ x(t) · e^(-j2πft) dt
   
   W praktyce używamy Dyskretnej Transformaty Fouriera (DFT):
   X[k] = Σ x[n] · e^(-j2πkn/N)
   
   gdzie:
   - x[n]: próbki sygnału w dziedzinie czasu
   - X[k]: współczynniki w dziedzinie częstotliwości
   - N: liczba próbek
   - k: indeks częstotliwości

2. SZYBKA TRANSFORMATA FOURIERA (FFT):
   FFT to efektywny algorytm obliczania DFT, redukujący złożoność
   z O(N²) do O(N log N). Wymaga liczby próbek będącej potęgą 2.

3. WIDMO MOCY (Power Spectrum):
   P(f) = |X(f)|²
   Pokazuje rozkład energii sygnału w funkcji częstotliwości.

4. WIDMO AMPLITUDOWE (Magnitude Spectrum):
   |X(f)| = sqrt(Re²(X) + Im²(X))
   Pokazuje amplitudy poszczególnych składowych częstotliwościowych.

5. WIDMO FAZOWE (Phase Spectrum):
   φ(f) = arctan(Im(X) / Re(X))
   Pokazuje przesunięcia fazowe składowych częstotliwościowych.

INTERPRETACJA:
--------------
- Piki w widmie odpowiadają częstotliwościom występującym w sygnale
- Wysokość piku pokazuje amplitudę danej składowej częstotliwościowej
- Sygnały okresowe dają dyskretne linie widmowe (harmoniczne)
- Sygnały nieokresowe dają ciągłe widmo

TWIERDZENIE O SPLOCIE:
----------------------
Splot w dziedzinie czasu odpowiada mnożeniu w dziedzinie częstotliwości:
x(t) * h(t) ←→ X(f) · H(f)

To właściwość jest kluczowa w projektowaniu filtrów.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy.fftpack import fft, fftfreq

# Parametry
fs = 1000  # Częstotliwość próbkowania [Hz]
duration = 1.0  # Czas trwania [s]
N = int(fs * duration)  # Liczba próbek

# Generowanie osi czasu
t = np.linspace(0, duration, N, endpoint=False)

# Przykład 1: Czysty sygnał sinusoidalny
f1 = 50  # Hz
signal1 = np.sin(2 * np.pi * f1 * t)

# Przykład 2: Suma dwóch sinusoid
f2_a = 50  # Hz
f2_b = 120  # Hz
signal2 = np.sin(2 * np.pi * f2_a * t) + 0.5 * np.sin(2 * np.pi * f2_b * t)

# Przykład 3: Sygnał z harmonicznymi (fala prostokątna)
signal3 = sig.square(2 * np.pi * f1 * t)

# Przykład 4: Sygnał z szumem
signal4 = np.sin(2 * np.pi * f1 * t) + 0.3 * np.random.randn(N)

# Funkcja do obliczania FFT i częstotliwości
def compute_fft(signal, fs):
    """Oblicza FFT i zwraca częstotliwości oraz amplitudy."""
    N = len(signal)
    # Okno Hanninga redukuje efekty brzegowe
    windowed_signal = signal * np.hanning(N)
    # Obliczanie FFT
    fft_vals = fft(windowed_signal)
    # Częstotliwości
    freqs = fftfreq(N, 1/fs)
    # Amplituda (normalizowana)
    amplitude = np.abs(fft_vals) * 2 / N
    # Widmo mocy
    power = np.abs(fft_vals) ** 2
    
    # Zwracamy tylko dodatnie częstotliwości
    positive_freq_idx = freqs >= 0
    return freqs[positive_freq_idx], amplitude[positive_freq_idx], power[positive_freq_idx]

# Obliczanie FFT dla wszystkich sygnałów
freqs1, amp1, power1 = compute_fft(signal1, fs)
freqs2, amp2, power2 = compute_fft(signal2, fs)
freqs3, amp3, power3 = compute_fft(signal3, fs)
freqs4, amp4, power4 = compute_fft(signal4, fs)

# Wizualizacja
fig, axes = plt.subplots(4, 2, figsize=(14, 12))
fig.suptitle('Transformata Fouriera - Dziedzina Czasu vs Częstotliwości', 
             fontsize=16, fontweight='bold')

# Przykład 1: Czysty sygnał sinusoidalny
axes[0, 0].plot(t[:200], signal1[:200], 'b-', linewidth=2)
axes[0, 0].set_title('Dziedzina Czasu: Czysty Sygnał Sinusoidalny')
axes[0, 0].set_ylabel('Amplituda')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].text(0.02, 0.85, f'f = {f1} Hz', transform=axes[0, 0].transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

axes[0, 1].stem(freqs1[:200], amp1[:200], basefmt=' ', linefmt='b-', markerfmt='bo')
axes[0, 1].set_title('Dziedzina Częstotliwości: Widmo Amplitudowe')
axes[0, 1].set_ylabel('Amplituda')
axes[0, 1].set_xlim(0, 200)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axvline(x=f1, color='r', linestyle='--', alpha=0.5, label=f'f = {f1} Hz')
axes[0, 1].legend()

# Przykład 2: Suma dwóch sinusoid
axes[1, 0].plot(t[:200], signal2[:200], 'g-', linewidth=2)
axes[1, 0].set_title('Dziedzina Czasu: Suma Dwóch Sinusoid')
axes[1, 0].set_ylabel('Amplituda')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].text(0.02, 0.85, f'f1 = {f2_a} Hz\nf2 = {f2_b} Hz', 
                transform=axes[1, 0].transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

axes[1, 1].stem(freqs2[:200], amp2[:200], basefmt=' ', linefmt='g-', markerfmt='go')
axes[1, 1].set_title('Dziedzina Częstotliwości: Dwa Piki')
axes[1, 1].set_ylabel('Amplituda')
axes[1, 1].set_xlim(0, 200)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axvline(x=f2_a, color='r', linestyle='--', alpha=0.5)
axes[1, 1].axvline(x=f2_b, color='r', linestyle='--', alpha=0.5)

# Przykład 3: Sygnał prostokątny (z harmonicznymi)
axes[2, 0].plot(t[:200], signal3[:200], 'm-', linewidth=2)
axes[2, 0].set_title('Dziedzina Czasu: Sygnał Prostokątny')
axes[2, 0].set_ylabel('Amplituda')
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].text(0.02, 0.85, 'Zawiera nieparzyste\nharmoniczne', 
                transform=axes[2, 0].transAxes,
                bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7))

axes[2, 1].stem(freqs3[:300], amp3[:300], basefmt=' ', linefmt='m-', markerfmt='mo')
axes[2, 1].set_title('Dziedzina Częstotliwości: Harmoniczne (f, 3f, 5f, ...)')
axes[2, 1].set_ylabel('Amplituda')
axes[2, 1].set_xlim(0, 300)
axes[2, 1].grid(True, alpha=0.3)
for harm in [1, 3, 5, 7]:
    axes[2, 1].axvline(x=f1*harm, color='r', linestyle='--', alpha=0.3)

# Przykład 4: Sygnał z szumem
axes[3, 0].plot(t[:200], signal4[:200], 'c-', linewidth=1.5)
axes[3, 0].set_title('Dziedzina Czasu: Sygnał z Szumem')
axes[3, 0].set_ylabel('Amplituda')
axes[3, 0].set_xlabel('Czas [s]')
axes[3, 0].grid(True, alpha=0.3)
axes[3, 0].text(0.02, 0.85, 'Sygnał użyteczny\n+ szum gaussowski', 
                transform=axes[3, 0].transAxes,
                bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.7))

axes[3, 1].plot(freqs4[:300], amp4[:300], 'c-', linewidth=1.5)
axes[3, 1].set_title('Dziedzina Częstotliwości: Pik + Szerokopasmowy Szum')
axes[3, 1].set_ylabel('Amplituda')
axes[3, 1].set_xlabel('Częstotliwość [Hz]')
axes[3, 1].set_xlim(0, 300)
axes[3, 1].grid(True, alpha=0.3)
axes[3, 1].axvline(x=f1, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('tutorial_02_fourier_transform.png', dpi=150, bbox_inches='tight')
print("✓ Wykres zapisany jako 'tutorial_02_fourier_transform.png'")
plt.show()

# Demonstracja widma mocy
fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8))
fig2.suptitle('Widmo Mocy (Power Spectrum)', fontsize=16, fontweight='bold')

# Sygnał złożony: suma kilku częstotliwości
f_components = [30, 75, 150, 220]
amplitudes = [1.0, 0.7, 0.5, 0.3]
complex_signal = np.zeros(N)
for f, a in zip(f_components, amplitudes):
    complex_signal += a * np.sin(2 * np.pi * f * t)

# Dodaj szum
complex_signal += 0.2 * np.random.randn(N)

# Oblicz FFT
freqs_complex, amp_complex, power_complex = compute_fft(complex_signal, fs)

# Wykres sygnału w czasie
axes2[0].plot(t[:500], complex_signal[:500], 'b-', linewidth=1.5)
axes2[0].set_title('Sygnał Złożony w Dziedzinie Czasu')
axes2[0].set_ylabel('Amplituda')
axes2[0].set_xlabel('Czas [s]')
axes2[0].grid(True, alpha=0.3)

# Wykres widma mocy
axes2[1].plot(freqs_complex[:300], 10*np.log10(power_complex[:300] + 1e-10), 'r-', linewidth=2)
axes2[1].set_title('Widmo Mocy [dB]')
axes2[1].set_ylabel('Moc [dB]')
axes2[1].set_xlabel('Częstotliwość [Hz]')
axes2[1].set_xlim(0, 300)
axes2[1].grid(True, alpha=0.3)

# Zaznacz składowe
for f, a in zip(f_components, amplitudes):
    axes2[1].axvline(x=f, color='g', linestyle='--', alpha=0.5)
    axes2[1].text(f, axes2[1].get_ylim()[1]*0.9, f'{f}Hz', 
                  ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('tutorial_02_power_spectrum.png', dpi=150, bbox_inches='tight')
print("✓ Wykres zapisany jako 'tutorial_02_power_spectrum.png'")
plt.show()

# Podsumowanie
print("\n" + "="*70)
print("PODSUMOWANIE TUTORIAL 2: TRANSFORMATA FOURIERA")
print("="*70)
print(f"\nParametry:")
print(f"  • Częstotliwość próbkowania: {fs} Hz")
print(f"  • Rozdzielczość częstotliwościowa: Δf = fs/N = {fs/N:.2f} Hz")
print(f"  • Zakres częstotliwości: 0 do {fs/2} Hz (częstotliwość Nyquista)")
print(f"\nWykryte składowe w sygnale złożonym:")
for f, a in zip(f_components, amplitudes):
    print(f"  • Częstotliwość: {f} Hz, Amplituda względna: {a}")
print(f"\nWażne wzory:")
print(f"  • DFT: X[k] = Σ x[n]·e^(-j2πkn/N)")
print(f"  • Widmo mocy: P(f) = |X(f)|²")
print(f"  • Rozdzielczość częstotliwości: Δf = fs/N")
print(f"\nZalety FFT:")
print(f"  • Identyfikacja składowych częstotliwościowych")
print(f"  • Wykrywanie harmonicznych")
print(f"  • Analiza filtrów i systemów")
print(f"  • Kompresja danych")
print("\n" + "="*70)
