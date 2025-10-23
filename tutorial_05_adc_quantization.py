#!/usr/bin/env python3
"""
Tutorial 5: Kwantyzacja ADC/DAC i ENOB

TEORIA:
-------
Przetworniki ADC (Analog-to-Digital Converter) i DAC (Digital-to-Analog Converter)
są mostami między światem analogowym a cyfrowym. Kwantyzacja jest kluczowym
procesem powodującym stratę informacji.

1. KWANTYZACJA:
   Proces przypisania wartości ciągłej do jednej z skończonej liczby
   wartości dyskretnych (poziomów kwantyzacji).
   
   Liczba poziomów: N = 2^b
   gdzie b to liczba bitów ADC
   
   Przykłady:
   - 8-bit ADC: 256 poziomów
   - 12-bit ADC: 4096 poziomów
   - 16-bit ADC: 65536 poziomów

2. BŁĄD KWANTYZACJI:
   Różnica między wartością rzeczywistą a kwantyzowaną:
   e[n] = x[n] - xq[n]
   
   Maksymalny błąd: ±LSB/2
   gdzie LSB (Least Significant Bit) = Vref / (2^b)
   
   Dla sygnału sinusoidalnego, błąd kwantyzacji ma moc:
   Pq = (LSB)² / 12 = (Vref / 2^b)² / 12

3. STOSUNEK SYGNAŁ/SZUM KWANTYZACJI (SNR):
   SNR = 6.02·b + 1.76 [dB]
   
   Każdy dodatkowy bit zwiększa SNR o ~6 dB (dwukrotnie)
   
   Dla sygnału sinusoidalnego o pełnej skali:
   SNR = 10·log₁₀(Psignal / Pquant)

4. ENOB (Effective Number of Bits):
   Rzeczywista rozdzielczość ADC uwzględniająca wszystkie niedoskonałości:
   szum, nieliniowość, zniekształcenia
   
   ENOB = (SINAD - 1.76) / 6.02
   
   gdzie SINAD to stosunek sygnału do szumu i zniekształceń
   
   Przykład: 12-bit ADC może mieć ENOB = 10.5 bitów

5. PARAMETRY ADC:
   
   a) SNR (Signal-to-Noise Ratio):
      Stosunek mocy sygnału do mocy szumu
      SNR = 10·log₁₀(Psignal / Pnoise) [dB]
   
   b) THD (Total Harmonic Distortion):
      Stosunek mocy harmonicznych do mocy podstawowej
      THD = 10·log₁₀(Σ P_harmonics / P_fundamental) [dB]
   
   c) SINAD (Signal-to-Noise And Distortion):
      Stosunek mocy sygnału do mocy szumu + zniekształceń
      SINAD = 10·log₁₀(Psignal / (Pnoise + Pdistortion)) [dB]
   
   d) SFDR (Spurious-Free Dynamic Range):
      Różnica między sygnałem a najsilniejszą składową pasożytniczą

6. TYPY KWANTYZACJI:
   - Uniform (równomierna): stałe LSB
   - Non-uniform: zmienne LSB (np. μ-law, A-law w telekomunikacji)
   - Midrise: zero nie jest poziomem kwantyzacji
   - Midtread: zero jest poziomem kwantyzacji

PRAKTYCZNE ZASTOSOWANIA:
-------------------------
- Audio: 16-bit (CD), 24-bit (studio), 32-bit float
- Video: 8-bit, 10-bit, 12-bit
- Pomiary: 12-bit, 16-bit, 24-bit
- Oscyloskopy: 8-bit (podstawowe), 12-bit (wysokiej klasy)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig

# Funkcja kwantyzacji
def quantize(signal_data, bits, v_ref=1.0):
    """
    Kwantyzuje sygnał do określonej liczby bitów.
    
    Args:
        signal_data: Sygnał wejściowy
        bits: Liczba bitów ADC
        v_ref: Napięcie referencyjne (zakres ±v_ref)
    
    Returns:
        Skwantyzowany sygnał
    """
    levels = 2 ** bits
    # Normalizacja do zakresu ±v_ref
    signal_normalized = np.clip(signal_data, -v_ref, v_ref)
    # Kwantyzacja
    lsb = (2 * v_ref) / levels
    quantized = np.round(signal_normalized / lsb) * lsb
    return quantized

# Funkcja do obliczania SNR
def calculate_snr(original, quantized):
    """Oblicza SNR między sygnałem oryginalnym a skwantyzowanym."""
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - quantized) ** 2)
    if noise_power == 0:
        return np.inf
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db

# Parametry sygnału
fs = 10000  # Hz
duration = 0.1  # s
f_signal = 100  # Hz
amplitude = 0.9  # Amplituda (poniżej pełnej skali)

t = np.linspace(0, duration, int(fs * duration), endpoint=False)
signal_original = amplitude * np.sin(2 * np.pi * f_signal * t)

# ============================================================================
# CZĘŚĆ 1: WPŁYW LICZBY BITÓW NA KWANTYZACJĘ
# ============================================================================

fig1, axes1 = plt.subplots(4, 2, figsize=(14, 12))
fig1.suptitle('Kwantyzacja ADC - Wpływ Liczby Bitów', fontsize=16, fontweight='bold')

bit_depths = [3, 6, 10, 16]
colors = ['r', 'orange', 'g', 'b']

for i, (bits, color) in enumerate(zip(bit_depths, colors)):
    # Kwantyzacja
    signal_quant = quantize(signal_original, bits)
    error = signal_original - signal_quant
    snr = calculate_snr(signal_original, signal_quant)
    snr_theoretical = 6.02 * bits + 1.76
    
    # Wykres sygnału
    time_window = 50  # próbek do wyświetlenia
    axes1[i, 0].plot(t[:time_window] * 1000, signal_original[:time_window], 
                     'k--', linewidth=1, alpha=0.5, label='Oryginalny')
    axes1[i, 0].plot(t[:time_window] * 1000, signal_quant[:time_window], 
                     color=color, linewidth=2, marker='o', markersize=3, label='Skwantyzowany')
    axes1[i, 0].set_title(f'{bits}-bit ADC (N = {2**bits} poziomów)')
    axes1[i, 0].set_ylabel('Amplituda [V]')
    if i == 3:
        axes1[i, 0].set_xlabel('Czas [ms]')
    axes1[i, 0].legend()
    axes1[i, 0].grid(True, alpha=0.3)
    
    # Informacje
    lsb = 2.0 / (2 ** bits)
    axes1[i, 0].text(0.02, 0.75, 
                     f'LSB = {lsb*1000:.3f} mV\nSNR = {snr:.1f} dB\nTeor. = {snr_theoretical:.1f} dB',
                     transform=axes1[i, 0].transAxes,
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Wykres błędu kwantyzacji
    axes1[i, 1].plot(t[:time_window] * 1000, error[:time_window] * 1000, 
                     color=color, linewidth=1.5)
    axes1[i, 1].axhline(y=lsb/2 * 1000, color='r', linestyle='--', alpha=0.5, label='±LSB/2')
    axes1[i, 1].axhline(y=-lsb/2 * 1000, color='r', linestyle='--', alpha=0.5)
    axes1[i, 1].set_title(f'Błąd Kwantyzacji')
    axes1[i, 1].set_ylabel('Błąd [mV]')
    if i == 3:
        axes1[i, 1].set_xlabel('Czas [ms]')
    axes1[i, 1].legend()
    axes1[i, 1].grid(True, alpha=0.3)
    axes1[i, 1].set_ylim(-lsb*1.5*1000, lsb*1.5*1000)

plt.tight_layout()
plt.savefig('tutorial_05_quantization_bits.png', dpi=150, bbox_inches='tight')
print("✓ Wykres zapisany jako 'tutorial_05_quantization_bits.png'")
plt.show()

# ============================================================================
# CZĘŚĆ 2: SNR vs LICZBA BITÓW (WERYFIKACJA TEORII)
# ============================================================================

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
fig2.suptitle('Analiza SNR i ENOB', fontsize=16, fontweight='bold')

# Zakres bitów do testowania
bits_range = np.arange(4, 17)
snr_measured = []
snr_theoretical = []

for bits in bits_range:
    signal_quant = quantize(signal_original, bits)
    snr = calculate_snr(signal_original, signal_quant)
    snr_measured.append(snr)
    snr_theoretical.append(6.02 * bits + 1.76)

# Wykres SNR vs liczba bitów
axes2[0, 0].plot(bits_range, snr_measured, 'bo-', linewidth=2, markersize=8, label='Zmierzone')
axes2[0, 0].plot(bits_range, snr_theoretical, 'r--', linewidth=2, label='Teoretyczne (6.02·b + 1.76)')
axes2[0, 0].set_title('SNR vs Liczba Bitów ADC')
axes2[0, 0].set_xlabel('Liczba bitów')
axes2[0, 0].set_ylabel('SNR [dB]')
axes2[0, 0].legend()
axes2[0, 0].grid(True, alpha=0.3)
axes2[0, 0].text(0.05, 0.85, 'Każdy bit dodaje\n~6 dB SNR',
                 transform=axes2[0, 0].transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

# ENOB dla różnych wartości SINAD
sinad_range = np.arange(30, 100, 5)
enob_values = (sinad_range - 1.76) / 6.02

axes2[0, 1].plot(sinad_range, enob_values, 'g-', linewidth=2, marker='s', markersize=6)
axes2[0, 1].set_title('ENOB vs SINAD')
axes2[0, 1].set_xlabel('SINAD [dB]')
axes2[0, 1].set_ylabel('ENOB [bity]')
axes2[0, 1].grid(True, alpha=0.3)
axes2[0, 1].text(0.05, 0.75, 'ENOB = (SINAD - 1.76) / 6.02',
                 transform=axes2[0, 1].transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Widmo błędu kwantyzacji dla różnych bitów
for bits, color, label in [(8, 'r', '8-bit'), (12, 'b', '12-bit')]:
    signal_quant = quantize(signal_original, bits)
    error = signal_original - signal_quant
    
    # FFT błędu
    N = len(error)
    fft_error = np.fft.fft(error * np.hanning(N))
    freqs = np.fft.fftfreq(N, 1/fs)
    magnitude = np.abs(fft_error) * 2 / N
    
    positive_idx = freqs >= 0
    axes2[1, 0].plot(freqs[positive_idx][:500], 20*np.log10(magnitude[positive_idx][:500] + 1e-10),
                     color=color, linewidth=2, label=label, alpha=0.7)

axes2[1, 0].set_title('Widmo Szumu Kwantyzacji')
axes2[1, 0].set_xlabel('Częstotliwość [Hz]')
axes2[1, 0].set_ylabel('Moc [dB]')
axes2[1, 0].legend()
axes2[1, 0].grid(True, alpha=0.3)
axes2[1, 0].set_xlim(0, 2000)

# Ilustracja poziomów kwantyzacji
bits_demo = 4
levels = 2 ** bits_demo
level_values = np.linspace(-1, 1, levels)
input_range = np.linspace(-1, 1, 1000)
output_quantized = quantize(input_range, bits_demo)

axes2[1, 1].plot(input_range, input_range, 'k--', linewidth=1, label='Idealny (nieskończona rozdzielczość)')
axes2[1, 1].plot(input_range, output_quantized, 'r-', linewidth=2, label=f'{bits_demo}-bit kwantyzacja')
axes2[1, 1].set_title(f'Charakterystyka Przenoszenia ADC ({bits_demo}-bit)')
axes2[1, 1].set_xlabel('Wejście [V]')
axes2[1, 1].set_ylabel('Wyjście [V]')
axes2[1, 1].legend()
axes2[1, 1].grid(True, alpha=0.3)
axes2[1, 1].set_aspect('equal')

plt.tight_layout()
plt.savefig('tutorial_05_snr_enob_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Wykres zapisany jako 'tutorial_05_snr_enob_analysis.png'")
plt.show()

# ============================================================================
# CZĘŚĆ 3: WPŁYW AMPLITUDY SYGNAŁU
# ============================================================================

fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8))
fig3.suptitle('Wpływ Amplitudy Sygnału na SNR', fontsize=16, fontweight='bold')

# Różne amplitudy
amplitudes = np.linspace(0.1, 1.0, 20)
bits_test = 12

snr_vs_amplitude = []
for amp in amplitudes:
    sig_test = amp * np.sin(2 * np.pi * f_signal * t)
    sig_quant = quantize(sig_test, bits_test)
    snr = calculate_snr(sig_test, sig_quant)
    snr_vs_amplitude.append(snr)

# SNR vs Amplituda
axes3[0, 0].plot(amplitudes * 100, snr_vs_amplitude, 'b-', linewidth=2, marker='o')
axes3[0, 0].set_title(f'SNR vs Amplituda Sygnału ({bits_test}-bit ADC)')
axes3[0, 0].set_xlabel('Amplituda [% pełnej skali]')
axes3[0, 0].set_ylabel('SNR [dB]')
axes3[0, 0].grid(True, alpha=0.3)
axes3[0, 0].text(0.05, 0.85, 'Większa amplituda\n→ lepszy SNR',
                 transform=axes3[0, 0].transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

# Przykłady dla małej i dużej amplitudy
amp_small = 0.2
amp_large = 0.9
sig_small = amp_small * np.sin(2 * np.pi * f_signal * t)
sig_large = amp_large * np.sin(2 * np.pi * f_signal * t)
sig_small_quant = quantize(sig_small, bits_test)
sig_large_quant = quantize(sig_large, bits_test)

time_window = 50
axes3[0, 1].plot(t[:time_window] * 1000, sig_small[:time_window], 'r--', 
                 linewidth=1, alpha=0.5, label='Oryginalny')
axes3[0, 1].plot(t[:time_window] * 1000, sig_small_quant[:time_window], 'r-', 
                 linewidth=2, marker='o', markersize=3, label='Skwantyzowany')
axes3[0, 1].set_title(f'Mała Amplituda ({amp_small*100:.0f}% FS)')
axes3[0, 1].set_ylabel('Amplituda [V]')
axes3[0, 1].legend()
axes3[0, 1].grid(True, alpha=0.3)
snr_small = calculate_snr(sig_small, sig_small_quant)
axes3[0, 1].text(0.5, 0.15, f'SNR = {snr_small:.1f} dB',
                 transform=axes3[0, 1].transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

axes3[1, 0].plot(t[:time_window] * 1000, sig_large[:time_window], 'g--',
                 linewidth=1, alpha=0.5, label='Oryginalny')
axes3[1, 0].plot(t[:time_window] * 1000, sig_large_quant[:time_window], 'g-',
                 linewidth=2, marker='o', markersize=3, label='Skwantyzowany')
axes3[1, 0].set_title(f'Duża Amplituda ({amp_large*100:.0f}% FS)')
axes3[1, 0].set_xlabel('Czas [ms]')
axes3[1, 0].set_ylabel('Amplituda [V]')
axes3[1, 0].legend()
axes3[1, 0].grid(True, alpha=0.3)
snr_large = calculate_snr(sig_large, sig_large_quant)
axes3[1, 0].text(0.5, 0.15, f'SNR = {snr_large:.1f} dB',
                 transform=axes3[1, 0].transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# Tabela porównawcza
axes3[1, 1].axis('off')
comparison_text = f"""
PORÓWNANIE PARAMETRÓW:

Parametr                  Wartość
{'='*45}
Częstotliwość próbkowania: {fs} Hz
Rozdzielczość ADC:         {bits_test} bitów
Liczba poziomów:           {2**bits_test}
LSB:                       {2.0/(2**bits_test)*1000:.3f} mV

SNR TEORETYCZNY:           {6.02*bits_test + 1.76:.1f} dB

WYNIKI POMIARÓW:
Amplituda {amp_small*100:.0f}% FS:       SNR = {snr_small:.1f} dB
Amplituda {amp_large*100:.0f}% FS:       SNR = {snr_large:.1f} dB

WNIOSEK:
Wykorzystanie pełnego zakresu ADC
maksymalizuje SNR i ENOB.

Niedostateczne wzmocnienie sygnału
przed ADC pogarsza jakość pomiaru!
"""
axes3[1, 1].text(0.1, 0.5, comparison_text, transform=axes3[1, 1].transAxes,
                 fontsize=10, verticalalignment='center', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('tutorial_05_amplitude_effect.png', dpi=150, bbox_inches='tight')
print("✓ Wykres zapisany jako 'tutorial_05_amplitude_effect.png'")
plt.show()

# Podsumowanie
print("\n" + "="*70)
print("PODSUMOWANIE TUTORIAL 5: KWANTYZACJA ADC/DAC I ENOB")
print("="*70)
print(f"\nPodstawowe wzory:")
print(f"  • Liczba poziomów: N = 2^b")
print(f"  • LSB = Vref / 2^b")
print(f"  • SNR = 6.02·b + 1.76 [dB]")
print(f"  • ENOB = (SINAD - 1.76) / 6.02")
print(f"\nPrzykład dla {bits_test}-bit ADC:")
print(f"  • Liczba poziomów: {2**bits_test}")
print(f"  • SNR teoretyczny: {6.02*bits_test + 1.76:.1f} dB")
print(f"  • LSB (dla Vref=2V): {2.0/(2**bits_test)*1000:.3f} mV")
print(f"  • SNR zmierzony (90% FS): {snr_large:.1f} dB")
print(f"\nWażne obserwacje:")
print(f"  • Każdy bit dodaje ~6 dB SNR")
print(f"  • Amplituda wpływa na SNR: większa amplituda → lepszy SNR")
print(f"  • ENOB < liczba bitów (z powodu niedoskonałości realnych ADC)")
print(f"  • Błąd kwantyzacji ≤ ±LSB/2")
print(f"\nTypowe zastosowania:")
print(f"  • Audio CD: 16-bit (SNR ≈ 98 dB)")
print(f"  • Audio Studio: 24-bit (SNR ≈ 146 dB)")
print(f"  • Oscyloskopy: 8-12 bit")
print(f"  • Multimetry: 16-24 bit")
print("\n" + "="*70)
