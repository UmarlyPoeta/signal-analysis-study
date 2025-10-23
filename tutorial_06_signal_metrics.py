#!/usr/bin/env python3
"""
Tutorial 6: Metryki Jakości Sygnału (SNR, THD, SINAD, SFDR)

TEORIA:
-------
Metryki jakości sygnału są kluczowe w ocenie wydajności systemów przetwarzania
sygnałów, szczególnie przetworników ADC/DAC, wzmacniaczy i systemów komunikacyjnych.

1. SNR (Signal-to-Noise Ratio) - Stosunek Sygnału do Szumu:
   
   SNR = 10·log₁₀(Psignal / Pnoise) [dB]
   
   gdzie:
   - Psignal: moc sygnału użytecznego
   - Pnoise: moc szumu
   
   Interpretacja:
   - SNR > 60 dB: bardzo dobra jakość
   - SNR 40-60 dB: dobra jakość
   - SNR 20-40 dB: akceptowalna jakość
   - SNR < 20 dB: słaba jakość
   
   Zastosowania:
   - Ocena jakości audio/wideo
   - Wydajność ADC/DAC
   - Systemy komunikacyjne

2. THD (Total Harmonic Distortion) - Zniekształcenia Harmoniczne:
   
   THD = 10·log₁₀(ΣP_harmonics / P_fundamental) [dB]
   
   lub w procentach: THD% = sqrt(ΣP_harmonics / P_fundamental) × 100%
   
   Harmoniczne to składowe o częstotliwościach n·f₀ (n = 2, 3, 4, ...)
   wynikające z nieliniowości systemu.
   
   Typowe wartości:
   - THD < -80 dB (<0.01%): doskonałe (Hi-Fi audio)
   - THD -60 do -80 dB (0.01-0.1%): bardzo dobre
   - THD -40 do -60 dB (0.1-1%): dobre
   - THD > -40 dB (>1%): słabe
   
   Zastosowania:
   - Ocena nieliniowości wzmacniaczy
   - Jakość generatorów sygnałów
   - Charakterystyka ADC/DAC

3. SINAD (Signal-to-Noise And Distortion):
   
   SINAD = 10·log₁₀(Psignal / (Pnoise + Pdistortion)) [dB]
   
   Uwzględnia zarówno szum jak i zniekształcenia.
   Ściśle związany z ENOB:
   
   ENOB = (SINAD - 1.76) / 6.02
   
   SINAD jest bardziej praktyczną miarą niż SNR, ponieważ uwzględnia
   wszystkie niedoskonałości systemu.

4. SFDR (Spurious-Free Dynamic Range):
   
   SFDR = 10·log₁₀(Psignal / Pmax_spurious) [dB]
   
   gdzie Pmax_spurious to moc najsilniejszej składowej pasożytniczej
   (harmonicznej lub intermodulacyjnej).
   
   Określa zakres dynamiki wolny od składowych niepożądanych.
   Kluczowy parametr w systemach komunikacyjnych.

5. ZWIĄZKI MIĘDZY METRYKAMI:
   
   - SNR ≥ SINAD (SINAD uwzględnia więcej zakłóceń)
   - SINAD ≈ SNR gdy THD jest bardzo małe
   - SFDR zwykle > THD (najsilniejsza harmoniczna vs suma wszystkich)
   
6. TECHNIKI POMIARU:
   
   W dziedzinie częstotliwości (FFT):
   - Identyfikacja piku sygnału podstawowego
   - Identyfikacja harmonicznych (2f, 3f, 4f, ...)
   - Obliczenie mocy w odpowiednich binach
   - Zastosowanie okna (Hanning, Hamming) dla dokładności

PRAKTYCZNE ZASTOSOWANIA:
-------------------------
- Specyfikacja przetworników ADC/DAC
- Testy wzmacniaczy audio
- Kontrola jakości systemów komunikacyjnych
- Certyfikacja sprzętu Hi-Fi
- Diagnostyka systemów RF
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig

# Funkcje pomocnicze
def compute_fft_metrics(signal_data, fs, f_fundamental, n_harmonics=10):
    """
    Oblicza metryki jakości sygnału w dziedzinie częstotliwości.
    
    Returns:
        dict: Słownik z metrykami (SNR, THD, SINAD, SFDR)
    """
    N = len(signal_data)
    # Okno Hanninga dla lepszej rozdzielczości
    windowed = signal_data * np.hanning(N)
    
    # FFT
    fft_vals = np.fft.fft(windowed)
    freqs = np.fft.fftfreq(N, 1/fs)
    power_spectrum = np.abs(fft_vals) ** 2
    
    # Znajdź pik podstawowy
    positive_freqs = freqs[:N//2]
    positive_power = power_spectrum[:N//2]
    
    # Indeks częstotliwości podstawowej
    fund_idx = np.argmin(np.abs(positive_freqs - f_fundamental))
    
    # Moc sygnału podstawowego (z sąsiednimi binami dla dokładności)
    signal_bins = range(max(0, fund_idx-2), min(len(positive_power), fund_idx+3))
    signal_power = np.sum(positive_power[signal_bins])
    
    # Moc harmonicznych
    harmonic_power = 0
    harmonic_frequencies = []
    harmonic_powers = []
    
    for n in range(2, n_harmonics + 1):
        harm_freq = n * f_fundamental
        if harm_freq < fs/2:  # Tylko poniżej Nyquista
            harm_idx = np.argmin(np.abs(positive_freqs - harm_freq))
            harm_bins = range(max(0, harm_idx-1), min(len(positive_power), harm_idx+2))
            harm_pow = np.sum(positive_power[harm_bins])
            harmonic_power += harm_pow
            harmonic_frequencies.append(harm_freq)
            harmonic_powers.append(harm_pow)
    
    # Całkowita moc
    total_power = np.sum(positive_power)
    
    # Moc szumu (wszystko oprócz sygnału)
    noise_power = total_power - signal_power
    
    # Moc szumu + zniekształcenia
    noise_and_distortion = noise_power
    
    # Metryki
    snr_db = 10 * np.log10(signal_power / max(noise_power - harmonic_power, 1e-12))
    thd_db = 10 * np.log10(harmonic_power / max(signal_power, 1e-12))
    sinad_db = 10 * np.log10(signal_power / max(noise_and_distortion, 1e-12))
    
    # SFDR - najsilniejsza składowa pasożytnicza
    # Usuń sygnał podstawowy i znajdź maksimum
    spurious_power = positive_power.copy()
    spurious_power[signal_bins] = 0
    max_spurious = np.max(spurious_power)
    sfdr_db = 10 * np.log10(signal_power / max(max_spurious, 1e-12))
    
    # ENOB
    enob = (sinad_db - 1.76) / 6.02
    
    return {
        'SNR': snr_db,
        'THD': thd_db,
        'SINAD': sinad_db,
        'SFDR': sfdr_db,
        'ENOB': enob,
        'harmonic_freqs': harmonic_frequencies,
        'harmonic_powers': harmonic_powers,
        'freqs': positive_freqs,
        'power_spectrum': positive_power,
        'signal_power': signal_power,
        'noise_power': noise_power,
        'harmonic_power': harmonic_power
    }

# ============================================================================
# CZĘŚĆ 1: DEMONSTRACJA METRYK DLA RÓŻNYCH SYGNAŁÓW
# ============================================================================

fs = 10000  # Hz
duration = 0.5  # s
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
f0 = 100  # Hz - częstotliwość podstawowa

# Sygnał 1: Idealny sinus (tylko szum termiczny)
signal_ideal = np.sin(2 * np.pi * f0 * t) + 0.001 * np.random.randn(len(t))

# Sygnał 2: Sinus z harmonicznymi (nieliniowość)
signal_distorted = (np.sin(2 * np.pi * f0 * t) + 
                   0.1 * np.sin(2 * np.pi * 2*f0 * t) +
                   0.05 * np.sin(2 * np.pi * 3*f0 * t))
signal_distorted += 0.001 * np.random.randn(len(t))

# Sygnał 3: Sinus z dużym szumem
signal_noisy = np.sin(2 * np.pi * f0 * t) + 0.1 * np.random.randn(len(t))

# Sygnał 4: Kwantyzowany (jak z ADC)
signal_quantized = np.sin(2 * np.pi * f0 * t)
# Kwantyzacja 8-bit
bits = 8
levels = 2 ** bits
signal_quantized = np.round(signal_quantized * (levels/2)) / (levels/2)

# Oblicz metryki dla wszystkich sygnałów
metrics_ideal = compute_fft_metrics(signal_ideal, fs, f0)
metrics_distorted = compute_fft_metrics(signal_distorted, fs, f0)
metrics_noisy = compute_fft_metrics(signal_noisy, fs, f0)
metrics_quantized = compute_fft_metrics(signal_quantized, fs, f0)

fig1, axes1 = plt.subplots(4, 2, figsize=(14, 12))
fig1.suptitle('Metryki Jakości Sygnału - Porównanie Różnych Przypadków', 
              fontsize=16, fontweight='bold')

signals = [
    (signal_ideal, metrics_ideal, 'Sygnał Idealny (minimalny szum)', 'green'),
    (signal_distorted, metrics_distorted, 'Sygnał ze Zniekształceniami (THD)', 'orange'),
    (signal_noisy, metrics_noisy, 'Sygnał Zaszumiony (niski SNR)', 'red'),
    (signal_quantized, metrics_quantized, 'Sygnał Skwantyzowany (8-bit ADC)', 'blue')
]

for i, (sig_data, metrics, title, color) in enumerate(signals):
    # Wykres w dziedzinie czasu
    axes1[i, 0].plot(t[:200] * 1000, sig_data[:200], color=color, linewidth=1.5)
    axes1[i, 0].set_title(title)
    axes1[i, 0].set_ylabel('Amplituda')
    if i == 3:
        axes1[i, 0].set_xlabel('Czas [ms]')
    axes1[i, 0].grid(True, alpha=0.3)
    
    # Metryki
    metrics_text = (f"SNR: {metrics['SNR']:.1f} dB\n"
                   f"THD: {metrics['THD']:.1f} dB\n"
                   f"SINAD: {metrics['SINAD']:.1f} dB\n"
                   f"SFDR: {metrics['SFDR']:.1f} dB\n"
                   f"ENOB: {metrics['ENOB']:.2f} bit")
    axes1[i, 0].text(0.62, 0.65, metrics_text, transform=axes1[i, 0].transAxes,
                    fontsize=9, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Wykres widma mocy
    freqs = metrics['freqs']
    power = metrics['power_spectrum']
    axes1[i, 1].semilogy(freqs[:1000], power[:1000], color=color, linewidth=1.5)
    axes1[i, 1].set_title('Widmo Mocy')
    axes1[i, 1].set_ylabel('Moc [skala log]')
    if i == 3:
        axes1[i, 1].set_xlabel('Częstotliwość [Hz]')
    axes1[i, 1].grid(True, alpha=0.3, which='both')
    axes1[i, 1].set_xlim(0, 1000)
    
    # Zaznacz częstotliwość podstawową i harmoniczne
    axes1[i, 1].axvline(x=f0, color='blue', linestyle='--', alpha=0.5, linewidth=2)
    for harm_freq in metrics['harmonic_freqs'][:5]:
        if harm_freq < 1000:
            axes1[i, 1].axvline(x=harm_freq, color='red', linestyle=':', alpha=0.3)

plt.tight_layout()
plt.savefig('tutorial_06_signal_metrics.png', dpi=150, bbox_inches='tight')
print("✓ Wykres zapisany jako 'tutorial_06_signal_metrics.png'")
plt.show()

# ============================================================================
# CZĘŚĆ 2: WPŁYW POZIOMU ZNIEKSZTAŁCEŃ NA METRYKI
# ============================================================================

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
fig2.suptitle('Analiza Wpływu Zniekształceń na Metryki', fontsize=16, fontweight='bold')

# Zakres zniekształceń do testowania
distortion_levels = np.linspace(0, 0.5, 20)  # amplituda 2. harmonicznej

snr_vals = []
thd_vals = []
sinad_vals = []
sfdr_vals = []

for dist_level in distortion_levels:
    sig_test = (np.sin(2 * np.pi * f0 * t) + 
                dist_level * np.sin(2 * np.pi * 2*f0 * t) +
                0.01 * np.random.randn(len(t)))
    
    metrics = compute_fft_metrics(sig_test, fs, f0)
    snr_vals.append(metrics['SNR'])
    thd_vals.append(metrics['THD'])
    sinad_vals.append(metrics['SINAD'])
    sfdr_vals.append(metrics['SFDR'])

# Wykres metryk vs poziom zniekształceń
axes2[0, 0].plot(distortion_levels * 100, snr_vals, 'b-', linewidth=2, marker='o', label='SNR')
axes2[0, 0].plot(distortion_levels * 100, sinad_vals, 'r--', linewidth=2, marker='s', label='SINAD')
axes2[0, 0].set_title('SNR i SINAD vs Poziom Zniekształceń')
axes2[0, 0].set_xlabel('Amplituda 2. harmonicznej [%]')
axes2[0, 0].set_ylabel('Wartość [dB]')
axes2[0, 0].legend()
axes2[0, 0].grid(True, alpha=0.3)
axes2[0, 0].text(0.5, 0.15, 'SINAD < SNR gdy\nrosną zniekształcenia',
                transform=axes2[0, 0].transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

axes2[0, 1].plot(distortion_levels * 100, thd_vals, 'g-', linewidth=2, marker='^')
axes2[0, 1].set_title('THD vs Poziom Zniekształceń')
axes2[0, 1].set_xlabel('Amplituda 2. harmonicznej [%]')
axes2[0, 1].set_ylabel('THD [dB]')
axes2[0, 1].grid(True, alpha=0.3)
axes2[0, 1].axhline(y=-60, color='orange', linestyle=':', label='Próg -60 dB')
axes2[0, 1].axhline(y=-80, color='red', linestyle=':', label='Próg -80 dB (Hi-Fi)')
axes2[0, 1].legend()

# Analiza spektralna dla wybranych poziomów zniekształceń
for dist, color, label in [(0.05, 'g', 'Małe (5%)'), (0.3, 'r', 'Duże (30%)')]:
    sig_test = (np.sin(2 * np.pi * f0 * t) + 
                dist * np.sin(2 * np.pi * 2*f0 * t) +
                0.01 * np.random.randn(len(t)))
    
    metrics = compute_fft_metrics(sig_test, fs, f0)
    axes2[1, 0].semilogy(metrics['freqs'][:500], metrics['power_spectrum'][:500],
                        color=color, linewidth=2, label=label, alpha=0.7)

axes2[1, 0].axvline(x=f0, color='blue', linestyle='--', alpha=0.5, linewidth=2, label=f'f₀={f0} Hz')
axes2[1, 0].axvline(x=2*f0, color='orange', linestyle=':', alpha=0.5, linewidth=2, label=f'2f₀')
axes2[1, 0].set_title('Widmo - Porównanie Zniekształceń')
axes2[1, 0].set_xlabel('Częstotliwość [Hz]')
axes2[1, 0].set_ylabel('Moc [skala log]')
axes2[1, 0].legend()
axes2[1, 0].grid(True, alpha=0.3, which='both')
axes2[1, 0].set_xlim(0, 500)

# Tabela interpretacji
axes2[1, 1].axis('off')
interpretation_text = """
INTERPRETACJA METRYK:

SNR (Signal-to-Noise Ratio):
  > 60 dB:  Doskonała jakość
  40-60 dB: Bardzo dobra
  20-40 dB: Dobra
  < 20 dB:  Słaba

THD (Total Harmonic Distortion):
  < -80 dB: Hi-Fi (< 0.01%)
  -60 to -80 dB: Bardzo dobre
  -40 to -60 dB: Dobre
  > -40 dB: Słabe (> 1%)

SINAD vs SNR:
  • SINAD ≤ SNR zawsze
  • SINAD ≈ SNR gdy THD niskie
  • SINAD << SNR gdy duże zniekształcenia

SFDR (Spurious-Free Dynamic Range):
  • Ważny w systemach RF
  • Określa wolny zakres dynamiki
  • Zwykle SFDR > THD

ENOB (Effective Number of Bits):
  • ENOB = (SINAD - 1.76) / 6.02
  • ENOB < liczba bitów ADC
  • Uwzględnia wszystkie niedoskonałości
"""
axes2[1, 1].text(0.1, 0.5, interpretation_text, transform=axes2[1, 1].transAxes,
                fontsize=9, verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('tutorial_06_metrics_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Wykres zapisany jako 'tutorial_06_metrics_analysis.png'")
plt.show()

# ============================================================================
# CZĘŚĆ 3: PRAKTYCZNE PRZYKŁADY Z RÓŻNYMI BITAMI ADC
# ============================================================================

fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8))
fig3.suptitle('Metryki dla Różnych Rozdzielczości ADC', fontsize=16, fontweight='bold')

# Generowanie sygnału testowego
sig_test = 0.9 * np.sin(2 * np.pi * f0 * t)

bit_depths = [6, 8, 12, 16]
colors_bits = ['red', 'orange', 'green', 'blue']
metrics_by_bits = {}

for bits, color in zip(bit_depths, colors_bits):
    # Kwantyzacja
    levels = 2 ** bits
    sig_quant = np.round(sig_test * (levels/2)) / (levels/2)
    
    # Oblicz metryki
    metrics = compute_fft_metrics(sig_quant, fs, f0)
    metrics_by_bits[bits] = metrics
    
    # Wykres ENOB
    axes3[0, 0].bar(bits, metrics['ENOB'], color=color, alpha=0.7, width=1.5)

axes3[0, 0].plot(bit_depths, bit_depths, 'k--', linewidth=2, label='Idealna (ENOB = bity)')
axes3[0, 0].set_title('ENOB dla Różnych Rozdzielczości ADC')
axes3[0, 0].set_xlabel('Liczba bitów ADC')
axes3[0, 0].set_ylabel('ENOB [bity]')
axes3[0, 0].legend()
axes3[0, 0].grid(True, alpha=0.3)
axes3[0, 0].text(0.5, 0.85, 'ENOB < bity\nz powodu\nkwantyzacji',
                transform=axes3[0, 0].transAxes,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

# Wykres SNR, SINAD
axes3[0, 1].plot(bit_depths, [metrics_by_bits[b]['SNR'] for b in bit_depths], 
                'bo-', linewidth=2, markersize=8, label='SNR')
axes3[0, 1].plot(bit_depths, [metrics_by_bits[b]['SINAD'] for b in bit_depths],
                'rs--', linewidth=2, markersize=8, label='SINAD')
axes3[0, 1].plot(bit_depths, [6.02*b + 1.76 for b in bit_depths],
                'k:', linewidth=2, label='Teoretyczne SNR')
axes3[0, 1].set_title('SNR i SINAD vs Liczba Bitów')
axes3[0, 1].set_xlabel('Liczba bitów ADC')
axes3[0, 1].set_ylabel('[dB]')
axes3[0, 1].legend()
axes3[0, 1].grid(True, alpha=0.3)

# Widmo dla 8-bit vs 12-bit
for bits, color, label in [(8, 'orange', '8-bit'), (12, 'green', '12-bit')]:
    levels = 2 ** bits
    sig_quant = np.round(sig_test * (levels/2)) / (levels/2)
    metrics = compute_fft_metrics(sig_quant, fs, f0)
    
    axes3[1, 0].semilogy(metrics['freqs'][:500], metrics['power_spectrum'][:500],
                        color=color, linewidth=2, label=label, alpha=0.7)

axes3[1, 0].axvline(x=f0, color='blue', linestyle='--', alpha=0.5, linewidth=2)
axes3[1, 0].set_title('Widmo Mocy - Porównanie Rozdzielczości')
axes3[1, 0].set_xlabel('Częstotliwość [Hz]')
axes3[1, 0].set_ylabel('Moc [skala log]')
axes3[1, 0].legend()
axes3[1, 0].grid(True, alpha=0.3, which='both')
axes3[1, 0].set_xlim(0, 500)
axes3[1, 0].text(0.5, 0.85, 'Wyższa rozdzielczość\n→ niższy szum kwantyzacji',
                transform=axes3[1, 0].transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Tabela podsumowująca
axes3[1, 1].axis('off')
table_text = "PODSUMOWANIE METRYK:\n\n"
table_text += f"{'Bity':<6} {'SNR':>6} {'THD':>6} {'SINAD':>6} {'ENOB':>6}\n"
table_text += "="*40 + "\n"
for bits in bit_depths:
    m = metrics_by_bits[bits]
    table_text += f"{bits:<6} {m['SNR']:>6.1f} {m['THD']:>6.1f} {m['SINAD']:>6.1f} {m['ENOB']:>6.2f}\n"

table_text += "\nWNIOSKI:\n"
table_text += "• Każdy bit dodaje ~6 dB SNR\n"
table_text += "• ENOB zawsze < liczba bitów\n"
table_text += "• SINAD ≈ SNR dla czystego sinusa\n"
table_text += "• THD wynika z kwantyzacji\n"
table_text += "\nAPLIKACJE:\n"
table_text += "• 8-bit: proste pomiary\n"
table_text += "• 12-bit: standardowe zastosowania\n"
table_text += "• 16-bit: audio, precyzyjne pomiary\n"
table_text += "• >16-bit: instrumentacja naukowa"

axes3[1, 1].text(0.1, 0.5, table_text, transform=axes3[1, 1].transAxes,
                fontsize=9, verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('tutorial_06_adc_metrics.png', dpi=150, bbox_inches='tight')
print("✓ Wykres zapisany jako 'tutorial_06_adc_metrics.png'")
plt.show()

# Podsumowanie
print("\n" + "="*70)
print("PODSUMOWANIE TUTORIAL 6: METRYKI JAKOŚCI SYGNAŁU")
print("="*70)
print(f"\nDefinicje metryk:")
print(f"  • SNR = 10·log₁₀(Psignal / Pnoise)")
print(f"  • THD = 10·log₁₀(ΣP_harmonics / P_fundamental)")
print(f"  • SINAD = 10·log₁₀(Psignal / (Pnoise + Pdistortion))")
print(f"  • SFDR = 10·log₁₀(Psignal / Pmax_spurious)")
print(f"  • ENOB = (SINAD - 1.76) / 6.02")
print(f"\nRelacje:")
print(f"  • SINAD ≤ SNR (zawsze)")
print(f"  • SINAD ≈ SNR gdy THD bardzo niskie")
print(f"  • SFDR > THD (typowo)")
print(f"\nPrzykładowe wyniki (8-bit ADC):")
m8 = metrics_by_bits[8]
print(f"  • SNR: {m8['SNR']:.1f} dB")
print(f"  • THD: {m8['THD']:.1f} dB")
print(f"  • SINAD: {m8['SINAD']:.1f} dB")
print(f"  • ENOB: {m8['ENOB']:.2f} bitów")
print(f"\nPraktyczne wytyczne:")
print(f"  • Audio Hi-Fi: THD < -80 dB, SNR > 90 dB")
print(f"  • Instrumentacja: ENOB > 90% liczby bitów")
print(f"  • RF systemy: SFDR > 60 dB")
print(f"  • Pomiary: SNR > 60 dB dla dokładności 0.1%")
print("\n" + "="*70)
