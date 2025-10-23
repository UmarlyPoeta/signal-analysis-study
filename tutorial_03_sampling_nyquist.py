#!/usr/bin/env python3
"""
Tutorial 3: Próbkowanie i Twierdzenie Nyquista-Shannona

TEORIA:
-------
Próbkowanie to proces konwersji sygnału ciągłego w czasie na dyskretny zbiór wartości
w określonych odstępach czasu. Jest to kluczowy krok w cyfrowym przetwarzaniu sygnałów.

1. TWIERDZENIE NYQUISTA-SHANNONA:
   Aby możliwa była dokładna rekonstrukcja sygnału analogowego z jego próbek,
   częstotliwość próbkowania musi być co najmniej dwukrotnie większa od
   najwyższej częstotliwości obecnej w sygnale:
   
   fs ≥ 2 · fmax
   
   gdzie:
   - fs: częstotliwość próbkowania [Hz]
   - fmax: maksymalna częstotliwość w sygnale [Hz]
   - fNyquist = fs/2: częstotliwość Nyquista [Hz]

2. ALIASING (Zakładkowanie):
   Gdy fs < 2·fmax, występuje zjawisko aliasingu - wysokie częstotliwości
   "odbijają się" i pojawiają jako niższe częstotliwości w próbkowanym sygnale.
   
   Częstotliwość pozorna (alias): falias = |f - n·fs|
   gdzie n jest liczbą całkowitą taką, aby falias < fs/2

3. CZĘSTOTLIWOŚĆ PRÓBKOWANIA:
   - Podpróbkowanie (undersampling): fs < 2·fmax → aliasing
   - Próbkowanie krytyczne: fs = 2·fmax → teoretycznie wystarczające
   - Nadpróbkowanie (oversampling): fs >> 2·fmax → zapas bezpieczeństwa
   
4. REKONSTRUKCJA SYGNAŁU:
   Teoretycznie, sygnał można idealnie zrekonstruować używając interpolacji sinc:
   
   x(t) = Σ x[n] · sinc((t - nTs) / Ts)
   
   gdzie Ts = 1/fs to okres próbkowania.
   
   W praktyce używa się filtrów dolnoprzepustowych (low-pass filters).

5. FILTR ANTYALIASINGOWY:
   Przed próbkowaniem stosuje się analogowy filtr dolnoprzepustowy,
   aby usunąć składowe powyżej fNyquist i zapobiec aliasingowi.

PRAKTYCZNE ZASTOSOWANIA:
-------------------------
- Audio CD: fs = 44.1 kHz (dla sygnału do ~20 kHz)
- Telefonia: fs = 8 kHz (dla sygnału do ~3.4 kHz)
- DAB radio: fs = 48 kHz
- Radar: fs zależy od maksymalnej częstotliwości Dopplera
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy.interpolate import interp1d

# Parametry sygnału oryginalnego
f_signal = 5  # Częstotliwość sygnału [Hz]
duration = 2.0  # Czas trwania [s]
fs_original = 1000  # Bardzo wysoka częstotliwość do symulacji sygnału ciągłego [Hz]

# Generowanie "sygnału ciągłego" (bardzo gęsto próbkowanego)
t_continuous = np.linspace(0, duration, int(fs_original * duration), endpoint=False)
signal_continuous = np.sin(2 * np.pi * f_signal * t_continuous)

# Demonstracja 1: Różne częstotliwości próbkowania
fig1, axes1 = plt.subplots(4, 1, figsize=(12, 10))
fig1.suptitle('Wpływ Częstotliwości Próbkowania na Reprezentację Sygnału', 
              fontsize=16, fontweight='bold')

sampling_rates = [50, 20, 10, 8]  # Hz
colors = ['green', 'blue', 'orange', 'red']
color_codes = ['g', 'b', 'C1', 'r']  # matplotlib color codes
labels = ['Nadpróbkowanie (fs=50Hz)', 'Dobre próbkowanie (fs=20Hz)', 
          'Próbkowanie krytyczne (fs=10Hz)', 'Podpróbkowanie (fs=8Hz)']

for i, (fs_sample, color, color_code, label) in enumerate(zip(sampling_rates, colors, color_codes, labels)):
    # Próbkowanie
    n_samples = int(fs_sample * duration)
    t_sampled = np.linspace(0, duration, n_samples, endpoint=False)
    signal_sampled = np.sin(2 * np.pi * f_signal * t_sampled)
    
    # Wykres
    axes1[i].plot(t_continuous, signal_continuous, 'k-', linewidth=1, alpha=0.3, label='Sygnał oryginalny')
    axes1[i].stem(t_sampled, signal_sampled, linefmt=color_code+'-', markerfmt=color_code+'o', 
                  basefmt=' ', label='Próbki')
    axes1[i].plot(t_sampled, signal_sampled, color=color_code, linestyle='--', linewidth=1, alpha=0.5)
    axes1[i].set_ylabel('Amplituda')
    axes1[i].grid(True, alpha=0.3)
    axes1[i].legend(loc='upper right')
    axes1[i].set_xlim(0, 1)
    
    # Informacja o spełnieniu twierdzenia Nyquista
    nyquist_check = "✓ DOBRA" if fs_sample >= 2 * f_signal else "✗ ALIASING"
    color_box = 'lightgreen' if fs_sample >= 2 * f_signal else 'lightcoral'
    axes1[i].text(0.02, 0.75, f'{label}\nfs/2 = {fs_sample/2} Hz\nf_signal = {f_signal} Hz\n{nyquist_check}',
                  transform=axes1[i].transAxes,
                  bbox=dict(boxstyle='round', facecolor=color_box, alpha=0.7))

axes1[3].set_xlabel('Czas [s]')
plt.tight_layout()
plt.savefig('tutorial_03_sampling_rates.png', dpi=150, bbox_inches='tight')
print("✓ Wykres zapisany jako 'tutorial_03_sampling_rates.png'")
plt.show()

# Demonstracja 2: Aliasing - podpróbkowanie sygnału wysokiej częstotliwości
fig2, axes2 = plt.subplots(3, 2, figsize=(14, 10))
fig2.suptitle('Demonstracja Zjawiska Aliasingu', fontsize=16, fontweight='bold')

# Sygnał wysokiej częstotliwości
f_high = 25  # Hz
signal_high = np.sin(2 * np.pi * f_high * t_continuous)

# Przypadek 1: Prawidłowe próbkowanie
fs_good = 60  # Hz (> 2*25)
t_good = np.linspace(0, duration, int(fs_good * duration), endpoint=False)
signal_good = np.sin(2 * np.pi * f_high * t_good)

# Przypadek 2: Podpróbkowanie powodujące aliasing
fs_bad = 30  # Hz (< 2*25)
t_bad = np.linspace(0, duration, int(fs_bad * duration), endpoint=False)
signal_bad = np.sin(2 * np.pi * f_high * t_bad)

# Obliczenie częstotliwości aliasu
f_alias = abs(f_high - fs_bad)

# Wykresy - Prawidłowe próbkowanie
axes2[0, 0].plot(t_continuous[:500], signal_high[:500], 'k-', linewidth=1, alpha=0.4, 
                 label=f'Oryginalny sygnał ({f_high} Hz)')
axes2[0, 0].stem(t_good[:30], signal_good[:30], linefmt='g-', markerfmt='go', basefmt=' ',
                 label=f'Próbki (fs={fs_good} Hz)')
axes2[0, 0].set_title('PRAWIDŁOWE Próbkowanie (fs > 2·f)')
axes2[0, 0].set_ylabel('Amplituda')
axes2[0, 0].legend()
axes2[0, 0].grid(True, alpha=0.3)
axes2[0, 0].set_xlim(0, 0.5)

# Widmo - Prawidłowe próbkowanie
fft_good = np.fft.fft(signal_good * np.hanning(len(signal_good)))
freqs_good = np.fft.fftfreq(len(signal_good), 1/fs_good)
axes2[0, 1].stem(freqs_good[:len(freqs_good)//2], np.abs(fft_good[:len(fft_good)//2]), 
                 linefmt='g-', markerfmt='go', basefmt=' ')
axes2[0, 1].set_title('Widmo - Pik przy Prawidłowej Częstotliwości')
axes2[0, 1].set_ylabel('Amplituda')
axes2[0, 1].axvline(x=f_high, color='r', linestyle='--', label=f'f = {f_high} Hz')
axes2[0, 1].axvline(x=fs_good/2, color='orange', linestyle='--', label=f'fNyquist = {fs_good/2} Hz')
axes2[0, 1].legend()
axes2[0, 1].grid(True, alpha=0.3)
axes2[0, 1].set_xlim(0, fs_good/2 + 10)

# Wykresy - Podpróbkowanie (aliasing)
axes2[1, 0].plot(t_continuous[:500], signal_high[:500], 'k-', linewidth=1, alpha=0.4,
                 label=f'Oryginalny sygnał ({f_high} Hz)')
axes2[1, 0].stem(t_bad[:15], signal_bad[:15], linefmt='r-', markerfmt='ro', basefmt=' ',
                 label=f'Próbki (fs={fs_bad} Hz)')
# Pokaż sygnał aliasu
signal_alias_continuous = np.sin(2 * np.pi * f_alias * t_continuous)
axes2[1, 0].plot(t_continuous[:500], signal_alias_continuous[:500], 'r--', linewidth=2, 
                 alpha=0.7, label=f'Sygnał aliasu ({f_alias} Hz)')
axes2[1, 0].set_title('PODPRÓBKOWANIE (fs < 2·f) - Występuje ALIASING!')
axes2[1, 0].set_ylabel('Amplituda')
axes2[1, 0].legend()
axes2[1, 0].grid(True, alpha=0.3)
axes2[1, 0].set_xlim(0, 0.5)

# Widmo - Podpróbkowanie
fft_bad = np.fft.fft(signal_bad * np.hanning(len(signal_bad)))
freqs_bad = np.fft.fftfreq(len(signal_bad), 1/fs_bad)
axes2[1, 1].stem(freqs_bad[:len(freqs_bad)//2], np.abs(fft_bad[:len(fft_bad)//2]),
                 linefmt='r-', markerfmt='ro', basefmt=' ')
axes2[1, 1].set_title('Widmo - Pik przy BŁĘDNEJ Częstotliwości (alias)')
axes2[1, 1].set_ylabel('Amplituda')
axes2[1, 1].axvline(x=f_alias, color='r', linestyle='--', label=f'falias = {f_alias} Hz')
axes2[1, 1].axvline(x=fs_bad/2, color='orange', linestyle='--', label=f'fNyquist = {fs_bad/2} Hz')
axes2[1, 1].legend()
axes2[1, 1].grid(True, alpha=0.3)
axes2[1, 1].set_xlim(0, fs_bad/2 + 5)

# Porównanie - oba przypadki razem
axes2[2, 0].plot(t_continuous[:500], signal_high[:500], 'k-', linewidth=2, alpha=0.5,
                 label=f'Sygnał oryginalny ({f_high} Hz)')
markerline, stemlines, baseline = axes2[2, 0].stem(t_good[:30], signal_good[:30], linefmt='g-', markerfmt='go', basefmt=' ',
                 label=f'Dobre próbkowanie (fs={fs_good} Hz)')
stemlines.set_alpha(0.6)
markerline.set_alpha(0.6)
markerline2, stemlines2, baseline2 = axes2[2, 0].stem(t_bad[:15], signal_bad[:15], linefmt='r-', markerfmt='ro', basefmt=' ',
                 label=f'Podpróbkowanie (fs={fs_bad} Hz)')
stemlines2.set_alpha(0.6)
markerline2.set_alpha(0.6)
axes2[2, 0].set_title('Porównanie: Prawidłowe vs Podpróbkowanie')
axes2[2, 0].set_xlabel('Czas [s]')
axes2[2, 0].set_ylabel('Amplituda')
axes2[2, 0].legend()
axes2[2, 0].grid(True, alpha=0.3)
axes2[2, 0].set_xlim(0, 0.5)

# Ilustracja wzoru aliasingu
axes2[2, 1].axis('off')
info_text = f"""
ANALIZA ALIASINGU:

Sygnał oryginalny: f = {f_high} Hz
Częstotliwość próbkowania: fs = {fs_bad} Hz
Częstotliwość Nyquista: fNyquist = fs/2 = {fs_bad/2} Hz

❌ Warunek Nyquista NIE spełniony:
   fs = {fs_bad} Hz < 2·f = {2*f_high} Hz

Obliczenie częstotliwości aliasu:
   falias = |f - fs| = |{f_high} - {fs_bad}| = {f_alias} Hz

Sygnał {f_high} Hz jest interpretowany jako {f_alias} Hz!

ROZWIĄZANIE:
✓ Zwiększyć fs do minimum {2*f_high} Hz
✓ Użyć filtru antyaliasingowego przed próbkowaniem
"""
axes2[2, 1].text(0.1, 0.5, info_text, transform=axes2[2, 1].transAxes,
                 fontsize=11, verticalalignment='center', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('tutorial_03_aliasing_demo.png', dpi=150, bbox_inches='tight')
print("✓ Wykres zapisany jako 'tutorial_03_aliasing_demo.png'")
plt.show()

# Demonstracja 3: Rekonstrukcja sygnału
fig3, axes3 = plt.subplots(2, 1, figsize=(12, 8))
fig3.suptitle('Rekonstrukcja Sygnału z Próbek', fontsize=16, fontweight='bold')

# Sygnał testowy
f_test = 3  # Hz
signal_test = np.sin(2 * np.pi * f_test * t_continuous)

# Próbkowanie
fs_recon = 20  # Hz
t_recon = np.linspace(0, duration, int(fs_recon * duration), endpoint=False)
signal_recon = np.sin(2 * np.pi * f_test * t_recon)

# Różne metody interpolacji
interp_linear = interp1d(t_recon, signal_recon, kind='linear', fill_value='extrapolate')
interp_cubic = interp1d(t_recon, signal_recon, kind='cubic', fill_value='extrapolate')

# Rekonstrukcja
signal_linear = interp_linear(t_continuous)
signal_cubic = interp_cubic(t_continuous)

# Wykres 1
axes3[0].plot(t_continuous, signal_test, 'k-', linewidth=2, label='Sygnał oryginalny', alpha=0.7)
axes3[0].stem(t_recon[:20], signal_recon[:20], linefmt='b-', markerfmt='bo', basefmt=' ',
              label=f'Próbki (fs={fs_recon} Hz)')
axes3[0].plot(t_continuous, signal_linear, 'g--', linewidth=2, label='Interpolacja liniowa', alpha=0.7)
axes3[0].set_title('Rekonstrukcja - Interpolacja Liniowa')
axes3[0].set_ylabel('Amplituda')
axes3[0].legend()
axes3[0].grid(True, alpha=0.3)
axes3[0].set_xlim(0, 1)

# Wykres 2
axes3[1].plot(t_continuous, signal_test, 'k-', linewidth=2, label='Sygnał oryginalny', alpha=0.7)
axes3[1].stem(t_recon[:20], signal_recon[:20], linefmt='b-', markerfmt='bo', basefmt=' ',
              label=f'Próbki (fs={fs_recon} Hz)')
axes3[1].plot(t_continuous, signal_cubic, 'r--', linewidth=2, label='Interpolacja kubiczna', alpha=0.7)
axes3[1].set_title('Rekonstrukcja - Interpolacja Kubiczna (lepsza)')
axes3[1].set_ylabel('Amplituda')
axes3[1].set_xlabel('Czas [s]')
axes3[1].legend()
axes3[1].grid(True, alpha=0.3)
axes3[1].set_xlim(0, 1)

plt.tight_layout()
plt.savefig('tutorial_03_reconstruction.png', dpi=150, bbox_inches='tight')
print("✓ Wykres zapisany jako 'tutorial_03_reconstruction.png'")
plt.show()

# Podsumowanie
print("\n" + "="*70)
print("PODSUMOWANIE TUTORIAL 3: PRÓBKOWANIE I TWIERDZENIE NYQUISTA")
print("="*70)
print(f"\nTwierdzenie Nyquista-Shannona:")
print(f"  • Minimalna częstotliwość próbkowania: fs ≥ 2·fmax")
print(f"  • Częstotliwość Nyquista: fNyquist = fs/2")
print(f"\nPrzykład z demonstracji:")
print(f"  • Częstotliwość sygnału: f = {f_high} Hz")
print(f"  • Minimalna wymagana fs: {2*f_high} Hz")
print(f"  • Użyta fs (dobra): {fs_good} Hz → brak aliasingu ✓")
print(f"  • Użyta fs (zła): {fs_bad} Hz → aliasing! ✗")
print(f"  • Częstotliwość aliasu: {f_alias} Hz")
print(f"\nTypowe częstotliwości próbkowania:")
print(f"  • Audio CD: 44.1 kHz (sygnał do ~20 kHz)")
print(f"  • Telefonia: 8 kHz (sygnał do ~3.4 kHz)")
print(f"  • Audio profesjonalne: 48 kHz, 96 kHz, 192 kHz")
print(f"\nMetody zapobiegania aliasingowi:")
print(f"  • Zwiększenie częstotliwości próbkowania")
print(f"  • Filtr antyaliasingowy (analogowy, przed ADC)")
print(f"  • Nadpróbkowanie + decymacja")
print("\n" + "="*70)
