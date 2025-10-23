#!/usr/bin/env python3
"""
Tutorial 4: Filtrowanie Sygnałów

TEORIA:
-------
Filtrowanie to proces modyfikacji sygnału poprzez wzmocnienie lub osłabienie
określonych składowych częstotliwościowych. Filtry są fundamentalnym narzędziem
w przetwarzaniu sygnałów.

1. TYPY FILTRÓW:

   a) FILTR DOLNOPRZEPUSTOWY (Low-Pass Filter, LPF):
      - Przepuszcza niskie częstotliwości (f < fc)
      - Tłumi wysokie częstotliwości (f > fc)
      - Zastosowania: usuwanie szumu, antyaliasing, wygładzanie
      
   b) FILTR GÓRNOPRZEPUSTOWY (High-Pass Filter, HPF):
      - Przepuszcza wysokie częstotliwości (f > fc)
      - Tłumi niskie częstotliwości (f < fc)
      - Zastosowania: usuwanie trendu, detekcja krawędzi
      
   c) FILTR PASMOWOPRZEPUSTOWY (Band-Pass Filter, BPF):
      - Przepuszcza częstotliwości w określonym paśmie (f1 < f < f2)
      - Tłumi częstotliwości poza pasmem
      - Zastosowania: selekcja sygnału, komunikacja radiowa
      
   d) FILTR PASMOWOZAPOROWY (Band-Stop/Notch Filter):
      - Tłumi częstotliwości w określonym paśmie
      - Przepuszcza częstotliwości poza pasmem
      - Zastosowania: usuwanie zakłóceń 50/60 Hz

2. CHARAKTERYSTYKA FILTRU:

   - Częstotliwość odcięcia (fc): Punkt gdzie sygnał jest tłumiony o -3 dB
   - Rząd filtru (n): Określa stromość charakterystyki
     * Wyższy rząd → stromi przejście, ale większe opóźnienie
   - Pasmo przejściowe: Zakres między pasmem przepustowym a zaporowym
   - Tłumienie w paśmie zaporowym: Określa skuteczność filtracji

3. RODZAJE FILTRÓW:

   a) FILTR BUTTERWORTHA:
      - Maksymalnie płaska charakterystyka w paśmie przepustowym
      - Łagodne przejście między pasmami
      - Brak tętnień w paśmie przepustowym
      
   b) FILTR CZEBYSZEWA typu I:
      - Tętnienia w paśmie przepustowym
      - Bardziej stromy spadek niż Butterworth
      - Lepsze tłumienie w paśmie zaporowym
      
   c) FILTR CZEBYSZEWA typu II:
      - Tętnienia w paśmie zaporowym
      - Płaska charakterystyka w paśmie przepustowym
      
   d) FILTR BESSELA:
      - Liniowa charakterystyka fazowa
      - Minimalne zniekształcenie fazowe
      - Najlepszy dla zachowania kształtu impulsu

4. FUNKCJA PRZEJŚCIA (Transfer Function):
   H(jω) = Vout(jω) / Vin(jω)
   
   Charakterystyka amplitudowa: |H(jω)|
   Charakterystyka fazowa: ∠H(jω)

5. ODPOWIEDŹ IMPULSOWA vs CZĘSTOTLIWOŚCIOWA:
   - Dziedzina czasu: splot z odpowiedzią impulsową h(t)
   - Dziedzina częstotliwości: mnożenie przez H(f)
   Związane transformatą Fouriera!

PRAKTYCZNE ZASTOSOWANIA:
-------------------------
- Audio: equalizery, usuwanie szumu
- Telekomunikacja: separacja kanałów
- Medycyna: filtrowanie EKG, EEG
- Radar: separacja sygnału od zakłóceń
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parametry
fs = 1000  # Częstotliwość próbkowania [Hz]
duration = 2.0  # Czas trwania [s]
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Generowanie sygnału testowego: suma trzech częstotliwości + szum
f1, f2, f3 = 5, 50, 150  # Hz
test_signal = (np.sin(2 * np.pi * f1 * t) + 
               0.5 * np.sin(2 * np.pi * f2 * t) + 
               0.3 * np.sin(2 * np.pi * f3 * t))
# Dodaj szum wysokoczęstotliwościowy
test_signal += 0.2 * np.random.randn(len(t))

# Funkcja do obliczania widma
def compute_spectrum(signal_data, fs):
    N = len(signal_data)
    fft_vals = np.fft.fft(signal_data * np.hanning(N))
    freqs = np.fft.fftfreq(N, 1/fs)
    amplitude = np.abs(fft_vals) * 2 / N
    positive_idx = freqs >= 0
    return freqs[positive_idx], amplitude[positive_idx]

# ============================================================================
# CZĘŚĆ 1: FILTRY DOLNOPRZEPUSTOWE
# ============================================================================

fig1, axes1 = plt.subplots(3, 2, figsize=(14, 10))
fig1.suptitle('Filtr Dolnoprzepustowy (Low-Pass Filter)', fontsize=16, fontweight='bold')

# Projektowanie filtru dolnoprzepustowego
fc_lowpass = 30  # Częstotliwość odcięcia [Hz]
order = 4  # Rząd filtru

# Butterworth
b_butter, a_butter = signal.butter(order, fc_lowpass, btype='low', fs=fs)
filtered_butter = signal.filtfilt(b_butter, a_butter, test_signal)

# Czebyszew typu I
b_cheby1, a_cheby1 = signal.cheby1(order, 1, fc_lowpass, btype='low', fs=fs)  # 1 dB ripple
filtered_cheby1 = signal.filtfilt(b_cheby1, a_cheby1, test_signal)

# Sygnał oryginalny
axes1[0, 0].plot(t[:500], test_signal[:500], 'b-', linewidth=1.5, alpha=0.7)
axes1[0, 0].set_title('Sygnał Oryginalny (5 Hz + 50 Hz + 150 Hz + szum)')
axes1[0, 0].set_ylabel('Amplituda')
axes1[0, 0].grid(True, alpha=0.3)
axes1[0, 0].text(0.02, 0.85, f'Zawiera:\n{f1} Hz, {f2} Hz,\n{f3} Hz + szum',
                 transform=axes1[0, 0].transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

# Widmo oryginalnego sygnału
freqs_orig, amp_orig = compute_spectrum(test_signal, fs)
axes1[0, 1].plot(freqs_orig[:250], amp_orig[:250], 'b-', linewidth=2)
axes1[0, 1].set_title('Widmo Sygnału Oryginalnego')
axes1[0, 1].set_ylabel('Amplituda')
axes1[0, 1].axvline(x=fc_lowpass, color='r', linestyle='--', label=f'fc = {fc_lowpass} Hz')
axes1[0, 1].legend()
axes1[0, 1].grid(True, alpha=0.3)

# Sygnał po filtracji (Butterworth)
axes1[1, 0].plot(t[:500], filtered_butter[:500], 'g-', linewidth=1.5)
axes1[1, 0].set_title('Po Filtrze Butterwortha')
axes1[1, 0].set_ylabel('Amplituda')
axes1[1, 0].grid(True, alpha=0.3)
axes1[1, 0].text(0.02, 0.85, f'fc = {fc_lowpass} Hz\nRząd: {order}\nPrzekazane:\n{f1} Hz',
                 transform=axes1[1, 0].transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Widmo przefiltrowanego sygnału
freqs_filt, amp_filt = compute_spectrum(filtered_butter, fs)
axes1[1, 1].plot(freqs_filt[:250], amp_filt[:250], 'g-', linewidth=2)
axes1[1, 1].set_title('Widmo Po Filtracji')
axes1[1, 1].set_ylabel('Amplituda')
axes1[1, 1].axvline(x=fc_lowpass, color='r', linestyle='--', label=f'fc = {fc_lowpass} Hz')
axes1[1, 1].legend()
axes1[1, 1].grid(True, alpha=0.3)

# Charakterystyka częstotliwościowa filtru
w_butter, h_butter = signal.freqz(b_butter, a_butter, worN=2000, fs=fs)
w_cheby1, h_cheby1 = signal.freqz(b_cheby1, a_cheby1, worN=2000, fs=fs)

axes1[2, 0].plot(w_butter, 20 * np.log10(abs(h_butter)), 'g-', linewidth=2, label='Butterworth')
axes1[2, 0].plot(w_cheby1, 20 * np.log10(abs(h_cheby1)), 'b--', linewidth=2, label='Czebyszew I')
axes1[2, 0].axhline(y=-3, color='r', linestyle=':', label='-3 dB')
axes1[2, 0].axvline(x=fc_lowpass, color='r', linestyle='--', alpha=0.5)
axes1[2, 0].set_title('Charakterystyka Amplitudowa Filtru')
axes1[2, 0].set_xlabel('Częstotliwość [Hz]')
axes1[2, 0].set_ylabel('Wzmocnienie [dB]')
axes1[2, 0].set_xlim(0, 200)
axes1[2, 0].set_ylim(-80, 5)
axes1[2, 0].legend()
axes1[2, 0].grid(True, alpha=0.3)

# Charakterystyka fazowa
axes1[2, 1].plot(w_butter, np.angle(h_butter, deg=True), 'g-', linewidth=2, label='Butterworth')
axes1[2, 1].plot(w_cheby1, np.angle(h_cheby1, deg=True), 'b--', linewidth=2, label='Czebyszew I')
axes1[2, 1].axvline(x=fc_lowpass, color='r', linestyle='--', alpha=0.5)
axes1[2, 1].set_title('Charakterystyka Fazowa Filtru')
axes1[2, 1].set_xlabel('Częstotliwość [Hz]')
axes1[2, 1].set_ylabel('Faza [stopnie]')
axes1[2, 1].set_xlim(0, 200)
axes1[2, 1].legend()
axes1[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tutorial_04_lowpass_filter.png', dpi=150, bbox_inches='tight')
print("✓ Wykres zapisany jako 'tutorial_04_lowpass_filter.png'")
plt.show()

# ============================================================================
# CZĘŚĆ 2: WSZYSTKIE TYPY FILTRÓW
# ============================================================================

fig2, axes2 = plt.subplots(4, 2, figsize=(14, 12))
fig2.suptitle('Porównanie Wszystkich Typów Filtrów', fontsize=16, fontweight='bold')

# Parametry filtrów
fc_low = 30  # Hz dla LPF
fc_high = 100  # Hz dla HPF
fc_band_low = 40  # Hz dla BPF
fc_band_high = 80  # Hz dla BPF
fc_notch = 50  # Hz dla notch
order_filt = 4

# 1. Filtr dolnoprzepustowy (Low-Pass)
b_lp, a_lp = signal.butter(order_filt, fc_low, btype='low', fs=fs)
signal_lp = signal.filtfilt(b_lp, a_lp, test_signal)
w_lp, h_lp = signal.freqz(b_lp, a_lp, worN=2000, fs=fs)

axes2[0, 0].plot(t[:500], signal_lp[:500], 'g-', linewidth=1.5)
axes2[0, 0].set_title('1. Filtr Dolnoprzepustowy (LPF)')
axes2[0, 0].set_ylabel('Amplituda')
axes2[0, 0].grid(True, alpha=0.3)
axes2[0, 0].text(0.02, 0.85, f'fc = {fc_low} Hz\nPrzepuszcza: f < fc',
                 transform=axes2[0, 0].transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

axes2[0, 1].plot(w_lp, 20 * np.log10(abs(h_lp)), 'g-', linewidth=2)
axes2[0, 1].axvline(x=fc_low, color='r', linestyle='--', label=f'fc = {fc_low} Hz')
axes2[0, 1].axhline(y=-3, color='orange', linestyle=':', label='-3 dB')
axes2[0, 1].set_title('Charakterystyka Częstotliwościowa')
axes2[0, 1].set_ylabel('Wzmocnienie [dB]')
axes2[0, 1].set_xlim(0, 200)
axes2[0, 1].set_ylim(-80, 5)
axes2[0, 1].legend()
axes2[0, 1].grid(True, alpha=0.3)

# 2. Filtr górnoprzepustowy (High-Pass)
b_hp, a_hp = signal.butter(order_filt, fc_high, btype='high', fs=fs)
signal_hp = signal.filtfilt(b_hp, a_hp, test_signal)
w_hp, h_hp = signal.freqz(b_hp, a_hp, worN=2000, fs=fs)

axes2[1, 0].plot(t[:500], signal_hp[:500], 'b-', linewidth=1.5)
axes2[1, 0].set_title('2. Filtr Górnoprzepustowy (HPF)')
axes2[1, 0].set_ylabel('Amplituda')
axes2[1, 0].grid(True, alpha=0.3)
axes2[1, 0].text(0.02, 0.85, f'fc = {fc_high} Hz\nPrzepuszcza: f > fc',
                 transform=axes2[1, 0].transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

axes2[1, 1].plot(w_hp, 20 * np.log10(abs(h_hp)), 'b-', linewidth=2)
axes2[1, 1].axvline(x=fc_high, color='r', linestyle='--', label=f'fc = {fc_high} Hz')
axes2[1, 1].axhline(y=-3, color='orange', linestyle=':', label='-3 dB')
axes2[1, 1].set_title('Charakterystyka Częstotliwościowa')
axes2[1, 1].set_ylabel('Wzmocnienie [dB]')
axes2[1, 1].set_xlim(0, 200)
axes2[1, 1].set_ylim(-80, 5)
axes2[1, 1].legend()
axes2[1, 1].grid(True, alpha=0.3)

# 3. Filtr pasmowoprzepustowy (Band-Pass)
b_bp, a_bp = signal.butter(order_filt, [fc_band_low, fc_band_high], btype='band', fs=fs)
signal_bp = signal.filtfilt(b_bp, a_bp, test_signal)
w_bp, h_bp = signal.freqz(b_bp, a_bp, worN=2000, fs=fs)

axes2[2, 0].plot(t[:500], signal_bp[:500], 'm-', linewidth=1.5)
axes2[2, 0].set_title('3. Filtr Pasmowoprzepustowy (BPF)')
axes2[2, 0].set_ylabel('Amplituda')
axes2[2, 0].grid(True, alpha=0.3)
axes2[2, 0].text(0.02, 0.85, f'Pasmo:\n{fc_band_low}-{fc_band_high} Hz',
                 transform=axes2[2, 0].transAxes,
                 bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7))

axes2[2, 1].plot(w_bp, 20 * np.log10(abs(h_bp)), 'm-', linewidth=2)
axes2[2, 1].axvline(x=fc_band_low, color='r', linestyle='--', alpha=0.5)
axes2[2, 1].axvline(x=fc_band_high, color='r', linestyle='--', alpha=0.5)
axes2[2, 1].axhline(y=-3, color='orange', linestyle=':', label='-3 dB')
axes2[2, 1].set_title('Charakterystyka Częstotliwościowa')
axes2[2, 1].set_ylabel('Wzmocnienie [dB]')
axes2[2, 1].set_xlim(0, 200)
axes2[2, 1].set_ylim(-80, 5)
axes2[2, 1].legend()
axes2[2, 1].grid(True, alpha=0.3)

# 4. Filtr pasmowozaporowy (Band-Stop/Notch)
b_bs, a_bs = signal.iirnotch(fc_notch, 30, fs)
signal_bs = signal.filtfilt(b_bs, a_bs, test_signal)
w_bs, h_bs = signal.freqz(b_bs, a_bs, worN=2000, fs=fs)

axes2[3, 0].plot(t[:500], signal_bs[:500], 'r-', linewidth=1.5)
axes2[3, 0].set_title('4. Filtr Pasmowozaporowy (Notch)')
axes2[3, 0].set_ylabel('Amplituda')
axes2[3, 0].set_xlabel('Czas [s]')
axes2[3, 0].grid(True, alpha=0.3)
axes2[3, 0].text(0.02, 0.85, f'Tłumi: {fc_notch} Hz\nUsuwa zakłócenia\nsieciowe',
                 transform=axes2[3, 0].transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

axes2[3, 1].plot(w_bs, 20 * np.log10(abs(h_bs)), 'r-', linewidth=2)
axes2[3, 1].axvline(x=fc_notch, color='g', linestyle='--', label=f'Notch = {fc_notch} Hz')
axes2[3, 1].set_title('Charakterystyka Częstotliwościowa')
axes2[3, 1].set_ylabel('Wzmocnienie [dB]')
axes2[3, 1].set_xlabel('Częstotliwość [Hz]')
axes2[3, 1].set_xlim(0, 200)
axes2[3, 1].set_ylim(-80, 5)
axes2[3, 1].legend()
axes2[3, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tutorial_04_all_filter_types.png', dpi=150, bbox_inches='tight')
print("✓ Wykres zapisany jako 'tutorial_04_all_filter_types.png'")
plt.show()

# ============================================================================
# CZĘŚĆ 3: WPŁYW RZĘDU FILTRU
# ============================================================================

fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8))
fig3.suptitle('Wpływ Rzędu Filtru na Charakterystykę', fontsize=16, fontweight='bold')

fc = 50  # Hz
orders = [2, 4, 6, 8]
colors = ['b', 'g', 'r', 'm']

# Porównanie charakterystyk dla różnych rzędów
for order, color in zip(orders, colors):
    b, a = signal.butter(order, fc, btype='low', fs=fs)
    w, h = signal.freqz(b, a, worN=2000, fs=fs)
    
    axes3[0, 0].plot(w, 20 * np.log10(abs(h)), color=color, linewidth=2, 
                     label=f'Rząd {order}', alpha=0.8)
    axes3[0, 1].plot(w, np.angle(h, deg=True), color=color, linewidth=2,
                     label=f'Rząd {order}', alpha=0.8)

axes3[0, 0].axvline(x=fc, color='k', linestyle='--', alpha=0.5, label=f'fc = {fc} Hz')
axes3[0, 0].axhline(y=-3, color='orange', linestyle=':', label='-3 dB')
axes3[0, 0].set_title('Charakterystyka Amplitudowa')
axes3[0, 0].set_xlabel('Częstotliwość [Hz]')
axes3[0, 0].set_ylabel('Wzmocnienie [dB]')
axes3[0, 0].set_xlim(0, 150)
axes3[0, 0].set_ylim(-100, 5)
axes3[0, 0].legend()
axes3[0, 0].grid(True, alpha=0.3)

axes3[0, 1].axvline(x=fc, color='k', linestyle='--', alpha=0.5, label=f'fc = {fc} Hz')
axes3[0, 1].set_title('Charakterystyka Fazowa')
axes3[0, 1].set_xlabel('Częstotliwość [Hz]')
axes3[0, 1].set_ylabel('Faza [stopnie]')
axes3[0, 1].set_xlim(0, 150)
axes3[0, 1].legend()
axes3[0, 1].grid(True, alpha=0.3)

# Odpowiedź impulsowa dla różnych rzędów
for i, (order, color) in enumerate(zip([2, 8], ['b', 'r'])):
    b, a = signal.butter(order, fc, btype='low', fs=fs)
    t_impulse = np.arange(100) / fs
    impulse = np.zeros(100)
    impulse[0] = 1
    impulse_response = signal.lfilter(b, a, impulse)
    
    axes3[1, i].stem(t_impulse * 1000, impulse_response, linefmt=f'{color}-', 
                     markerfmt=f'{color}o', basefmt=' ')
    axes3[1, i].set_title(f'Odpowiedź Impulsowa - Rząd {order}')
    axes3[1, i].set_xlabel('Czas [ms]')
    axes3[1, i].set_ylabel('Amplituda')
    axes3[1, i].grid(True, alpha=0.3)
    axes3[1, i].text(0.5, 0.85, f'Wyższy rząd →\nDłuższe osiadanie',
                     transform=axes3[1, i].transAxes,
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

plt.tight_layout()
plt.savefig('tutorial_04_filter_order.png', dpi=150, bbox_inches='tight')
print("✓ Wykres zapisany jako 'tutorial_04_filter_order.png'")
plt.show()

# Podsumowanie
print("\n" + "="*70)
print("PODSUMOWANIE TUTORIAL 4: FILTROWANIE SYGNAŁÓW")
print("="*70)
print(f"\nTypy filtrów:")
print(f"  • Low-Pass (LPF): przepuszcza f < fc")
print(f"  • High-Pass (HPF): przepuszcza f > fc")
print(f"  • Band-Pass (BPF): przepuszcza f1 < f < f2")
print(f"  • Band-Stop (Notch): tłumi określone pasmo")
print(f"\nParametry filtru:")
print(f"  • Częstotliwość odcięcia: {fc_low} Hz (LPF)")
print(f"  • Punkt -3 dB: gdzie amplituda spada o 3 dB (70.7%)")
print(f"  • Rząd filtru: {order_filt} → stromość charakterystyki")
print(f"\nRodzaje filtrów:")
print(f"  • Butterworth: maksymalnie płaska charakterystyka")
print(f"  • Czebyszew I: tętnienia w paśmie przepustowym")
print(f"  • Czebyszew II: tętnienia w paśmie zaporowym")
print(f"  • Bessel: liniowa faza, minimalne zniekształcenia")
print(f"\nZastosowania praktyczne:")
print(f"  • Audio: equalizer, usuwanie szumów")
print(f"  • EKG/EEG: separacja zakłóceń sieciowych (50/60 Hz)")
print(f"  • Telekomunikacja: separacja kanałów")
print(f"  • Antyaliasing: przed próbkowaniem ADC")
print("\n" + "="*70)
