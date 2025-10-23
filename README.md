# Signal Analysis Study - Nauka Analizy Sygnałów

Kompleksowy zestaw tutoriali w języku Python do nauki analizy sygnałów. Każdy skrypt działa jako interaktywny tutorial z wytłumaczeniem teorii matematycznej i praktycznymi przykładami.

## 📚 Zawartość

### Tutorial 1: Podstawowe Sygnały
**Plik:** `tutorial_01_basic_signals.py`

Wprowadzenie do podstawowych typów sygnałów używanych w analizie:
- Sygnał sinusoidalny i cosinusoidalny
- Sygnał prostokątny (square wave)
- Sygnał trójkątny (triangle wave)
- Sygnał piłokształtny (sawtooth wave)
- Szum biały (white noise)

**Teoria:** Wzory matematyczne, parametry sygnałów (amplituda, częstotliwość, okres), zastosowania praktyczne.

### Tutorial 2: Transformata Fouriera
**Plik:** `tutorial_02_fourier_transform.py`

Analiza sygnałów w dziedzinie częstotliwości:
- Dyskretna Transformata Fouriera (DFT)
- Szybka Transformata Fouriera (FFT)
- Widmo amplitudowe i mocy
- Widmo fazowe
- Analiza harmonicznych

**Teoria:** Wzory DFT/FFT, interpretacja widma, rozdzielczość częstotliwościowa, twierdzenie o splocie.

### Tutorial 3: Próbkowanie i Twierdzenie Nyquista
**Plik:** `tutorial_03_sampling_nyquist.py`

Konwersja sygnałów analogowych na cyfrowe:
- Twierdzenie Nyquista-Shannona
- Zjawisko aliasingu (zakładkowanie)
- Częstotliwość próbkowania
- Rekonstrukcja sygnału
- Filtr antyaliasingowy

**Teoria:** Warunek Nyquista (fs ≥ 2·fmax), obliczanie częstotliwości aliasu, metody interpolacji.

### Tutorial 4: Filtrowanie Sygnałów
**Plik:** `tutorial_04_filtering.py`

Projektowanie i zastosowanie filtrów cyfrowych:
- Filtr dolnoprzepustowy (Low-Pass Filter)
- Filtr górnoprzepustowy (High-Pass Filter)
- Filtr pasmowoprzepustowy (Band-Pass Filter)
- Filtr pasmowozaporowy (Notch Filter)
- Rodzaje filtrów: Butterworth, Czebyszew, Bessel

**Teoria:** Charakterystyki częstotliwościowe, rząd filtru, częstotliwość odcięcia, odpowiedź impulsowa.

### Tutorial 5: Kwantyzacja ADC/DAC i ENOB
**Plik:** `tutorial_05_adc_quantization.py`

Przetworniki analogowo-cyfrowe i cyfrowo-analogowe:
- Proces kwantyzacji
- Błąd kwantyzacji
- SNR teoretyczny (6.02·b + 1.76 dB)
- ENOB (Effective Number of Bits)
- Wpływ liczby bitów i amplitudy

**Teoria:** Poziomy kwantyzacji, LSB, szum kwantyzacji, rozdzielczość ADC.

### Tutorial 6: Metryki Jakości Sygnału
**Plik:** `tutorial_06_signal_metrics.py`

Ocena jakości systemów przetwarzania sygnałów:
- SNR (Signal-to-Noise Ratio)
- THD (Total Harmonic Distortion)
- SINAD (Signal-to-Noise And Distortion)
- SFDR (Spurious-Free Dynamic Range)
- ENOB (Effective Number of Bits)

**Teoria:** Definicje matematyczne, interpretacja wyników, wartości praktyczne, zastosowania.

## 🚀 Instalacja

### Wymagania
- Python 3.7 lub nowszy
- pip (menedżer pakietów Python)

### Instalacja zależności

```bash
pip install -r requirements.txt
```

Lub ręcznie:
```bash
pip install numpy scipy matplotlib pandas
```

## 💻 Uruchamianie

Każdy tutorial można uruchomić niezależnie:

```bash
python tutorial_01_basic_signals.py
python tutorial_02_fourier_transform.py
python tutorial_03_sampling_nyquist.py
python tutorial_04_filtering.py
python tutorial_05_adc_quantization.py
python tutorial_06_signal_metrics.py
```

Każdy skrypt:
1. Wyświetli interaktywne wykresy
2. Zapisze wykresy jako pliki PNG
3. Wypisze podsumowanie z kluczowymi wzorami i wynikami

## 📊 Wygenerowane Wykresy

Po uruchomieniu tutoriali, w katalogu pojawią się następujące pliki graficzne:

### Tutorial 1:
- `tutorial_01_basic_signals.png` - Wszystkie podstawowe typy sygnałów
- `tutorial_01_frequency_comparison.png` - Porównanie różnych częstotliwości

### Tutorial 2:
- `tutorial_02_fourier_transform.png` - Analiza FFT różnych sygnałów
- `tutorial_02_power_spectrum.png` - Widmo mocy sygnału złożonego

### Tutorial 3:
- `tutorial_03_sampling_rates.png` - Wpływ częstotliwości próbkowania
- `tutorial_03_aliasing_demo.png` - Demonstracja zjawiska aliasingu
- `tutorial_03_reconstruction.png` - Rekonstrukcja sygnału z próbek

### Tutorial 4:
- `tutorial_04_lowpass_filter.png` - Filtr dolnoprzepustowy
- `tutorial_04_all_filter_types.png` - Wszystkie typy filtrów
- `tutorial_04_filter_order.png` - Wpływ rzędu filtru

### Tutorial 5:
- `tutorial_05_quantization_bits.png` - Kwantyzacja dla różnych bitów
- `tutorial_05_snr_enob_analysis.png` - Analiza SNR i ENOB
- `tutorial_05_amplitude_effect.png` - Wpływ amplitudy na SNR

### Tutorial 6:
- `tutorial_06_signal_metrics.png` - Porównanie metryk dla różnych sygnałów
- `tutorial_06_metrics_analysis.png` - Analiza wpływu zniekształceń
- `tutorial_06_adc_metrics.png` - Metryki dla różnych rozdzielczości ADC

## 📖 Kluczowe Wzory

### Transformata Fouriera
```
DFT: X[k] = Σ x[n]·e^(-j2πkn/N)
SNR = 10·log₁₀(Psignal / Pnoise) [dB]
```

### Próbkowanie Nyquista
```
fs ≥ 2·fmax
fNyquist = fs/2
```

### Kwantyzacja
```
SNR = 6.02·b + 1.76 [dB]
ENOB = (SINAD - 1.76) / 6.02
LSB = Vref / 2^b
```

### Metryki
```
THD = 10·log₁₀(ΣP_harmonics / P_fundamental) [dB]
SINAD = 10·log₁₀(Psignal / (Pnoise + Pdistortion)) [dB]
```

## 🎯 Zastosowania Praktyczne

- **Audio:** Analiza jakości dźwięku, projektowanie equalizerów
- **Telekomunikacja:** Analiza kanałów, modulacja, demodulacja
- **Pomiary:** Oscyloskopy, multimetry cyfrowe
- **RF/Radar:** Analiza widma, detekcja sygnałów
- **Medycyna:** Przetwarzanie EKG, EEG, obrazowanie medyczne
- **IoT:** Czujniki, akwizycja danych

## 📝 Dodatkowe Skrypty

### automated_rf_adc_dac.py
Automatyczne testy parametrów ADC/DAC z różnymi konfiguracjami:
- Testowanie różnych głębokości bitowych (8, 10, 12 bitów)
- Analiza wpływu szumu
- Generowanie raportów CSV i wykresów

## 🔧 Wymagane Pakiety

- **NumPy:** Operacje numeryczne, tablice, matematyka
- **SciPy:** FFT, filtry cyfrowe, przetwarzanie sygnałów
- **Matplotlib:** Wizualizacja danych, wykresy
- **Pandas:** Analiza danych, eksport CSV

## 📚 Źródła i Literatura

1. Oppenheim, A. V., & Schafer, R. W. (2009). *Discrete-Time Signal Processing*
2. Proakis, J. G., & Manolakis, D. G. (2006). *Digital Signal Processing*
3. Smith, S. W. (1997). *The Scientist and Engineer's Guide to Digital Signal Processing*
4. Lyons, R. G. (2004). *Understanding Digital Signal Processing*

## 🤝 Współpraca

Projekt edukacyjny dla studiowania analizy sygnałów. Każdy tutorial zawiera:
- Szczegółowe komentarze w kodzie
- Wyjaśnienia matematyczne
- Przykłady praktyczne
- Wizualizacje

## 📄 Licencja

Ten projekt jest dostępny do celów edukacyjnych.

## ✨ Autor

Projekt stworzony jako materiał edukacyjny do nauki przetwarzania sygnałów cyfrowych.

## 🐛 Zgłaszanie problemów

W przypadku problemów z uruchomieniem skryptów:
1. Sprawdź wersję Pythona: `python --version` (wymagany ≥ 3.7)
2. Zainstaluj ponownie zależności: `pip install -r requirements.txt`
3. Upewnij się, że wszystkie pakiety są aktualne

## 📞 Kontakt

Pytania i sugestie mile widziane!

---

**Miłej nauki analizy sygnałów! 🎓📊**
