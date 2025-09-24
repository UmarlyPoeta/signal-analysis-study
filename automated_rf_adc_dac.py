import numpy as np
import pandas as pd
import os
import math
from scipy import fftpack
import matplotlib.pyplot as plt

# dodatkowe funkcje 
def generate_signal(freq=1e3, fs=100e3, duration=0.01, amp=1.0, harmonics=None, noise_std=0.0):
    t = np.arange(0, duration, 1.0/fs)
    sig = np.zeros_like(t)
    sig += amp * np.sin(2*np.pi*freq*t)
    if harmonics:
        for h_amp, h_mult in harmonics:
            sig += h_amp * np.sin(2*np.pi*freq*h_mult*t)
    if noise_std>0:
        sig += np.random.normal(scale=noise_std, size=t.shape)
    return t, sig
def quantize(signal, bits):
    # uni midrise quantizer symmetric 0
    q_levels = 2**bits
    vmax = np.max(np.abs(signal)) if np.max(np.abs(signal))>0 else 1.0
    # scale signal -> [-q_levels/2, q_levels/2]
    scale = (q_levels/2 - 1)
    scaled = signal / vmax * scale
    q = np.round(scaled)
    dq = q / scale * vmax
    return dq
def compute_snr(signal, fs, fundamental_freq):
    # estimate snr przez fft
    N = len(signal)
    window = np.hanning(N)
    xw = signal * window
    X = np.abs(np.fft.rfft(xw))**2
    freqs = np.fft.rfftfreq(N, 1.0/fs)
    fund_idx = np.argmin(np.abs(freqs - fundamental_freq))
    sig_bins = np.arange(max(0, fund_idx-1), min(len(X), fund_idx+2))
    signal_power = X[sig_bins].sum()
    total_power = X.sum()
    noise_power = total_power - signal_power
    if noise_power<=0:
        return np.inf
    snr = 10*np.log10(signal_power/noise_power)
    return snr

def compute_thd(signal, fs, fundamental_freq, nharm=5):
    N = len(signal)
    window = np.hanning(N)
    xw = signal * window
    X = np.abs(np.fft.rfft(xw))**2
    freqs = np.fft.rfftfreq(N, 1.0/fs)
    fund_idx = np.argmin(np.abs(freqs - fundamental_freq))
    fund_power = X[fund_idx]
    harm_power = 0.0
    for n in range(2, nharm+1):
        idx = np.argmin(np.abs(freqs - fundamental_freq*n))
        if idx < len(X):
            harm_power += X[idx]
    if fund_power <= 0:
        return np.inf
    thd = 10*np.log10(harm_power / fund_power) if harm_power>0 else -np.inf
    return thd

# testy automatyczne 
results = []
fs = 200e3
duration = 0.01
f0 = 10e3  # czestot
harmonics = [(0.05,2), (0.02,3)] 
bit_depths = [8, 10, 12]
noise_stds = [0.0001, 0.001, 0.005, 0.01]
amps = [0.8, 0.5]  
for bits in bit_depths:
    for noise in noise_stds:
        for amp in amps:
            t, sig = generate_signal(freq=f0, fs=fs, duration=duration, amp=amp, harmonics=harmonics, noise_std=noise)
            qsig = quantize(sig, bits)
            snr = compute_snr(qsig, fs, f0)
            thd = compute_thd(qsig, fs, f0, nharm=5)
            results.append({
                "bits": bits,
                "noise_std": noise,
                "amp": amp,
                "snr_db": round(float(snr),3) if np.isfinite(snr) else None,
                "thd_db": round(float(thd),3) if np.isfinite(thd) else None
            })

df = pd.DataFrame(results)
csv_path = "adc_test_results.csv"
df.to_csv(csv_path, index=False)

# SNR - noise bit depth
plt.figure(figsize=(6,4))
for bits in sorted(df['bits'].unique()):
    sub = df[df['bits']==bits].groupby('noise_std')['snr_db'].mean().reset_index()
    plt.plot(sub['noise_std'], sub['snr_db'], marker='o', label=f"{bits} bits")
plt.xscale('log')
plt.xlabel("Noise std (log scale)")
plt.ylabel("SNR (dB)")
plt.title("SNR vs input noise for different ADC bit depths")
plt.legend()
plot_path = "snr_vs_noise.png"
plt.tight_layout()
plt.savefig(plot_path)
plt.close()
