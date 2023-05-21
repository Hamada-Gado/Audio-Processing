import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.fftpack import fft


C3: float = 130.81
D3: float = 146.83
E3: float = 164.81
F3: float = 174.61
G3: float = 196
A3: float = 220
B3: float = 246.93
C4: float = 261.63
D4: float = 293.66
E4: float = 329.63
F4: float = 349.23
G4: float = 392
A4: float = 440
B4: float = 493.88

# Song generation
def song(N: int, t: np.ndarray, a1: list[float], a2: list[float], ti: list[float], Ti: list[float], f1: list[float], f2: list[float]) -> np.ndarray:
    assert len(a1) == len(a2) == len(ti) == len(Ti) == len(f1) == len(f2) == N, "Invalid input"
    signal: np.ndarray  = np.zeros(t.shape)
     
    for i in range(N):
        wave1 = a1[i] * np.sin(2 * np.pi * f1[i] * t)
        wave2 = a2[i] * np.sin(2 * np.pi * f2[i] * t)
        wave = wave1 + wave2
        unit_step = ((t >= ti[i]) & (t <= Ti[i]))
        signal += wave * unit_step
        
    return signal

# Converting from time domain to frequency domain
def to_freq_domain(signal: np.ndarray, N: int) -> np.ndarray:
    signal = fft(signal)
    signal = 2/N * np.abs(signal[0:N//2])
    return signal

songDuration: int = 6
t: np.ndarray = np.linspace(0, songDuration, 24 * 1024)

x: np.ndarray = song(14,
                t,
                [1]*14,
                [1]*14,
                [0.00, 0.33, 0.81, 1.14, 1.62, 1.95, 2.43, 3.02, 3.35, 3.83, 4.16, 4.64, 4.97, 5.45], 
                [0.21, 0.69, 1.02, 1.50, 1.83, 2.31, 2.90, 3.23, 3.71, 4.04, 4.52, 4.85, 5.33, 5.92], 
                [0, 0, D4, D4, E4, E4, D4, C4, C4, 0, 0, 0, 0, 0],
                [G3, G3, 0, 0, 0, 0, 0, 0, 0, B3, B3, A3, A3, G3]
                )

plt.plot(t, x)
sd.play(x, 4 * 1024)


# Noise Cancellation
N: int = songDuration * 1024
f: np.ndarray = np.linspace(0, 512, N//2)

x_f: np.ndarray = to_freq_domain(x, N)
plt.plot(f, x_f)

rand_f1, rand_f2 = np.random.randint(0, 512, 2)
noise: np.ndarray = np.sin(2 * rand_f1 * np.pi * t) + np.sin(2 * rand_f2 * np.pi * t)
print(rand_f1, rand_f2)

x_n: np.ndarray = x + noise
plt.plot(t, x_n)
sd.play(x_n, 4 * 1024)

x_n_f: np.ndarray = to_freq_domain(x_n, N)
plt.plot(f, x_n_f)

# Filtering out the noise

f1, f2 = f[np.argsort(x_n_f)[-2:]]
f1, f2 = int(f1), int(f2)
print(f1, f2)

x_filtered: np.ndarray = x_n - np.sin(2 * f1 * np.pi * t) - np.sin(2 * f2 * np.pi * t)
plt.plot(t, x_filtered)
sd.play(x_filtered, 4 * 1024)

x_filtered_f: np.ndarray = to_freq_domain(x_filtered, N)
plt.plot(f, x_filtered_f)
