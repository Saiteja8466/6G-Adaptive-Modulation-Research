
import numpy as np
import matplotlib.pyplot as plt
from math import erfc, sqrt

# 1. Simulation settings
rng = np.random.default_rng(42)

num_bits = 1_000_000
snr_db = np.arange(0, 21, 2)
target_ber = 1e-3

# 2. Theoretical BER functions
def bpsk_ber(snr):
    return 0.5 * erfc(sqrt(snr))

def qpsk_ber(snr):
    return 0.5 * erfc(sqrt(snr))

def qam16_ber(snr):
    return 0.375 * erfc(sqrt(0.4 * snr))

# 3. Calculate theoretical SNR thresholds
def find_threshold(ber_function, target):
    low = 0.01
    high = 1000

    for _ in range(100):
        mid = (low + high) / 2

        if ber_function(mid) <= target:
            high = mid
        else:
            low = mid

    return 10 * np.log10(high)

bpsk_threshold = find_threshold(bpsk_ber, target_ber)
qpsk_threshold = find_threshold(qpsk_ber, target_ber)
qam16_threshold = find_threshold(qam16_ber, target_ber)

print("Target BER:", target_ber)
print("BPSK threshold:", round(bpsk_threshold, 2), "dB")
print("QPSK threshold:", round(qpsk_threshold, 2), "dB")
print("16-QAM threshold:", round(qam16_threshold, 2), "dB")

# 4. Adaptive modulation selection
def select_modulation(snr):
    if snr >= qam16_threshold:
        return "16-QAM"
    elif snr >= qpsk_threshold:
        return "QPSK"
    elif snr >= bpsk_threshold:
        return "BPSK"
    else:
        return "OUTAGE"

selected_modulations = []

for snr in snr_db:
    modulation = select_modulation(snr)
    selected_modulations.append(modulation)

    print(f"SNR = {snr:2d} dB -> {modulation}")

# 5. Display spectral efficiency
spectral_efficiency = {
    "BPSK": 1,
    "QPSK": 2,
    "16-QAM": 4,
    "OUTAGE": 0
}

efficiency = [
    spectral_efficiency[m] for m in selected_modulations
]

# 6. Plot adaptive modulation selection
plt.figure(figsize=(10, 5))

plt.step(
    snr_db,
    efficiency,
    where="mid",
    linewidth=2,
    marker="o"
)

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Bits per Symbol")
plt.title("Adaptive Modulation Based on Channel SNR")
plt.yticks([0, 1, 2, 4], ["Outage", "BPSK", "QPSK", "16-QAM"])
plt.grid(True)
plt.tight_layout()
plt.show()
