import numpy as np
import matplotlib.pyplot as plt
from math import erfc, sqrt

# 1. Simulation settings
num_bits = 1_000_000
snr_db = np.arange(0, 13, 2)

rng = np.random.default_rng(42)

# 2. Generate random information bits
bits = rng.integers(0, 2, num_bits)

# 3. BPSK modulation: 0 -> -1, 1 -> +1
transmitted_signal = 2 * bits - 1

simulated_ber = []
theoretical_ber = []

# 4. Transmit through AWGN channel
for snr in snr_db:

    snr_linear = 10 ** (snr / 10)

    noise_std = sqrt(1 / (2 * snr_linear))

    noise = rng.normal(0, noise_std, num_bits)

    received_signal = transmitted_signal + noise

    # 5. BPSK demodulation
    received_bits = (received_signal > 0).astype(int)

    # 6. Calculate BER
    errors = np.sum(bits != received_bits)

    ber = errors / num_bits
    simulated_ber.append(ber)

    theory = 0.5 * erfc(sqrt(snr_linear))
    theoretical_ber.append(theory)

    print(
        f"SNR: {snr:2d} dB | "
        f"Errors: {errors:5d} | "
        f"Simulated BER: {ber:.6e} | "
        f"Theoretical BER: {theory:.6e}"
    )

# 7. Plot BER comparison
plt.figure(figsize=(9, 5))

plt.semilogy(
    snr_db, simulated_ber,
    'o-', label="Simulated BPSK"
)

plt.semilogy(
    snr_db, theoretical_ber,
    's--', label="Theoretical BPSK"
)

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("BPSK BER Performance over AWGN Channel")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()
