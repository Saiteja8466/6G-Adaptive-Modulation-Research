
import numpy as np
import matplotlib.pyplot as plt
from math import erfc, sqrt

# 1. Simulation settings
num_bits = 1_000_000
snr_db = np.arange(0, 13, 2)

rng = np.random.default_rng(42)

simulated_ber = []
theoretical_ber = []

# 2. Generate random bits
bits = rng.integers(0, 2, num_bits)

# 3. QPSK modulation
bit_pairs = bits.reshape(-1, 2)

i_tx = 2 * bit_pairs[:, 0] - 1
q_tx = 2 * bit_pairs[:, 1] - 1

# 4. AWGN channel
for snr in snr_db:

    eb_n0 = 10 ** (snr / 10)

    noise_std = sqrt(1 / (2 * eb_n0))

    i_noise = rng.normal(0, noise_std, len(i_tx))
    q_noise = rng.normal(0, noise_std, len(q_tx))

    i_rx = i_tx + i_noise
    q_rx = q_tx + q_noise

    # 5. QPSK demodulation
    detected = np.column_stack((
        (i_rx >= 0).astype(int),
        (q_rx >= 0).astype(int)
    )).reshape(-1)

    # 6. BER calculation
    errors = np.sum(bits != detected)
    ber = errors / num_bits

    simulated_ber.append(ber)

    theory = 0.5 * erfc(sqrt(eb_n0))
    theoretical_ber.append(theory)

    print(
        f"SNR: {snr:2d} dB | "
        f"Errors: {errors:5d} | "
        f"Simulated BER: {ber:.6e} | "
        f"Theoretical BER: {theory:.6e}"
    )

# 7. Plot results
plt.figure(figsize=(9, 5))

plt.semilogy(
    snr_db, np.maximum(simulated_ber, 0.5 / num_bits),
    'o-', label='Simulated QPSK'
)

plt.semilogy(
    snr_db, theoretical_ber,
    's--', label='Theoretical QPSK'
)

plt.xlabel('Eb/N0 (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('QPSK BER Performance over AWGN Channel')

plt.grid(True, which='both')
plt.legend()
plt.tight_layout()
plt.show()
