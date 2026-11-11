
import numpy as np
import matplotlib.pyplot as plt
from math import erfc, sqrt

# 1. Simulation settings
num_bits = 1_000_000
snr_db = np.arange(0, 15, 2)

rng = np.random.default_rng(42)

simulated_ber = []
theoretical_ber = []

# 2. Generate random information bits
bits = rng.integers(0, 2, num_bits)

# 3. 16-QAM Gray-coded constellation levels
levels = np.array([-3, -1, 3, 1])
thresholds = [-2, 0, 2]

gray_bits = np.array([
    [0, 0],
    [0, 1],
    [1, 1],
    [1, 0]
])

# 4. Group four bits per symbol
groups = bits.reshape(-1, 4)

# 5. Separate I and Q bit pairs
i_indices = 2 * groups[:, 0] + groups[:, 1]
q_indices = 2 * groups[:, 2] + groups[:, 3]

# 6. 16-QAM modulation
i_tx = levels[i_indices] / sqrt(10)
q_tx = levels[q_indices] / sqrt(10)

# 7. AWGN channel and BER calculation
for snr in snr_db:

    eb_n0 = 10 ** (snr / 10)

    noise_std = sqrt(1 / (8 * eb_n0))

    # Add Gaussian noise
    i_noise = rng.normal(0, noise_std, len(i_tx))
    q_noise = rng.normal(0, noise_std, len(q_tx))

    i_rx = i_tx + i_noise
    q_rx = q_tx + q_noise

    # 8. Receiver detection
    i_detect = np.digitize(
        i_rx * sqrt(10), thresholds
    )

    q_detect = np.digitize(
        q_rx * sqrt(10), thresholds
    )

    # 9. Convert detected symbols into bits
    detected = np.column_stack((
        gray_bits[i_detect],
        gray_bits[q_detect]
    )).reshape(-1)

    # 10. Calculate BER
    errors = np.sum(bits != detected)

    ber = errors / num_bits

    simulated_ber.append(ber)

    # Approximate theoretical BER
    theory = 0.375 * erfc(sqrt(0.4 * eb_n0))

    theoretical_ber.append(theory)

    print(
        f"Eb/N0: {snr:2d} dB | "
        f"Errors: {errors:5d} | "
        f"Simulated BER: {ber:.6e} | "
        f"Theoretical BER: {theory:.6e}"
    )

# 11. Plot BER performance
plt.figure(figsize=(9, 5))

plt.semilogy(
    snr_db,
    np.maximum(simulated_ber, 0.5 / num_bits),
    'o-',
    label='Simulated 16-QAM'
)

plt.semilogy(
    snr_db,
    theoretical_ber,
    's--',
    label='Theoretical 16-QAM (approx.)'
)

plt.xlabel('Eb/N0 (dB)')
plt.ylabel('Bit Error Rate (BER)')

plt.title('16-QAM BER Performance over AWGN Channel')

plt.grid(True, which='both')
plt.legend()
plt.tight_layout()

plt.show()
