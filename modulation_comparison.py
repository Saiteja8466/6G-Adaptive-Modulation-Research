
import numpy as np
import matplotlib.pyplot as plt
from math import erfc, sqrt

# 1. Simulation settings
rng = np.random.default_rng(42)

num_bits = 1_000_000
snr_db = np.arange(0, 17, 2)

ber_bpsk = []
ber_qpsk = []
ber_16qam = []

# 2. BPSK simulation
for snr in snr_db:
    bits = rng.integers(0, 2, num_bits)

    eb_n0 = 10 ** (snr / 10)
    noise_std = sqrt(1 / (2 * eb_n0))

    tx = 2 * bits - 1
    rx = tx + noise_std * rng.standard_normal(num_bits)

    detected = (rx >= 0).astype(int)
    ber_bpsk.append(np.mean(bits != detected))


# 3. QPSK simulation
for snr in snr_db:
    bits = rng.integers(0, 2, num_bits)
    pairs = bits.reshape(-1, 2)

    eb_n0 = 10 ** (snr / 10)
    noise_std = sqrt(1 / (4 * eb_n0))

    i_tx = (2 * pairs[:, 0] - 1) / sqrt(2)
    q_tx = (2 * pairs[:, 1] - 1) / sqrt(2)

    i_rx = i_tx + noise_std * rng.standard_normal(len(i_tx))
    q_rx = q_tx + noise_std * rng.standard_normal(len(q_tx))

    detected = np.column_stack((
        (i_rx >= 0).astype(int),
        (q_rx >= 0).astype(int)
    )).reshape(-1)

    ber_qpsk.append(np.mean(bits != detected))


# 4. 16-QAM simulation
levels = np.array([-3, -1, 3, 1])
thresholds = [-2, 0, 2]

gray_bits = np.array([
    [0, 0],
    [0, 1],
    [1, 1],
    [1, 0]
])

for snr in snr_db:
    bits = rng.integers(0, 2, num_bits)
    groups = bits.reshape(-1, 4)

    eb_n0 = 10 ** (snr / 10)
    noise_std = sqrt(1 / (8 * eb_n0))

    # Separate bit pairs for I and Q
    i_indices = 2 * groups[:, 0] + groups[:, 1]
    q_indices = 2 * groups[:, 2] + groups[:, 3]

    i_tx = levels[i_indices] / sqrt(10)
    q_tx = levels[q_indices] / sqrt(10)

    i_rx = i_tx + noise_std * rng.standard_normal(len(i_tx))
    q_rx = q_tx + noise_std * rng.standard_normal(len(q_tx))

    # Detect received amplitude levels
    i_detect = np.digitize(i_rx * sqrt(10), thresholds)
    q_detect = np.digitize(q_rx * sqrt(10), thresholds)

    # Convert detected levels into Gray-coded bits
    detected = np.column_stack((
        gray_bits[i_detect],
        gray_bits[q_detect]
    )).reshape(-1)

    ber_16qam.append(np.mean(bits != detected))


# 5. Theoretical BER calculations
snr_linear = 10 ** (snr_db / 10)

theory_bpsk = [
    0.5 * erfc(sqrt(x)) for x in snr_linear
]

theory_qpsk = theory_bpsk.copy()

# Approximate BER formula for Gray-coded 16-QAM
theory_16qam = [
    0.375 * erfc(sqrt(0.4 * x)) for x in snr_linear
]


# 6. Display BER results
print("\nBER Comparison")
print("SNR | BPSK | QPSK | 16-QAM")

for i, snr in enumerate(snr_db):
    print(
        f"{snr:2d} dB | "
        f"{ber_bpsk[i]:.6e} | "
        f"{ber_qpsk[i]:.6e} | "
        f"{ber_16qam[i]:.6e}"
    )


# 7. Plot BER comparison
# Floor is used only to display zero-error simulation points.
plot_floor = 0.5 / num_bits

def safe_ber(values):
    return np.maximum(values, plot_floor)


plt.figure(figsize=(10, 6))

plt.semilogy(
    snr_db, safe_ber(ber_bpsk),
    'o-', label='BPSK Simulation'
)

plt.semilogy(
    snr_db, safe_ber(ber_qpsk),
    's-', label='QPSK Simulation'
)

plt.semilogy(
    snr_db, safe_ber(ber_16qam),
    '^-', label='16-QAM Simulation'
)

plt.semilogy(
    snr_db, theory_bpsk,
    '--', label='BPSK Theory'
)

plt.semilogy(
    snr_db, theory_qpsk,
    ':', label='QPSK Theory'
)

plt.semilogy(
    snr_db, theory_16qam,
    '-.', label='16-QAM Theory Approx.'
)

plt.xlabel('Eb/N0 (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('BER Performance Comparison of Digital Modulations')

plt.grid(True, which='both')
plt.legend()
plt.tight_layout()
plt.show()
