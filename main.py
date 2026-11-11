import numpy as np
import matplotlib.pyplot as plt
from math import erfc, sqrt

# Simulation parameters
num_bits = 1000000
snr_values = np.arange(0, 17, 2)

# Generate random bits
bits = np.random.randint(0, 2, num_bits)

# Group bits into 4-bit symbols
bit_groups = bits.reshape(-1, 4)

# Gray-coded 16-QAM mapping
levels = np.array([-3, -1, 1, 3])

I_index = 2 * bit_groups[:, 0] + bit_groups[:, 1]
Q_index = 2 * bit_groups[:, 2] + bit_groups[:, 3]

I_map = np.array([-3, -1, 3, 1])
Q_map = np.array([-3, -1, 3, 1])

I = I_map[I_index] / np.sqrt(10)
Q = Q_map[Q_index] / np.sqrt(10)

# Create 16-QAM symbols
qam_signal = I + 1j * Q

simulated_ber = []
theoretical_ber = []

for snr_db in snr_values:

    # Convert Eb/N0 from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)

    # Add AWGN noise
    noise_std = np.sqrt(1 / (8 * snr_linear))

    noise_I = np.random.normal(0, noise_std, len(I))
    noise_Q = np.random.normal(0, noise_std, len(Q))

    received = qam_signal + noise_I + 1j * noise_Q

    # Demodulation: find nearest amplitude level
    received_I = received.real * np.sqrt(10)
    received_Q = received.imag * np.sqrt(10)

    I_detect = np.digitize(received_I, [-2, 0, 2])
    Q_detect = np.digitize(received_Q, [-2, 0, 2])

    # Convert detected levels back to bits
    bit_lookup_1 = np.array([0, 0, 1, 1])
    bit_lookup_2 = np.array([0, 1, 1, 0])

    detected_bits = np.column_stack((
        bit_lookup_1[I_detect],
        bit_lookup_2[I_detect],
        bit_lookup_1[Q_detect],
        bit_lookup_2[Q_detect]
    )).flatten()

    # Calculate BER
    errors = np.sum(bits != detected_bits)
    ber = errors / num_bits
    simulated_ber.append(ber)

    # Theoretical Gray-coded 16-QAM BER approximation
    theoretical = (
        0.75 * 0.5 *
        erfc(sqrt(0.4 * snr_linear))
    )

    theoretical_ber.append(theoretical)

    print(
        f"SNR: {snr_db} dB | "
        f"Simulated BER: {ber:.8f} | "
        f"Theoretical BER: {theoretical:.8f}"
    )

# Plot BER comparison
plt.figure(figsize=(9, 6))

plt.semilogy(
    snr_values, simulated_ber,
    'bo-', label='16-QAM Simulated BER'
)

plt.semilogy(
    snr_values, theoretical_ber,
    'r--s', label='16-QAM Theoretical BER (Approx.)'
)

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Bit Error Rate")
plt.title("16-QAM: Simulated vs Theoretical BER")
plt.legend()
plt.grid(True, which="both")
plt.show()

# Plot constellation diagram
plt.figure(figsize=(7, 7))

plt.scatter(
    qam_signal.real[:500],
    qam_signal.imag[:500],
    marker='o'
)

plt.xlabel("In-phase (I)")
plt.ylabel("Quadrature (Q)")
plt.title("16-QAM Constellation Diagram")
plt.grid(True)
plt.axis("equal")
plt.show()