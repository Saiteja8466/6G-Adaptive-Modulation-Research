
import numpy as np
import matplotlib.pyplot as plt
from math import erfc, sqrt

# ==========================================
# 1. SIMULATION SETTINGS
# ==========================================

num_bits = 400_000
snr_db = np.arange(0, 15, 2)

rng = np.random.default_rng(42)

target_ber = 1e-3

# System parameters
bandwidth_hz = 1e6       # 1 MHz bandwidth
rolloff = 0.25           # Raised cosine roll-off
coding_rate = 1.0        # Uncoded transmission

bpsk_ber = []
qpsk_ber = []
qam16_ber = []
adaptive_ber = []

selected_modulations = []

# ==========================================
# 2. THEORETICAL BER FUNCTIONS
# ==========================================

def theoretical_bpsk(snr):
    return 0.5 * erfc(sqrt(snr))


def theoretical_qam16(snr):
    return 0.375 * erfc(sqrt(0.4 * snr))


# ==========================================
# 3. CALCULATE SNR THRESHOLDS
# ==========================================

def find_threshold(ber_function, target):

    low = 0.001
    high = 1000

    for _ in range(100):

        mid = (low + high) / 2

        if ber_function(mid) <= target:
            high = mid
        else:
            low = mid

    return 10 * np.log10(high)


bpsk_threshold = find_threshold(
    theoretical_bpsk, target_ber
)

qam16_threshold = find_threshold(
    theoretical_qam16, target_ber
)

print("\n========== SNR THRESHOLDS ==========")

print("Target BER:", target_ber)

print(
    "BPSK/QPSK threshold:",
    round(bpsk_threshold, 2), "dB"
)

print(
    "16-QAM threshold:",
    round(qam16_threshold, 2), "dB"
)


# ==========================================
# 4. BPSK TRANSMISSION
# ==========================================

def simulate_bpsk(bits, snr):

    gamma = 10 ** (snr / 10)

    tx = 2 * bits - 1

    noise_std = sqrt(1 / (2 * gamma))

    noise = rng.normal(
        0, noise_std, len(bits)
    )

    rx = tx + noise

    detected = (rx >= 0).astype(int)

    return np.mean(bits != detected)


# ==========================================
# 5. QPSK TRANSMISSION
# ==========================================

def simulate_qpsk(bits, snr):

    gamma = 10 ** (snr / 10)

    pairs = bits.reshape(-1, 2)

    i_tx = 2 * pairs[:, 0] - 1
    q_tx = 2 * pairs[:, 1] - 1

    noise_std = sqrt(1 / (2 * gamma))

    i_rx = i_tx + rng.normal(
        0, noise_std, len(i_tx)
    )

    q_rx = q_tx + rng.normal(
        0, noise_std, len(q_tx)
    )

    detected = np.column_stack((
        (i_rx >= 0).astype(int),
        (q_rx >= 0).astype(int)
    )).reshape(-1)

    return np.mean(bits != detected)


# ==========================================
# 6. 16-QAM TRANSMISSION
# ==========================================

def simulate_16qam(bits, snr):

    gamma = 10 ** (snr / 10)

    groups = bits.reshape(-1, 4)

    levels = np.array([-3, -1, 3, 1])

    thresholds = [-2, 0, 2]

    gray_bits = np.array([
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0]
    ])

    i_indices = (
        2 * groups[:, 0] + groups[:, 1]
    )

    q_indices = (
        2 * groups[:, 2] + groups[:, 3]
    )

    i_tx = levels[i_indices] / sqrt(10)
    q_tx = levels[q_indices] / sqrt(10)

    noise_std = sqrt(1 / (8 * gamma))

    i_rx = i_tx + rng.normal(
        0, noise_std, len(i_tx)
    )

    q_rx = q_tx + rng.normal(
        0, noise_std, len(q_tx)
    )

    i_detect = np.digitize(
        i_rx * sqrt(10), thresholds
    )

    q_detect = np.digitize(
        q_rx * sqrt(10), thresholds
    )

    detected = np.column_stack((
        gray_bits[i_detect],
        gray_bits[q_detect]
    )).reshape(-1)

    return np.mean(bits != detected)


# ==========================================
# 7. ADAPTIVE MODULATION SELECTION
# ==========================================

def select_modulation(snr):

    if snr >= qam16_threshold:
        return "16-QAM"

    elif snr >= bpsk_threshold:
        return "QPSK"

    else:
        return "OUTAGE"


# ==========================================
# 8. MAIN SIMULATION
# ==========================================

print("\n========== BER COMPARISON ==========")

for snr in snr_db:

    bits = rng.integers(
        0, 2, num_bits
    )

    # Fixed BPSK
    ber_bpsk = simulate_bpsk(bits, snr)

    # Fixed QPSK
    ber_qpsk = simulate_qpsk(bits, snr)

    # Fixed 16-QAM
    ber_qam16 = simulate_16qam(bits, snr)

    bpsk_ber.append(ber_bpsk)
    qpsk_ber.append(ber_qpsk)
    qam16_ber.append(ber_qam16)

    # Adaptive selection
    selected = select_modulation(snr)

    selected_modulations.append(selected)

    if selected == "BPSK":

        adaptive = simulate_bpsk(bits, snr)

    elif selected == "QPSK":

        adaptive = simulate_qpsk(bits, snr)

    elif selected == "16-QAM":

        adaptive = simulate_16qam(bits, snr)

    else:

        adaptive = np.nan

    adaptive_ber.append(adaptive)

    if np.isnan(adaptive):

        print(
            f"Eb/N0: {snr:2d} dB | "
            "Adaptive: OUTAGE"
        )

    else:

        print(
            f"Eb/N0: {snr:2d} dB | "
            f"BPSK: {ber_bpsk:.6e} | "
            f"QPSK: {ber_qpsk:.6e} | "
            f"16-QAM: {ber_qam16:.6e} | "
            f"Adaptive: {adaptive:.6e}"
        )


# ==========================================
# 9. DISPLAY ADAPTIVE SELECTION
# ==========================================

print("\n========== ADAPTIVE SELECTION ==========")

for snr, modulation in zip(
    snr_db, selected_modulations
):

    print(
        f"{snr:2d} dB -> {modulation}"
    )


# ==========================================
# 10. BER COMPARISON GRAPH
# ==========================================

plot_floor = 0.5 / num_bits

plt.figure(figsize=(11, 6))

plt.semilogy(
    snr_db,
    np.maximum(bpsk_ber, plot_floor),
    'o-',
    label='BPSK'
)

plt.semilogy(
    snr_db,
    np.maximum(qpsk_ber, plot_floor),
    's-',
    label='QPSK'
)

plt.semilogy(
    snr_db,
    np.maximum(qam16_ber, plot_floor),
    '^-',
    label='16-QAM'
)

plt.semilogy(
    snr_db,
    np.maximum(adaptive_ber, plot_floor),
    'D-',
    linewidth=2,
    label='Adaptive Modulation'
)

plt.axvline(
    bpsk_threshold,
    linestyle=':',
    label='QPSK threshold'
)

plt.axvline(
    qam16_threshold,
    linestyle='--',
    label='16-QAM threshold'
)

plt.xlabel('Eb/N0 (dB)')
plt.ylabel('Bit Error Rate (BER)')

plt.title(
    'BPSK vs QPSK vs 16-QAM vs Adaptive Modulation'
)

plt.grid(True, which='both')

plt.legend()

plt.tight_layout()

plt.savefig("BER_Comparison.png", dpi=300)

plt.show()


# ==========================================
# 11. SPECTRAL EFFICIENCY ANALYSIS
# ==========================================

print("\n========== SPECTRAL EFFICIENCY ANALYSIS ==========")

bpsk_efficiency = 1
qpsk_efficiency = 2
qam16_efficiency = 4

adaptive_efficiency = []

for modulation in selected_modulations:

    if modulation == "BPSK":

        efficiency = bpsk_efficiency

    elif modulation == "QPSK":

        efficiency = qpsk_efficiency

    elif modulation == "16-QAM":

        efficiency = qam16_efficiency

    else:

        efficiency = 0

    adaptive_efficiency.append(efficiency)


for snr, modulation, efficiency in zip(
    snr_db,
    selected_modulations,
    adaptive_efficiency
):

    print(
        f"Eb/N0: {snr:2d} dB | "
        f"Selected: {modulation:7s} | "
        f"Spectral Efficiency: {efficiency} bits/symbol"
    )


# ==========================================
# 12. THROUGHPUT CALCULATION
# ==========================================

print("\n========== THROUGHPUT ANALYSIS ==========")

symbol_rate = bandwidth_hz / (1 + rolloff)

adaptive_throughput = []

for efficiency in adaptive_efficiency:

    throughput = (
        efficiency * symbol_rate * coding_rate
    )

    adaptive_throughput.append(
        throughput / 1e6
    )


for snr, modulation, efficiency, throughput in zip(
    snr_db,
    selected_modulations,
    adaptive_efficiency,
    adaptive_throughput
):

    print(
        f"Eb/N0: {snr:2d} dB | "
        f"{modulation:7s} | "
        f"Efficiency: {efficiency} bits/symbol | "
        f"Throughput: {throughput:.3f} Mbps"
    )


# ==========================================
# 13. SPECTRAL EFFICIENCY GRAPH
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    snr_db,
    adaptive_efficiency,
    'o-',
    linewidth=2,
    markersize=8,
    label='Adaptive Spectral Efficiency'
)

plt.xlabel('Eb/N0 (dB)')

plt.ylabel('Spectral Efficiency (bits/symbol)')

plt.title('Adaptive Spectral Efficiency vs Eb/N0')

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig("Spectral_Efficiency.png", dpi=300)

plt.show()


# ==========================================
# 14. THROUGHPUT GRAPH
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    snr_db,
    adaptive_throughput,
    's-',
    linewidth=2,
    markersize=8,
    label='Adaptive Throughput'
)

plt.xlabel('Eb/N0 (dB)')

plt.ylabel('Throughput (Mbps)')

plt.title('Adaptive Throughput vs Eb/N0')

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig("Adaptive_Throughput.png", dpi=300)

plt.show()


# ==========================================
# 15. FINAL RESULTS
# ==========================================

print("\n========== FINAL SIMULATION RESULTS ==========")

print("Target BER:", target_ber)

print("Bandwidth:", bandwidth_hz / 1e6, "MHz")

print("Roll-off factor:", rolloff)

print(
    "Maximum spectral efficiency:",
    max(adaptive_efficiency),
    "bits/symbol"
)

print(
    "Maximum throughput:",
    f"{max(adaptive_throughput):.3f} Mbps"
)

print("\nGenerated Graphs:")

print("1. BER_Comparison.png")
print("2. Spectral_Efficiency.png")
print("3. Adaptive_Throughput.png")

print("\n========== SIMULATION COMPLETED ==========")
