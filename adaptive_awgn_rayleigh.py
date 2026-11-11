
import numpy as np
import matplotlib.pyplot as plt
from math import erfc, sqrt

# ==========================================
# 1. SIMULATION SETTINGS
# ==========================================

num_symbols = 100_000
snr_db = np.arange(0, 21, 2)

rng = np.random.default_rng(42)

target_ber = 1e-3

bandwidth_hz = 1e6
rolloff = 0.25
coding_rate = 1.0

modulations = ["BPSK", "QPSK", "16-QAM"]

awgn_results = {m: [] for m in modulations}
rayleigh_results = {m: [] for m in modulations}

adaptive_ber = []
adaptive_selection = []
adaptive_efficiency = []
adaptive_throughput = []

# ==========================================
# 2. THEORETICAL BER
# ==========================================

def theoretical_bpsk(snr):
    return 0.5 * erfc(sqrt(snr))


def theoretical_qam16(snr):
    return 0.375 * erfc(sqrt(0.4 * snr))


# ==========================================
# 3. SNR THRESHOLDS
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


qpsk_threshold = find_threshold(
    theoretical_bpsk, target_ber
)

qam16_threshold = find_threshold(
    theoretical_qam16, target_ber
)

print("\n========== SNR THRESHOLDS ==========")
print("Target BER:", target_ber)
print("QPSK threshold:", round(qpsk_threshold, 2), "dB")
print("16-QAM threshold:", round(qam16_threshold, 2), "dB")


# ==========================================
# 4. MODULATION
# ==========================================

def modulate(bits, modulation):

    if modulation == "BPSK":
        return 1 - 2 * bits

    elif modulation == "QPSK":

        pairs = bits.reshape(-1, 2)

        i = 1 - 2 * pairs[:, 0]
        q = 1 - 2 * pairs[:, 1]

        return (i + 1j * q) / sqrt(2)

    elif modulation == "16-QAM":

        groups = bits.reshape(-1, 4)

        levels = np.array([-3, -1, 3, 1])

        i_index = 2 * groups[:, 0] + groups[:, 1]
        q_index = 2 * groups[:, 2] + groups[:, 3]

        i = levels[i_index]
        q = levels[q_index]

        return (i + 1j * q) / sqrt(10)


# ==========================================
# 5. DEMODULATION
# ==========================================

def demodulate(received, modulation):

    if modulation == "BPSK":
        return (received.real < 0).astype(int)

    elif modulation == "QPSK":

        detected = np.column_stack((
            (received.real < 0).astype(int),
            (received.imag < 0).astype(int)
        ))

        return detected.reshape(-1)

    elif modulation == "16-QAM":

        thresholds = [-2, 0, 2]

        gray_bits = np.array([
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0]
        ])

        i_index = np.digitize(
            received.real * sqrt(10),
            thresholds
        )

        q_index = np.digitize(
            received.imag * sqrt(10),
            thresholds
        )

        detected = np.column_stack((
            gray_bits[i_index],
            gray_bits[q_index]
        ))

        return detected.reshape(-1)


# ==========================================
# 6. CHANNEL SIMULATION
# ==========================================

def simulate_channel(bits, snr, modulation, channel):

    gamma = 10 ** (snr / 10)

    bits_per_symbol = {
        "BPSK": 1,
        "QPSK": 2,
        "16-QAM": 4
    }

    k = bits_per_symbol[modulation]

    symbols = modulate(bits, modulation)

    noise_power = 1 / (k * gamma)

    noise = np.sqrt(noise_power / 2) * (
        rng.normal(size=len(symbols))
        + 1j * rng.normal(size=len(symbols))
    )

    if channel == "AWGN":

        h = np.ones(len(symbols), dtype=complex)

    elif channel == "RAYLEIGH":

        h = (
            rng.normal(size=len(symbols))
            + 1j * rng.normal(size=len(symbols))
        ) / sqrt(2)

    else:
        raise ValueError("Unknown channel")

    received = h * symbols + noise

    # Perfect channel estimation assumption
    equalized = received / h

    detected = demodulate(
        equalized, modulation
    )

    return np.mean(bits != detected)


# ==========================================
# 7. ADAPTIVE MODULATION
# ==========================================

def select_modulation(snr):

    if snr >= qam16_threshold:
        return "16-QAM"

    elif snr >= qpsk_threshold:
        return "QPSK"

    else:
        return "OUTAGE"


# ==========================================
# 8. AWGN VS RAYLEIGH
# ==========================================

print("\n========== CHANNEL COMPARISON ==========")

for snr in snr_db:

    for modulation in modulations:

        k = {
            "BPSK": 1,
            "QPSK": 2,
            "16-QAM": 4
        }[modulation]

        bits = rng.integers(
            0, 2, num_symbols * k
        )

        ber_awgn = simulate_channel(
            bits, snr, modulation, "AWGN"
        )

        ber_rayleigh = simulate_channel(
            bits, snr, modulation, "RAYLEIGH"
        )

        awgn_results[modulation].append(ber_awgn)
        rayleigh_results[modulation].append(ber_rayleigh)

        print(
            f"SNR: {snr:2d} dB | "
            f"{modulation:7s} | "
            f"AWGN: {ber_awgn:.6e} | "
            f"Rayleigh: {ber_rayleigh:.6e}"
        )


# ==========================================
# 9. ADAPTIVE RAYLEIGH SIMULATION
# ==========================================

print("\n========== ADAPTIVE RAYLEIGH ANALYSIS ==========")

for snr in snr_db:

    selected = select_modulation(snr)

    adaptive_selection.append(selected)

    if selected == "OUTAGE":

        adaptive_ber.append(np.nan)
        adaptive_efficiency.append(0)
        adaptive_throughput.append(0)

        print(f"SNR: {snr:2d} dB | OUTAGE")

        continue

    k = {
        "BPSK": 1,
        "QPSK": 2,
        "16-QAM": 4
    }[selected]

    bits = rng.integers(
        0, 2, num_symbols * k
    )

    ber = simulate_channel(
        bits, snr, selected, "RAYLEIGH"
    )

    adaptive_ber.append(ber)

    efficiency = k

    adaptive_efficiency.append(efficiency)

    spectral_efficiency = efficiency / (1 + rolloff)

    throughput = (
        spectral_efficiency
        * bandwidth_hz
        * coding_rate
    )

    adaptive_throughput.append(throughput / 1e6)

    print(
        f"SNR: {snr:2d} dB | "
        f"Selected: {selected:7s} | "
        f"BER: {ber:.6e} | "
        f"Throughput: {throughput / 1e6:.3f} Mbps"
    )


# ==========================================
# 10. AWGN VS RAYLEIGH BER GRAPH
# ==========================================

plot_floor = 0.5 / (num_symbols * 4)

plt.figure(figsize=(11, 6))

for modulation in modulations:

    plt.semilogy(
        snr_db,
        np.maximum(awgn_results[modulation], plot_floor),
        marker='o',
        label=f"{modulation} AWGN"
    )

    plt.semilogy(
        snr_db,
        np.maximum(rayleigh_results[modulation], plot_floor),
        marker='s',
        linestyle='--',
        label=f"{modulation} Rayleigh"
    )

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Bit Error Rate")
plt.title("AWGN vs Rayleigh Fading Channel BER Comparison")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig("AWGN_vs_Rayleigh_BER.png", dpi=300)
plt.show()


# ==========================================
# 11. ADAPTIVE RAYLEIGH BER GRAPH
# ==========================================

plt.figure(figsize=(10, 5))

plt.semilogy(
    snr_db,
    np.maximum(adaptive_ber, plot_floor),
    'o-',
    linewidth=2,
    label="Adaptive Rayleigh BER"
)

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Bit Error Rate")
plt.title("Adaptive Modulation Under Rayleigh Fading")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig("Adaptive_Rayleigh_BER.png", dpi=300)
plt.show()


# ==========================================
# 12. SPECTRAL EFFICIENCY GRAPH
# ==========================================

spectral_efficiency = (
    np.array(adaptive_efficiency) / (1 + rolloff)
)

plt.figure(figsize=(10, 5))

plt.plot(
    snr_db,
    spectral_efficiency,
    'o-',
    linewidth=2
)

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Spectral Efficiency (bits/s/Hz)")
plt.title("Adaptive Spectral Efficiency Under Rayleigh")
plt.grid(True)
plt.tight_layout()
plt.savefig("Rayleigh_Spectral_Efficiency.png", dpi=300)
plt.show()


# ==========================================
# 13. THROUGHPUT GRAPH
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    snr_db,
    adaptive_throughput,
    's-',
    linewidth=2
)

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Throughput (Mbps)")
plt.title("Adaptive Throughput Under Rayleigh Fading")
plt.grid(True)
plt.tight_layout()
plt.savefig("Rayleigh_Adaptive_Throughput.png", dpi=300)
plt.show()


# ==========================================
# 14. FINAL RESULTS
# ==========================================

print("\n========== FINAL SIMULATION RESULTS ==========")

print("Target BER:", target_ber)
print("Bandwidth:", bandwidth_hz / 1e6, "MHz")
print("Roll-off factor:", rolloff)

print(
    "Maximum throughput:",
    f"{max(adaptive_throughput):.3f} Mbps"
)

print("\nGenerated Graphs:")

print("1. AWGN_vs_Rayleigh_BER.png")
print("2. Adaptive_Rayleigh_BER.png")
print("3. Rayleigh_Spectral_Efficiency.png")
print("4. Rayleigh_Adaptive_Throughput.png")

print("\n========== SIMULATION COMPLETED ==========")
