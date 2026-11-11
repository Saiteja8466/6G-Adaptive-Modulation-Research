
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from adaptive_rayleigh_optimized import modulate, demodulate

# ==========================================================
# AWGN vs RAYLEIGH vs ADAPTIVE MODULATION COMPARISON
# ==========================================================

TARGET_BER = 1e-3

SNR_DB = np.arange(0, 46, 2)
NUM_SYMBOLS = 200_000

BANDWIDTH_HZ = 1_000_000
ROLLOFF = 0.25
CODE_RATE = 1.0

BITS_PER_SYMBOL = {
    "BPSK": 1,
    "QPSK": 2,
    "16-QAM": 4
}

# Previously estimated Rayleigh thresholds.
# These are preliminary Monte Carlo estimates.
THRESHOLDS = {
    "BPSK": 26,
    "QPSK": 26,
    "16-QAM": 28
}

rng = np.random.default_rng(2026)


def simulate_channel(modulation, snr_db, channel):
    k = BITS_PER_SYMBOL[modulation]

    bits = rng.integers(0, 2, NUM_SYMBOLS * k)
    tx = modulate(bits, modulation)

    ebn0 = 10 ** (snr_db / 10)
    noise_variance = 1 / (k * ebn0)

    noise = np.sqrt(noise_variance / 2) * (
        rng.normal(size=NUM_SYMBOLS)
        + 1j * rng.normal(size=NUM_SYMBOLS)
    )

    if channel == "AWGN":
        received = tx + noise

    elif channel == "Rayleigh":
        h = (
            rng.normal(0, 1 / np.sqrt(2), NUM_SYMBOLS)
            + 1j * rng.normal(0, 1 / np.sqrt(2), NUM_SYMBOLS)
        )
        received = h * tx + noise
        received = received / h

    else:
        raise ValueError("Unknown channel")

    detected = demodulate(received, modulation)

    errors = np.count_nonzero(bits != detected)
    ber = errors / len(bits)

    return ber, errors, len(bits)


def select_modulation(snr):
    if snr >= THRESHOLDS["16-QAM"]:
        return "16-QAM"
    elif snr >= THRESHOLDS["QPSK"]:
        return "QPSK"
    elif snr >= THRESHOLDS["BPSK"]:
        return "BPSK"
    else:
        return "OUTAGE"


def main():

    results = []

    print("\nAWGN VS RAYLEIGH COMPARISON")
    print("Target BER:", TARGET_BER)

    # ------------------------------------------------------
    # 1. Fixed modulation comparison
    # ------------------------------------------------------

    for modulation in BITS_PER_SYMBOL:

        for channel in ["AWGN", "Rayleigh"]:

            print(f"\nTesting {modulation} - {channel}")

            for snr in SNR_DB:

                ber, errors, total_bits = simulate_channel(
                    modulation, snr, channel
                )

                results.append({
                    "SNR_dB": snr,
                    "Modulation": modulation,
                    "Channel": channel,
                    "BER": ber,
                    "Errors": errors,
                    "Total_Bits": total_bits
                })

                print(
                    f"{snr:2d} dB | BER={ber:.6e} | "
                    f"Errors={errors}"
                )

    fixed_df = pd.DataFrame(results)

    # ------------------------------------------------------
    # 2. Adaptive Rayleigh comparison
    # ------------------------------------------------------

    adaptive_rows = []

    print("\nOPTIMIZED ADAPTIVE RAYLEIGH RESULTS")

    for snr in SNR_DB:

        modulation = select_modulation(snr)

        if modulation == "OUTAGE":
            ber = np.nan
            throughput = 0.0
            efficiency = 0.0
            errors = 0
            total_bits = 0

        else:
            ber, errors, total_bits = simulate_channel(
                modulation, snr, "Rayleigh"
            )

            k = BITS_PER_SYMBOL[modulation]

            symbol_rate = BANDWIDTH_HZ / (1 + ROLLOFF)

            throughput = (
                symbol_rate * k * CODE_RATE
            )

            efficiency = (
                k * CODE_RATE / (1 + ROLLOFF)
            )

        adaptive_rows.append({
            "SNR_dB": snr,
            "Selected_Modulation": modulation,
            "BER": ber,
            "Throughput_Mbps": throughput / 1e6,
            "Spectral_Efficiency": efficiency,
            "Errors": errors,
            "Total_Bits": total_bits
        })

        print(
            f"{snr:2d} dB | {modulation:7s} | "
            f"BER={ber} | "
            f"Throughput={throughput / 1e6:.2f} Mbps"
        )

    adaptive_df = pd.DataFrame(adaptive_rows)

    # ------------------------------------------------------
    # 3. BER comparison graph
    # ------------------------------------------------------

    plt.figure(figsize=(10, 7))

    for modulation in BITS_PER_SYMBOL:

        for channel in ["AWGN", "Rayleigh"]:

            subset = fixed_df[
                (fixed_df["Modulation"] == modulation)
                & (fixed_df["Channel"] == channel)
            ]

            # Plotting floor only: avoids zero on log axis.
            floor = 0.5 / (
                NUM_SYMBOLS * BITS_PER_SYMBOL[modulation]
            )

            ber_plot = np.maximum(
                subset["BER"].to_numpy(), floor
            )

            plt.semilogy(
                subset["SNR_dB"],
                ber_plot,
                marker="o",
                label=f"{modulation} - {channel}"
            )

    plt.axhline(
        TARGET_BER,
        linestyle="--",
        label="Target BER = 1e-3"
    )

    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Bit Error Rate")
    plt.title("AWGN vs Rayleigh BER Performance")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig("AWGN_vs_Rayleigh_BER.png", dpi=300)
    plt.show()

    # ------------------------------------------------------
    # 4. Adaptive throughput graph
    # ------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        adaptive_df["SNR_dB"],
        adaptive_df["Throughput_Mbps"],
        marker="s"
    )

    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Throughput (Mbps)")
    plt.title("Optimized Adaptive Throughput - Rayleigh")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Final_Adaptive_Throughput.png", dpi=300)
    plt.show()

    # ------------------------------------------------------
    # 5. Adaptive BER graph
    # ------------------------------------------------------

    valid = adaptive_df.dropna(subset=["BER"])

    plt.figure(figsize=(10, 6))

    plt.semilogy(
        valid["SNR_dB"],
        valid["BER"],
        marker="o",
        label="Adaptive Rayleigh"
    )

    plt.axhline(
        TARGET_BER,
        linestyle="--",
        label="Target BER = 1e-3"
    )

    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Bit Error Rate")
    plt.title("Adaptive Rayleigh BER Performance")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Final_Adaptive_BER.png", dpi=300)
    plt.show()

    # ------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------

    fixed_df.to_csv(
        "AWGN_Rayleigh_Comparison.csv",
        index=False
    )

    adaptive_df.to_csv(
        "Final_Adaptive_Results.csv",
        index=False
    )

    print("\n========== COMPARISON COMPLETED ==========")
    print("1. AWGN_vs_Rayleigh_BER.png")
    print("2. Final_Adaptive_Throughput.png")
    print("3. Final_Adaptive_BER.png")
    print("4. AWGN_Rayleigh_Comparison.csv")
    print("5. Final_Adaptive_Results.csv")


if __name__ == "__main__":
    main()
