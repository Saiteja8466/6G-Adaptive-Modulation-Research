
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from math import sqrt

# ============================================================
# 6G ADAPTIVE MODULATION
# MULTI-SEED PILOT LENGTH STATISTICAL VALIDATION
# ============================================================

OUTPUT_DIR = Path("multiseed_results")
OUTPUT_DIR.mkdir(exist_ok=True)

PILOT_LENGTHS = [4, 8, 16, 32]
DATA_BLOCK_LENGTH = 128

MODULATIONS = {
    "BPSK": 1,
    "QPSK": 2,
    "16-QAM": 4
}

SNR_VALUES = {
    "BPSK": np.arange(20, 27, 1),
    "QPSK": np.arange(21, 28, 1),
    "16-QAM": np.arange(25, 33, 1)
}

NUM_SEEDS = 10
SYMBOLS_PER_SEED = 100_000

TARGET_BER = 1e-3

BANDWIDTH_HZ = 1e6
ROLLOFF = 0.25
CODE_RATE = 1.0

# ============================================================
# MODULATION
# ============================================================


def modulate(bits, modulation):

    bits = np.asarray(bits, dtype=int)

    if modulation == "BPSK":
        return (2 * bits[:, 0] - 1).astype(complex)

    if modulation == "QPSK":

        i = 2 * bits[:, 0] - 1
        q = 2 * bits[:, 1] - 1

        return (i + 1j * q) / sqrt(2)

    if modulation == "16-QAM":

        gray_map = {
            (0, 0): -3,
            (0, 1): -1,
            (1, 1): 1,
            (1, 0): 3
        }

        i = np.array([
            gray_map[tuple(pair)]
            for pair in bits[:, :2]
        ])

        q = np.array([
            gray_map[tuple(pair)]
            for pair in bits[:, 2:]
        ])

        return (i + 1j * q) / sqrt(10)

# ============================================================
# DEMODULATION
# ============================================================

def demodulate(symbols, modulation):

    if modulation == "BPSK":
        return (symbols.real > 0).astype(int).reshape(-1, 1)

    if modulation == "QPSK":

        i = (symbols.real > 0).astype(int)
        q = (symbols.imag > 0).astype(int)

        return np.column_stack((i, q))

    if modulation == "16-QAM":

        levels = np.array([-3, -1, 1, 3]) / sqrt(10)

        gray_bits = {
            -3: (0, 0),
            -1: (0, 1),
            1: (1, 1),
            3: (1, 0)
        }

        i_index = np.argmin(
            np.abs(symbols.real[:, None] - levels), axis=1
        )

        q_index = np.argmin(
            np.abs(symbols.imag[:, None] - levels), axis=1
        )

        i_levels = np.array([-3, -1, 1, 3])[i_index]
        q_levels = np.array([-3, -1, 1, 3])[q_index]

        i_bits = np.array([gray_bits[x] for x in i_levels])
        q_bits = np.array([gray_bits[x] for x in q_levels])

        return np.column_stack((i_bits, q_bits))

# ============================================================
# PILOT-BASED LS CHANNEL ESTIMATION
# ============================================================

def simulate_ber(
    modulation,
    pilot_length,
    snr_db,
    seed
):

    rng = np.random.default_rng(seed)

    bits_per_symbol = MODULATIONS[modulation]

    number_of_blocks = int(
        np.ceil(SYMBOLS_PER_SEED / DATA_BLOCK_LENGTH)
    )

    total_errors = 0
    total_bits = 0

    noise_variance = (
        1 / (bits_per_symbol * 10 ** (snr_db / 10))
    )

    for _ in range(number_of_blocks):

        # Generate data bits
        bits = rng.integers(
            0, 2,
            size=(DATA_BLOCK_LENGTH, bits_per_symbol)
        )

        tx_symbols = modulate(bits, modulation)

        # Block Rayleigh fading
        h = (
            rng.normal() + 1j * rng.normal()
        ) / sqrt(2)

        # AWGN during pilot transmission
        pilot_noise = sqrt(noise_variance / 2) * (
            rng.normal(size=pilot_length)
            + 1j * rng.normal(size=pilot_length)
        )

        # Known unit-amplitude pilot symbols
        pilot_rx = h + pilot_noise

        # Least Squares channel estimate
        h_estimated = np.mean(pilot_rx)

        # AWGN during data transmission
        data_noise = sqrt(noise_variance / 2) * (
            rng.normal(size=DATA_BLOCK_LENGTH)
            + 1j * rng.normal(size=DATA_BLOCK_LENGTH)
        )

        rx_symbols = h * tx_symbols + data_noise

        # Equalization using estimated CSI
        if abs(h_estimated) < 1e-12:
            equalized = np.zeros_like(rx_symbols)
        else:
            equalized = rx_symbols / h_estimated

        # Demodulation
        received_bits = demodulate(
            equalized, modulation
        )

        total_errors += np.count_nonzero(
            bits != received_bits
        )

        total_bits += bits.size

    ber = total_errors / total_bits

    return total_errors, total_bits, ber

# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("MULTI-SEED PILOT LENGTH VALIDATION")
    print("=" * 65)

    print("Pilot lengths:", PILOT_LENGTHS)
    print("Independent seeds:", NUM_SEEDS)
    print("Symbols per seed:", SYMBOLS_PER_SEED)

    results = []

    total_tests = sum(
        len(SNR_VALUES[m]) * len(PILOT_LENGTHS) * NUM_SEEDS
        for m in MODULATIONS
    )

    test_number = 0

    for modulation, bits_per_symbol in MODULATIONS.items():

        print(f"\nTesting {modulation}")

        for pilot_length in PILOT_LENGTHS:

            for snr_db in SNR_VALUES[modulation]:

                for seed in range(NUM_SEEDS):

                    test_number += 1

                    errors, total_bits, ber = simulate_ber(
                        modulation,
                        pilot_length,
                        int(snr_db),
                        seed=1000 + seed
                    )

                    results.append({
                        "Modulation": modulation,
                        "Pilot_Length": pilot_length,
                        "SNR_dB": int(snr_db),
                        "Seed": seed + 1,
                        "Errors": errors,
                        "Total_Bits": total_bits,
                        "BER": ber
                    })

                    print(
                        f"\rProgress: {test_number}/{total_tests}",
                        end=""
                    )

    print("\n\nSimulation completed.")

    # ========================================================
    # SAVE PER-SEED RESULTS
    # ========================================================

    raw_df = pd.DataFrame(results)

    raw_df.to_csv(
        OUTPUT_DIR / "multiseed_raw_results.csv",
        index=False
    )

    # ========================================================
    # STATISTICAL SUMMARY
    # ========================================================

    summary = []

    grouped = raw_df.groupby(
        ["Modulation", "Pilot_Length", "SNR_dB"]
    )

    for (modulation, pilot_length, snr_db), group in grouped:

        seed_bers = group["BER"].to_numpy()

        mean_ber = np.mean(seed_bers)
        std_ber = np.std(seed_bers, ddof=1)

        n = len(seed_bers)

        # Approximate 95% confidence interval
        margin = 2.262 * std_ber / sqrt(n)

        lower_ci = max(0, mean_ber - margin)
        upper_ci = mean_ber + margin

        total_errors = int(group["Errors"].sum())
        total_bits = int(group["Total_Bits"].sum())

        pooled_ber = total_errors / total_bits

        # Wilson confidence interval for pooled BER
        z = 1.96
        denominator = 1 + z**2 / total_bits

        center = (
            pooled_ber + z**2 / (2 * total_bits)
        ) / denominator

        half_width = (
            z * sqrt(
                pooled_ber * (1 - pooled_ber) / total_bits
                + z**2 / (4 * total_bits**2)
            ) / denominator
        )

        wilson_lower = max(0, center - half_width)
        wilson_upper = min(1, center + half_width)

        summary.append({
            "Modulation": modulation,
            "Pilot_Length": pilot_length,
            "SNR_dB": snr_db,
            "Mean_BER": mean_ber,
            "Std_BER": std_ber,
            "CI95_Lower": lower_ci,
            "CI95_Upper": upper_ci,
            "Pooled_BER": pooled_ber,
            "Wilson_Lower": wilson_lower,
            "Wilson_Upper": wilson_upper,
            "Total_Errors": total_errors,
            "Total_Bits": total_bits
        })

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        OUTPUT_DIR / "multiseed_ber_summary.csv",
        index=False
    )

    # ========================================================
    # ROBUST BER THRESHOLD
    # ========================================================

    threshold_results = []

    for modulation in MODULATIONS:

        for pilot_length in PILOT_LENGTHS:

            subset = summary_df[
                (summary_df["Modulation"] == modulation)
                & (summary_df["Pilot_Length"] == pilot_length)
            ].sort_values("SNR_dB")

            threshold = None

            snr_list = subset["SNR_dB"].tolist()

            for index, row in subset.iterrows():

                remaining = subset[
                    subset["SNR_dB"] >= row["SNR_dB"]
                ]

                # Require the upper Wilson bound to pass
                # at this and all higher tested SNR points.
                if (
                    row["Wilson_Upper"] <= TARGET_BER
                    and np.all(
                        remaining["Wilson_Upper"].to_numpy()
                        <= TARGET_BER
                    )
                ):
                    threshold = int(row["SNR_dB"])
                    break

            threshold_results.append({
                "Modulation": modulation,
                "Pilot_Length": pilot_length,
                "Robust_Threshold_dB": threshold
            })

    threshold_df = pd.DataFrame(threshold_results)

    threshold_df.to_csv(
        OUTPUT_DIR / "robust_thresholds.csv",
        index=False
    )

    # ========================================================
    # ADAPTIVE THROUGHPUT
    # ========================================================

    throughput_results = []

    symbol_rate = BANDWIDTH_HZ / (1 + ROLLOFF)

    for pilot_length in PILOT_LENGTHS:

        pilot_efficiency = (
            DATA_BLOCK_LENGTH /
            (DATA_BLOCK_LENGTH + pilot_length)
        )

        for snr_db in range(20, 33):

            selected_modulation = "OUTAGE"
            throughput = 0.0

            # Choose the highest-order modulation
            # with a validated robust threshold.
            for modulation in ["BPSK", "QPSK", "16-QAM"]:

                row = threshold_df[
                    (threshold_df["Modulation"] == modulation)
                    & (threshold_df["Pilot_Length"] == pilot_length)
                ]

                threshold = row[
                    "Robust_Threshold_dB"
                ].iloc[0]

                if pd.notna(threshold) and snr_db >= threshold:

                    selected_modulation = modulation

                    throughput = (
                        symbol_rate
                        * MODULATIONS[modulation]
                        * CODE_RATE
                        * pilot_efficiency
                    )

            throughput_results.append({
                "Pilot_Length": pilot_length,
                "SNR_dB": snr_db,
                "Selected_Modulation": selected_modulation,
                "Throughput_Mbps": throughput / 1e6
            })

    throughput_df = pd.DataFrame(throughput_results)

    throughput_df.to_csv(
        OUTPUT_DIR / "multiseed_adaptive_throughput.csv",
        index=False
    )

    # ========================================================
    # PLOT 1: MEAN BER WITH 95% CONFIDENCE INTERVALS
    # ========================================================

    for modulation in MODULATIONS:

        plt.figure(figsize=(10, 6))

        for pilot_length in PILOT_LENGTHS:

            subset = summary_df[
                (summary_df["Modulation"] == modulation)
                & (summary_df["Pilot_Length"] == pilot_length)
            ].sort_values("SNR_dB")

            plt.errorbar(
                subset["SNR_dB"],
                subset["Mean_BER"],
                yerr=[
                    subset["Mean_BER"] - subset["CI95_Lower"],
                    subset["CI95_Upper"] - subset["Mean_BER"]
                ],
                marker="o",
                capsize=3,
                label=f"{pilot_length} pilots"
            )

        plt.axhline(
            TARGET_BER,
            linestyle="--",
            label="Target BER = 10^-3"
        )

        plt.yscale("log")
        plt.xlabel("SNR (dB)")
        plt.ylabel("Mean BER")
        plt.title(
            f"{modulation}: Multi-Seed BER Validation"
        )

        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"{modulation}_multiseed_BER.png",
            dpi=300
        )

        plt.close()

    # ========================================================
    # PLOT 2: ADAPTIVE THROUGHPUT
    # ========================================================

    plt.figure(figsize=(10, 6))

    for pilot_length in PILOT_LENGTHS:

        subset = throughput_df[
            throughput_df["Pilot_Length"] == pilot_length
        ]

        plt.plot(
            subset["SNR_dB"],
            subset["Throughput_Mbps"],
            marker="o",
            label=f"{pilot_length} pilots"
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Adaptive Throughput (Mbps)")
    plt.title("Adaptive Throughput vs SNR")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "multiseed_adaptive_throughput.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 65)
    print("ROBUST BER THRESHOLDS")
    print("=" * 65)

    print(
        threshold_df.to_string(index=False)
    )

    print("\n" + "=" * 65)
    print("MAXIMUM VALIDATED ADAPTIVE THROUGHPUT")
    print("=" * 65)

    maximum_throughput = throughput_df.groupby(
        "Pilot_Length"
    )["Throughput_Mbps"].max()

    print(maximum_throughput)

    print("\nAll output files saved in:")
    print(OUTPUT_DIR.resolve())

    print("\nExperiment completed successfully.")

if __name__ == "__main__":
    main()
