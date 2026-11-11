
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from math import sqrt

# ============================================================
# TARGETED BPSK VALIDATION
# Pilot lengths: 4 and 16
# SNR: 27 to 30 dB
# ============================================================

OUTPUT_DIR = Path("targeted_bpsk_results")
OUTPUT_DIR.mkdir(exist_ok=True)

PILOT_LENGTHS = [4, 16]
SNR_VALUES = [27, 28, 29, 30]

NUM_SEEDS = 10
SYMBOLS_PER_SEED = 200_000

DATA_BLOCK_LENGTH = 128
TARGET_BER = 1e-3

# ============================================================
# BPSK MODULATION AND DEMODULATION
# ============================================================

def modulate(bits):
    bits = np.asarray(bits, dtype=int)
    return (2 * bits[:, 0] - 1).astype(complex)


def demodulate(symbols):
    return (symbols.real > 0).astype(int).reshape(-1, 1)


# ============================================================
# PILOT-BASED LS SIMULATION
# ============================================================

def simulate_bpsk(pilot_length, snr_db, seed):

    rng = np.random.default_rng(seed)

    number_of_blocks = int(
        np.ceil(SYMBOLS_PER_SEED / DATA_BLOCK_LENGTH)
    )

    noise_variance = 1 / (10 ** (snr_db / 10))

    total_errors = 0
    total_bits = 0

    for _ in range(number_of_blocks):

        bits = rng.integers(
            0, 2, size=(DATA_BLOCK_LENGTH, 1)
        )

        tx_symbols = modulate(bits)

        # Block Rayleigh fading
        h = (
            rng.normal() + 1j * rng.normal()
        ) / sqrt(2)

        # Known unit-amplitude pilots
        pilot_noise = sqrt(noise_variance / 2) * (
            rng.normal(size=pilot_length)
            + 1j * rng.normal(size=pilot_length)
        )

        pilot_rx = h + pilot_noise

        # Least Squares channel estimate
        h_estimated = np.mean(pilot_rx)

        # AWGN data channel
        data_noise = sqrt(noise_variance / 2) * (
            rng.normal(size=DATA_BLOCK_LENGTH)
            + 1j * rng.normal(size=DATA_BLOCK_LENGTH)
        )

        rx_symbols = h * tx_symbols + data_noise

        # Equalization
        if abs(h_estimated) < 1e-12:
            equalized = np.zeros_like(rx_symbols)
        else:
            equalized = rx_symbols / h_estimated

        received_bits = demodulate(equalized)

        total_errors += np.count_nonzero(
            bits != received_bits
        )

        total_bits += bits.size

    ber = total_errors / total_bits

    return total_errors, total_bits, ber


# ============================================================
# WILSON CONFIDENCE INTERVAL
# ============================================================

def wilson_interval(errors, total_bits):

    z = 1.96
    p = errors / total_bits

    denominator = 1 + z**2 / total_bits

    center = (
        p + z**2 / (2 * total_bits)
    ) / denominator

    margin = (
        z * sqrt(
            p * (1 - p) / total_bits
            + z**2 / (4 * total_bits**2)
        ) / denominator
    )

    return max(0, center - margin), min(1, center + margin)


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("TARGETED BPSK PILOT VALIDATION")
    print("=" * 65)

    print("Pilot lengths:", PILOT_LENGTHS)
    print("SNR values:", SNR_VALUES)
    print("Independent seeds:", NUM_SEEDS)
    print("Symbols per seed:", SYMBOLS_PER_SEED)

    raw_results = []
    summary_results = []

    total_tests = (
        len(PILOT_LENGTHS)
        * len(SNR_VALUES)
        * NUM_SEEDS
    )

    progress = 0

    for pilot_length in PILOT_LENGTHS:

        for snr_db in SNR_VALUES:

            seed_bers = []
            total_errors = 0
            total_bits = 0

            for seed in range(NUM_SEEDS):

                progress += 1

                errors, bits, ber = simulate_bpsk(
                    pilot_length,
                    snr_db,
                    seed=5000 + seed
                )

                seed_bers.append(ber)

                total_errors += errors
                total_bits += bits

                raw_results.append({
                    "Pilot_Length": pilot_length,
                    "SNR_dB": snr_db,
                    "Seed": seed + 1,
                    "Errors": errors,
                    "Total_Bits": bits,
                    "BER": ber
                })

                print(
                    f"\rProgress: {progress}/{total_tests}",
                    end=""
                )

            # Seed-level statistics
            mean_ber = np.mean(seed_bers)
            std_ber = np.std(seed_bers, ddof=1)

            margin = (
                2.262 * std_ber / sqrt(NUM_SEEDS)
            )

            seed_ci_lower = max(0, mean_ber - margin)
            seed_ci_upper = mean_ber + margin

            # Pooled statistics
            pooled_ber = total_errors / total_bits

            wilson_lower, wilson_upper = wilson_interval(
                total_errors,
                total_bits
            )

            summary_results.append({
                "Pilot_Length": pilot_length,
                "SNR_dB": snr_db,
                "Mean_BER": mean_ber,
                "Std_BER": std_ber,
                "Seed_CI_Lower": seed_ci_lower,
                "Seed_CI_Upper": seed_ci_upper,
                "Pooled_BER": pooled_ber,
                "Wilson_Lower": wilson_lower,
                "Wilson_Upper": wilson_upper,
                "Total_Errors": total_errors,
                "Total_Bits": total_bits
            })

    print("\n\nSimulation completed.")

    # ========================================================
    # SAVE CSV FILES
    # ========================================================

    raw_df = pd.DataFrame(raw_results)
    summary_df = pd.DataFrame(summary_results)

    raw_df.to_csv(
        OUTPUT_DIR / "targeted_bpsk_raw.csv",
        index=False
    )

    summary_df.to_csv(
        OUTPUT_DIR / "targeted_bpsk_summary.csv",
        index=False
    )

    # ========================================================
    # CONSERVATIVE THRESHOLD
    # ========================================================

    thresholds = []

    for pilot_length in PILOT_LENGTHS:

        subset = summary_df[
            summary_df["Pilot_Length"] == pilot_length
        ].sort_values("SNR_dB").reset_index(drop=True)

        threshold = None

        for i, row in subset.iterrows():

            higher_snr = subset.iloc[i:]

            seed_pass = np.all(
                higher_snr["Seed_CI_Upper"].to_numpy()
                <= TARGET_BER
            )

            wilson_pass = np.all(
                higher_snr["Wilson_Upper"].to_numpy()
                <= TARGET_BER
            )

            if seed_pass and wilson_pass:

                threshold = int(row["SNR_dB"])
                break

        thresholds.append({
            "Modulation": "BPSK",
            "Pilot_Length": pilot_length,
            "Conservative_Threshold_dB": threshold,
            "Status": (
                "PASSED"
                if threshold is not None
                else "NOT_ESTABLISHED"
            )
        })

    threshold_df = pd.DataFrame(thresholds)

    threshold_df.to_csv(
        OUTPUT_DIR / "targeted_bpsk_thresholds.csv",
        index=False
    )

    # ========================================================
    # BER GRAPH
    # ========================================================

    plt.figure(figsize=(10, 6))

    for pilot_length in PILOT_LENGTHS:

        subset = summary_df[
            summary_df["Pilot_Length"] == pilot_length
        ]

        lower_error = (
            subset["Mean_BER"] - subset["Seed_CI_Lower"]
        )

        upper_error = (
            subset["Seed_CI_Upper"] - subset["Mean_BER"]
        )

        plt.errorbar(
            subset["SNR_dB"],
            subset["Mean_BER"],
            yerr=[lower_error, upper_error],
            marker="o",
            capsize=4,
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
        "Targeted BPSK Pilot Length Validation"
    )

    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "targeted_bpsk_BER.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 65)
    print("TARGETED BPSK BER SUMMARY")
    print("=" * 65)

    print(
        summary_df.to_string(index=False)
    )

    print("\n" + "=" * 65)
    print("CONSERVATIVE BPSK THRESHOLDS")
    print("=" * 65)

    print(
        threshold_df.to_string(index=False)
    )

    print("\nFiles saved in:")
    print(OUTPUT_DIR.resolve())

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()
