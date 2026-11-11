
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# INDEPENDENT DYNAMIC-SYSTEM VALIDATION
# Fresh Rayleigh fading + LS pilot-based channel estimation
# ============================================================

OUTPUT_DIR = "independent_dynamic_validation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PILOT_LENGTHS = [4, 8, 16, 32]
MODULATIONS = ["BPSK", "QPSK", "16-QAM"]
SNR_VALUES = list(range(25, 33))

DATA_BLOCK_LENGTH = 128
SYMBOLS_PER_SEED = 100_000
NUMBER_OF_SEEDS = 10

TARGET_BER = 1e-3
BANDWIDTH_HZ = 1_000_000
ROLLOFF = 0.25
CODE_RATE = 1.0

# Conservative thresholds established in earlier experiments.
# Rows: modulation; columns: pilot length.
THRESHOLDS = {
    "BPSK":  {4: 27, 8: 26, 16: 27, 32: 26},
    "QPSK":  {4: 27, 8: 25, 16: 25, 32: 27},
    "16-QAM": {4: 28, 8: 29, 16: 28, 32: 29},
}

BITS_PER_SYMBOL = {
    "BPSK": 1,
    "QPSK": 2,
    "16-QAM": 4,
}

# 95% two-sided t critical value for 9 degrees of freedom.
T_CRITICAL = 2.262

# ============================================================
# MODULATION
# ============================================================

def modulate(bits, modulation):
    if modulation == "BPSK":
        return (2 * bits[:, 0] - 1).astype(complex)

    if modulation == "QPSK":
        i = 2 * bits[:, 0] - 1
        q = 2 * bits[:, 1] - 1
        return (i + 1j * q) / np.sqrt(2)

    if modulation == "16-QAM":
        # Gray mapping:
        # 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3
        mapping = np.array([-3, -1, 3, 1])
        i_index = 2 * bits[:, 0] + bits[:, 1]
        q_index = 2 * bits[:, 2] + bits[:, 3]

        i = mapping[i_index]
        q = mapping[q_index]
        return (i + 1j * q) / np.sqrt(10)

    raise ValueError("Unknown modulation")


def demodulate(symbols, modulation):
    if modulation == "BPSK":
        return (symbols.real > 0).astype(int).reshape(-1, 1)

    if modulation == "QPSK":
        return np.column_stack((
            symbols.real > 0,
            symbols.imag > 0
        )).astype(int)

    if modulation == "16-QAM":
        levels = np.array([-3, -1, 1, 3]) / np.sqrt(10)
        gray_bits = np.array([
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0]
        ])

        i_index = np.argmin(
            np.abs(symbols.real[:, None] - levels[None, :]),
            axis=1
        )
        q_index = np.argmin(
            np.abs(symbols.imag[:, None] - levels[None, :]),
            axis=1
        )

        return np.column_stack((
            gray_bits[i_index],
            gray_bits[q_index]
        )).astype(int)

    raise ValueError("Unknown modulation")


# ============================================================
# ONE FRESH SIMULATION RUN
# ============================================================

def simulate_configuration(snr_db, pilot_length, modulation, seed):
    rng = np.random.default_rng(seed)

    bits_per_symbol = BITS_PER_SYMBOL[modulation]

    number_of_blocks = int(np.ceil(
        SYMBOLS_PER_SEED / DATA_BLOCK_LENGTH
    ))

    total_symbols = number_of_blocks * DATA_BLOCK_LENGTH
    total_bits = total_symbols * bits_per_symbol

    bits = rng.integers(
        0, 2, size=(total_symbols, bits_per_symbol)
    )

    tx_data = modulate(bits, modulation).reshape(
        number_of_blocks, DATA_BLOCK_LENGTH
    )

    snr_linear = 10 ** (snr_db / 10)
    noise_variance = 1 / (bits_per_symbol * snr_linear)

    # Independent block Rayleigh fading.
    channel = (
        rng.normal(size=number_of_blocks)
        + 1j * rng.normal(size=number_of_blocks)
    ) / np.sqrt(2)

    # Unit-amplitude known pilot symbols.
    pilot_noise = np.sqrt(noise_variance / 2) * (
        rng.normal(size=(number_of_blocks, pilot_length))
        + 1j * rng.normal(size=(number_of_blocks, pilot_length))
    )

    received_pilots = channel[:, None] + pilot_noise

    # LS estimate for unit-amplitude pilots.
    channel_estimate = np.mean(received_pilots, axis=1)

    data_noise = np.sqrt(noise_variance / 2) * (
        rng.normal(size=tx_data.shape)
        + 1j * rng.normal(size=tx_data.shape)
    )

    received_data = (
        channel[:, None] * tx_data + data_noise
    )

    # Equalization using estimated channel.
    safe_estimate = channel_estimate.copy()
    safe_estimate[np.abs(safe_estimate) < 1e-12] = 1e-12

    equalized = received_data / safe_estimate[:, None]

    detected_bits = demodulate(
        equalized.reshape(-1), modulation
    )

    errors = int(np.count_nonzero(detected_bits != bits))
    ber = errors / total_bits

    return {
        "errors": errors,
        "total_bits": total_bits,
        "ber": ber
    }


# ============================================================
# STATISTICAL CONFIDENCE CHECKS
# ============================================================

def wilson_upper(errors, total, z=1.96):
    if total == 0:
        return 1.0

    p = errors / total
    denominator = 1 + z**2 / total

    centre = p + z**2 / (2 * total)
    margin = z * np.sqrt(
        p * (1 - p) / total + z**2 / (4 * total**2)
    )

    return (centre + margin) / denominator


def calculate_confidence(seed_bers, total_errors, total_bits):
    seed_bers = np.asarray(seed_bers, dtype=float)

    mean_ber = float(np.mean(seed_bers))

    if len(seed_bers) > 1:
        standard_deviation = float(np.std(seed_bers, ddof=1))
        seed_ci_upper = (
            mean_ber
            + T_CRITICAL * standard_deviation
            / np.sqrt(len(seed_bers))
        )
    else:
        standard_deviation = 0.0
        seed_ci_upper = mean_ber

    pooled_wilson_upper = wilson_upper(
        total_errors, total_bits
    )

    passed = (
        seed_ci_upper <= TARGET_BER
        and pooled_wilson_upper <= TARGET_BER
    )

    return {
        "mean_ber": mean_ber,
        "seed_sd": standard_deviation,
        "seed_ci_upper": seed_ci_upper,
        "wilson_upper": pooled_wilson_upper,
        "fresh_pass": passed
    }


# ============================================================
# THROUGHPUT
# ============================================================

def calculate_throughput(modulation, pilot_length):
    symbol_rate = BANDWIDTH_HZ / (1 + ROLLOFF)

    data_efficiency = (
        DATA_BLOCK_LENGTH /
        (DATA_BLOCK_LENGTH + pilot_length)
    )

    return (
        symbol_rate
        * BITS_PER_SYMBOL[modulation]
        * CODE_RATE
        * data_efficiency
        / 1e6
    )


# ============================================================
# STRATEGY SELECTION USING EXISTING THRESHOLDS
# ============================================================

def eligible_configurations(snr_db):
    candidates = []

    for pilot_length in PILOT_LENGTHS:
        for modulation in MODULATIONS:
            if snr_db >= THRESHOLDS[modulation][pilot_length]:
                candidates.append({
                    "pilot_length": pilot_length,
                    "modulation": modulation,
                    "throughput_mbps": calculate_throughput(
                        modulation, pilot_length
                    )
                })

    return candidates


def select_fixed_strategy(snr_db, pilot_length):
    candidates = []

    for modulation in MODULATIONS:
        if snr_db >= THRESHOLDS[modulation][pilot_length]:
            candidates.append({
                "pilot_length": pilot_length,
                "modulation": modulation,
                "throughput_mbps": calculate_throughput(
                    modulation, pilot_length
                )
            })

    if not candidates:
        return None

    return max(candidates, key=lambda x: x["throughput_mbps"])


def select_dynamic_strategy(snr_db):
    candidates = eligible_configurations(snr_db)

    if not candidates:
        return None

    return max(candidates, key=lambda x: x["throughput_mbps"])


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    print("\nINDEPENDENT DYNAMIC-SYSTEM VALIDATION")
    print("--------------------------------------")
    print("Fresh random seeds:", NUMBER_OF_SEEDS)
    print("Symbols per seed:", SYMBOLS_PER_SEED)
    print("SNR range:", SNR_VALUES[0], "to", SNR_VALUES[-1], "dB")
    print("Target BER:", TARGET_BER)

    raw_results = []
    configuration_results = {}

    total_runs = (
        len(SNR_VALUES)
        * len(PILOT_LENGTHS)
        * len(MODULATIONS)
        * NUMBER_OF_SEEDS
    )
    run_number = 0

    for snr_db in SNR_VALUES:
        for pilot_length in PILOT_LENGTHS:
            for modulation in MODULATIONS:

                seed_results = []

                for seed_number in range(NUMBER_OF_SEEDS):
                    run_number += 1

                    # Fresh, reproducible seed for every run.
                    seed = (
                        20261112
                        + snr_db * 100000
                        + pilot_length * 1000
                        + MODULATIONS.index(modulation) * 100
                        + seed_number
                    )

                    result = simulate_configuration(
                        snr_db,
                        pilot_length,
                        modulation,
                        seed
                    )

                    seed_results.append(result)

                    raw_results.append({
                        "snr_db": snr_db,
                        "pilot_length": pilot_length,
                        "modulation": modulation,
                        "seed": seed_number + 1,
                        "errors": result["errors"],
                        "total_bits": result["total_bits"],
                        "ber": result["ber"]
                    })

                    if run_number % 20 == 0:
                        print(
                            f"Progress: {run_number}/{total_runs}"
                        )

                total_errors = sum(
                    item["errors"] for item in seed_results
                )
                total_bits = sum(
                    item["total_bits"] for item in seed_results
                )

                confidence = calculate_confidence(
                    [item["ber"] for item in seed_results],
                    total_errors,
                    total_bits
                )

                configuration_results[
                    (snr_db, pilot_length, modulation)
                ] = confidence

    # Save every individual seed result.
    raw_df = pd.DataFrame(raw_results)
    raw_path = os.path.join(
        OUTPUT_DIR, "independent_raw_results.csv"
    )
    raw_df.to_csv(raw_path, index=False)

    # Save configuration-level validation results.
    config_rows = []

    for key, result in configuration_results.items():
        snr_db, pilot_length, modulation = key

        config_rows.append({
            "snr_db": snr_db,
            "pilot_length": pilot_length,
            "modulation": modulation,
            **result
        })

    config_df = pd.DataFrame(config_rows)
    config_path = os.path.join(
        OUTPUT_DIR, "configuration_validation.csv"
    )
    config_df.to_csv(config_path, index=False)

    # ========================================================
    # EVALUATE STRATEGIES ON FRESH VALIDATION RESULTS
    # ========================================================

    strategy_rows = []

    for snr_db in SNR_VALUES:

        strategies = {
            "Dynamic": select_dynamic_strategy(snr_db)
        }

        for pilot_length in PILOT_LENGTHS:
            strategies[f"Fixed_{pilot_length}"] = (
                select_fixed_strategy(snr_db, pilot_length)
            )

        for strategy_name, selection in strategies.items():

            if selection is None:
                strategy_rows.append({
                    "snr_db": snr_db,
                    "strategy": strategy_name,
                    "pilot_length": 0,
                    "modulation": "OUTAGE",
                    "fresh_ber": np.nan,
                    "throughput_mbps": 0.0,
                    "outage": 1
                })
                continue

            pilot_length = selection["pilot_length"]
            modulation = selection["modulation"]

            fresh_result = configuration_results[
                (snr_db, pilot_length, modulation)
            ]

            # Do not switch configurations after seeing validation.
            # A failed fresh BER check is counted as an outage.
            if fresh_result["fresh_pass"]:
                throughput = selection["throughput_mbps"]
                outage = 0
            else:
                throughput = 0.0
                outage = 1

            strategy_rows.append({
                "snr_db": snr_db,
                "strategy": strategy_name,
                "pilot_length": pilot_length,
                "modulation": modulation,
                "fresh_ber": fresh_result["mean_ber"],
                "seed_ci_upper": fresh_result["seed_ci_upper"],
                "wilson_upper": fresh_result["wilson_upper"],
                "throughput_mbps": throughput,
                "outage": outage
            })

    strategy_df = pd.DataFrame(strategy_rows)

    strategy_path = os.path.join(
        OUTPUT_DIR, "independent_strategy_results.csv"
    )
    strategy_df.to_csv(strategy_path, index=False)

    summary_df = (
        strategy_df.groupby("strategy")
        .agg(
            average_throughput_mbps=("throughput_mbps", "mean"),
            maximum_throughput_mbps=("throughput_mbps", "max"),
            outage_percentage=("outage", lambda x: 100 * x.mean())
        )
        .reset_index()
    )

    summary_path = os.path.join(
        OUTPUT_DIR, "independent_strategy_summary.csv"
    )
    summary_df.to_csv(summary_path, index=False)

    # ========================================================
    # PLOT
    # ========================================================

    plt.figure(figsize=(10, 6))

    for strategy_name, group in strategy_df.groupby("strategy"):
        plt.plot(
            group["snr_db"],
            group["throughput_mbps"],
            marker="o",
            label=strategy_name
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Validated Throughput (Mbps)")
    plt.title("Independent Dynamic vs Fixed Pilot Validation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(
        OUTPUT_DIR, "independent_dynamic_validation.png"
    )
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\nCONFIGURATION VALIDATION")
    print(config_df.to_string(index=False))

    print("\nINDEPENDENT STRATEGY SUMMARY")
    print(summary_df.to_string(index=False))

    print("\nFiles saved in:", OUTPUT_DIR)
    print("1. independent_raw_results.csv")
    print("2. configuration_validation.csv")
    print("3. independent_strategy_results.csv")
    print("4. independent_strategy_summary.csv")
    print("5. independent_dynamic_validation.png")

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()
