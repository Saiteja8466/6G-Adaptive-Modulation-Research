
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# RELIABILITY-AWARE DYNAMIC PILOT AND MODULATION SELECTION
# Two-stage experiment: selection + independent holdout testing
# ============================================================

OUTPUT_DIR = "reliability_aware_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PILOT_LENGTHS = [4, 8, 16, 32]
MODULATIONS = ["BPSK", "QPSK", "16-QAM"]
SNR_VALUES = list(range(25, 33))

DATA_BLOCK_LENGTH = 128
SYMBOLS_PER_SEED = 50_000
NUMBER_OF_SEEDS = 10

TARGET_BER = 1e-3
BANDWIDTH_HZ = 1_000_000
ROLLOFF = 0.25
CODE_RATE = 1.0

BITS_PER_SYMBOL = {
    "BPSK": 1,
    "QPSK": 2,
    "16-QAM": 4
}

T_CRITICAL = 2.262
Z_VALUE = 1.96

# ============================================================
# MODULATION AND DEMODULATION
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
# BLOCK RAYLEIGH FADING + LS PILOT CHANNEL ESTIMATION
# ============================================================

def simulate_configuration(
    snr_db,
    pilot_length,
    modulation,
    seed
):

    rng = np.random.default_rng(seed)

    bps = BITS_PER_SYMBOL[modulation]

    number_of_blocks = int(np.ceil(
        SYMBOLS_PER_SEED / DATA_BLOCK_LENGTH
    ))

    total_symbols = number_of_blocks * DATA_BLOCK_LENGTH

    total_bits = total_symbols * bps

    bits = rng.integers(
        0, 2,
        size=(total_symbols, bps)
    )

    tx = modulate(bits, modulation).reshape(
        number_of_blocks,
        DATA_BLOCK_LENGTH
    )

    snr_linear = 10 ** (snr_db / 10)

    noise_variance = 1 / (bps * snr_linear)

    # One complex Rayleigh coefficient per block.
    h = (
        rng.normal(size=number_of_blocks)
        + 1j * rng.normal(size=number_of_blocks)
    ) / np.sqrt(2)

    # Known unit-amplitude pilots.
    pilot_noise = np.sqrt(noise_variance / 2) * (
        rng.normal(size=(number_of_blocks, pilot_length))
        + 1j * rng.normal(size=(number_of_blocks, pilot_length))
    )

    received_pilots = h[:, None] + pilot_noise

    # LS channel estimate.
    h_est = np.mean(received_pilots, axis=1)

    # Data transmission.
    data_noise = np.sqrt(noise_variance / 2) * (
        rng.normal(size=tx.shape)
        + 1j * rng.normal(size=tx.shape)
    )

    received_data = h[:, None] * tx + data_noise

    # Channel equalization.
    safe_h_est = h_est.copy()
    safe_h_est[np.abs(safe_h_est) < 1e-12] = 1e-12

    equalized = received_data / safe_h_est[:, None]

    detected_bits = demodulate(
        equalized.reshape(-1),
        modulation
    )

    errors = int(np.count_nonzero(detected_bits != bits))

    return {
        "errors": errors,
        "total_bits": total_bits,
        "ber": errors / total_bits
    }


# ============================================================
# CONFIDENCE INTERVALS
# ============================================================

def wilson_upper(errors, total):

    if total == 0:
        return 1.0

    p = errors / total
    z = Z_VALUE

    denominator = 1 + z**2 / total

    centre = p + z**2 / (2 * total)

    margin = z * np.sqrt(
        p * (1 - p) / total
        + z**2 / (4 * total**2)
    )

    return (centre + margin) / denominator


def summarize_runs(results):

    seed_bers = np.array([
        result["ber"] for result in results
    ])

    total_errors = sum(
        result["errors"] for result in results
    )

    total_bits = sum(
        result["total_bits"] for result in results
    )

    mean_ber = float(np.mean(seed_bers))

    sd = float(np.std(seed_bers, ddof=1))

    seed_ci_upper = (
        mean_ber
        + T_CRITICAL * sd / np.sqrt(len(seed_bers))
    )

    pooled_wilson_upper = wilson_upper(
        total_errors,
        total_bits
    )

    passed = (
        seed_ci_upper <= TARGET_BER
        and pooled_wilson_upper <= TARGET_BER
    )

    return {
        "mean_ber": mean_ber,
        "seed_sd": sd,
        "seed_ci_upper": seed_ci_upper,
        "wilson_upper": pooled_wilson_upper,
        "total_errors": total_errors,
        "total_bits": total_bits,
        "reliable": passed
    }


# ============================================================
# THROUGHPUT WITH PILOT OVERHEAD
# ============================================================

def throughput_mbps(modulation, pilot_length):

    symbol_rate = BANDWIDTH_HZ / (1 + ROLLOFF)

    pilot_efficiency = (
        DATA_BLOCK_LENGTH /
        (DATA_BLOCK_LENGTH + pilot_length)
    )

    return (
        symbol_rate
        * BITS_PER_SYMBOL[modulation]
        * CODE_RATE
        * pilot_efficiency
        / 1e6
    )


def all_configurations():

    configurations = []

    for pilot in PILOT_LENGTHS:
        for modulation in MODULATIONS:
            configurations.append({
                "pilot_length": pilot,
                "modulation": modulation,
                "throughput_mbps": throughput_mbps(
                    modulation, pilot
                )
            })

    return configurations


# ============================================================
# STAGE 1: CONFIGURATION SELECTION
# ============================================================

def run_selection_stage():

    print("\nSTAGE 1: RELIABILITY-BASED CONFIGURATION SELECTION")

    selection_results = {}
    selection_rows = []

    configs = all_configurations()

    total = (
        len(SNR_VALUES)
        * len(configs)
        * NUMBER_OF_SEEDS
    )

    completed = 0

    for snr in SNR_VALUES:

        for config in configs:

            pilot = config["pilot_length"]
            modulation = config["modulation"]

            results = []

            for seed_number in range(NUMBER_OF_SEEDS):

                seed = (
                    41000000
                    + snr * 100000
                    + pilot * 1000
                    + MODULATIONS.index(modulation) * 100
                    + seed_number
                )

                result = simulate_configuration(
                    snr,
                    pilot,
                    modulation,
                    seed
                )

                results.append(result)

                completed += 1

                if completed % 100 == 0:
                    print(f"Selection progress: {completed}/{total}")

            summary = summarize_runs(results)

            selection_results[
                (snr, pilot, modulation)
            ] = summary

            selection_rows.append({
                "snr_db": snr,
                "pilot_length": pilot,
                "modulation": modulation,
                "throughput_mbps": config["throughput_mbps"],
                **summary
            })

    selection_df = pd.DataFrame(selection_rows)

    selection_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "selection_stage_results.csv"
        ),
        index=False
    )

    return selection_results, selection_df


# ============================================================
# STRATEGY SELECTION
# ============================================================

def choose_best(candidates):

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item: item["throughput_mbps"]
    )


def select_reliable_strategies(snr, selection_results):

    eligible = []

    for config in all_configurations():

        key = (
            snr,
            config["pilot_length"],
            config["modulation"]
        )

        if selection_results[key]["reliable"]:
            eligible.append(config)

    strategies = {
        "Dynamic_Reliability_Aware": choose_best(eligible)
    }

    # Fixed pilot strategies use the same reliability criterion.
    for pilot in PILOT_LENGTHS:

        fixed_candidates = [
            item for item in eligible
            if item["pilot_length"] == pilot
        ]

        strategies[f"Fixed_{pilot}"] = choose_best(
            fixed_candidates
        )

    return strategies


# ============================================================
# STAGE 2: INDEPENDENT HOLDOUT VALIDATION
# ============================================================

def run_holdout_stage(selection_results):

    print("\nSTAGE 2: INDEPENDENT HOLDOUT VALIDATION")

    strategy_rows = []
    holdout_rows = []

    for snr in SNR_VALUES:

        strategies = select_reliable_strategies(
            snr,
            selection_results
        )

        # Cache identical selected configurations so that
        # the same holdout data is not simulated repeatedly.
        holdout_cache = {}

        for strategy_name, selected in strategies.items():

            if selected is None:

                strategy_rows.append({
                    "snr_db": snr,
                    "strategy": strategy_name,
                    "pilot_length": 0,
                    "modulation": "OUTAGE",
                    "fresh_ber": np.nan,
                    "seed_ci_upper": np.nan,
                    "wilson_upper": np.nan,
                    "throughput_mbps": 0.0,
                    "outage": 1
                })

                continue

            pilot = selected["pilot_length"]
            modulation = selected["modulation"]

            config_key = (snr, pilot, modulation)

            if config_key not in holdout_cache:

                results = []

                for seed_number in range(NUMBER_OF_SEEDS):

                    # Completely different seed range from Stage 1.
                    seed = (
                        92000000
                        + snr * 100000
                        + pilot * 1000
                        + MODULATIONS.index(modulation) * 100
                        + seed_number
                    )

                    result = simulate_configuration(
                        snr,
                        pilot,
                        modulation,
                        seed
                    )

                    results.append(result)

                    holdout_rows.append({
                        "snr_db": snr,
                        "pilot_length": pilot,
                        "modulation": modulation,
                        "seed": seed_number + 1,
                        "ber": result["ber"],
                        "errors": result["errors"],
                        "total_bits": result["total_bits"]
                    })

                holdout_cache[config_key] = summarize_runs(
                    results
                )

            test = holdout_cache[config_key]

            # Only independently validated configurations
            # contribute throughput.
            if test["reliable"]:
                delivered_throughput = selected["throughput_mbps"]
                outage = 0
            else:
                delivered_throughput = 0.0
                outage = 1

            strategy_rows.append({
                "snr_db": snr,
                "strategy": strategy_name,
                "pilot_length": pilot,
                "modulation": modulation,
                "fresh_ber": test["mean_ber"],
                "seed_ci_upper": test["seed_ci_upper"],
                "wilson_upper": test["wilson_upper"],
                "throughput_mbps": delivered_throughput,
                "outage": outage
            })

        print(f"Holdout completed at {snr} dB")

    strategy_df = pd.DataFrame(strategy_rows)
    holdout_df = pd.DataFrame(holdout_rows)

    strategy_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "holdout_strategy_results.csv"
        ),
        index=False
    )

    holdout_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "holdout_raw_results.csv"
        ),
        index=False
    )

    return strategy_df, holdout_df


# ============================================================
# SUMMARY AND GRAPHS
# ============================================================

def create_summary(strategy_df):

    summary = (
        strategy_df.groupby("strategy")
        .agg(
            average_throughput_mbps=("throughput_mbps", "mean"),
            maximum_throughput_mbps=("throughput_mbps", "max"),
            outage_percentage=("outage", lambda x: 100 * x.mean())
        )
        .reset_index()
    )

    summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "reliability_aware_summary.csv"
        ),
        index=False
    )

    plt.figure(figsize=(10, 6))

    for name, group in strategy_df.groupby("strategy"):

        plt.plot(
            group["snr_db"],
            group["throughput_mbps"],
            marker="o",
            label=name
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Independently Validated Throughput (Mbps)")
    plt.title("Reliability-Aware Dynamic vs Fixed Pilot Strategies")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "reliability_aware_throughput.png"
        ),
        dpi=300
    )

    plt.close()

    plt.figure(figsize=(10, 6))

    plt.bar(
        summary["strategy"],
        summary["average_throughput_mbps"]
    )

    plt.ylabel("Average Validated Throughput (Mbps)")
    plt.title("Average Throughput Comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "reliability_aware_average_throughput.png"
        ),
        dpi=300
    )

    plt.close()

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nRELIABILITY-AWARE DYNAMIC SELECTION")
    print("-----------------------------------")
    print("SNR range:", SNR_VALUES)
    print("Pilot lengths:", PILOT_LENGTHS)
    print("Modulations:", MODULATIONS)
    print("Seeds per stage:", NUMBER_OF_SEEDS)
    print("Symbols per seed:", SYMBOLS_PER_SEED)
    print("Target BER:", TARGET_BER)

    selection_results, selection_df = run_selection_stage()

    strategy_df, holdout_df = run_holdout_stage(
        selection_results
    )

    summary = create_summary(strategy_df)

    print("\nSELECTION STAGE RESULTS")
    print(selection_df.to_string(index=False))

    print("\nINDEPENDENT HOLDOUT STRATEGY RESULTS")
    print(strategy_df.to_string(index=False))

    print("\nFINAL STRATEGY SUMMARY")
    print(summary.to_string(index=False))

    print("\nAll files saved in:", OUTPUT_DIR)

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()
