
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# MULTI-SEED STATISTICAL ANALYSIS - VERSION 2
# Uses existing simulation results. No new simulation required.
# ============================================================

INPUT_DIR = Path("multiseed_results")
OUTPUT_DIR = Path("multiseed_statistical_v2")
OUTPUT_DIR.mkdir(exist_ok=True)

INPUT_FILE = INPUT_DIR / "multiseed_ber_summary.csv"

TARGET_BER = 1e-3

PILOT_LENGTHS = [4, 8, 16, 32]

MODULATIONS = {
    "BPSK": 1,
    "QPSK": 2,
    "16-QAM": 4
}

DATA_BLOCK_LENGTH = 128

BANDWIDTH_HZ = 1e6
ROLLOFF = 0.25
CODE_RATE = 1.0

# ============================================================
# LOAD EXISTING RESULTS
# ============================================================

def load_results():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE}. "
            "Check your project folder."
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "Modulation",
        "Pilot_Length",
        "SNR_dB",
        "Mean_BER",
        "CI95_Upper",
        "Wilson_Upper"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing CSV columns: {missing}"
        )

    return df


# ============================================================
# CONSERVATIVE THRESHOLD CALCULATION
# ============================================================

def calculate_thresholds(df):

    results = []

    for modulation in MODULATIONS:

        for pilot_length in PILOT_LENGTHS:

            subset = df[
                (df["Modulation"] == modulation)
                & (df["Pilot_Length"] == pilot_length)
            ].sort_values("SNR_dB").reset_index(drop=True)

            threshold = None

            for i, row in subset.iterrows():

                higher_snr = subset.iloc[i:]

                seed_ci_pass = np.all(
                    higher_snr["CI95_Upper"].to_numpy()
                    <= TARGET_BER
                )

                wilson_pass = np.all(
                    higher_snr["Wilson_Upper"].to_numpy()
                    <= TARGET_BER
                )

                if seed_ci_pass and wilson_pass:

                    threshold = int(row["SNR_dB"])
                    break

            results.append({
                "Modulation": modulation,
                "Pilot_Length": pilot_length,
                "Conservative_Threshold_dB": threshold,
                "Threshold_Status": (
                    "PASSED"
                    if threshold is not None
                    else "NOT_ESTABLISHED"
                )
            })

    return pd.DataFrame(results)


# ============================================================
# CONFIDENCE INTERVAL ACCEPTANCE TABLE
# ============================================================

def create_acceptance_table(df):

    output = df.copy()

    output["Seed_CI_Pass"] = (
        output["CI95_Upper"] <= TARGET_BER
    )

    output["Wilson_CI_Pass"] = (
        output["Wilson_Upper"] <= TARGET_BER
    )

    output["Both_Criteria_Pass"] = (
        output["Seed_CI_Pass"]
        & output["Wilson_CI_Pass"]
    )

    output.to_csv(
        OUTPUT_DIR / "confidence_interval_acceptance.csv",
        index=False
    )

    return output


# ============================================================
# ADAPTIVE THROUGHPUT
# ============================================================

def calculate_throughput(threshold_df):

    symbol_rate = BANDWIDTH_HZ / (1 + ROLLOFF)

    results = []

    for pilot_length in PILOT_LENGTHS:

        pilot_efficiency = (
            DATA_BLOCK_LENGTH
            / (DATA_BLOCK_LENGTH + pilot_length)
        )

        thresholds = {}

        for modulation in MODULATIONS:

            row = threshold_df[
                (threshold_df["Modulation"] == modulation)
                & (threshold_df["Pilot_Length"] == pilot_length)
            ]

            thresholds[modulation] = (
                row["Conservative_Threshold_dB"].iloc[0]
            )

        for snr_db in range(20, 33):

            selected = "OUTAGE"
            throughput = 0.0

            # Highest modulation order meeting both criteria
            for modulation in ["BPSK", "QPSK", "16-QAM"]:

                threshold = thresholds[modulation]

                if (
                    pd.notna(threshold)
                    and snr_db >= threshold
                ):

                    selected = modulation

                    throughput = (
                        symbol_rate
                        * MODULATIONS[modulation]
                        * CODE_RATE
                        * pilot_efficiency
                    )

            results.append({
                "Pilot_Length": pilot_length,
                "SNR_dB": snr_db,
                "Selected_Modulation": selected,
                "Throughput_Mbps": throughput / 1e6
            })

    return pd.DataFrame(results)


# ============================================================
# PLOT 1: MEAN BER WITH SEED-LEVEL CONFIDENCE INTERVALS
# ============================================================

def plot_ber(df):

    for modulation in MODULATIONS:

        plt.figure(figsize=(10, 6))

        for pilot_length in PILOT_LENGTHS:

            subset = df[
                (df["Modulation"] == modulation)
                & (df["Pilot_Length"] == pilot_length)
            ].sort_values("SNR_dB")

            lower_error = np.maximum(
                0,
                subset["Mean_BER"].to_numpy()
                - subset["CI95_Lower"].to_numpy()
            )

            upper_error = np.maximum(
                0,
                subset["CI95_Upper"].to_numpy()
                - subset["Mean_BER"].to_numpy()
            )

            plt.errorbar(
                subset["SNR_dB"],
                subset["Mean_BER"],
                yerr=[lower_error, upper_error],
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
            f"{modulation}: Conservative Statistical Analysis"
        )

        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"{modulation}_statistical_BER.png",
            dpi=300
        )

        plt.close()


# ============================================================
# PLOT 2: ADAPTIVE THROUGHPUT
# ============================================================

def plot_throughput(throughput_df):

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
    plt.title(
        "Adaptive Throughput Using Conservative BER Thresholds"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "conservative_adaptive_throughput.png",
        dpi=300
    )

    plt.close()


# ============================================================
# PLOT 3: MAXIMUM THROUGHPUT COMPARISON
# ============================================================

def plot_maximum_throughput(throughput_df):

    maximum = throughput_df.groupby(
        "Pilot_Length"
    )["Throughput_Mbps"].max()

    plt.figure(figsize=(9, 6))

    plt.bar(
        maximum.index.astype(str),
        maximum.values
    )

    plt.xlabel("Pilot Length")
    plt.ylabel("Maximum Adaptive Throughput (Mbps)")
    plt.title("Maximum Throughput Comparison")

    plt.grid(axis="y")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "maximum_throughput_comparison.png",
        dpi=300
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("CONSERVATIVE STATISTICAL ANALYSIS - VERSION 2")
    print("=" * 65)

    df = load_results()

    print("\nExisting simulation data loaded successfully.")

    acceptance_df = create_acceptance_table(df)

    threshold_df = calculate_thresholds(df)

    threshold_df.to_csv(
        OUTPUT_DIR / "conservative_thresholds.csv",
        index=False
    )

    throughput_df = calculate_throughput(threshold_df)

    throughput_df.to_csv(
        OUTPUT_DIR / "conservative_adaptive_throughput.csv",
        index=False
    )

    # Generate plots
    plot_ber(df)
    plot_throughput(throughput_df)
    plot_maximum_throughput(throughput_df)

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\nCONSERVATIVE BER THRESHOLDS")
    print("-" * 65)

    print(
        threshold_df.to_string(index=False)
    )

    print("\nCONFIDENCE INTERVAL ACCEPTANCE")
    print("-" * 65)

    passed = acceptance_df[
        "Both_Criteria_Pass"
    ].sum()

    total = len(acceptance_df)

    print(f"Passing individual SNR tests: {passed}/{total}")

    print("\nMAXIMUM ADAPTIVE THROUGHPUT")
    print("-" * 65)

    maximum = throughput_df.groupby(
        "Pilot_Length"
    )["Throughput_Mbps"].max()

    print(maximum.to_string())

    print("\nOUTPUT DIRECTORY")
    print(OUTPUT_DIR.resolve())

    print("\nStatistical analysis completed successfully.")


if __name__ == "__main__":
    main()
