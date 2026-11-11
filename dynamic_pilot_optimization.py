
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# DYNAMIC PILOT LENGTH AND ADAPTIVE MODULATION OPTIMIZATION
# ============================================================

OUTPUT_DIR = Path("dynamic_pilot_results")
OUTPUT_DIR.mkdir(exist_ok=True)

PILOT_LENGTHS = [4, 8, 16, 32]

MODULATION_BITS = {
    "BPSK": 1,
    "QPSK": 2,
    "16-QAM": 4
}

# Conservative thresholds obtained from previous experiments
THRESHOLDS = {
    4: {
        "BPSK": 27,
        "QPSK": 27,
        "16-QAM": 28
    },
    8: {
        "BPSK": 26,
        "QPSK": 25,
        "16-QAM": 29
    },
    16: {
        "BPSK": 27,
        "QPSK": 25,
        "16-QAM": 28
    },
    32: {
        "BPSK": 26,
        "QPSK": 27,
        "16-QAM": 29
    }
}

SNR_VALUES = np.arange(20, 33, 1)

DATA_BLOCK_LENGTH = 128
BANDWIDTH_HZ = 1e6
ROLLOFF = 0.25
CODE_RATE = 1.0

TARGET_BER = 1e-3

# ============================================================
# THROUGHPUT CALCULATION
# ============================================================

def calculate_throughput(pilot_length, modulation):

    symbol_rate = BANDWIDTH_HZ / (1 + ROLLOFF)

    data_efficiency = (
        DATA_BLOCK_LENGTH /
        (DATA_BLOCK_LENGTH + pilot_length)
    )

    bits_per_symbol = MODULATION_BITS[modulation]

    throughput = (
        symbol_rate
        * bits_per_symbol
        * CODE_RATE
        * data_efficiency
    )

    return throughput / 1e6


# ============================================================
# DYNAMIC OPTIMIZATION
# ============================================================

def optimize_dynamic_system():

    results = []

    for snr_db in SNR_VALUES:

        feasible_options = []

        for pilot_length in PILOT_LENGTHS:

            for modulation in MODULATION_BITS:

                required_snr = THRESHOLDS[
                    pilot_length
                ][modulation]

                if snr_db >= required_snr:

                    throughput = calculate_throughput(
                        pilot_length,
                        modulation
                    )

                    feasible_options.append({

                        "SNR_dB": int(snr_db),

                        "Pilot_Length": pilot_length,

                        "Modulation": modulation,

                        "Required_SNR_dB": required_snr,

                        "Throughput_Mbps": throughput

                    })

        if feasible_options:

            best_option = max(
                feasible_options,
                key=lambda x: x["Throughput_Mbps"]
            )

            results.append({

                "SNR_dB": int(snr_db),

                "Selected_Pilot_Length":
                    best_option["Pilot_Length"],

                "Selected_Modulation":
                    best_option["Modulation"],

                "Required_SNR_dB":
                    best_option["Required_SNR_dB"],

                "Dynamic_Throughput_Mbps":
                    best_option["Throughput_Mbps"],

                "Outage": 0

            })

        else:

            results.append({

                "SNR_dB": int(snr_db),

                "Selected_Pilot_Length": None,

                "Selected_Modulation": "OUTAGE",

                "Required_SNR_dB": None,

                "Dynamic_Throughput_Mbps": 0.0,

                "Outage": 1

            })

    return pd.DataFrame(results)


# ============================================================
# FIXED PILOT STRATEGY FOR COMPARISON
# ============================================================

def calculate_fixed_strategies():

    results = []

    for pilot_length in PILOT_LENGTHS:

        for snr_db in SNR_VALUES:

            selected_modulation = "OUTAGE"

            throughput = 0.0

            for modulation in MODULATION_BITS:

                required_snr = THRESHOLDS[
                    pilot_length
                ][modulation]

                if snr_db >= required_snr:

                    selected_modulation = modulation

                    throughput = calculate_throughput(
                        pilot_length,
                        modulation
                    )

            results.append({

                "Pilot_Length": pilot_length,

                "SNR_dB": int(snr_db),

                "Modulation": selected_modulation,

                "Throughput_Mbps": throughput,

                "Outage": int(throughput == 0)

            })

    return pd.DataFrame(results)


# ============================================================
# SUMMARY COMPARISON
# ============================================================

def calculate_summary(dynamic_df, fixed_df):

    results = []

    dynamic_average = (
        dynamic_df["Dynamic_Throughput_Mbps"].mean()
    )

    dynamic_outage = (
        dynamic_df["Outage"].mean() * 100
    )

    results.append({

        "Strategy": "Dynamic Pilot Selection",

        "Average_Throughput_Mbps": dynamic_average,

        "Maximum_Throughput_Mbps":
            dynamic_df["Dynamic_Throughput_Mbps"].max(),

        "Outage_Percent": dynamic_outage

    })

    for pilot_length in PILOT_LENGTHS:

        subset = fixed_df[
            fixed_df["Pilot_Length"] == pilot_length
        ]

        results.append({

            "Strategy": f"Fixed {pilot_length} Pilots",

            "Average_Throughput_Mbps":
                subset["Throughput_Mbps"].mean(),

            "Maximum_Throughput_Mbps":
                subset["Throughput_Mbps"].max(),

            "Outage_Percent":
                subset["Outage"].mean() * 100

        })

    return pd.DataFrame(results)


# ============================================================
# GRAPH 1: DYNAMIC VS FIXED THROUGHPUT
# ============================================================

def plot_throughput(dynamic_df, fixed_df):

    plt.figure(figsize=(11, 6))

    plt.plot(
        dynamic_df["SNR_dB"],
        dynamic_df["Dynamic_Throughput_Mbps"],
        marker="o",
        linewidth=2.5,
        label="Dynamic Pilot Selection"
    )

    for pilot_length in PILOT_LENGTHS:

        subset = fixed_df[
            fixed_df["Pilot_Length"] == pilot_length
        ]

        plt.plot(
            subset["SNR_dB"],
            subset["Throughput_Mbps"],
            marker=".",
            linestyle="--",
            label=f"Fixed {pilot_length} pilots"
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Throughput (Mbps)")

    plt.title(
        "Dynamic Pilot Selection vs Fixed Pilot Strategies"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "dynamic_vs_fixed_throughput.png",
        dpi=300
    )

    plt.close()


# ============================================================
# GRAPH 2: SELECTED PILOT LENGTH
# ============================================================

def plot_pilot_selection(dynamic_df):

    valid = dynamic_df[
        dynamic_df["Outage"] == 0
    ]

    plt.figure(figsize=(10, 5))

    plt.step(
        valid["SNR_dB"],
        valid["Selected_Pilot_Length"],
        where="mid",
        marker="o"
    )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Selected Pilot Length")

    plt.title("Dynamic Pilot Length Selection vs SNR")

    plt.yticks(PILOT_LENGTHS)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "dynamic_pilot_selection.png",
        dpi=300
    )

    plt.close()


# ============================================================
# GRAPH 3: SELECTED MODULATION
# ============================================================

def plot_modulation_selection(dynamic_df):

    mapping = {
        "OUTAGE": 0,
        "BPSK": 1,
        "QPSK": 2,
        "16-QAM": 3
    }

    values = dynamic_df[
        "Selected_Modulation"
    ].map(mapping)

    plt.figure(figsize=(10, 5))

    plt.step(
        dynamic_df["SNR_dB"],
        values,
        where="mid",
        marker="o"
    )

    plt.yticks(
        [0, 1, 2, 3],
        ["OUTAGE", "BPSK", "QPSK", "16-QAM"]
    )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Selected Modulation")

    plt.title("Adaptive Modulation Selection")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "dynamic_modulation_selection.png",
        dpi=300
    )

    plt.close()


# ============================================================
# GRAPH 4: STRATEGY SUMMARY
# ============================================================

def plot_strategy_summary(summary):

    plt.figure(figsize=(11, 6))

    plt.bar(
        summary["Strategy"],
        summary["Average_Throughput_Mbps"]
    )

    plt.xlabel("Strategy")
    plt.ylabel("Average Throughput (Mbps)")

    plt.title("Average Throughput: Dynamic vs Fixed Strategies")

    plt.xticks(rotation=20)
    plt.grid(axis="y")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "dynamic_strategy_comparison.png",
        dpi=300
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("DYNAMIC PILOT LENGTH OPTIMIZATION")
    print("=" * 65)

    print("\nOptimizing pilot length and modulation by SNR.")

    dynamic_df = optimize_dynamic_system()

    fixed_df = calculate_fixed_strategies()

    summary_df = calculate_summary(
        dynamic_df,
        fixed_df
    )

    # Save CSV files
    dynamic_df.to_csv(
        OUTPUT_DIR / "dynamic_optimization_results.csv",
        index=False
    )

    fixed_df.to_csv(
        OUTPUT_DIR / "fixed_strategy_results.csv",
        index=False
    )

    summary_df.to_csv(
        OUTPUT_DIR / "dynamic_strategy_summary.csv",
        index=False
    )

    # Generate graphs
    plot_throughput(dynamic_df, fixed_df)

    plot_pilot_selection(dynamic_df)

    plot_modulation_selection(dynamic_df)

    plot_strategy_summary(summary_df)

    # Display results
    print("\nDYNAMIC OPTIMIZATION RESULTS")
    print("-" * 65)

    print(dynamic_df.to_string(index=False))

    print("\nFIXED VS DYNAMIC STRATEGY SUMMARY")
    print("-" * 65)

    print(summary_df.round(4).to_string(index=False))

    print("\nFiles saved in:")
    print(OUTPUT_DIR.resolve())

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()
