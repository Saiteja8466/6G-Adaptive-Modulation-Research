
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# COMPREHENSIVE 6G ADAPTIVE SYSTEM COMPARISON
# ============================================================

OUTPUT_DIR = Path("comprehensive_adaptive_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Validated conservative thresholds (dB)
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

PILOT_LENGTHS = [4, 8, 16, 32]

MODULATION_BITS = {
    "BPSK": 1,
    "QPSK": 2,
    "16-QAM": 4
}

DATA_BLOCK_LENGTH = 128
BANDWIDTH_HZ = 1e6
ROLLOFF = 0.25
CODE_RATE = 1.0
TARGET_BER = 1e-3

SNR_RANGE = np.arange(20, 33, 1)

# ============================================================
# SYSTEM CALCULATIONS
# ============================================================

def calculate_system():

    results = []

    symbol_rate = BANDWIDTH_HZ / (1 + ROLLOFF)

    for pilot_length in PILOT_LENGTHS:

        pilot_overhead = (
            pilot_length /
            (pilot_length + DATA_BLOCK_LENGTH)
        )

        data_efficiency = 1 - pilot_overhead

        for snr_db in SNR_RANGE:

            selected_modulation = "OUTAGE"
            bits_per_symbol = 0

            # Select highest modulation whose
            # validated BER threshold is satisfied.
            for modulation in ["BPSK", "QPSK", "16-QAM"]:

                required_snr = THRESHOLDS[
                    pilot_length
                ][modulation]

                if snr_db >= required_snr:

                    selected_modulation = modulation

                    bits_per_symbol = MODULATION_BITS[
                        modulation
                    ]

            if selected_modulation == "OUTAGE":
                throughput = 0.0
                spectral_efficiency = 0.0
                outage = 1
            else:

                throughput = (
                    symbol_rate
                    * bits_per_symbol
                    * CODE_RATE
                    * data_efficiency
                )

                spectral_efficiency = (
                    throughput / BANDWIDTH_HZ
                )

                outage = 0

            results.append({

                "Pilot_Length": pilot_length,

                "SNR_dB": snr_db,

                "Selected_Modulation": selected_modulation,

                "Bits_Per_Symbol": bits_per_symbol,

                "Pilot_Overhead_Percent": (
                    pilot_overhead * 100
                ),

                "Throughput_Mbps": (
                    throughput / 1e6
                ),

                "Spectral_Efficiency_bps_Hz": (
                    spectral_efficiency
                ),

                "Outage": outage

            })

    return pd.DataFrame(results)


# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

def calculate_summary(df):

    summary = []

    for pilot_length in PILOT_LENGTHS:

        subset = df[
            df["Pilot_Length"] == pilot_length
        ]

        average_throughput = (
            subset["Throughput_Mbps"].mean()
        )

        maximum_throughput = (
            subset["Throughput_Mbps"].max()
        )

        average_spectral_efficiency = (
            subset["Spectral_Efficiency_bps_Hz"].mean()
        )

        outage_fraction = (
            subset["Outage"].mean()
        )

        summary.append({

            "Pilot_Length": pilot_length,

            "Pilot_Overhead_Percent": (
                pilot_length /
                (pilot_length + DATA_BLOCK_LENGTH)
                * 100
            ),

            "Average_Throughput_Mbps": (
                average_throughput
            ),

            "Maximum_Throughput_Mbps": (
                maximum_throughput
            ),

            "Average_Spectral_Efficiency_bps_Hz": (
                average_spectral_efficiency
            ),

            "Outage_Fraction_Percent": (
                outage_fraction * 100
            )

        })

    return pd.DataFrame(summary)


# ============================================================
# GRAPH 1: ADAPTIVE THROUGHPUT
# ============================================================

def plot_throughput(df):

    plt.figure(figsize=(10, 6))

    for pilot_length in PILOT_LENGTHS:

        subset = df[
            df["Pilot_Length"] == pilot_length
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
        "Adaptive Throughput Comparison"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "adaptive_throughput_comparison.png",
        dpi=300
    )

    plt.close()


# ============================================================
# GRAPH 2: SPECTRAL EFFICIENCY
# ============================================================

def plot_spectral_efficiency(df):

    plt.figure(figsize=(10, 6))

    for pilot_length in PILOT_LENGTHS:

        subset = df[
            df["Pilot_Length"] == pilot_length
        ]

        plt.plot(
            subset["SNR_dB"],
            subset["Spectral_Efficiency_bps_Hz"],
            marker="o",
            label=f"{pilot_length} pilots"
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Spectral Efficiency (bps/Hz)")

    plt.title(
        "Adaptive Spectral Efficiency Comparison"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "spectral_efficiency_comparison.png",
        dpi=300
    )

    plt.close()


# ============================================================
# GRAPH 3: PILOT OVERHEAD
# ============================================================

def plot_pilot_overhead():

    overhead = [
        p / (p + DATA_BLOCK_LENGTH) * 100
        for p in PILOT_LENGTHS
    ]

    plt.figure(figsize=(9, 6))

    plt.bar(
        [str(p) for p in PILOT_LENGTHS],
        overhead
    )

    plt.xlabel("Pilot Length")
    plt.ylabel("Pilot Overhead (%)")

    plt.title("Pilot Overhead Comparison")

    plt.grid(axis="y")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "pilot_overhead_comparison.png",
        dpi=300
    )

    plt.close()


# ============================================================
# GRAPH 4: MAXIMUM AND AVERAGE THROUGHPUT
# ============================================================

def plot_summary(summary):

    x = np.arange(len(PILOT_LENGTHS))
    width = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(
        x - width / 2,
        summary["Average_Throughput_Mbps"],
        width,
        label="Average"
    )

    plt.bar(
        x + width / 2,
        summary["Maximum_Throughput_Mbps"],
        width,
        label="Maximum"
    )

    plt.xticks(
        x,
        [str(p) for p in PILOT_LENGTHS]
    )

    plt.xlabel("Pilot Length")
    plt.ylabel("Throughput (Mbps)")

    plt.title("Average vs Maximum Throughput")

    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "throughput_summary.png",
        dpi=300
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("COMPREHENSIVE ADAPTIVE SYSTEM COMPARISON")
    print("=" * 65)

    print("\nUsing validated conservative thresholds.")

    df = calculate_system()

    summary = calculate_summary(df)

    # Save CSV results
    df.to_csv(
        OUTPUT_DIR / "adaptive_system_detailed_results.csv",
        index=False
    )

    summary.to_csv(
        OUTPUT_DIR / "adaptive_system_summary.csv",
        index=False
    )

    # Generate graphs
    plot_throughput(df)

    plot_spectral_efficiency(df)

    plot_pilot_overhead()

    plot_summary(summary)

    # Display detailed results
    print("\nADAPTIVE MODULATION RESULTS")
    print("-" * 65)

    print(
        df.to_string(index=False)
    )

    print("\nPILOT LENGTH PERFORMANCE SUMMARY")
    print("-" * 65)

    print(
        summary.round(4).to_string(index=False)
    )

    # Identify highest average throughput
    best_row = summary.loc[
        summary["Average_Throughput_Mbps"].idxmax()
    ]

    print("\nBEST CONFIGURATION BY AVERAGE THROUGHPUT")

    print(
        f"Pilot length: {int(best_row['Pilot_Length'])}"
    )

    print(
        "Average throughput:",
        round(
            best_row["Average_Throughput_Mbps"], 4
        ),
        "Mbps"
    )

    print("\nAll files saved in:")
    print(OUTPUT_DIR.resolve())

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()
