
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from adaptive_rayleigh_refined import (
    MODULATIONS,
    modulate,
    demodulate,
    BANDWIDTH_HZ,
    ROLLOFF,
    CODE_RATE,
    TARGET_BER
)

# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================

CSI_CONDITIONS = {
    "Perfect CSI": 0.0,
    "Very Accurate CSI": 0.001,
    "Accurate CSI": 0.005,
    "Moderate CSI": 0.01,
    "Poor CSI": 0.02,
    "Very Poor CSI": 0.05
}

CALIBRATION_SNR_DB = np.arange(18, 41, 1)
VALIDATION_SYMBOLS = 200_000
CALIBRATION_SYMBOLS = 50_000

BASE_SEED = 20261111

# Throughput assumptions
SYMBOL_RATE = BANDWIDTH_HZ / (1 + ROLLOFF)

# ============================================================
# 2. BER SIMULATION WITH IMPERFECT CSI
# ============================================================

def simulate_ber_csi(
    modulation,
    snr_db,
    num_symbols,
    csi_error_variance,
    seed
):
    rng = np.random.default_rng(seed)

    bits_per_symbol = MODULATIONS[modulation]

    total_bits = num_symbols * bits_per_symbol

    # Generate random information bits
    bits = rng.integers(0, 2, total_bits)

    # Modulate
    transmitted = modulate(bits, modulation)

    # Rayleigh fading channel: CN(0,1)
    h = (
        rng.normal(0, 1 / np.sqrt(2), num_symbols)
        + 1j * rng.normal(0, 1 / np.sqrt(2), num_symbols)
    )

    # Convert Eb/N0 to linear scale
    ebn0_linear = 10 ** (snr_db / 10)

    # AWGN noise variance
    noise_variance = 1 / (
        bits_per_symbol * ebn0_linear
    )

    noise = np.sqrt(noise_variance / 2) * (
        rng.normal(size=num_symbols)
        + 1j * rng.normal(size=num_symbols)
    )

    # Received signal
    received = h * transmitted + noise

    # Channel estimation error
    estimation_error = np.sqrt(
        csi_error_variance / 2
    ) * (
        rng.normal(size=num_symbols)
        + 1j * rng.normal(size=num_symbols)
    )

    # Estimated channel
    h_estimated = h + estimation_error

    # Avoid numerical division by zero
    safe_h = np.where(
        np.abs(h_estimated) < 1e-12,
        1e-12 + 0j,
        h_estimated
    )

    # Equalization using estimated channel
    equalized = received / safe_h

    # Demodulation
    detected_bits = demodulate(equalized, modulation)

    # Calculate BER
    errors = np.count_nonzero(bits != detected_bits)

    ber = errors / total_bits

    return ber, errors, total_bits


# ============================================================
# 3. CALIBRATION OF SNR THRESHOLDS
# ============================================================

def calibrate_thresholds():

    print("\n" + "=" * 65)
    print("SNR THRESHOLD CALIBRATION")
    print("=" * 65)

    threshold_records = []
    calibration_records = []

    for condition_index, (condition, error_variance) in enumerate(
        CSI_CONDITIONS.items()
    ):

        print(f"\nCSI Condition: {condition}")
        print(f"Error Variance: {error_variance}")

        for modulation_index, modulation in enumerate(MODULATIONS):

            print(f"\n  Testing {modulation}")

            first_valid_snr = None

            for snr_index, snr_db in enumerate(CALIBRATION_SNR_DB):

                seed = (
                    BASE_SEED
                    + condition_index * 100000
                    + modulation_index * 10000
                    + snr_index
                )

                ber, errors, total_bits = simulate_ber_csi(
                    modulation,
                    snr_db,
                    CALIBRATION_SYMBOLS,
                    error_variance,
                    seed
                )

                calibration_records.append({
                    "CSI Condition": condition,
                    "Error Variance": error_variance,
                    "Modulation": modulation,
                    "SNR (dB)": snr_db,
                    "BER": ber,
                    "Errors": errors,
                    "Total Bits": total_bits
                })

                print(
                    f"    SNR = {snr_db:2.0f} dB"
                    f" | BER = {ber:.6e}"
                )

                if ber <= TARGET_BER:
                    first_valid_snr = snr_db
                    break

            if first_valid_snr is not None:

                # Add a 0.5 dB safety margin
                proposed_threshold = first_valid_snr + 0.5

                print(
                    f"  Initial threshold: "
                    f"{proposed_threshold:.1f} dB"
                )

            else:
                proposed_threshold = None

                print(
                    "  No threshold found in calibration range."
                )

            threshold_records.append({
                "CSI Condition": condition,
                "Error Variance": error_variance,
                "Modulation": modulation,
                "Initial Threshold (dB)": proposed_threshold
            })

    threshold_df = pd.DataFrame(threshold_records)
    calibration_df = pd.DataFrame(calibration_records)

    threshold_df.to_csv(
        "Imperfect_CSI_Initial_Thresholds.csv",
        index=False
    )

    calibration_df.to_csv(
        "Imperfect_CSI_Calibration_Results.csv",
        index=False
    )

    return threshold_df, calibration_df


# ============================================================
# 4. INDEPENDENT THRESHOLD VALIDATION
# ============================================================

def validate_thresholds(threshold_df):

    print("\n" + "=" * 65)
    print("INDEPENDENT THRESHOLD VALIDATION")
    print("=" * 65)

    validation_records = []

    for index, row in threshold_df.iterrows():

        condition = row["CSI Condition"]
        error_variance = row["Error Variance"]
        modulation = row["Modulation"]

        threshold = row["Initial Threshold (dB)"]

        if pd.isna(threshold):

            validation_records.append({
                "CSI Condition": condition,
                "Error Variance": error_variance,
                "Modulation": modulation,
                "Validated Threshold (dB)": np.nan,
                "Validation BER": np.nan,
                "Errors": np.nan,
                "Total Bits": np.nan,
                "Validation Status": "NO THRESHOLD FOUND"
            })

            continue

        validated_threshold = None
        validation_ber = np.nan
        validation_errors = np.nan
        validation_total_bits = np.nan

        # If validation fails, increase SNR gradually
        candidate_thresholds = np.arange(
            threshold,
            41,
            0.5
        )

        for candidate_index, candidate_snr in enumerate(
            candidate_thresholds
        ):

            seed = (
                BASE_SEED
                + 900000
                + index * 10000
                + candidate_index
            )

            ber, errors, total_bits = simulate_ber_csi(
                modulation,
                candidate_snr,
                VALIDATION_SYMBOLS,
                error_variance,
                seed
            )

            print(
                f"{condition:20s} | "
                f"{modulation:6s} | "
                f"{candidate_snr:5.1f} dB | "
                f"BER = {ber:.6e}"
            )

            if ber <= TARGET_BER:

                validated_threshold = candidate_snr
                validation_ber = ber
                validation_errors = errors
                validation_total_bits = total_bits

                break

        if validated_threshold is not None:
            status = "PASSED"

        else:
            status = "FAILED - TARGET NOT MET"

        validation_records.append({
            "CSI Condition": condition,
            "Error Variance": error_variance,
            "Modulation": modulation,
            "Validated Threshold (dB)": validated_threshold,
            "Validation BER": validation_ber,
            "Errors": validation_errors,
            "Total Bits": validation_total_bits,
            "Validation Status": status
        })

    validation_df = pd.DataFrame(validation_records)

    validation_df.to_csv(
        "Imperfect_CSI_Validation_Results.csv",
        index=False
    )

    return validation_df


# ============================================================
# 5. ADAPTIVE MODULATION SELECTION
# ============================================================

def select_modulation(snr_db, condition_validation):

    eligible_modulations = []

    for _, row in condition_validation.iterrows():

        threshold = row["Validated Threshold (dB)"]

        if pd.isna(threshold):
            continue

        if row["Validation Status"] != "PASSED":
            continue

        if snr_db >= threshold:

            eligible_modulations.append(
                row["Modulation"]
            )

    if not eligible_modulations:
        return "OUTAGE"

    # Select the modulation with the highest spectral efficiency
    return max(
        eligible_modulations,
        key=lambda modulation: MODULATIONS[modulation]
    )


# ============================================================
# 6. ADAPTIVE THROUGHPUT AND SPECTRAL EFFICIENCY
# ============================================================

def evaluate_adaptive_system(validation_df):

    print("\n" + "=" * 65)
    print("ADAPTIVE SYSTEM PERFORMANCE")
    print("=" * 65)

    snr_values = np.arange(18, 41, 1)

    adaptive_records = []

    for condition in CSI_CONDITIONS:

        condition_validation = validation_df[
            validation_df["CSI Condition"] == condition
        ]

        print(f"\nCSI Condition: {condition}")

        for snr_db in snr_values:

            selected = select_modulation(
                snr_db,
                condition_validation
            )

            if selected == "OUTAGE":

                spectral_efficiency = 0.0
                throughput_mbps = 0.0

            else:

                bits_per_symbol = MODULATIONS[selected]

                spectral_efficiency = (
                    bits_per_symbol * CODE_RATE
                ) / (1 + ROLLOFF)

                throughput_bps = (
                    SYMBOL_RATE
                    * bits_per_symbol
                    * CODE_RATE
                )

                throughput_mbps = throughput_bps / 1e6

            adaptive_records.append({
                "CSI Condition": condition,
                "SNR (dB)": snr_db,
                "Selected Modulation": selected,
                "Spectral Efficiency (bits/s/Hz)": spectral_efficiency,
                "Throughput (Mbps)": throughput_mbps
            })

            print(
                f"SNR = {snr_db:2.0f} dB"
                f" | Mode = {selected:7s}"
                f" | Throughput = {throughput_mbps:.3f} Mbps"
            )

    adaptive_df = pd.DataFrame(adaptive_records)

    adaptive_df.to_csv(
        "Imperfect_CSI_Adaptive_Performance.csv",
        index=False
    )

    return adaptive_df


# ============================================================
# 7. FIXED MODULATION VS ADAPTIVE MODULATION
# ============================================================

def fixed_vs_adaptive(adaptive_df):

    records = []

    for condition in CSI_CONDITIONS:

        condition_df = adaptive_df[
            adaptive_df["CSI Condition"] == condition
        ]

        for _, row in condition_df.iterrows():

            snr_db = row["SNR (dB)"]

            adaptive_throughput = row["Throughput (Mbps)"]

            for modulation, bits_per_symbol in MODULATIONS.items():

                fixed_throughput = (
                    SYMBOL_RATE
                    * bits_per_symbol
                    * CODE_RATE
                ) / 1e6

                records.append({
                    "CSI Condition": condition,
                    "SNR (dB)": snr_db,
                    "Modulation": modulation,
                    "Fixed Throughput (Mbps)": fixed_throughput,
                    "Adaptive Throughput (Mbps)": adaptive_throughput
                })

    comparison_df = pd.DataFrame(records)

    comparison_df.to_csv(
        "Imperfect_CSI_Fixed_vs_Adaptive.csv",
        index=False
    )

    return comparison_df


# ============================================================
# 8. PLOT CALIBRATION BER RESULTS
# ============================================================

def plot_calibration_results(calibration_df):

    for modulation in MODULATIONS:

        plt.figure(figsize=(10, 6))

        for condition in CSI_CONDITIONS:

            data = calibration_df[
                (calibration_df["CSI Condition"] == condition)
                & (calibration_df["Modulation"] == modulation)
            ]

            if data.empty:
                continue

            plt.semilogy(
                data["SNR (dB)"],
                data["BER"],
                marker="o",
                markersize=3,
                label=condition
            )

        plt.axhline(
            TARGET_BER,
            linestyle="--",
            label="Target BER = 10^-3"
        )

        plt.xlabel("SNR (dB)")
        plt.ylabel("Bit Error Rate (BER)")
        plt.title(
            f"{modulation}: BER Under Different CSI Conditions"
        )

        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()

        filename = (
            modulation.replace("-", "")
            .replace(" ", "")
        )

        plt.savefig(
            f"Imperfect_CSI_{filename}_BER.png",
            dpi=300
        )

        plt.show()


# ============================================================
# 9. PLOT ADAPTIVE THROUGHPUT
# ============================================================

def plot_adaptive_throughput(adaptive_df):

    plt.figure(figsize=(10, 6))

    for condition in CSI_CONDITIONS:

        data = adaptive_df[
            adaptive_df["CSI Condition"] == condition
        ]

        plt.plot(
            data["SNR (dB)"],
            data["Throughput (Mbps)"],
            marker="o",
            markersize=3,
            label=condition
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Throughput (Mbps)")
    plt.title("Adaptive Throughput Under Different CSI Conditions")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "Imperfect_CSI_Adaptive_Throughput.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

def print_final_summary(validation_df):

    print("\n" + "=" * 75)
    print("FINAL VALIDATED SNR THRESHOLDS")
    print("=" * 75)

    print(
        validation_df[
            [
                "CSI Condition",
                "Modulation",
                "Validated Threshold (dB)",
                "Validation BER",
                "Validation Status"
            ]
        ].to_string(index=False)
    )

    print("\nExperiment assumptions:")
    print(f"Target BER       : {TARGET_BER}")
    print(f"Bandwidth        : {BANDWIDTH_HZ / 1e6:.2f} MHz")
    print(f"Roll-off factor  : {ROLLOFF}")
    print(f"Code rate        : {CODE_RATE}")
    print("Channel          : Rayleigh fading")
    print("CSI model        : h_estimated = h + estimation_error")
    print("Validation       : Independent random seeds")


# ============================================================
# 11. MAIN PROGRAM
# ============================================================

def main():

    print("\n")
    print("=" * 75)
    print("6G ADAPTIVE MODULATION WITH IMPERFECT CSI")
    print("=" * 75)

    print("\nModulations:", list(MODULATIONS.keys()))
    print("Target BER:", TARGET_BER)

    # Step 1: Calibration
    threshold_df, calibration_df = calibrate_thresholds()

    # Step 2: Independent validation
    validation_df = validate_thresholds(threshold_df)

    # Step 3: Adaptive modulation
    adaptive_df = evaluate_adaptive_system(validation_df)

    # Step 4: Fixed vs adaptive comparison
    comparison_df = fixed_vs_adaptive(adaptive_df)

    # Step 5: Generate plots
    plot_calibration_results(calibration_df)

    plot_adaptive_throughput(adaptive_df)

    # Step 6: Final summary
    print_final_summary(validation_df)

    print("\n" + "=" * 75)
    print("EXPERIMENT COMPLETED")
    print("=" * 75)

    print("\nGenerated CSV files:")
    print("1. Imperfect_CSI_Initial_Thresholds.csv")
    print("2. Imperfect_CSI_Calibration_Results.csv")
    print("3. Imperfect_CSI_Validation_Results.csv")
    print("4. Imperfect_CSI_Adaptive_Performance.csv")
    print("5. Imperfect_CSI_Fixed_vs_Adaptive.csv")

    print("\nGenerated plots:")
    print("1. Imperfect_CSI_BPSK_BER.png")
    print("2. Imperfect_CSI_QPSK_BER.png")
    print("3. Imperfect_CSI_16QAM_BER.png")
    print("4. Imperfect_CSI_Adaptive_Throughput.png")


if __name__ == "__main__":
    main()
