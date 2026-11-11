
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

PILOT_LENGTH = 16
DATA_BLOCK_LENGTH = 128

CALIBRATION_SYMBOLS = 100_000
VALIDATION_SYMBOLS = 300_000

SNR_RANGE_DB = np.arange(18, 41, 1)

BASE_SEED = 20261112

SYMBOL_RATE = BANDWIDTH_HZ / (1 + ROLLOFF)

# Pilot overhead
PILOT_OVERHEAD = (
    PILOT_LENGTH / (PILOT_LENGTH + DATA_BLOCK_LENGTH)
)

DATA_EFFICIENCY = 1 - PILOT_OVERHEAD

CSI_CONDITIONS = [
    "Perfect CSI",
    "Pilot-Based LS CSI"
]


# ============================================================
# 2. BER SIMULATION WITH PILOT-BASED LS ESTIMATION
# ============================================================

def simulate_pilot_ber(
    modulation,
    snr_db,
    num_data_symbols,
    csi_condition,
    seed
):

    rng = np.random.default_rng(seed)

    bits_per_symbol = MODULATIONS[modulation]

    # Ensure complete data blocks
    num_blocks = int(
        np.ceil(num_data_symbols / DATA_BLOCK_LENGTH)
    )

    actual_data_symbols = (
        num_blocks * DATA_BLOCK_LENGTH
    )

    total_bits = (
        actual_data_symbols * bits_per_symbol
    )

    # Generate information bits
    bits = rng.integers(0, 2, total_bits)

    # Modulation
    transmitted = modulate(bits, modulation)

    # Reshape into blocks
    transmitted_blocks = transmitted.reshape(
        num_blocks,
        DATA_BLOCK_LENGTH
    )

    # Rayleigh fading channel, one channel per block
    h = (
        rng.normal(
            0,
            1 / np.sqrt(2),
            num_blocks
        )
        +
        1j * rng.normal(
            0,
            1 / np.sqrt(2),
            num_blocks
        )
    )

    # Convert Eb/N0 to linear
    ebn0_linear = 10 ** (snr_db / 10)

    # Complex AWGN variance
    noise_variance = (
        1 / (bits_per_symbol * ebn0_linear)
    )

    # --------------------------------------------------------
    # Pilot transmission
    # --------------------------------------------------------

    pilot_symbols = np.ones(
        (num_blocks, PILOT_LENGTH),
        dtype=complex
    )

    pilot_noise = np.sqrt(
        noise_variance / 2
    ) * (
        rng.normal(
            size=(num_blocks, PILOT_LENGTH)
        )
        +
        1j * rng.normal(
            size=(num_blocks, PILOT_LENGTH)
        )
    )

    received_pilots = (
        h[:, None] * pilot_symbols
        + pilot_noise
    )

    # --------------------------------------------------------
    # Channel estimation
    # --------------------------------------------------------

    if csi_condition == "Perfect CSI":

        h_estimated = h.copy()

    elif csi_condition == "Pilot-Based LS CSI":

        # LS estimator:
        # h_hat = sum(conj(x_p)*y_p) / sum(|x_p|^2)

        h_estimated = (
            np.sum(
                np.conjugate(pilot_symbols)
                * received_pilots,
                axis=1
            )
            /
            np.sum(
                np.abs(pilot_symbols) ** 2,
                axis=1
            )
        )

    else:
        raise ValueError("Unknown CSI condition")

    # --------------------------------------------------------
    # Data transmission
    # --------------------------------------------------------

    data_noise = np.sqrt(
        noise_variance / 2
    ) * (
        rng.normal(
            size=(num_blocks, DATA_BLOCK_LENGTH)
        )
        +
        1j * rng.normal(
            size=(num_blocks, DATA_BLOCK_LENGTH)
        )
    )

    received_data = (
        h[:, None] * transmitted_blocks
        + data_noise
    )

    # Equalization
    safe_h = np.where(
        np.abs(h_estimated) < 1e-12,
        1e-12 + 0j,
        h_estimated
    )

    equalized = (
        received_data / safe_h[:, None]
    )

    # Demodulation
    detected_bits = demodulate(
        equalized.flatten(),
        modulation
    )

    # BER
    errors = np.count_nonzero(
        bits != detected_bits
    )

    ber = errors / total_bits

    return ber, errors, total_bits


# ============================================================
# 3. CALIBRATION
# ============================================================

def calibrate_thresholds():

    print("\n" + "=" * 70)
    print("PILOT-BASED CSI: SNR CALIBRATION")
    print("=" * 70)

    records = []

    for condition_index, condition in enumerate(
        CSI_CONDITIONS
    ):

        print(f"\nCSI CONDITION: {condition}")

        for modulation_index, modulation in enumerate(
            MODULATIONS
        ):

            print(f"\nModulation: {modulation}")

            threshold = None

            for snr_index, snr_db in enumerate(
                SNR_RANGE_DB
            ):

                seed = (
                    BASE_SEED
                    + condition_index * 100000
                    + modulation_index * 10000
                    + snr_index
                )

                ber, errors, total_bits = simulate_pilot_ber(
                    modulation,
                    snr_db,
                    CALIBRATION_SYMBOLS,
                    condition,
                    seed
                )

                records.append({
                    "CSI Condition": condition,
                    "Modulation": modulation,
                    "SNR (dB)": snr_db,
                    "BER": ber,
                    "Errors": errors,
                    "Total Bits": total_bits
                })

                print(
                    f"SNR = {snr_db:2.0f} dB"
                    f" | BER = {ber:.6e}"
                )

                if ber <= TARGET_BER:
                    threshold = snr_db + 0.5

                    print(
                        f"Initial threshold: "
                        f"{threshold:.1f} dB"
                    )

                    break

            if threshold is None:
                print("No threshold found up to 40 dB.")

            records[-1]["Initial Threshold"] = threshold

    calibration_df = pd.DataFrame(records)

    # Extract one threshold per modulation and condition
    threshold_df = (
        calibration_df.groupby(
            ["CSI Condition", "Modulation"],
            sort=False
        )["Initial Threshold"]
        .first()
        .reset_index()
    )

    calibration_df.to_csv(
        "Pilot_CSI_Calibration_Results.csv",
        index=False
    )

    threshold_df.to_csv(
        "Pilot_CSI_Initial_Thresholds.csv",
        index=False
    )

    return calibration_df, threshold_df


# ============================================================
# 4. INDEPENDENT VALIDATION
# ============================================================

def validate_thresholds(threshold_df):

    print("\n" + "=" * 70)
    print("INDEPENDENT VALIDATION")
    print("=" * 70)

    validation_records = []

    for index, row in threshold_df.iterrows():

        condition = row["CSI Condition"]
        modulation = row["Modulation"]

        threshold = row["Initial Threshold"]

        if pd.isna(threshold):

            validation_records.append({
                "CSI Condition": condition,
                "Modulation": modulation,
                "Validated Threshold (dB)": np.nan,
                "Validation BER": np.nan,
                "Validation Status": "NO THRESHOLD FOUND"
            })

            continue

        validated_threshold = None
        final_ber = np.nan

        candidates = np.arange(
            threshold,
            41,
            0.5
        )

        for candidate_index, snr_db in enumerate(
            candidates
        ):

            seed = (
                BASE_SEED
                + 900000
                + index * 10000
                + candidate_index
            )

            ber, errors, total_bits = simulate_pilot_ber(
                modulation,
                snr_db,
                VALIDATION_SYMBOLS,
                condition,
                seed
            )

            print(
                f"{condition:22s} | "
                f"{modulation:6s} | "
                f"{snr_db:5.1f} dB | "
                f"BER = {ber:.6e}"
            )

            if ber <= TARGET_BER:

                validated_threshold = snr_db
                final_ber = ber

                break

        if validated_threshold is not None:
            status = "PASSED"
        else:
            status = "FAILED - TARGET NOT MET"

        validation_records.append({
            "CSI Condition": condition,
            "Modulation": modulation,
            "Validated Threshold (dB)": validated_threshold,
            "Validation BER": final_ber,
            "Validation Status": status
        })

    validation_df = pd.DataFrame(
        validation_records
    )

    validation_df.to_csv(
        "Pilot_CSI_Validation_Results.csv",
        index=False
    )

    return validation_df


# ============================================================
# 5. ADAPTIVE MODULATION SELECTION
# ============================================================

def select_modulation(snr_db, condition_df):

    eligible = []

    for _, row in condition_df.iterrows():

        threshold = row[
            "Validated Threshold (dB)"
        ]

        if pd.isna(threshold):
            continue

        if row["Validation Status"] != "PASSED":
            continue

        if snr_db >= threshold:
            eligible.append(row["Modulation"])

    if not eligible:
        return "OUTAGE"

    return max(
        eligible,
        key=lambda modulation: MODULATIONS[modulation]
    )


# ============================================================
# 6. ADAPTIVE THROUGHPUT
# ============================================================

def calculate_adaptive_performance(validation_df):

    print("\n" + "=" * 70)
    print("ADAPTIVE MODULATION PERFORMANCE")
    print("=" * 70)

    records = []

    for condition in CSI_CONDITIONS:

        condition_df = validation_df[
            validation_df["CSI Condition"] == condition
        ]

        print(f"\n{condition}")

        for snr_db in np.arange(18, 41, 1):

            selected = select_modulation(
                snr_db,
                condition_df
            )

            if selected == "OUTAGE":

                spectral_efficiency = 0.0
                throughput = 0.0

            else:

                bits_per_symbol = MODULATIONS[selected]

                # Account for pilot overhead
                spectral_efficiency = (
                    bits_per_symbol
                    * CODE_RATE
                    * DATA_EFFICIENCY
                    / (1 + ROLLOFF)
                )

                throughput = (
                    SYMBOL_RATE
                    * bits_per_symbol
                    * CODE_RATE
                    * DATA_EFFICIENCY
                    / 1e6
                )

            records.append({
                "CSI Condition": condition,
                "SNR (dB)": snr_db,
                "Selected Modulation": selected,
                "Spectral Efficiency (bits/s/Hz)": spectral_efficiency,
                "Throughput (Mbps)": throughput
            })

            print(
                f"SNR = {snr_db:2.0f} dB"
                f" | Mode = {selected:7s}"
                f" | Throughput = {throughput:.3f} Mbps"
            )

    adaptive_df = pd.DataFrame(records)

    adaptive_df.to_csv(
        "Pilot_CSI_Adaptive_Performance.csv",
        index=False
    )

    return adaptive_df


# ============================================================
# 7. BER GRAPH
# ============================================================

def plot_ber(calibration_df):

    for modulation in MODULATIONS:

        plt.figure(figsize=(10, 6))

        for condition in CSI_CONDITIONS:

            data = calibration_df[
                (calibration_df["CSI Condition"] == condition)
                &
                (calibration_df["Modulation"] == modulation)
            ]

            plt.semilogy(
                data["SNR (dB)"],
                data["BER"],
                marker="o",
                markersize=4,
                label=condition
            )

        plt.axhline(
            TARGET_BER,
            linestyle="--",
            label="Target BER = 10^-3"
        )

        plt.xlabel("SNR (dB)")
        plt.ylabel("Bit Error Rate")

        plt.title(
            f"{modulation}: Perfect CSI vs Pilot-Based LS CSI"
        )

        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()

        filename = modulation.replace("-", "").replace(" ", "")

        plt.savefig(
            f"Pilot_CSI_{filename}_BER.png",
            dpi=300
        )

        plt.show()


# ============================================================
# 8. THROUGHPUT GRAPH
# ============================================================

def plot_throughput(adaptive_df):

    plt.figure(figsize=(10, 6))

    for condition in CSI_CONDITIONS:

        data = adaptive_df[
            adaptive_df["CSI Condition"] == condition
        ]

        plt.plot(
            data["SNR (dB)"],
            data["Throughput (Mbps)"],
            marker="o",
            markersize=4,
            label=condition
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Throughput (Mbps)")

    plt.title(
        "Adaptive Throughput: Perfect CSI vs Pilot-Based CSI"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "Pilot_CSI_Adaptive_Throughput.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

def final_summary(validation_df):

    print("\n" + "=" * 75)
    print("FINAL VALIDATED THRESHOLDS")
    print("=" * 75)

    print(
        validation_df.to_string(index=False)
    )

    print("\nEXPERIMENT PARAMETERS")
    print("---------------------")

    print(f"Target BER: {TARGET_BER}")
    print(f"Channel: Rayleigh fading")
    print(f"Estimation: Least Squares (LS)")
    print(f"Pilot length: {PILOT_LENGTH}")
    print(f"Data block length: {DATA_BLOCK_LENGTH}")
    print(f"Pilot overhead: {PILOT_OVERHEAD * 100:.2f}%")
    print(f"Data efficiency: {DATA_EFFICIENCY * 100:.2f}%")
    print(f"Bandwidth: {BANDWIDTH_HZ / 1e6:.2f} MHz")
    print(f"Roll-off: {ROLLOFF}")
    print(f"Code rate: {CODE_RATE}")


# ============================================================
# 10. MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 75)
    print("6G ADAPTIVE MODULATION WITH PILOT-BASED CHANNEL ESTIMATION")
    print("=" * 75)

    print("Modulations:", list(MODULATIONS.keys()))

    print("Pilot length:", PILOT_LENGTH)
    print("Data block length:", DATA_BLOCK_LENGTH)

    print("Target BER:", TARGET_BER)

    # Step 1: Calibration
    calibration_df, threshold_df = calibrate_thresholds()

    # Step 2: Independent validation
    validation_df = validate_thresholds(threshold_df)

    # Step 3: Adaptive modulation
    adaptive_df = calculate_adaptive_performance(
        validation_df
    )

    # Step 4: BER plots
    plot_ber(calibration_df)

    # Step 5: Throughput plot
    plot_throughput(adaptive_df)

    # Step 6: Summary
    final_summary(validation_df)

    print("\n" + "=" * 75)
    print("PILOT-BASED CSI EXPERIMENT COMPLETED")
    print("=" * 75)

    print("\nGenerated CSV files:")
    print("1. Pilot_CSI_Calibration_Results.csv")
    print("2. Pilot_CSI_Initial_Thresholds.csv")
    print("3. Pilot_CSI_Validation_Results.csv")
    print("4. Pilot_CSI_Adaptive_Performance.csv")

    print("\nGenerated plots:")
    print("1. Pilot_CSI_BPSK_BER.png")
    print("2. Pilot_CSI_QPSK_BER.png")
    print("3. Pilot_CSI_16QAM_BER.png")
    print("4. Pilot_CSI_Adaptive_Throughput.png")


if __name__ == "__main__":
    main()
