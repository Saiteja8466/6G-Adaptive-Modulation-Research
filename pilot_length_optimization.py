
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
# 1. EXPERIMENT SETTINGS
# ============================================================

PILOT_LENGTHS = [4, 8, 16, 32]

DATA_BLOCK_LENGTH = 128

CALIBRATION_SYMBOLS = 50_000
VALIDATION_SYMBOLS = 200_000

SNR_RANGE_DB = np.arange(18, 41, 1)

BASE_SEED = 20261113

SYMBOL_RATE = BANDWIDTH_HZ / (1 + ROLLOFF)


# ============================================================
# 2. PILOT-BASED LS CHANNEL ESTIMATION
# ============================================================

def simulate_ber(
    modulation,
    snr_db,
    num_data_symbols,
    pilot_length,
    seed
):

    rng = np.random.default_rng(seed)

    bits_per_symbol = MODULATIONS[modulation]

    # Complete data blocks
    num_blocks = int(
        np.ceil(num_data_symbols / DATA_BLOCK_LENGTH)
    )

    actual_data_symbols = (
        num_blocks * DATA_BLOCK_LENGTH
    )

    total_bits = (
        actual_data_symbols * bits_per_symbol
    )

    # Generate bits
    bits = rng.integers(0, 2, total_bits)

    # Modulation
    transmitted = modulate(bits, modulation)

    transmitted_blocks = transmitted.reshape(
        num_blocks,
        DATA_BLOCK_LENGTH
    )

    # Independent Rayleigh fading per block
    h = (
        rng.normal(
            0, 1 / np.sqrt(2), num_blocks
        )
        +
        1j * rng.normal(
            0, 1 / np.sqrt(2), num_blocks
        )
    )

    # Eb/N0
    ebn0_linear = 10 ** (snr_db / 10)

    noise_variance = (
        1 / (bits_per_symbol * ebn0_linear)
    )

    # --------------------------------------------------------
    # Pilot transmission
    # --------------------------------------------------------

    pilots = np.ones(
        (num_blocks, pilot_length),
        dtype=complex
    )

    pilot_noise = np.sqrt(
        noise_variance / 2
    ) * (
        rng.normal(
            size=(num_blocks, pilot_length)
        )
        +
        1j * rng.normal(
            size=(num_blocks, pilot_length)
        )
    )

    received_pilots = (
        h[:, None] * pilots + pilot_noise
    )

    # LS channel estimate
    h_estimated = (
        np.sum(
            np.conjugate(pilots) * received_pilots,
            axis=1
        )
        /
        np.sum(
            np.abs(pilots) ** 2,
            axis=1
        )
    )

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

    safe_h = np.where(
        np.abs(h_estimated) < 1e-12,
        1e-12 + 0j,
        h_estimated
    )

    equalized = (
        received_data / safe_h[:, None]
    )

    detected_bits = demodulate(
        equalized.flatten(),
        modulation
    )

    errors = np.count_nonzero(
        bits != detected_bits
    )

    ber = errors / total_bits

    return ber, errors, total_bits


# ============================================================
# 3. CALIBRATION FOR ALL PILOT LENGTHS
# ============================================================

def calibrate_all():

    print("\n" + "=" * 75)
    print("PILOT LENGTH OPTIMIZATION - CALIBRATION")
    print("=" * 75)

    calibration_records = []
    threshold_records = []

    for pilot_index, pilot_length in enumerate(
        PILOT_LENGTHS
    ):

        print("\n" + "-" * 65)
        print(f"PILOT LENGTH: {pilot_length}")
        print("-" * 65)

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
                    + pilot_index * 100000
                    + modulation_index * 10000
                    + snr_index
                )

                ber, errors, total_bits = simulate_ber(
                    modulation,
                    snr_db,
                    CALIBRATION_SYMBOLS,
                    pilot_length,
                    seed
                )

                calibration_records.append({
                    "Pilot Length": pilot_length,
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

            threshold_records.append({
                "Pilot Length": pilot_length,
                "Modulation": modulation,
                "Initial Threshold (dB)": threshold
            })

    calibration_df = pd.DataFrame(
        calibration_records
    )

    threshold_df = pd.DataFrame(
        threshold_records
    )

    calibration_df.to_csv(
        "Pilot_Length_Calibration_Results.csv",
        index=False
    )

    threshold_df.to_csv(
        "Pilot_Length_Initial_Thresholds.csv",
        index=False
    )

    return calibration_df, threshold_df


# ============================================================
# 4. INDEPENDENT VALIDATION
# ============================================================

def validate_thresholds(threshold_df):

    print("\n" + "=" * 75)
    print("PILOT LENGTH - INDEPENDENT VALIDATION")
    print("=" * 75)

    validation_records = []

    for index, row in threshold_df.iterrows():

        pilot_length = int(row["Pilot Length"])
        modulation = row["Modulation"]

        threshold = row["Initial Threshold (dB)"]

        if pd.isna(threshold):

            validation_records.append({
                "Pilot Length": pilot_length,
                "Modulation": modulation,
                "Validated Threshold (dB)": np.nan,
                "Validation BER": np.nan,
                "Validation Status": "NO THRESHOLD FOUND"
            })

            continue

        validated_threshold = None
        validation_ber = np.nan

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

            ber, errors, total_bits = simulate_ber(
                modulation,
                snr_db,
                VALIDATION_SYMBOLS,
                pilot_length,
                seed
            )

            print(
                f"Pilot = {pilot_length:2d} | "
                f"{modulation:6s} | "
                f"SNR = {snr_db:5.1f} dB | "
                f"BER = {ber:.6e}"
            )

            if ber <= TARGET_BER:

                validated_threshold = snr_db
                validation_ber = ber

                break

        if validated_threshold is not None:
            status = "PASSED"
        else:
            status = "FAILED - TARGET NOT MET"

        validation_records.append({
            "Pilot Length": pilot_length,
            "Modulation": modulation,
            "Validated Threshold (dB)": validated_threshold,
            "Validation BER": validation_ber,
            "Validation Status": status
        })

    validation_df = pd.DataFrame(
        validation_records
    )

    validation_df.to_csv(
        "Pilot_Length_Validation_Results.csv",
        index=False
    )

    return validation_df


# ============================================================
# 5. ADAPTIVE MODULATION SELECTION
# ============================================================

def select_modulation(snr_db, pilot_validation):

    eligible = []

    for _, row in pilot_validation.iterrows():

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
# 6. ADAPTIVE THROUGHPUT FOR EACH PILOT LENGTH
# ============================================================

def calculate_adaptive_performance(validation_df):

    print("\n" + "=" * 75)
    print("ADAPTIVE THROUGHPUT COMPARISON")
    print("=" * 75)

    records = []

    for pilot_length in PILOT_LENGTHS:

        pilot_validation = validation_df[
            validation_df["Pilot Length"] == pilot_length
        ]

        # Pilot overhead for this configuration
        pilot_overhead = (
            pilot_length
            / (pilot_length + DATA_BLOCK_LENGTH)
        )

        data_efficiency = 1 - pilot_overhead

        print(f"\nPilot length: {pilot_length}")
        print(
            f"Pilot overhead: "
            f"{pilot_overhead * 100:.2f}%"
        )

        for snr_db in np.arange(18, 41, 1):

            selected = select_modulation(
                snr_db,
                pilot_validation
            )

            if selected == "OUTAGE":

                spectral_efficiency = 0.0
                throughput = 0.0

            else:

                bits_per_symbol = MODULATIONS[selected]

                spectral_efficiency = (
                    bits_per_symbol
                    * CODE_RATE
                    * data_efficiency
                    / (1 + ROLLOFF)
                )

                throughput = (
                    SYMBOL_RATE
                    * bits_per_symbol
                    * CODE_RATE
                    * data_efficiency
                    / 1e6
                )

            records.append({
                "Pilot Length": pilot_length,
                "SNR (dB)": snr_db,
                "Selected Modulation": selected,
                "Pilot Overhead (%)": pilot_overhead * 100,
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
        "Pilot_Length_Adaptive_Performance.csv",
        index=False
    )

    return adaptive_df


# ============================================================
# 7. BER COMPARISON PLOTS
# ============================================================

def plot_ber_comparison(calibration_df):

    for modulation in MODULATIONS:

        plt.figure(figsize=(10, 6))

        for pilot_length in PILOT_LENGTHS:

            data = calibration_df[
                (calibration_df["Pilot Length"] == pilot_length)
                &
                (calibration_df["Modulation"] == modulation)
            ]

            if data.empty:
                continue

            plt.semilogy(
                data["SNR (dB)"],
                data["BER"],
                marker="o",
                markersize=3,
                label=f"{pilot_length} pilots"
            )

        plt.axhline(
            TARGET_BER,
            linestyle="--",
            label="Target BER = 10^-3"
        )

        plt.xlabel("SNR (dB)")
        plt.ylabel("Bit Error Rate")

        plt.title(
            f"{modulation}: BER for Different Pilot Lengths"
        )

        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()

        filename = modulation.replace("-", "").replace(" ", "")

        plt.savefig(
            f"Pilot_Length_{filename}_BER.png",
            dpi=300
        )

        plt.show()


# ============================================================
# 8. THRESHOLD COMPARISON
# ============================================================

def plot_threshold_comparison(validation_df):

    for modulation in MODULATIONS:

        data = validation_df[
            validation_df["Modulation"] == modulation
        ].copy()

        plt.figure(figsize=(9, 5))

        valid_data = data.dropna(
            subset=["Validated Threshold (dB)"]
        )

        plt.bar(
            valid_data["Pilot Length"].astype(str),
            valid_data["Validated Threshold (dB)"]
        )

        plt.xlabel("Pilot Length")
        plt.ylabel("Validated SNR Threshold (dB)")

        plt.title(
            f"{modulation}: SNR Threshold vs Pilot Length"
        )

        plt.grid(axis="y")
        plt.tight_layout()

        filename = modulation.replace("-", "").replace(" ", "")

        plt.savefig(
            f"Pilot_Length_{filename}_Threshold.png",
            dpi=300
        )

        plt.show()


# ============================================================
# 9. THROUGHPUT COMPARISON
# ============================================================

def plot_throughput(adaptive_df):

    plt.figure(figsize=(10, 6))

    for pilot_length in PILOT_LENGTHS:

        data = adaptive_df[
            adaptive_df["Pilot Length"] == pilot_length
        ]

        plt.plot(
            data["SNR (dB)"],
            data["Throughput (Mbps)"],
            marker="o",
            markersize=3,
            label=f"{pilot_length} pilots"
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Throughput (Mbps)")

    plt.title(
        "Adaptive Throughput for Different Pilot Lengths"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "Pilot_Length_Adaptive_Throughput.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

def print_summary(validation_df, adaptive_df):

    print("\n" + "=" * 75)
    print("FINAL PILOT LENGTH OPTIMIZATION RESULTS")
    print("=" * 75)

    print("\nVALIDATED SNR THRESHOLDS")

    print(
        validation_df.to_string(index=False)
    )

    print("\nPILOT OVERHEAD")

    for pilot_length in PILOT_LENGTHS:

        overhead = (
            pilot_length
            / (pilot_length + DATA_BLOCK_LENGTH)
        ) * 100

        print(
            f"{pilot_length:2d} pilots: "
            f"{overhead:.2f}% overhead"
        )

    print("\nMAXIMUM SIMULATED ADAPTIVE THROUGHPUT")

    for pilot_length in PILOT_LENGTHS:

        data = adaptive_df[
            adaptive_df["Pilot Length"] == pilot_length
        ]

        maximum = data["Throughput (Mbps)"].max()

        print(
            f"{pilot_length:2d} pilots: "
            f"{maximum:.3f} Mbps"
        )

    print("\nExperiment settings:")
    print(f"Target BER: {TARGET_BER}")
    print(f"Channel: Rayleigh block fading")
    print("Estimation: Least Squares")
    print(f"Data block length: {DATA_BLOCK_LENGTH}")
    print(f"Bandwidth: {BANDWIDTH_HZ / 1e6:.2f} MHz")
    print(f"Roll-off: {ROLLOFF}")
    print(f"Code rate: {CODE_RATE}")


# ============================================================
# 11. MAIN PROGRAM
# ============================================================

def main():

    print("\n")
    print("=" * 75)
    print("6G ADAPTIVE MODULATION - PILOT LENGTH OPTIMIZATION")
    print("=" * 75)

    print("Pilot lengths:", PILOT_LENGTHS)
    print("Modulations:", list(MODULATIONS.keys()))
    print("Target BER:", TARGET_BER)

    # Step 1: Calibration
    calibration_df, threshold_df = calibrate_all()

    # Step 2: Independent validation
    validation_df = validate_thresholds(
        threshold_df
    )

    # Step 3: Adaptive throughput
    adaptive_df = calculate_adaptive_performance(
        validation_df
    )

    # Step 4: BER graphs
    plot_ber_comparison(calibration_df)

    # Step 5: Threshold graphs
    plot_threshold_comparison(validation_df)

    # Step 6: Throughput graph
    plot_throughput(adaptive_df)

    # Step 7: Summary
    print_summary(
        validation_df,
        adaptive_df
    )

    print("\n" + "=" * 75)
    print("PILOT LENGTH OPTIMIZATION COMPLETED")
    print("=" * 75)

    print("\nGenerated CSV files:")
    print("1. Pilot_Length_Calibration_Results.csv")
    print("2. Pilot_Length_Initial_Thresholds.csv")
    print("3. Pilot_Length_Validation_Results.csv")
    print("4. Pilot_Length_Adaptive_Performance.csv")

    print("\nGenerated plots:")
    print("1. Pilot_Length_BPSK_BER.png")
    print("2. Pilot_Length_QPSK_BER.png")
    print("3. Pilot_Length_16QAM_BER.png")
    print("4. Pilot_Length_Adaptive_Throughput.png")
    print("5. Pilot_Length_*_Threshold.png")


if __name__ == "__main__":
    main()
