
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from adaptive_rayleigh_refined import (
    MODULATIONS,
    modulate,
    demodulate
)

# ---------------------------------------------
# CSI ESTIMATION ERROR SENSITIVITY EXPERIMENT
# ---------------------------------------------

TARGET_BER = 1e-3

SNR_RANGE_DB = np.arange(18, 33, 1)

NUM_SYMBOLS = 100_000

BASE_SEED = 2026

# Complex channel estimation error variance:
# E[|h_hat - h|^2] = error variance

CSI_ERROR_VARIANCES = {
    "Perfect CSI": 0.0,
    "Very Accurate CSI": 0.001,
    "Accurate CSI": 0.005,
    "Moderate CSI": 0.01,
    "Poor CSI": 0.02,
    "Very Poor CSI": 0.05
}


def simulate_csi_ber(
    modulation,
    ebn0_db,
    num_symbols,
    error_variance,
    rng
):
    bits_per_symbol = MODULATIONS[modulation]

    # Generate random information bits
    bits = rng.integers(
        0,
        2,
        num_symbols * bits_per_symbol
    )

    # Modulate transmitted bits
    tx = modulate(bits, modulation)

    # Convert Eb/N0 from dB to linear
    ebn0_linear = 10 ** (ebn0_db / 10)

    # AWGN noise variance
    noise_variance = (
        1 / (bits_per_symbol * ebn0_linear)
    )

    # Generate flat Rayleigh fading channel
    h = (
        rng.normal(0, 1 / np.sqrt(2), num_symbols)
        + 1j * rng.normal(0, 1 / np.sqrt(2), num_symbols)
    )

    # Generate AWGN
    noise = np.sqrt(noise_variance / 2) * (
        rng.normal(size=num_symbols)
        + 1j * rng.normal(size=num_symbols)
    )

    # Received signal
    received = h * tx + noise

    # Channel estimation
    if error_variance == 0:
        h_estimated = h.copy()
    else:
        estimation_error = np.sqrt(
            error_variance / 2
        ) * (
            rng.normal(size=num_symbols)
            + 1j * rng.normal(size=num_symbols)
        )

        h_estimated = h + estimation_error

    # Equalization using estimated channel
    equalized = received / h_estimated

    # Demodulation
    detected_bits = demodulate(
        equalized,
        modulation
    )

    # Calculate BER
    errors = np.count_nonzero(
        bits != detected_bits
    )

    ber = errors / len(bits)

    return ber, errors, len(bits)


def run_experiment():

    print("\nCSI ESTIMATION ERROR SENSITIVITY ANALYSIS")

    print("Target BER:", TARGET_BER)
    print("Symbols per SNR point:", NUM_SYMBOLS)

    rng = np.random.default_rng(BASE_SEED)

    results = []

    for modulation in ["BPSK", "QPSK", "16-QAM"]:

        print(f"\nTesting {modulation}...")

        for condition, error_variance in (
            CSI_ERROR_VARIANCES.items()
        ):

            print(f"\n{condition}")

            for snr in SNR_RANGE_DB:

                ber, errors, total_bits = simulate_csi_ber(
                    modulation,
                    snr,
                    NUM_SYMBOLS,
                    error_variance,
                    rng
                )

                results.append({
                    "Modulation": modulation,
                    "CSI Condition": condition,
                    "CSI Error Variance": error_variance,
                    "SNR (dB)": snr,
                    "BER": ber,
                    "Bit Errors": errors,
                    "Total Bits": total_bits,
                    "Target BER Met": ber <= TARGET_BER
                })

                print(
                    f"SNR={snr:2d} dB | "
                    f"BER={ber:.6e} | "
                    f"Errors={errors}"
                )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "CSI_Error_Sensitivity_Results.csv",
        index=False
    )

    return results_df


def plot_results(results_df):

    for modulation in ["BPSK", "QPSK", "16-QAM"]:

        plt.figure(figsize=(10, 7))

        modulation_data = results_df[
            results_df["Modulation"] == modulation
        ]

        for condition in CSI_ERROR_VARIANCES:

            data = modulation_data[
                modulation_data["CSI Condition"] == condition
            ]

            # Avoid zero values on logarithmic BER axis
            plot_ber = np.maximum(
                data["BER"].to_numpy(),
                0.5 / data["Total Bits"].to_numpy()
            )

            plt.semilogy(
                data["SNR (dB)"],
                plot_ber,
                marker="o",
                label=condition
            )

        plt.axhline(
            TARGET_BER,
            linestyle="--",
            label="Target BER = 1e-3"
        )

        plt.xlabel("Eb/N0 (dB)")
        plt.ylabel("Bit Error Rate")

        plt.title(
            f"{modulation}: CSI Estimation Error Sensitivity"
        )

        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()

        filename = (
            modulation.replace("-", "")
            + "_CSI_Error_Sensitivity.png"
        )

        plt.savefig(
            filename,
            dpi=300
        )

        plt.show()
        plt.close()

        print("Saved:", filename)


def main():

    results_df = run_experiment()

    plot_results(results_df)

    print("\nEXPERIMENT COMPLETED SUCCESSFULLY")

    print("1. CSI_Error_Sensitivity_Results.csv")
    print("2. BPSK_CSI_Error_Sensitivity.png")
    print("3. QPSK_CSI_Error_Sensitivity.png")
    print("4. 16QAM_CSI_Error_Sensitivity.png")


if __name__ == "__main__":
    main()
