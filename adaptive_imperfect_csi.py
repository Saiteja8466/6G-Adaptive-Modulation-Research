
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PERFECT CSI VS IMPERFECT CSI - RAYLEIGH CHANNEL
# BPSK, QPSK, 16-QAM
# ============================================================

TARGET_BER = 1e-3

MODULATIONS = {
    "BPSK": 1,
    "QPSK": 2,
    "16-QAM": 4
}

SNR_DB = np.arange(18, 33, 1)

NUM_SYMBOLS = 100_000
RANDOM_SEED = 2026

# Channel estimation error variances
CSI_ERROR_VARIANCES = {
    "Perfect CSI": 0.0,
    "Imperfect CSI - Low Error": 0.01,
    "Imperfect CSI - Higher Error": 0.05
}


# -------------------- Modulation --------------------

def modulate(bits, modulation):

    bits = bits.reshape(-1)

    if modulation == "BPSK":
        return (2 * bits - 1).astype(complex)

    if modulation == "QPSK":
        pairs = bits.reshape(-1, 2)

        i = 2 * pairs[:, 0] - 1
        q = 2 * pairs[:, 1] - 1

        return (i + 1j * q) / np.sqrt(2)

    if modulation == "16-QAM":

        groups = bits.reshape(-1, 4)

        def gray_level(b1, b0):

            if b1 == 0 and b0 == 0:
                return -3

            if b1 == 0 and b0 == 1:
                return -1

            if b1 == 1 and b0 == 1:
                return 1

            return 3

        i = np.array([
            gray_level(b[0], b[1]) for b in groups
        ])

        q = np.array([
            gray_level(b[2], b[3]) for b in groups
        ])

        return (i + 1j * q) / np.sqrt(10)

    raise ValueError("Unknown modulation")


# -------------------- Demodulation --------------------

def demodulate(symbols, modulation):

    if modulation == "BPSK":
        return (symbols.real > 0).astype(int)

    if modulation == "QPSK":

        i_bits = (symbols.real > 0).astype(int)
        q_bits = (symbols.imag > 0).astype(int)

        output = np.empty(2 * len(symbols), dtype=int)

        output[0::2] = i_bits
        output[1::2] = q_bits

        return output

    if modulation == "16-QAM":

        levels = np.array([-3, -1, 1, 3])

        i_level = levels[
            np.argmin(
                np.abs(
                    symbols.real[:, None]
                    - levels[None, :] / np.sqrt(10)
                ),
                axis=1
            )
        ]

        q_level = levels[
            np.argmin(
                np.abs(
                    symbols.imag[:, None]
                    - levels[None, :] / np.sqrt(10)
                ),
                axis=1
            )
        ]

        inverse_gray = {
            -3: (0, 0),
            -1: (0, 1),
             1: (1, 1),
             3: (1, 0)
        }

        output = np.empty((len(symbols), 4), dtype=int)

        for index, (iv, qv) in enumerate(
            zip(i_level, q_level)
        ):

            output[index, 0:2] = inverse_gray[iv]
            output[index, 2:4] = inverse_gray[qv]

        return output.reshape(-1)

    raise ValueError("Unknown modulation")


# -------------------- CSI Simulation --------------------

def simulate_csi_comparison(
    modulation,
    snr_db,
    num_symbols,
    rng
):

    bits_per_symbol = MODULATIONS[modulation]

    bits = rng.integers(
        0,
        2,
        num_symbols * bits_per_symbol
    )

    tx = modulate(bits, modulation)

    ebn0_linear = 10 ** (snr_db / 10)

    noise_variance = 1 / (
        bits_per_symbol * ebn0_linear
    )

    # Flat Rayleigh channel
    h = (
        rng.normal(0, 1 / np.sqrt(2), num_symbols)
        + 1j * rng.normal(0, 1 / np.sqrt(2), num_symbols)
    )

    # AWGN
    noise = np.sqrt(noise_variance / 2) * (
        rng.normal(size=num_symbols)
        + 1j * rng.normal(size=num_symbols)
    )

    received = h * tx + noise

    results = {}

    for condition, error_variance in (
        CSI_ERROR_VARIANCES.items()
    ):

        if error_variance == 0:

            # Perfect channel knowledge
            h_estimated = h

        else:

            # Independent channel estimation error
            estimation_error = np.sqrt(
                error_variance / 2
            ) * (
                rng.normal(size=num_symbols)
                + 1j * rng.normal(size=num_symbols)
            )

            h_estimated = h + estimation_error

        # Equalization using estimated channel
        equalized = received / h_estimated

        detected_bits = demodulate(
            equalized,
            modulation
        )

        errors = np.count_nonzero(
            bits != detected_bits
        )

        ber = errors / len(bits)

        results[condition] = {
            "BER": ber,
            "Errors": int(errors),
            "Total Bits": len(bits)
        }

    return results


# -------------------- Main Experiment --------------------

def main():

    print("\nPERFECT CSI VS IMPERFECT CSI")
    print("Target BER:", TARGET_BER)

    rng = np.random.default_rng(RANDOM_SEED)

    rows = []

    for modulation in MODULATIONS:

        print(f"\nTesting {modulation}...")

        for snr in SNR_DB:

            results = simulate_csi_comparison(
                modulation,
                snr,
                NUM_SYMBOLS,
                rng
            )

            for condition, values in results.items():

                ber = values["BER"]

                rows.append({
                    "Modulation": modulation,
                    "SNR (dB)": snr,
                    "CSI Condition": condition,
                    "BER": ber,
                    "Bit Errors": values["Errors"],
                    "Total Bits": values["Total Bits"],
                    "Target BER Met": ber <= TARGET_BER
                })

                print(
                    f"{snr:2d} dB | "
                    f"{condition:27s} | "
                    f"BER={ber:.6e}"
                )

    df = pd.DataFrame(rows)

    # Save all numerical results
    df.to_csv(
        "Perfect_vs_Imperfect_CSI_Results.csv",
        index=False
    )

    # -------------------- Plotting --------------------

    for modulation in MODULATIONS:

        plt.figure(figsize=(9, 6))

        subset = df[
            df["Modulation"] == modulation
        ]

        for condition in CSI_ERROR_VARIANCES:

            condition_data = subset[
                subset["CSI Condition"] == condition
            ]

            plt.semilogy(
                condition_data["SNR (dB)"],
                condition_data["BER"],
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
            f"{modulation}: Perfect CSI vs Imperfect CSI"
        )

        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()

        filename = (
            modulation.replace("-", "")
            + "_Perfect_vs_Imperfect_CSI.png"
        )

        plt.savefig(filename, dpi=300)
        plt.show()

    print("\nEXPERIMENT COMPLETED SUCCESSFULLY")

    print("1. Perfect_vs_Imperfect_CSI_Results.csv")
    print("2. BPSK_Perfect_vs_Imperfect_CSI.png")
    print("3. QPSK_Perfect_vs_Imperfect_CSI.png")
    print("4. 16QAM_Perfect_vs_Imperfect_CSI.png")


if __name__ == "__main__":
    main()
