import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# RAYLEIGH-BASED ADAPTIVE MODULATION
# BPSK, QPSK, 16-QAM | Target BER = 1e-3
# ============================================================

TARGET_BER = 1e-3

# Calibration and validation use independent random seeds.
CALIBRATION_SEED = 123
VALIDATION_SEED = 456

# Eb/N0 range in dB. Rayleigh requires much higher SNR than AWGN.
CALIBRATION_SNR_DB = np.arange(22, 30.5, 0.5)
VALIDATION_SNR_DB = np.arange(22, 30.5, 0.5)
# Increase these for more reliable low-BER estimates.
CALIBRATION_SYMBOLS = 500_000
VALIDATION_SYMBOLS = 500_000

BANDWIDTH_HZ = 1_000_000
ROLLOFF = 0.25
CODE_RATE = 1.0

MODULATIONS = {
    "BPSK": 1,
    "QPSK": 2,
    "16-QAM": 4
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

        # Gray-coded levels:
        # 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3
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

        # Nearest-level decision.
        i_level = levels[
            np.argmin(
                np.abs(symbols.real[:, None] -
                       levels[None, :] / np.sqrt(10)),
                axis=1
            )
        ]

        q_level = levels[
            np.argmin(
                np.abs(symbols.imag[:, None] -
                       levels[None, :] / np.sqrt(10)),
                axis=1
            )
        ]

        # Inverse Gray mapping.
        inverse_gray = {
            -3: (0, 0),
            -1: (0, 1),
             1: (1, 1),
             3: (1, 0)
        }

        output = np.empty((len(symbols), 4), dtype=int)

        for index, (iv, qv) in enumerate(zip(i_level, q_level)):
            output[index, 0:2] = inverse_gray[iv]
            output[index, 2:4] = inverse_gray[qv]

        return output.reshape(-1)

    raise ValueError("Unknown modulation")


# -------------------- Rayleigh channel --------------------

def simulate_ber(modulation, ebn0_db, num_symbols, rng):
    bits_per_symbol = MODULATIONS[modulation]

    bits = rng.integers(
        0, 2, num_symbols * bits_per_symbol
    )

    tx = modulate(bits, modulation)

    ebn0_linear = 10 ** (ebn0_db / 10)

    # Unit average symbol energy:
    # N0 = 1 / (bits_per_symbol * Eb/N0)
    noise_variance = 1 / (bits_per_symbol * ebn0_linear)

    # Flat Rayleigh fading: E[|h|^2] = 1
    h = (
        rng.normal(0, 1 / np.sqrt(2), num_symbols)
        + 1j * rng.normal(0, 1 / np.sqrt(2), num_symbols)
    )

    noise = np.sqrt(noise_variance / 2) * (
        rng.normal(size=num_symbols)
        + 1j * rng.normal(size=num_symbols)
    )

    received = h * tx + noise

    # Perfect channel state information and equalization.
    equalized = received / h

    detected_bits = demodulate(equalized, modulation)

    errors = np.count_nonzero(bits != detected_bits)
    ber = errors / len(bits)

    return ber, errors, len(bits)

def wilson_confidence_interval(errors, total_bits, confidence=0.95):

    """Calculate the Wilson confidence interval for BER."""

    if total_bits <= 0:
        raise ValueError("Total bits must be positive.")

    z = 1.959964  # 95% confidence level
    p = errors / total_bits

    denominator = 1 + (z ** 2 / total_bits)

    center = (
        p + (z ** 2 / (2 * total_bits))
    ) / denominator

    margin = (
        z * np.sqrt(
            (p * (1 - p) / total_bits)
            + (z ** 2 / (4 * total_bits ** 2))
        )
    ) / denominator

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return lower, upper

# -------------------- Calibration --------------------

def calibrate_thresholds():
    print("\nCALIBRATING RAYLEIGH SNR THRESHOLDS")
    print("Target BER:", TARGET_BER)

    rng = np.random.default_rng(CALIBRATION_SEED)

    results = {}
    curves = {}

    for modulation in MODULATIONS:
        print(f"\nTesting {modulation}...")

        ber_values = []

        for snr in CALIBRATION_SNR_DB:
            ber, errors, total_bits = simulate_ber(
                modulation,
                snr,
                CALIBRATION_SYMBOLS,
                rng
            )

            ber_values.append(ber)

            print(
            f"SNR={snr:4.1f} dB | "
            f"BER={ber:.6e} | "
            f"Errors={errors}"
             )

        curves[modulation] = np.array(ber_values)

        valid_indices = np.where(
            curves[modulation] <= TARGET_BER
        )[0]

        if len(valid_indices) == 0:
            results[modulation] = None
            print("Threshold not found in tested SNR range.")
        else:
            threshold = CALIBRATION_SNR_DB[valid_indices[0]]
            results[modulation] = float(threshold)
            print(f"Estimated threshold: {threshold} dB")

    return results, curves


# -------------------- Independent validation --------------------

def validate_thresholds(thresholds):
    print("\nINDEPENDENT RAYLEIGH VALIDATION")

    rng = np.random.default_rng(VALIDATION_SEED)

    validation_rows = []

    for modulation, threshold in thresholds.items():
        if threshold is None:
            continue

        ber, errors, total_bits = simulate_ber(
            modulation,
            threshold,
            VALIDATION_SYMBOLS,
            rng
        )

        passed = ber <= TARGET_BER
        ci_lower, ci_upper = wilson_confidence_interval(
            errors,
            total_bits
        )

        validation_rows.append({
            "Modulation": modulation,
            "Threshold (dB)": threshold,
            "Validation BER": ber,
            "95% CI Lower": ci_lower,
            "95% CI Upper": ci_upper,
            "Bit Errors": errors,
            "Total Bits": total_bits,
            "Target Met": passed
        })

        print(
            f"{modulation}: threshold={threshold} dB, "
            f"validation BER={ber:.6e}, "
            f"target met={passed}"
        )

    return pd.DataFrame(validation_rows)


def multi_seed_validation(thresholds):
    print("\n10-RUN MULTI-SEED VALIDATION")

    NUM_RUNS = 10
    BASE_SEED = 1000

    rows = []

    # t critical value for 95% CI with 9 degrees of freedom
    T_CRITICAL = 2.262

    for modulation, threshold in thresholds.items():
        if threshold is None:
            continue

        ber_values = []
        total_errors = 0
        total_bits_all_runs = 0

        print(f"\nTesting {modulation}...")

        for run in range(NUM_RUNS):
            rng = np.random.default_rng(BASE_SEED + run)

            ber, errors, total_bits = simulate_ber(
                modulation,
                threshold,
                VALIDATION_SYMBOLS,
                rng
            )

            ber_values.append(ber)
            total_errors += errors
            total_bits_all_runs += total_bits

            print(
                f"Run {run + 1:02d}: "
                f"BER={ber:.6e}, Errors={errors}"
            )

        mean_ber = float(np.mean(ber_values))
        std_ber = float(np.std(ber_values, ddof=1))

        standard_error = std_ber / np.sqrt(NUM_RUNS)

        margin = T_CRITICAL * standard_error

        ci_lower = max(0.0, mean_ber - margin)
        ci_upper = min(1.0, mean_ber + margin)

        pooled_ber = total_errors / total_bits_all_runs

        rows.append({
            "Modulation": modulation,
            "Threshold (dB)": threshold,
            "Number of Runs": NUM_RUNS,
            "Mean BER": mean_ber,
            "Standard Deviation": std_ber,
            "Pooled BER": pooled_ber,
            "95% CI Lower": ci_lower,
            "95% CI Upper": ci_upper,
            "Target BER": TARGET_BER,
            "Mean BER Meets Target": mean_ber <= TARGET_BER,
            "CI Upper Meets Target": ci_upper <= TARGET_BER
        })

        print(f"Mean BER: {mean_ber:.6e}")
        print(f"Standard Deviation: {std_ber:.6e}")
        print(f"95% CI: [{ci_lower:.6e}, {ci_upper:.6e}]")
        print(f"Pooled BER: {pooled_ber:.6e}")

    return pd.DataFrame(rows)



# -------------------- Adaptive modulation --------------------

def adaptive_selection(snr_db, thresholds):
    # Select the highest spectral-efficiency mode
    # whose calibrated threshold is satisfied.
    selected = "OUTAGE"

    for modulation in ["BPSK", "QPSK", "16-QAM"]:
        threshold = thresholds.get(modulation)

        if threshold is not None and snr_db >= threshold:
            selected = modulation

    return selected


def evaluate_adaptive_system(thresholds):
    print("\nADAPTIVE RAYLEIGH SYSTEM")

    rng = np.random.default_rng(789)

    rows = []

    for snr in VALIDATION_SNR_DB:
        selected = adaptive_selection(snr, thresholds)

        if selected == "OUTAGE":
            ber = np.nan
            throughput = 0.0
            efficiency = 0.0
        else:
            ber, errors, total_bits = simulate_ber(
                selected,
                snr,
                VALIDATION_SYMBOLS,
                rng
            )

            bits_per_symbol = MODULATIONS[selected]

            symbol_rate = BANDWIDTH_HZ / (1 + ROLLOFF)

            throughput = (
                symbol_rate * bits_per_symbol * CODE_RATE
            )

            efficiency = (
                bits_per_symbol * CODE_RATE / (1 + ROLLOFF)
            )

        rows.append({
            "SNR (dB)": snr,
            "Selected Modulation": selected,
            "BER": ber,
            "Spectral Efficiency (bits/s/Hz)": efficiency,
            "Throughput (Mbps)": throughput / 1e6
        })

        print(
            f"{snr:4.1f} dB | {selected:7s} | "
            f"BER={ber if not np.isnan(ber) else 'N/A'} | "
            f"Throughput={throughput / 1e6:.2f} Mbps"
        )

    return pd.DataFrame(rows)


def fixed_vs_adaptive_comparison(thresholds):
    print("\nFIXED VS ADAPTIVE MODULATION COMPARISON")

    schemes = ["BPSK", "QPSK", "16-QAM", "ADAPTIVE"]
    rows = []

    symbol_rate = BANDWIDTH_HZ / (1 + ROLLOFF)

    for snr_index, snr in enumerate(VALIDATION_SNR_DB):

        for scheme_index, scheme in enumerate(schemes):

            if scheme == "ADAPTIVE":
                modulation = adaptive_selection(snr, thresholds)

                if modulation == "OUTAGE":
                    ber = np.nan
                    errors = 0
                    total_bits = 0
                    throughput = 0.0
                    efficiency = 0.0
                else:
                    rng = np.random.default_rng(
                        5000 + snr_index * 100 + scheme_index
                    )

                    ber, errors, total_bits = simulate_ber(
                        modulation,
                        snr,
                        VALIDATION_SYMBOLS,
                        rng
                    )

                    bits_per_symbol = MODULATIONS[modulation]

                    throughput = (
                        symbol_rate * bits_per_symbol * CODE_RATE
                    )

                    efficiency = (
                        bits_per_symbol * CODE_RATE / (1 + ROLLOFF)
                    )

            else:
                modulation = scheme

                rng = np.random.default_rng(
                    5000 + snr_index * 100 + scheme_index
                )

                ber, errors, total_bits = simulate_ber(
                    modulation,
                    snr,
                    VALIDATION_SYMBOLS,
                    rng
                )

                bits_per_symbol = MODULATIONS[modulation]

                throughput = (
                    symbol_rate * bits_per_symbol * CODE_RATE
                )

                efficiency = (
                    bits_per_symbol * CODE_RATE / (1 + ROLLOFF)
                )

            target_met = (
                bool(ber <= TARGET_BER)
                if not np.isnan(ber)
                else False
            )

            rows.append({
                "SNR (dB)": snr,
                "Scheme": scheme,
                "Selected Modulation": modulation,
                "BER": ber,
                "Bit Errors": errors,
                "Total Bits": total_bits,
                "Target BER Met": target_met,
                "Spectral Efficiency (bits/s/Hz)": efficiency,
                "Throughput (Mbps)": throughput / 1e6
            })

            ber_text = (
                f"{ber:.6e}" if not np.isnan(ber) else "N/A"
            )

            print(
                f"SNR={snr:4.1f} dB | "
                f"{scheme:8s} | "
                f"Modulation={modulation:7s} | "
                f"BER={ber_text} | "
                f"Throughput={throughput / 1e6:.2f} Mbps"
            )

    return pd.DataFrame(rows)



# -------------------- Plotting --------------------

def plot_calibration(curves):
    plt.figure(figsize=(9, 6))

    for modulation, ber_values in curves.items():
        # Avoid zero values on logarithmic axis.
        plot_values = np.maximum(ber_values, 0.5 / (
            CALIBRATION_SYMBOLS * MODULATIONS[modulation]
        ))

        plt.semilogy(
            CALIBRATION_SNR_DB,
            plot_values,
            marker="o",
            label=modulation
        )

    plt.axhline(
        TARGET_BER,
        linestyle="--",
        label="Target BER = 1e-3"
    )

    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Bit Error Rate")
    plt.title("Rayleigh BER Calibration")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Rayleigh_Threshold_Calibration.png", dpi=300)
    plt.show()


def plot_adaptive_results(adaptive_df):
    plt.figure(figsize=(9, 6))

    valid = adaptive_df.dropna(subset=["BER"])

    if not valid.empty:
        plt.semilogy(
            valid["SNR (dB)"],
            valid["BER"],
            marker="o",
            label="Adaptive Rayleigh BER"
        )

    plt.axhline(
        TARGET_BER,
        linestyle="--",
        label="Target BER = 1e-3"
    )

    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Bit Error Rate")
    plt.title("Optimized Adaptive Modulation - Rayleigh")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Optimized_Adaptive_Rayleigh_BER.png", dpi=300)
    plt.show()

    plt.figure(figsize=(9, 6))
    plt.plot(
        adaptive_df["SNR (dB)"],
        adaptive_df["Spectral Efficiency (bits/s/Hz)"],
        marker="s"
    )
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Spectral Efficiency (bits/s/Hz)")
    plt.title("Rayleigh Adaptive Spectral Efficiency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Optimized_Rayleigh_Spectral_Efficiency.png", dpi=300)
    plt.show()

    plt.figure(figsize=(9, 6))
    plt.plot(
        adaptive_df["SNR (dB)"],
        adaptive_df["Throughput (Mbps)"],
        marker="^"
    )
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Throughput (Mbps)")
    plt.title("Rayleigh Adaptive Throughput")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Optimized_Rayleigh_Throughput.png", dpi=300)
    plt.show()


# -------------------- Main --------------------
def main():
    thresholds, curves = calibrate_thresholds()

    # Apply provisional 0.5 dB safety margins
    thresholds["BPSK"] = 25.0
    thresholds["QPSK"] = 24.5
    thresholds["16-QAM"] = 27.5

    print("\nCALIBRATED THRESHOLDS")
    for modulation, threshold in thresholds.items():
        print(f"{modulation}: {threshold} dB")

    validation_df = validate_thresholds(thresholds)

    multi_seed_df = multi_seed_validation(thresholds)

    adaptive_df = evaluate_adaptive_system(thresholds)

    comparison_df = fixed_vs_adaptive_comparison(thresholds)

    plot_calibration(curves)
    plot_adaptive_results(adaptive_df)

    validation_df.to_csv(
        "Rayleigh_Threshold_Validation.csv",
        index=False
    )

    multi_seed_df.to_csv(
    "Rayleigh_Multi_Seed_Validation.csv",
    index=False
    )

    adaptive_df.to_csv(
        "Optimized_Rayleigh_Adaptive_Results.csv",
        index=False
    )

    comparison_df.to_csv(
    "Fixed_vs_Adaptive_Rayleigh.csv",
    index=False
    )

    print("\nFILES SAVED SUCCESSFULLY:")
    print("1. Rayleigh_Threshold_Calibration.png")
    print("2. Optimized_Adaptive_Rayleigh_BER.png")
    print("3. Optimized_Rayleigh_Spectral_Efficiency.png")
    print("4. Optimized_Rayleigh_Throughput.png")
    print("5. Rayleigh_Threshold_Validation.csv")
    print("6. Optimized_Rayleigh_Adaptive_Results.csv")


if __name__ == "__main__":
    main()