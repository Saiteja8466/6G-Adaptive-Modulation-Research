@'
# 6G Adaptive Modulation Research

## Research Overview

A simulation-based research study on adaptive modulation and pilot-based channel estimation over Rayleigh fading channels.

## Research Objectives

- Evaluate BPSK, QPSK, and 16-QAM modulation.
- Compare AWGN and Rayleigh fading channels.
- Study perfect and imperfect Channel State Information (CSI).
- Implement Least Squares (LS) pilot-based channel estimation.
- Investigate pilot-length optimization.
- Analyze BER, throughput, and spectral efficiency.
- Compare fixed and dynamic pilot-selection strategies.
- Evaluate reliability using multi-seed and independent validation.

## Simulation Parameters

- Channel model: Block Rayleigh fading in pilot-based experiments
- Modulation: BPSK, QPSK, 16-QAM
- Pilot lengths: 4, 8, 16, 32 symbols
- Data block length: 128 symbols
- Target BER: 1e-3
- Bandwidth: 1 MHz
- Roll-off factor: 0.25
- Code rate: 1

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib

## Experiments

1. Modulation comparison
2. AWGN versus Rayleigh fading
3. Perfect and imperfect CSI analysis
4. LS pilot-based channel estimation
5. Pilot-length optimization
6. Multi-seed statistical validation
7. Fixed versus dynamic pilot selection
8. Independent dynamic-system validation
9. Reliability-aware selection and holdout validation

## Repository Contents

- Python scripts: simulation implementations
- CSV files: experimental data and statistical results
- PNG files: BER, throughput, and spectral-efficiency graphs
- Experiment folders: detailed outputs from individual studies

## Installation

Install the required packages:

```bash
py -m pip install numpy pandas matplotlib