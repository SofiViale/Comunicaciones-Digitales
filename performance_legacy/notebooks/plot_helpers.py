#Heuristic to drop unreliable points
ERRORS_PER_SNR_POINT = 100

import numpy as np
import matplotlib.pyplot as plt

def plot_BER_SNR_from_binary(filename, sf, spc, label_fill, title_fill='BER vs SNR', ax=None, num_sims=None):
    """Plot the BER vs SNR from a binary file, skipping unreliable points"""
    
    if ax is None:
        ax = plt.gca()
    
    data = np.load(filename)
    SNR_values = data[0, :]
    BER_values = data[1, :]

    # Mask values that are too small (unreliable due to low trial count)
    if num_sims is not None:
        limit = ERRORS_PER_SNR_POINT / num_sims
        mask = BER_values >= limit
        SNR_values = SNR_values[mask]
        BER_values = BER_values[mask]

    ax.plot(SNR_values, BER_values, marker='o', linestyle='-', label=label_fill)
    ax.set_xlim([SNR_values.min(), SNR_values.max()])
    ax.set_yscale('log')
    ax.set_xticks(np.arange(-30 + 12 - sf, 2, 1))
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('BER (log scale)')
    ax.grid(which='both', linestyle='--', linewidth=0.5)
    ax.set_title(title_fill)

def plot_SER_SNR_from_binary(filename, sf, spc, label_fill, title_fill='SER vs SNR', ax=None, num_sims=None):
    """Plot the SER vs SNR from a binary file, skipping unreliable points"""
    
    if ax is None:
        ax = plt.gca()
    
    data = np.load(filename)
    SNR_values = data[0, :]
    SER_values = data[1, :]

    # Mask values that are too small (unreliable due to low trial count)}
    if num_sims is not None:
        limit = ERRORS_PER_SNR_POINT / num_sims
        mask = SER_values >= limit
        SNR_values = SNR_values[mask]
        SER_values = SER_values[mask]

    ax.plot(SNR_values, SER_values, marker='o', linestyle='-', label=label_fill)
    ax.set_xlim([SNR_values.min(), SNR_values.max()])
    ax.set_yscale('log')
    ax.set_xticks(np.arange(-30 + 12 - sf, 2, 1))
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('SER (log scale)')
    ax.grid(which='both', linestyle='--', linewidth=0.5)
    ax.set_title(title_fill)

def plot_all_snr_ber():
    """Plot all SNR vs BER and SER from binary files in subplots"""
    # Crear los subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 16))
    fig.suptitle('SNRxBER', fontsize=16)

    # Primer subplot: BER vs SNR para un Sample per Chip y diferentes Spreading Factors en canal AWGN
    axs[0, 0].set_title('BER vs SNR para 1 Sample per Chip y diferentes Spreading Factors en canal AWGN')
    for sf in range(7, 13):
        plot_BER_SNR_from_binary(f'../data/SNRxBER_AWGN/spc1/BER_SNR_sf{sf}.npy', sf, 1,
                                f'SF={sf}, BW=125kHz', title_fill='Different SF & AWGN Channel', ax=axs[0, 0], num_sims=10000000)
    axs[0, 0].legend()

    # Segundo subplot: BER vs SNR para SF=7 y diferentes Samples per Chip en canal AWGN
    axs[0, 1].set_title('BER vs SNR para SF=7 y diferentes Samples per Chip en canal AWGN')
    for spc in [1, 2, 4, 8, 10]:
        plot_BER_SNR_from_binary(f'../data/SNRxBER_AWGN/spc{spc}/BER_SNR_sf7.npy', 7, spc,
                                f'SPC={spc}, BW=125kHz', title_fill='Different SPC & AWGN Channel', ax=axs[0, 1], num_sims=10000000)
    axs[0, 1].legend()

    # Tercer subplot: BER vs SNR para un Sample per Chip y diferentes Spreading Factors en canal Frequency Selective
    axs[1, 0].set_title('BER vs SNR para 1 Sample per Chip y diferentes Spreading Factors en canal Frequency Selective')
    for sf in range(7, 13):
        plot_BER_SNR_from_binary(f'../data/SNRxBER_FreqSel/spc1/BER_SNR_sf{sf}.npy', sf, 1,
                                f'SF={sf}, BW=125kHz', title_fill='Different SF & Frequency Selective Channel', ax=axs[1, 0], num_sims=10000000)
    axs[1, 0].legend()

    # Cuarto subplot: BER vs SNR para SF=7 y diferentes Samples per Chip en canal Frequency Selective
    axs[1, 1].set_title('BER vs SNR para SF=7 y diferentes Samples per Chip en canal Frequency Selective')
    for spc in [1, 2, 4, 8, 10]:
        plot_BER_SNR_from_binary(f'../data/SNRxBER_FreqSel/spc{spc}/BER_SNR_sf7.npy', 7, spc,
                                f'SPC={spc}, BW=125kHz', title_fill='Different SPC & Frequency Selective Channel', ax=axs[1, 1], num_sims=10000000)
    axs[1, 1].legend()

    # Ecuación en LaTeX colocada en el centro debajo del último subplot
    equation = r"$h(nT) = \sqrt{0.8} \delta(nT) + \sqrt{0.2} \delta(nT - T)$"
    fig.text(0.5, 0.02, "Impulse Response of Frequency Selective Channel: "+equation, fontsize=14, ha='center', va='center')

    for ax in axs.flat:  # Itera sobre todos los subplots
        ax.set_xticklabels(ax.get_xticks(), rotation=45)  # Rota las etiquetas 45 grados


    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    plt.show()

def plot_all_snr_ser():
    """Plot all SNR vs SER from binary files in subplots"""
    # Crear los subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 16))
    fig.suptitle('SNRxSER', fontsize=16)

    # Primer subplot: SER vs SNR para un Sample per Chip y diferentes Spreading Factors en canal AWGN
    axs[0, 0].set_title('SER vs SNR para 1 Sample per Chip y diferentes Spreading Factors en canal AWGN')
    for sf in range(7, 13):
        plot_SER_SNR_from_binary(f'../data/SNRxSER_AWGN/spc1/SER_SNR_sf{sf}.npy', sf, 1,
                                f'SF={sf}, BW=125kHz', title_fill='Different SF & AWGN Channel', ax=axs[0, 0], num_sims=10000000)
    axs[0, 0].legend()

    # Segundo subplot: SER vs SNR para SF=7 y diferentes Samples per Chip en canal AWGN
    axs[0, 1].set_title('SER vs SNR para SF=7 y diferentes Samples per Chip en canal AWGN')
    for spc in [1, 2, 4, 8, 10]:
        plot_SER_SNR_from_binary(f'../data/SNRxSER_AWGN/spc{spc}/SER_SNR_sf7.npy', 7, spc,
                                f'SPC={spc}, BW=125kHz', title_fill='Different SPC & AWGN Channel', ax=axs[0, 1], num_sims=10000000)
    axs[0, 1].legend()

    # Tercer subplot: SER vs SNR para un Sample per Chip y diferentes Spreading Factors en canal Frequency Selective
    axs[1, 0].set_title('SER vs SNR para 1 Sample per Chip y diferentes Spreading Factors en canal Frequency Selective')
    for sf in range(7, 13):
        plot_SER_SNR_from_binary(f'../data/SNRxSER_FreqSel/spc1/SER_SNR_sf{sf}.npy', sf, 1,
                                f'SF={sf}, BW=125kHz', title_fill='Different SF & Frequency Selective Channel', ax=axs[1, 0], num_sims=10000000)
    axs[1, 0].legend()

    # Cuarto subplot: SER vs SNR para SF=7 y diferentes Samples per Chip en canal Frequency Selective
    axs[1, 1].set_title('SER vs SNR para SF=7 y diferentes Samples per Chip en canal Frequency Selective')
    for spc in [1, 2, 4, 8, 10]:
        plot_SER_SNR_from_binary(f'../data/SNRxSER_FreqSel/spc{spc}/SER_SNR_sf7.npy', 7, spc,
                                f'SPC={spc}, BW=125kHz', title_fill='Different SPC & Frequency Selective Channel', ax=axs[1, 1], num_sims=10000000)
    axs[1, 1].legend()

    # Ecuación en LaTeX colocada en el centro debajo del último subplot
    equation = r"$h(nT) = \sqrt{0.8} \delta(nT) + \sqrt{0.2} \delta(nT - T)$"
    fig.text(0.5, 0.02, "Impulse Response of Frequency Selective Channel: "+equation, fontsize=14, ha='center', va='center')

    for ax in axs.flat:  # Itera sobre todos los subplots
        ax.set_xticklabels(ax.get_xticks(), rotation=45)  # Rota las etiquetas 45 grados


    plt.tight_layout(rect=[0, 0.05, 1, 0.96])  # Ajusta para dejar espacio para la ecuación
    plt.show()

def plot_vangelista_comparison():
    """Plot SNR vs BER comparison between current implementation and Vangelista's implementation"""
    fig, axs = plt.subplots(1, 1, figsize=(15, 8))
    fig.suptitle('SNRxBER Comparison between Different LoRa Implementations', fontsize=16)

    # Primer sublplot: Implementación normal, con decimación sin el uso de abs()
    plot_BER_SNR_from_binary(f'../data/SNRxBER_AWGN/spc1/BER_SNR_sf7.npy', 7, 1,
                            'Current Implementation: Use of integration and phase retainer', title_fill='', ax=axs, num_sims=10000000)

    # Implementación con decimación y uso de abs()
    plot_BER_SNR_from_binary(f'../data/vangelista_comparison/BER_SNR_sf7_bw125k_spc1_mod_simple.npy', 7, 1,
                            'Vangelista Implementation', title_fill='', ax=axs, num_sims=10000000)
    axs.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_FER_SNR_from_binary(filename, sf, spc, label_fill, title_fill='FER vs SNR', ax=None, num_sims= None):
    """Plot the FER vs SNR from a binary file"""
    
    if ax is None:
        ax = plt.gca()
    
    # Cargar el archivo binario
    data = np.load(filename)
    SNR_values = data[0, :]
    FER_values = data[1, :]

    if num_sims is not None:
        limit = ERRORS_PER_SNR_POINT / num_sims
        mask = FER_values >= limit
        SNR_values = SNR_values[mask]
        FER_values = FER_values[mask]
    
    # Graficar los datos
    ax.plot(SNR_values, FER_values, marker='o', linestyle='-', label=label_fill)
    ax.set_xlim([SNR_values.min(), SNR_values.max()])
    ax.set_yscale('log')  # Escala logarítmica para PER
    ax.set_xticks(np.arange(-30 + 12 - sf, 2, 1))
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('FER (log scale)')
    ax.grid(which='both', linestyle='--', linewidth=0.5)
    ax.set_title(title_fill)

def plot_relative_SER_SNR_from_binary(filename, sf, spc, label_fill, title_fill='Relative SER vs SNR', ax=None, num_sims=None):
    """Plot the relative SER vs SNR from a binary file"""
    
    if ax is None:
        ax = plt.gca()
    
    # Cargar el archivo binario
    data = np.load(filename)
    SNR_values = data[0, :]
    SER_values = data[2, :]

    if num_sims is not None:
        limit = ERRORS_PER_SNR_POINT / num_sims
        mask = SER_values >= limit
        SNR_values = SNR_values[mask]
        SER_values = SER_values[mask]
    
    # Graficar los datos
    ax.plot(SNR_values, SER_values, marker='o', linestyle='-', label=label_fill)
    ax.set_xlim([SNR_values.min(), SNR_values.max()])
    ax.set_yscale('log')  # Escala logarítmica para SER
    ax.set_xticks(np.arange(-30 + 12 - sf, 2, 1))
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('SER (log scale)')
    ax.grid(which='both', linestyle='--', linewidth=0.5)
    ax.set_title(title_fill)

def plot_frame_corr_performance():
    fig, axs = plt.subplots(1, 1, figsize=(18, 8))
    fig.suptitle('Frame Performance for Correlation', fontsize=16)

    plot_FER_SNR_from_binary('../data/SNRxSER_PER_Corr/spc2/SNRxSER_PER_Corr_sf7.npy', 7, 2,
                            'FER vs SNR', ax=axs, title_fill="SNR vs SER & FER", num_sims=10000000)



    plot_relative_SER_SNR_from_binary('../data/SNRxSER_PER_Corr/spc2/SNRxSER_PER_Corr_sf7.npy', 7, 2,
                            'SER vs SNR', ax=axs, title_fill="SNR vs SER & FER", num_sims=10000000)


    axs.set_ylabel('FER/SER (log scale)')
    axs.legend()
    axs.plot()


def plot_frame_corr_mode_comparison():
    """Plot the comparison of frame performance between compact and full correlation modes."""
    fig, axs = plt.subplots(1, 1, figsize=(18, 8))
    fig.suptitle('Frame Performance: Full vs Compact Correlation Mode', fontsize=16)
    axs.set_title('SNRxFER')
    plot_FER_SNR_from_binary('../data/SNRxSER_PER_Corr/spc2/SNRxSER_PER_Corr_sf7.npy', 7, 2,
                            'FER vs SNR - Compact', ax=axs, num_sims=10000000)
    plot_FER_SNR_from_binary('../data/SNRxSER_PER_Corr/spc2/SNRxSER_PER_Corr_sf_full7.npy', 7, 2,
                            'FER vs SNR - Full', ax=axs, num_sims=10000000)

    axs.legend()
    axs.plot()

def plot_frame_dechirp_performance():
    fig, axs = plt.subplots(1, 1, figsize=(18, 8))
    fig.suptitle('Frame Performance for a Dechirping Synchronizer', fontsize=16)
    axs.set_title('SNRxFER')
    plot_FER_SNR_from_binary('../data/SNRxSER_PER_Sync/spc2/SNRxSER_PER_Sync_sf7.npy', 7, 2,
                            'FER vs SNR', ax=axs, num_sims=10000000)


    axs.set_title('SNRxRelative SER')
    plot_relative_SER_SNR_from_binary('../data/SNRxSER_PER_Sync/spc2/SNRxSER_PER_Sync_sf7.npy', 7, 2,
                            'SER vs SNR', ax=axs, title_fill="SNR vs SER", num_sims=10000000)


    axs.set_ylabel('FER/SER (log scale)')
    axs.legend()
    axs.plot()

def plot_frame_dechirp_vs_corr_performance():
    """Plot the performance comparison between Dechirping Synchronizer and Correlation"""
    fig, axs = plt.subplots(1, 1, figsize=(18, 8))
    fig.suptitle('Frame Performance: Correlation vs Dechirping Synchronizer (SF 7 & SPC 2)', fontsize=16)
    axs.set_title('SNRxFER')
    plot_FER_SNR_from_binary('../data/SNRxSER_PER_Sync/spc2/SNRxSER_PER_Sync_sf7.npy', 7, 2,
                            'FER vs SNR: Dechirping Synchronizer', ax=axs, title_fill="SNR x FER", num_sims=10000000)

    plot_FER_SNR_from_binary('../data/SNRxSER_PER_Corr/spc2/SNRxSER_PER_Corr_sf7.npy', 7, 2,
                            'FER vs SNR: Correlation', ax=axs, title_fill="SNR x FER: Dechirp-based vs Correlation", num_sims=10000000)

    axs.legend()
    axs.plot()

def plot_SER_SNR_from_binary_with_shift(filename, sf, spc, label_fill, title_fill='SER vs SNR', ax=None, num_sims=None, shift_dB: float = 0.5):
    """
    Plot the SER vs SNR from a binary file, optionally applying a vertical shift (in log scale).
    
    Parameters:
    - filename: path to the .npy file
    - sf: spreading factor
    - spc: samples per chip
    - label_fill: label for the legend
    - title_fill: plot title
    - ax: matplotlib Axes object
    - num_sims: number of simulations (for reliability threshold)
    - shift_dB: vertical shift in decades (log10 scale). Positive values shift the curve down.
    """
    if ax is None:
        ax = plt.gca()

    data = np.load(filename)
    SNR_values = data[0, :]
    SER_values = data[1, :]

    # Apply log-scale vertical shift (e.g. shift=0.5 -> divide by 10^0.5)
    if shift_dB != 0.0:
        SER_values = SER_values / (10 ** shift_dB)

    # Mask unreliable values
    if num_sims is not None:
        limit = ERRORS_PER_SNR_POINT / num_sims
        mask = SER_values >= limit
        SNR_values = SNR_values[mask]
        SER_values = SER_values[mask]

    ax.plot(SNR_values, SER_values, marker='o', linestyle='-', label=label_fill)
    ax.set_xlim([SNR_values.min(), SNR_values.max()])
    ax.set_yscale('log')
    ax.set_xticks(np.arange(-30 + 12 - sf, 2, 1))
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('SER (log scale)')
    ax.grid(which='both', linestyle='--', linewidth=0.5)
    ax.set_title(title_fill)


def plot_all_sync_relative_SER_comparison():
    """Compare relative SER for: no sync, correlation-based sync, and dechirping sync"""
    fig, axs = plt.subplots(1, 1, figsize=(18, 8))
    fig.suptitle('Relative Symbol Error Rate: Unsynchronized vs Correlation vs Dechirping', fontsize=16)
    axs.set_title('SNR x Relative SER (Payload)')

    # Caso base: sin sincronizador (modulación/demodulación directa sin detección de trama)
    plot_SER_SNR_from_binary(f'../data/SNRxSER_AWGN/spc2/SER_SNR_sf7.npy', 7, 2,
                                f'SPC=2, BW=125kHz', title_fill='Different SPC & AWGN Channel', ax=axs, num_sims=10000000)

    # Sincronizador por correlación
    plot_relative_SER_SNR_from_binary(
        '../data/SNRxSER_PER_Corr/spc2/SNRxSER_PER_Corr_sf7.npy', 7, 2,
        'Correlation-based Synchronizer', ax=axs, title_fill='', num_sims=10000000)

    # Sincronizador por dechirping
    plot_relative_SER_SNR_from_binary(
        '../data/SNRxSER_PER_Sync/spc2/SNRxSER_PER_Sync_sf7.npy', 7, 2,
        'Dechirping Synchronizer', ax=axs, title_fill='', num_sims=10000000)

    axs.set_xlabel('SNR (dB)')
    axs.set_ylabel('Relative SER (log scale)')
    axs.grid(which='both', linestyle='--', linewidth=0.5)
    axs.legend()
    plt.tight_layout()
    plt.show()
