import numpy as np

class SimpleLoraMoDem():
    def __init__(self, spreading_factor, bandwidth, samples_per_chip = 1):
        self.spreading_factor = spreading_factor
        self.bandwidth = bandwidth
        self.samples_per_chip = samples_per_chip
        self.downchirp = self.generate_downchirp()
    
    def generate_downchirp(self):
        chips_number = 2**self.spreading_factor  # Número de chips por símbolo
        k = np.arange(chips_number * self.samples_per_chip)  # Índices de tiempo
        downchirp = np.exp(-1j * 2 * np.pi * (k**2) / (chips_number * (self.samples_per_chip**2)))  # Término e^{-j 2π k² / 2^SF}
        downchirp_power = np.mean(np.abs(downchirp)**2)
        downchirp = downchirp / np.sqrt(downchirp_power)
        return downchirp

    def modulate_symbols(self, symbols):
        signal = np.array([])
        for symbol in symbols:
            """
            Genera la señal LoRa para un símbolo dado.

            Parameters:
            symbols (int): Símbolos a modular.
            Returns:
            np.array: Señal LoRa generada.
            """
            chips_number = 2**self.spreading_factor  # Número de chips por símbolo
            k = np.arange(chips_number * self.samples_per_chip)  # Índices de tiempo

            # Generar la señal LoRa
            parcial_signal = (1 / np.sqrt(chips_number)) * np.exp(
                1j * 2 * np.pi * ((symbol + k / self.samples_per_chip) % chips_number)  * (k / (self.samples_per_chip * chips_number))
            )
            signal = np.concatenate((signal, parcial_signal))
        
        power = np.mean(np.abs(signal)**2)
        signal = signal / np.sqrt(power)
        return signal
    
    def demodulate_symbols(self, received_signal):
        """
        Demodula múltiples símbolos de la señal LoRa recibida, siguiendo la fórmula de la imagen.

        Parameters:
        received_signal (np.array): Señal LoRa recibida.

        Returns:
        list: Lista de símbolos demodulados.
        """
        symbols = []
        chips_number = 2**self.spreading_factor  # Número de chips por símbolo
        samples_per_symbol = chips_number * self.samples_per_chip  # Número de muestras por símbolo
        
        # Procesar cada símbolo en la señal recibida
        for i in range(0, len(received_signal), samples_per_symbol):
            # Extraer la señal de un símbolo
            symbol_signal = received_signal[i:i + samples_per_symbol]

            # Evitar procesar símbolos incompletos
            if len(symbol_signal) < samples_per_symbol:
                break

            # Multiplicar la señal recibida por el downchirp
            dechirped_signal = symbol_signal * self.downchirp

            # Aplicar la FFT para obtener la proyección en cada base
            fft_result = np.fft.fft(dechirped_signal)

            # Seleccionar el índice del símbolo (máximo de la magnitud de la FFT)
            symbol = np.argmax(np.abs(fft_result)) % chips_number

            # Agregar el símbolo a la lista de resultados
            symbols.append(symbol)

        return symbols
    