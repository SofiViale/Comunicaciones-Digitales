# Comunicaciones Digitales – Framework LoRa PHY

Este repositorio contiene un framework modular para el estudio, simulación y validación experimental de la capa física de LoRa (LoRa PHY), incluyendo modulación, demodulación, sincronización y pruebas con canales reales y simulados. La arquitectura está diseñada para experimentación reproducible, análisis comparativo y validación teórica basada en los trabajos de Vangelista (2017) y Xu et al. (2022).



##  Estructura del Proyecto

### `src/`
Código fuente principal del framework.

#### `core/`
Componentes esenciales del sistema:
- *LoRaPhyParams*, *LoRaFrameParams*: modelos de parámetros PHY y de trama.  
- `sdr_utils.py`: emisión/recepción con SDR (PlutoSDR).  
- Utilidades varias: estimación de SNR, networking, misc helpers.

#### `mod/`
Módulos de **modulación LoRa**:
- Generación de chirps.  
- Construcción de símbolos y tramas.  
- Herramientas de visualización (`plot_frame()`).

#### `demod/`
Implementación de **demodulación**:
- Dechirping.  
- Extracción de simbolos basado en la FFT.  
- Pipeline completo y herramientas de análisis (`plot_demodulation()`).

#### `sync/`
Algoritmos de **sincronización temporal y frecuencial**:
- `DechirpBasedSynchronizer` y `CorrelationBasedSynchronizer`.  
- Alineamiento de ventanas y detección de tramas.  
- Visualización (`plot_synchronization()`).

#### `codec/`
Codificación y decodificación de payload:
- Interleaving, whitening, operaciones específicas del PHY LoRa.


### `notebooks/`
Notebooks experimentales para:
- Validación del PHY.  
- Evaluación de sincronización.  
- Pruebas comparativas de SNR, PER, BER.  
- Análisis con canales reales y simulados.



### Modelado de canales en `quantitative_tests.ipynb`
Modelos de canal:
- AWGN.  
- Multipath y selectividad en frecuencia.  
- Desalineación de fase.
- Canales reales con integracion VPN y SDRs.



### Generacion de curvas de rendimiento `quantitative_tests.ipynb`
Scripts automatizados para:  
- Evaluación de métricas (FER/BER/SER).  
- Comparativas entre módulos o parámetros.



### Visualizacion de curvas de rendimiento
Resultados generados por los experimentos:
- Métricas (JSON).  
- Figuras, plots y logs.

## Referencias

- **[Vangelista, L. (2017)]**  
  *Frequency Shift Chirp Modulation: The LoRa Modulation.*  
  IEEE Signal Processing Letters, 24(12), 1818–1822.  
  

- **[Xu, Z., Tong, S., Xie, P., & Wang, J. (2022)]**  
  *From Demodulation to Decoding: Toward Complete LoRa PHY Understanding and Implementation.*  
  ACM Transactions on Sensor Networks, 18(4), Article 64.  
