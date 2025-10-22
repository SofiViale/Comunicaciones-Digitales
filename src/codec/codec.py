
from typing import Union, List

"""
Optimized LoRa Codec Implementation
===================================

This module provides an efficient implementation of a LoRa codec for encoding
and decoding data into symbols using the specified spreading factor.

Key optimizations:
- Efficient bit manipulation using bitwise operations
- Memory-efficient processing without string concatenation
- Reduced computational complexity
- Proper error handling and validation
- Clean, dependency-free implementation
"""

from src.core import LoRaPhyParams

class LoRaCodec:
    """
    Optimized LoRa Codec for encoding/decoding data to/from symbols.
    """

    def __init__(self, phy_params: LoRaPhyParams):
        """
        Initialize the LoRa codec with optimized parameters.
        
        Args:
            phy_params.spreading_factor (int): Number of bits per symbol (typically 7-12)
            bandwidth (float): Signal bandwidth in Hz
            phy_params.samples_per_chip (int): Number of samples per chip
        """
        self.phy_params = phy_params
    
    def encode(self, data: Union[bytes, str]) -> List[int]:
        """
        Efficiently encode data into LoRa symbols.
        
        Args:
            data: Input data as bytes or string
            
        Returns:
            List of integers representing encoded symbols, with padding info as first element
            
        Raises:
            ValueError: If input data is invalid
        """
        if not data:
            return [0]  # Empty data case
            
        # Convert to bytes efficiently
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, bytes):
            data_bytes = data
        else:
            raise ValueError("Data must be string or bytes")
        
        # Convert bytes to integer for efficient bit manipulation
        data_int = int.from_bytes(data_bytes, byteorder='big')
        total_bits = len(data_bytes) * 8
        
        # Calculate padding needed
        bits_per_symbol = self.phy_params.spreading_factor
        padding = (bits_per_symbol - (total_bits % bits_per_symbol)) % bits_per_symbol
        
        # Add padding bits (zeros at the end)
        if padding > 0:
            data_int <<= padding
            total_bits += padding
        
        # Extract symbols efficiently using bit shifting
        symbols = []
        mask = (2**self.phy_params.spreading_factor - 1)
        
        for i in range(0, total_bits, bits_per_symbol):
            # Extract symbol from the rightmost bits
            symbol = data_int & mask
            symbols.append(symbol)
            # Shift right for next symbol
            data_int >>= bits_per_symbol
        
        # Reverse to maintain correct order (since we extracted from right to left)
        symbols.reverse()
        
        # Prepend padding information
        result = [padding] + symbols
        return result
    
    def decode(self, symbols: List[int]) -> bytes:
        """
        Efficiently decode LoRa symbols back to original data.
        
        Args:
            symbols: List of symbols to decode
            
        Returns:
            Decoded data as bytes
            
        Raises:
            ValueError: If symbols are invalid
        """
        if not symbols:
            return b''
        
        # Validate input
        if not isinstance(symbols, list) or len(symbols) < 1:
            raise ValueError("Symbols must be a non-empty list")
        
        # Extract padding and data symbols
        padding = symbols[0]
        data_symbols = symbols[1:]
        
        if not data_symbols:
            return b''
        
        # Validate symbols
        for i, symbol in enumerate(data_symbols):
            if not isinstance(symbol, int) or symbol < 0 or symbol > (2**self.phy_params.spreading_factor - 1):
                raise ValueError(f"Invalid symbol at index {i}: {symbol}")
        
        # Reconstruct data efficiently
        bits_per_symbol = self.phy_params.spreading_factor
        total_bits = len(data_symbols) * bits_per_symbol
        
        # Build integer from symbols using bit shifting
        data_int = 0
        for symbol in data_symbols:
            data_int = (data_int << bits_per_symbol) | symbol
        
        # Remove padding bits
        if padding > 0:
            data_int >>= padding
            total_bits -= padding
        
        # Convert back to bytes
        if total_bits == 0:
            return b''
        
        # Calculate number of bytes needed
        num_bytes = (total_bits + 7) // 8  # Ceiling division
        
        try:
            result_bytes = data_int.to_bytes(num_bytes, byteorder='big')
        except OverflowError:
            raise ValueError("Decoded data is too large")
        
        return result_bytes
    
    def get_efficiency_metrics(self, data_size_bytes: int) -> dict:
        """
        Calculate efficiency metrics for given data size.
        
        Args:
            data_size_bytes: Size of input data in bytes
            
        Returns:
            Dictionary with efficiency metrics
        """
        data_bits = data_size_bytes * 8
        bits_per_symbol = self.phy_params.spreading_factor
        
        # Calculate symbols needed (with padding)
        symbols_needed = (data_bits + bits_per_symbol - 1) // bits_per_symbol
        total_bits_transmitted = symbols_needed * bits_per_symbol
        padding_bits = total_bits_transmitted - data_bits
        
        efficiency = (data_bits / total_bits_transmitted) * 100 if total_bits_transmitted > 0 else 0
        
        return {
            'input_bytes': data_size_bytes,
            'input_bits': data_bits,
            'symbols_needed': symbols_needed,
            'padding_bits': padding_bits,
            'total_bits_transmitted': total_bits_transmitted,
            'efficiency_percent': round(efficiency, 2),
            'overhead_percent': round((padding_bits / data_bits) * 100, 2) if data_bits > 0 else 0
        }
    
    def __repr__(self) -> str:
        return (f"LoRaCodec(SF={self.phy_params.spreading_factor}, "
                f"BW={self.phy_params.bandwidth}Hz, SPC={self.phy_params.samples_per_chip})")

