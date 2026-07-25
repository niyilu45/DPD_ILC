"""Shared Wi-Fi modulation and waveform metadata.

This module contains data-only structures consumed by waveform generation,
frame processing, and result analysis. Keeping these structures independent
prevents the receiver-side analysis path from importing the waveform
generator implementation.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class MCSInfo:
    """Describe the modulation and nominal coding parameters of one Wi-Fi MCS.

    Attributes:
        index: Integer modulation-and-coding-scheme index.
        modulation: Human-readable modulation name.
        qamOrder: Number of constellation points.
        codeRate: Nominal forward-error-correction code rate.
        bitsPerSubcarrier: Uncoded bits carried by one data subcarrier.
    """

    index: int
    modulation: str
    qamOrder: int
    codeRate: float
    bitsPerSubcarrier: int


@dataclass
class WifiWaveform:
    """Store waveform samples and metadata required by downstream processing.

    The class intentionally contains no waveform-generation or receiver
    algorithms. ``WaveGenWifi`` produces it, ``FrameProcess`` consumes the
    OFDM and MIMO fields, and ``Analysis`` reads common timing and bandwidth
    fields without depending on the generator implementation.

    Attributes:
        samples: Unit-RMS complex waveform, shaped samples for SISO or
            samples-by-transmit-antennas for MIMO.
        sampleRateHz: Complex-baseband sample rate in hertz.
        bandwidthHz: Nominal occupied channel bandwidth in hertz.
        fftLength: OFDM FFT length in samples.
        cpLength: Data-symbol cyclic-prefix length in samples.
        oversampling: Resolved sample-rate-to-bandwidth ratio.
        activeSubcarriers: Centered signed indices for all active tones.
        dataSubcarriers: Centered signed indices for data tones.
        pilotSubcarriers: Centered signed indices for pilot tones.
        referenceDataSymbols: Ideal per-symbol data constellation values.
        fieldSlices: Named half-open packet-field sample ranges.
        dataSymbolStarts: Start sample of every data OFDM symbol.
        symbolLength: Data OFDM symbol length including cyclic prefix.
        mcsInfo: Resolved modulation and coding metadata.
        normalizationScale: Scale used to make the complete packet unit RMS.
        codedBitsPerSymbol: Coded payload bits per OFDM symbol.
        informationBitsPerSymbol: Nominal information bits per OFDM symbol.
        frameFormat: Canonical VHT, HE, or EHT PHY name.
        dataFieldName: Format-specific data-field key in ``fieldSlices``.
        formatName: Human-readable generation and PHY description.
        numTransmitAntennas: Number of physical transmit chains.
        numSpatialStreams: Number of independently modulated spatial streams.
        spatialMapping: Selected direct, DFT, or custom mapping name.
        spatialMappingMatrix: Orthonormal antenna-by-stream mapping matrix.
        cyclicShiftsSeconds: Per-antenna cyclic shifts in seconds.
        ltfSymbolCount: Number of format-specific long-training symbols.
        seed: Unsigned 32-bit random seed used for deterministic regeneration.
        cyclicShiftEnabled: Original generator CSD enable configuration.
    """

    samples: np.ndarray
    sampleRateHz: float
    bandwidthHz: float
    fftLength: int
    cpLength: int
    oversampling: float
    activeSubcarriers: np.ndarray
    dataSubcarriers: np.ndarray
    pilotSubcarriers: np.ndarray
    referenceDataSymbols: np.ndarray
    fieldSlices: Dict[str, slice]
    dataSymbolStarts: np.ndarray
    symbolLength: int
    mcsInfo: MCSInfo
    normalizationScale: float
    codedBitsPerSymbol: int
    informationBitsPerSymbol: int
    frameFormat: str
    dataFieldName: str
    formatName: str
    numTransmitAntennas: int
    numSpatialStreams: int
    spatialMapping: str
    spatialMappingMatrix: np.ndarray
    cyclicShiftsSeconds: np.ndarray
    ltfSymbolCount: int
    seed: int
    cyclicShiftEnabled: bool
