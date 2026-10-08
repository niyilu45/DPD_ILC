"""Shared Wi-Fi OFDM frame demodulation and spatial-stream processing.

The module contains receiver-side frame operations shared by waveform
analysis and any future Wi-Fi receiver. It depends only on neutral waveform
metadata, never on the waveform generator or PA implementation.
"""

from typing import Optional

import numpy as np

from .WifiMetadata import WifiWaveform


def BuildCsdPhaseMatrix(
    subcarrierIndices: np.ndarray,
    subcarrierSpacingHz: float,
    cyclicShiftsSeconds: np.ndarray,
) -> np.ndarray:
    """Build frequency-dependent CSD phases for all tones and chains.

    Processing details:
        Algorithm: Evaluate ``exp(-j*2*pi*k*deltaF*tau)`` for every centered
        subcarrier index and transmit-chain cyclic shift. The same matrix is
        used by the transmitter for applying CSD and by the receiver through
        complex conjugation for removing CSD.

    Args:
        subcarrierIndices: Centered signed OFDM tone indices.
        subcarrierSpacingHz: Positive tone spacing in hertz.
        cyclicShiftsSeconds: Finite per-chain cyclic shifts in seconds.

    Returns:
        result: Complex phase matrix shaped tones by transmit antennas.
    """

    toneIndices = np.asarray(subcarrierIndices, dtype=float).reshape(-1)
    cyclicShifts = np.asarray(
        cyclicShiftsSeconds, dtype=float
    ).reshape(-1)
    if toneIndices.size == 0:
        raise ValueError("subcarrierIndices cannot be empty")
    if cyclicShifts.size == 0:
        raise ValueError("cyclicShiftsSeconds cannot be empty")
    if not np.all(np.isfinite(toneIndices)):
        raise ValueError("subcarrierIndices must contain finite values")
    if not np.all(np.isfinite(cyclicShifts)):
        raise ValueError("cyclicShiftsSeconds must contain finite values")
    if (
        not isinstance(subcarrierSpacingHz, (int, float))
        or isinstance(subcarrierSpacingHz, bool)
        or not np.isfinite(subcarrierSpacingHz)
        or subcarrierSpacingHz <= 0.0
    ):
        raise ValueError("subcarrierSpacingHz must be finite and positive")
    toneFrequencies = (
        toneIndices.reshape(-1, 1) * float(subcarrierSpacingHz)
    )
    return np.exp(
        -1j
        * 2.0
        * np.pi
        * toneFrequencies
        * cyclicShifts.reshape(1, -1)
    )


class FrameProcess:
    """Demodulate Wi-Fi data symbols using shared waveform metadata.

    The processor removes the cyclic prefix, performs a unitary FFT, selects
    data tones, removes transmitter cyclic-shift diversity, and reverses the
    known orthonormal spatial mapping. It does not perform timing, CFO, SFO, or
    common-gain correction; callers must apply ``SigProc`` first.
    """

    def __init__(self, waveform: WifiWaveform) -> None:
        """Initialize one receiver-side frame-processing context.

        Processing details:
            Algorithm: Retain the immutable-by-convention waveform metadata
            and validate every field needed by OFDM and MIMO demodulation.

        Args:
            waveform: Shared Wi-Fi samples and frame metadata.

        Returns:
            result: None. The instance is ready to demodulate aligned samples.
        """

        if not isinstance(waveform, WifiWaveform):
            raise TypeError("waveform must be a WifiWaveform")
        self.waveform = waveform
        self.ValidateMetadata()

    def ValidateMetadata(self) -> None:
        """Validate metadata required for frame demodulation.

        Processing details:
            Algorithm: Check FFT and CP domains, antenna dimensions, tone
            indices, symbol starts, cyclic shifts, and the spatial-map shape
            before any signal is transformed.

        Returns:
            result: None. Invalid metadata raises a descriptive exception.
        """

        if (
            not isinstance(self.waveform.sampleRateHz, (int, float))
            or isinstance(self.waveform.sampleRateHz, bool)
            or not np.isfinite(self.waveform.sampleRateHz)
            or self.waveform.sampleRateHz <= 0.0
        ):
            raise ValueError(
                "waveform.sampleRateHz must be finite and positive"
            )
        if (
            not isinstance(self.waveform.fftLength, int)
            or isinstance(self.waveform.fftLength, bool)
            or self.waveform.fftLength < 1
        ):
            raise ValueError("waveform.fftLength must be a positive integer")
        if (
            not isinstance(self.waveform.cpLength, int)
            or isinstance(self.waveform.cpLength, bool)
            or self.waveform.cpLength < 0
        ):
            raise ValueError(
                "waveform.cpLength must be a nonnegative integer"
            )
        if (
            not isinstance(self.waveform.numTransmitAntennas, int)
            or isinstance(self.waveform.numTransmitAntennas, bool)
            or self.waveform.numTransmitAntennas < 1
        ):
            raise ValueError(
                "waveform.numTransmitAntennas must be a positive integer"
            )
        if (
            not isinstance(self.waveform.numSpatialStreams, int)
            or isinstance(self.waveform.numSpatialStreams, bool)
            or self.waveform.numSpatialStreams < 1
            or (
                self.waveform.numSpatialStreams
                > self.waveform.numTransmitAntennas
            )
        ):
            raise ValueError(
                "waveform.numSpatialStreams must be between one and the "
                "transmit-antenna count"
            )
        waveformSamples = np.asarray(self.waveform.samples)
        if waveformSamples.ndim == 1:
            if self.waveform.numTransmitAntennas != 1:
                raise ValueError(
                    "one-dimensional waveform.samples requires one antenna"
                )
        elif (
            waveformSamples.ndim != 2
            or waveformSamples.shape[1]
            != self.waveform.numTransmitAntennas
        ):
            raise ValueError(
                "waveform.samples must have one column per transmit antenna"
            )
        if waveformSamples.size == 0 or not np.all(
            np.isfinite(waveformSamples)
        ):
            raise ValueError(
                "waveform.samples must be nonempty and finite"
            )
        dataSubcarriers = np.asarray(
            self.waveform.dataSubcarriers
        ).reshape(-1)
        dataSymbolStarts = np.asarray(
            self.waveform.dataSymbolStarts
        ).reshape(-1)
        cyclicShifts = np.asarray(
            self.waveform.cyclicShiftsSeconds
        ).reshape(-1)
        spatialMappingMatrix = np.asarray(
            self.waveform.spatialMappingMatrix
        )
        if dataSubcarriers.size == 0:
            raise ValueError("waveform.dataSubcarriers cannot be empty")
        if dataSymbolStarts.size == 0:
            raise ValueError("waveform.dataSymbolStarts cannot be empty")
        if not np.all(np.isfinite(dataSubcarriers)):
            raise ValueError(
                "waveform.dataSubcarriers must contain finite values"
            )
        if not np.all(dataSubcarriers == np.rint(dataSubcarriers)):
            raise ValueError(
                "waveform.dataSubcarriers must contain integer tone indices"
            )
        if np.any(
            np.abs(dataSubcarriers) > self.waveform.fftLength // 2
        ):
            raise ValueError(
                "waveform.dataSubcarriers exceed the centered FFT grid"
            )
        if not np.all(np.isfinite(dataSymbolStarts)):
            raise ValueError(
                "waveform.dataSymbolStarts must contain finite values"
            )
        if (
            not np.all(dataSymbolStarts == np.rint(dataSymbolStarts))
            or np.any(dataSymbolStarts < 0)
        ):
            raise ValueError(
                "waveform.dataSymbolStarts must be nonnegative integers"
            )
        usefulStops = (
            dataSymbolStarts
            + self.waveform.cpLength
            + self.waveform.fftLength
        )
        if np.any(usefulStops > waveformSamples.shape[0]):
            raise ValueError(
                "waveform.dataSymbolStarts exceed the waveform sample range"
            )
        if cyclicShifts.size != self.waveform.numTransmitAntennas:
            raise ValueError(
                "waveform.cyclicShiftsSeconds must match transmit antennas"
            )
        if not np.all(np.isfinite(cyclicShifts)):
            raise ValueError(
                "waveform.cyclicShiftsSeconds must contain finite values"
            )
        expectedMappingShape = (
            self.waveform.numTransmitAntennas,
            self.waveform.numSpatialStreams,
        )
        if spatialMappingMatrix.shape != expectedMappingShape:
            raise ValueError(
                "waveform.spatialMappingMatrix must be shaped antennas by "
                "spatial streams"
            )
        if not np.all(np.isfinite(spatialMappingMatrix)):
            raise ValueError(
                "waveform.spatialMappingMatrix must contain finite values"
            )
        mappingGramMatrix = (
            spatialMappingMatrix.conj().T @ spatialMappingMatrix
        )
        if not np.allclose(
            mappingGramMatrix,
            np.eye(self.waveform.numSpatialStreams),
            rtol=1.0e-10,
            atol=1.0e-12,
        ):
            raise ValueError(
                "waveform.spatialMappingMatrix columns must be orthonormal"
            )

    def ValidatePreparedSignal(
        self, preparedSignal: np.ndarray
    ) -> np.ndarray:
        """Validate an aligned signal against the waveform sample grid.

        Processing details:
            Algorithm: Convert to complex double precision, require the exact
            SISO or samples-by-antennas shape, and reject non-finite values.

        Args:
            preparedSignal: Synchronized signal on the reference sample grid.

        Returns:
            result: Validated complex array without changing sample values.
        """

        complexPrepared = np.asarray(
            preparedSignal, dtype=np.complex128
        )
        expectedShape = np.asarray(self.waveform.samples).shape
        if complexPrepared.shape != expectedShape:
            raise ValueError(
                "preparedSignal shape must match waveform.samples"
            )
        if complexPrepared.ndim not in (1, 2):
            raise ValueError("preparedSignal must be a vector or matrix")
        if not np.all(np.isfinite(complexPrepared)):
            raise ValueError("preparedSignal must contain finite samples")
        return complexPrepared

    def DemodulatePreparedWifiData(
        self,
        preparedSignal: np.ndarray,
        maximumSymbolCount: Optional[int] = None,
        symbolIndices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Demodulate data tones and recover transmitted spatial streams.

        Processing details:
            Algorithm: Gather requested data symbols in bounded batches,
            remove the cyclic prefix, apply unitary FFTs along the sample
            axis, select data subcarriers, remove CSD, and invert the
            orthonormal spatial mapping. Each batch holds at most 131072
            complex input points across all antennas, or one symbol when
            that symbol alone exceeds the bound. Consecutive symbols use
            buffer views; irregular selections use bounded indexed copies.
            Tone indices and inverse CSD/mapping factors are shared within
            this call, while caller symbol order and duplicates are preserved.

        Args:
            preparedSignal: Synchronized signal on the reference sample grid.
            maximumSymbolCount: Optional positive limit for debugging or
                partial-frame analysis; ``None`` processes every data symbol.
            symbolIndices: Optional one-dimensional integer indices selecting
                fully captured data symbols before maximumSymbolCount applies.

        Returns:
            result: SISO matrix shaped symbols by data tones, or MIMO tensor
                shaped symbols by data tones by spatial streams.
        """

        complexPrepared = self.ValidatePreparedSignal(preparedSignal)
        symbolStarts = np.asarray(
            self.waveform.dataSymbolStarts, dtype=int
        ).reshape(-1)
        if symbolIndices is not None:
            indexArray = np.asarray(symbolIndices)
            if (
                indexArray.ndim != 1
                or indexArray.size == 0
                or not np.issubdtype(indexArray.dtype, np.integer)
                or np.any(indexArray < 0)
                or np.any(indexArray >= symbolStarts.size)
            ):
                raise ValueError("symbolIndices must contain valid data-symbol indices")
            symbolStarts = symbolStarts[indexArray]
        if maximumSymbolCount is not None:
            if (
                not isinstance(maximumSymbolCount, int)
                or isinstance(maximumSymbolCount, bool)
                or maximumSymbolCount < 1
            ):
                raise ValueError(
                    "maximumSymbolCount must be a positive integer or None"
                )
            symbolStarts = symbolStarts[:maximumSymbolCount]
        csdPhaseMatrix = BuildCsdPhaseMatrix(
            self.waveform.dataSubcarriers,
            self.waveform.sampleRateHz / self.waveform.fftLength,
            self.waveform.cyclicShiftsSeconds,
        )
        fftLength = self.waveform.fftLength
        usefulStarts = symbolStarts + self.waveform.cpLength
        if np.any(usefulStarts + fftLength > complexPrepared.shape[0]):
            raise ValueError(
                "preparedSignal is shorter than the Wi-Fi data field"
            )
        dataToneIndices = np.mod(self.waveform.dataSubcarriers, fftLength)
        inverseCsdPhase = np.conj(csdPhaseMatrix)
        inverseSpatialMapping = np.conj(self.waveform.spatialMappingMatrix)
        fftScale = np.sqrt(fftLength)
        preparedMatrix = (
            complexPrepared.reshape(-1, 1)
            if complexPrepared.ndim == 1
            else complexPrepared
        )
        batchSymbolCount = max(
            1, 131072 // (fftLength * self.waveform.numTransmitAntennas)
        )
        symbolLength = fftLength + self.waveform.cpLength
        sampleOffsets = np.arange(fftLength)
        demodulatedArray = np.empty(
            (
                symbolStarts.size,
                np.asarray(dataToneIndices).size,
                self.waveform.numSpatialStreams,
            ),
            dtype=np.complex128,
        )
        for batchStart in range(0, symbolStarts.size, batchSymbolCount):
            batchStop = min(batchStart + batchSymbolCount, symbolStarts.size)
            batchStarts = usefulStarts[batchStart:batchStop]
            if np.all(np.diff(batchStarts) == symbolLength):
                # Consecutive OFDM symbols share a strided view of the
                # original buffer, so their useful samples need no copy.
                batchField = preparedMatrix[
                    int(batchStarts[0]) - self.waveform.cpLength:
                    int(batchStarts[-1]) + fftLength
                ]
                usefulSamples = batchField.reshape(
                    batchStop - batchStart,
                    symbolLength,
                    self.waveform.numTransmitAntennas,
                )[:, self.waveform.cpLength:, :]
            else:
                sampleIndices = batchStarts[:, None] + sampleOffsets[None, :]
                usefulSamples = preparedMatrix[sampleIndices]
            frequencyGrid = np.fft.fft(usefulSamples, axis=1)
            frequencyGrid /= fftScale
            antennaData = frequencyGrid[:, dataToneIndices, :]
            # The transmit mapping is y = s Q^T D_csd. Because Q has
            # orthonormal columns and D_csd is unitary, conjugating both terms
            # in reverse order gives the corresponding left inverse.
            antennaData *= inverseCsdPhase
            demodulatedArray[batchStart:batchStop] = (
                antennaData @ inverseSpatialMapping
            )
        if self.waveform.numTransmitAntennas == 1:
            return demodulatedArray[:, :, 0]
        return demodulatedArray
