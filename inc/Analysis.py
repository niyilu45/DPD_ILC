"""Object-oriented SNR, EVM, and ACLR analysis for VHT/HE/EHT simulations."""

import csv
import json
from collections import ChainMap
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .SigProcess import SigProcess, SignalProcessingResult
from .waveGen import BuildCsdPhaseMatrix, WifiWaveform


@dataclass(frozen=True)
class SignalMetrics:
    """Collect the requested signal-quality and spectral-quality metrics."""

    snrDb: float
    evmDb: float
    evmPercent: float
    aclrLowerDb: float
    aclrUpperDb: float
    aclrWorstDb: float

    def ToDict(self) -> Dict[str, float]:
        """Convert metrics to a JSON/CSV-ready dictionary.

        Processing details:
            Algorithm: Convert validated in-memory results into a stable reporting format without altering later numerical calculations.

        Returns:
            result: Dict[str, float]. The computed value described by the summary, with documented units, shape, and normalization.
        """

        return {key: float(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class MimoSignalMetrics:
    """Collect per-chain and per-spatial-stream MIMO metric details.

    Aggregate values remain available through ``SignalMetrics`` so existing
    SISO callers keep the same API. This companion record exposes conducted
    RF-chain SNR/ACLR and post-spatial-demapping EVM for each stream.
    """

    snrDbPerChain: Tuple[float, ...]
    evmDbPerSpatialStream: Tuple[float, ...]
    evmPercentPerSpatialStream: Tuple[float, ...]
    aclrLowerDbPerChain: Tuple[float, ...]
    aclrUpperDbPerChain: Tuple[float, ...]
    aclrWorstDbPerChain: Tuple[float, ...]

    def ToDict(self) -> Dict[str, object]:
        """Convert MIMO metric tuples to a JSON-ready dictionary.

        Processing details:
            Algorithm: Convert each immutable numeric tuple into ordinary
            float lists while preserving chain and stream ordering.

        Returns:
            result: Mapping containing all per-chain and per-stream metrics.
        """

        return {
            metricName: [float(value) for value in metricValues]
            for metricName, metricValues in asdict(self).items()
        }


@dataclass
class PowerEvmCurve:
    """Store a multi-method EVM sweep over PA input RMS drive levels."""

    driveRmsValues: np.ndarray
    inputPowerDb: np.ndarray
    evmDbByMethod: Dict[str, np.ndarray]
    evmPercentByMethod: Dict[str, np.ndarray]

    def ToDict(self) -> Dict[str, object]:
        """Convert all curve samples to a JSON-ready dictionary.

        Processing details:
            Algorithm: Convert validated in-memory results into a stable reporting format without altering later numerical calculations.

        Returns:
            result: Dict[str, object]. The computed value described by the summary, with documented units, shape, and normalization.
        """

        return {
            "driveRmsValues": self.driveRmsValues.astype(float).tolist(),
            "inputPowerDb": self.inputPowerDb.astype(float).tolist(),
            "methods": {
                methodName: {
                    "evmDb": self.evmDbByMethod[methodName]
                    .astype(float)
                    .tolist(),
                    "evmPercent": self.evmPercentByMethod[methodName]
                    .astype(float)
                    .tolist(),
                }
                for methodName in self.evmDbByMethod
            },
        }


def AveragePeriodogram(
    inputSignal: np.ndarray,
    sampleRateHz: float,
    maxSegmentLength: int = 16384,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate a low-variance PSD using overlapping Hann-windowed segments.

    Processing details:
        Algorithm: Perform the numerical calculation with explicit power, shape, and normalization handling for comparable results.

    Args:
        inputSignal: One-dimensional complex baseband samples supplied to the operation.
        sampleRateHz: Complex sample rate in samples per second.
        maxSegmentLength: Caller-supplied value consumed according to the function contract.

    Returns:
        result: Tuple[np.ndarray, np.ndarray]. The computed value described by the summary, with documented units, shape, and normalization.
    """

    complexInput = np.asarray(inputSignal, dtype=np.complex128).reshape(-1)
    if complexInput.size < 16:
        raise ValueError("inputSignal is too short for spectral analysis")

    segmentLength = min(maxSegmentLength, complexInput.size)
    segmentLength = 1 << int(np.floor(np.log2(segmentLength)))
    segmentStep = max(segmentLength // 2, 1)
    analysisWindow = np.hanning(segmentLength)
    windowPower = max(np.sum(analysisWindow**2), np.finfo(float).tiny)
    accumulatedPsd = np.zeros(segmentLength, dtype=float)
    segmentCount = 0

    for startIndex in range(
        0, complexInput.size - segmentLength + 1, segmentStep
    ):
        signalSegment = complexInput[startIndex : startIndex + segmentLength]
        signalSpectrum = np.fft.fftshift(
            np.fft.fft(signalSegment * analysisWindow)
        )
        accumulatedPsd += np.abs(signalSpectrum) ** 2 / windowPower
        segmentCount += 1

    if segmentCount == 0:
        raise RuntimeError("unable to create a PSD segment")
    averagePsd = accumulatedPsd / segmentCount
    frequencyBins = np.fft.fftshift(
        np.fft.fftfreq(segmentLength, d=1.0 / sampleRateHz)
    )
    return frequencyBins, averagePsd


class Analysis:
    """Analyze and save PA/DPD results for one reference Wi-Fi waveform.

    Callers construct one instance from the transmitted reference signal and
    its ``WifiWaveform`` metadata. Every PA, ILC, or deployed-DPD output is
    then evaluated through the same stored analysis context.

    Example:
        ``resultAnalysis = Analysis(referenceSignal, waveform)``
        ``metrics = resultAnalysis.Analyze(paOutput)``
    """

    def __init__(
        self,
        referenceSignal: np.ndarray,
        waveform: WifiWaveform,
        parameters: Optional[Mapping[str, object]] = None,
        **parameterOverrides: object,
    ) -> None:
        """Initialize a reusable signal-analysis context and its live parameter layers.

        Processing details:
            Algorithm: Define immutable analysis defaults inside the
            constructor and layer direct and external overrides ahead of them,
            keeping every default out of caller-side construction code.

        Args:
            referenceSignal: Ideal complex baseband samples used as the target or regression input.
            waveform: Wi-Fi metadata defining field locations, FFT sizes, and subcarriers.
            parameters: Optional external mapping layered ahead of the built-in defaults.
            parameterOverrides: Highest-priority keyword values applied to the local ChainMap layer.

        Returns:
            result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
        """
        self.defaultParameters: Mapping[str, object] = MappingProxyType(
            {
                "maxSegmentLength": 16384,
                "minimumAclrOversampling": 3.0,
                "powerEvmFileStem": "power_evm_curve",
                "signalProcessingParameters": None,
            }
        )
        complexReference = np.asarray(referenceSignal, dtype=np.complex128)
        if complexReference.size == 0:
            raise ValueError("referenceSignal cannot be empty")
        expectedShape = np.asarray(waveform.samples).shape
        if complexReference.shape != expectedShape:
            raise ValueError(
                "referenceSignal shape must match the Wi-Fi waveform"
            )
        if complexReference.ndim not in (1, 2):
            raise ValueError("referenceSignal must be a vector or matrix")
        if (
            complexReference.ndim == 2
            and complexReference.shape[1] != waveform.numTransmitAntennas
        ):
            raise ValueError(
                "referenceSignal must contain one column per transmit chain"
            )
        if not np.all(np.isfinite(complexReference)):
            raise ValueError("referenceSignal contains NaN or infinite values")
        self.referenceSignal = complexReference
        self.waveform = waveform
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping or None")
        externalParameters = {} if parameters is None else parameters
        self.parameters: ChainMap[str, object] = ChainMap(
            dict(parameterOverrides),
            externalParameters,
            self.defaultParameters,
        )
        self.ValidateParameters()
        self.stageMetrics: Dict[str, SignalMetrics] = {}
        self.stageSignalProcessingResults: Dict[
            str, Tuple[SignalProcessingResult, ...]
        ] = {}
        self.stageMimoMetrics: Dict[str, MimoSignalMetrics] = {}
        self.lastSignalProcessingResult: Optional[
            SignalProcessingResult
        ] = None
        self.lastSignalProcessingResults: Tuple[
            SignalProcessingResult, ...
        ] = tuple()
        self.lastMimoMetrics: Optional[MimoSignalMetrics] = None
        self.powerEvmCurve: Optional[PowerEvmCurve] = None

    def GetParameters(self) -> Dict[str, object]:
        """Return a flattened snapshot of all resolved analysis parameters.

        Processing details:
            Algorithm: Resolve values according to state and ChainMap precedence, keeping caller-owned configuration behavior explicit.

        Returns:
            result: Dict[str, object]. The computed value described by the summary, with documented units, shape, and normalization.
        """

        return dict(self.parameters)

    def UpdateParameters(self, **parameterOverrides: object) -> None:
        """Apply validated high-priority analysis parameter overrides.

        Processing details:
            Algorithm: Resolve values according to state and ChainMap precedence, keeping caller-owned configuration behavior explicit.

        Args:
            parameterOverrides: Highest-priority keyword values applied to the local ChainMap layer.

        Returns:
            result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
        """

        previousOverrides = dict(self.parameters.maps[0])
        self.parameters.maps[0].update(parameterOverrides)
        try:
            self.ValidateParameters()
        except (TypeError, ValueError):
            self.parameters.maps[0].clear()
            self.parameters.maps[0].update(previousOverrides)
            raise

    def ValidateParameters(self) -> None:
        """Validate the currently resolved ChainMap analysis settings.

        Processing details:
            Algorithm: Evaluate every documented constraint in deterministic order and stop at the first invalid condition without changing valid state.

        Returns:
            result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
        """

        unknownParameters = set(self.parameters).difference(
            self.defaultParameters
        )
        if unknownParameters:
            unknownNames = ", ".join(
                sorted(str(parameterName) for parameterName in unknownParameters)
            )
            raise TypeError(f"unknown Analysis parameters: {unknownNames}")
        maxSegmentLength = self.parameters["maxSegmentLength"]
        if (
            not isinstance(maxSegmentLength, int)
            or isinstance(maxSegmentLength, bool)
            or maxSegmentLength < 16
        ):
            raise ValueError(
                "maxSegmentLength must be an integer of at least 16"
            )
        minimumAclrOversampling = self.parameters["minimumAclrOversampling"]
        if not isinstance(minimumAclrOversampling, (int, float)) or isinstance(
            minimumAclrOversampling, bool
        ):
            raise TypeError("minimumAclrOversampling must be numeric")
        if minimumAclrOversampling < 3.0:
            raise ValueError("minimumAclrOversampling cannot be less than 3.0")
        powerEvmFileStem = self.parameters["powerEvmFileStem"]
        if not isinstance(powerEvmFileStem, str):
            raise TypeError("powerEvmFileStem must be a string")
        if not powerEvmFileStem or any(
            character in powerEvmFileStem for character in '<>:"/\\|?*'
        ):
            raise ValueError("powerEvmFileStem must be a valid simple file name")
        signalProcessingParameters = self.parameters[
            "signalProcessingParameters"
        ]
        if signalProcessingParameters is not None and not isinstance(
            signalProcessingParameters, Mapping
        ):
            raise TypeError(
                "signalProcessingParameters must be a mapping or None"
            )
        # Constructing one temporary processor per conducted chain validates
        # nested settings without duplicating synchronization constraints.
        referenceMatrix = (
            self.referenceSignal.reshape(-1, 1)
            if self.referenceSignal.ndim == 1
            else self.referenceSignal
        )
        for chainIndex in range(referenceMatrix.shape[1]):
            SigProcess(
                referenceMatrix[:, chainIndex],
                self.waveform.sampleRateHz,
                parameters=signalProcessingParameters,
            )

    def PrepareMeasuredSignal(self, measuredSignal: np.ndarray) -> np.ndarray:
        """Synchronize and compensate one signal before metric processing.

        Processing details:
            Algorithm: Construct ``SigProcess`` with the current nested
            settings, estimate and compensate timing, carrier frequency,
            sampling frequency, and complex gain, then retain all estimates.

        Args:
            measuredSignal: Measured or simulated complex samples. The input
                may be longer or shorter than the reference before alignment.

        Returns:
            result: Reference-length synchronized and compensated samples.
        """

        self.ValidateParameters()
        signalProcessingParameters = self.parameters[
            "signalProcessingParameters"
        ]
        measuredArray = np.asarray(measuredSignal, dtype=np.complex128)
        referenceMatrix = (
            self.referenceSignal.reshape(-1, 1)
            if self.referenceSignal.ndim == 1
            else self.referenceSignal
        )
        inputWasVector = self.referenceSignal.ndim == 1
        if inputWasVector and measuredArray.ndim == 1:
            measuredMatrix = measuredArray.reshape(-1, 1)
        elif measuredArray.ndim == 2:
            measuredMatrix = measuredArray
        else:
            raise ValueError(
                "measuredSignal must have one column per transmit chain"
            )
        if measuredMatrix.shape[1] != referenceMatrix.shape[1]:
            raise ValueError(
                "measuredSignal must have one column per transmit chain"
            )
        if measuredMatrix.shape[0] == 0 or not np.all(
            np.isfinite(measuredMatrix)
        ):
            raise ValueError("measuredSignal must contain finite samples")
        dataSlice = self.waveform.fieldSlices[self.waveform.dataFieldName]
        processingResults = []
        processedColumns = []
        for chainIndex in range(referenceMatrix.shape[1]):
            signalProcessor = SigProcess(
                referenceMatrix[:, chainIndex],
                self.waveform.sampleRateHz,
                parameters=signalProcessingParameters,
            )
            processingResult = signalProcessor.Process(
                measuredMatrix[:, chainIndex],
                estimationSlice=dataSlice,
            )
            processingResults.append(processingResult)
            processedColumns.append(processingResult.processedSignal)
        self.lastSignalProcessingResults = tuple(processingResults)
        self.lastSignalProcessingResult = processingResults[0]
        processedMatrix = np.column_stack(processedColumns)
        return processedMatrix[:, 0] if inputWasVector else processedMatrix

    def GetLastSignalProcessingResult(
        self,
    ) -> Optional[SignalProcessingResult]:
        """Return the most recent synchronization and compensation result.

        Processing details:
            Algorithm: Return the immutable result object retained by the most
            recent ``PrepareMeasuredSignal`` or ``Analyze`` call.

        Returns:
            result: Last ``SignalProcessingResult``, or ``None`` before any
                measured signal has been processed.
        """

        return self.lastSignalProcessingResult

    def GetLastSignalProcessingResults(
        self,
    ) -> Tuple[SignalProcessingResult, ...]:
        """Return the latest compensation result for every transmit chain.

        Processing details:
            Algorithm: Return the immutable chain-ordered tuple retained by
            the most recent synchronization pass.

        Returns:
            result: Empty tuple before processing, otherwise one result per PA.
        """

        return tuple(self.lastSignalProcessingResults)

    def GetLastMimoMetrics(self) -> Optional[MimoSignalMetrics]:
        """Return per-chain and per-stream details from the latest analysis.

        Processing details:
            Algorithm: Return the immutable MIMO detail record created after
            the aggregate SNR, EVM, and ACLR calculations.

        Returns:
            result: MIMO details, or None before a MIMO analysis call.
        """

        return self.lastMimoMetrics

    def GetStageSignalProcessingResults(
        self,
    ) -> Dict[str, Tuple[SignalProcessingResult, ...]]:
        """Return synchronization results retained by ``AnalyzeStages``.

        Processing details:
            Algorithm: Copy the stage-to-result mapping while reusing its
            immutable ``SignalProcessingResult`` values.

        Returns:
            result: Mapping from stage names to chain-ordered estimates.
        """

        return dict(self.stageSignalProcessingResults)

    def GetStageMimoMetrics(self) -> Dict[str, MimoSignalMetrics]:
        """Return per-chain and per-stream details retained by stage name.

        Processing details:
            Algorithm: Copy the stage mapping while reusing immutable MIMO
            metric records created during ``AnalyzeStages``.

        Returns:
            result: Mapping from stage labels to detailed MIMO metrics.
        """

        return dict(self.stageMimoMetrics)

    def ValidatePreparedSignal(self, preparedSignal: np.ndarray) -> np.ndarray:
        """Validate a signal already mapped onto the reference sample grid.

        Processing details:
            Algorithm: Delegate finite-array conversion to ``SigProcess`` and
            require the exact stored reference length.

        Args:
            preparedSignal: Synchronized and compensated complex samples.

        Returns:
            result: Valid one-dimensional complex128 array.
        """

        complexPrepared = np.asarray(preparedSignal, dtype=np.complex128)
        if complexPrepared.shape != self.referenceSignal.shape:
            raise ValueError(
                "preparedSignal and referenceSignal must have equal shape"
            )
        if complexPrepared.ndim not in (1, 2) or not np.all(
            np.isfinite(complexPrepared)
        ):
            raise ValueError(
                "preparedSignal must be a finite vector or matrix"
            )
        return complexPrepared

    def CalculateSnr(self, measuredSignal: np.ndarray) -> float:
        """Calculate data-field SNR after removing one complex gain and phase.

        Processing details:
            Algorithm: Perform the numerical calculation with explicit power, shape, and normalization handling for comparable results.

        Args:
            measuredSignal: Measured or simulated complex samples evaluated against the reference.

        Returns:
            result: float. The computed value described by the summary, with documented units, shape, and normalization.
        """

        complexMeasured = self.PrepareMeasuredSignal(measuredSignal)
        return self.CalculatePreparedSnr(complexMeasured)

    def CalculatePreparedSnr(self, preparedSignal: np.ndarray) -> float:
        """Calculate data-field SNR from a compensated signal.

        Processing details:
            Algorithm: Compare the prepared data field directly with the
            stored reference because ``SigProcess`` has already removed the
            deterministic complex gain and synchronization impairments.

        Args:
            preparedSignal: Signal returned by ``PrepareMeasuredSignal``.

        Returns:
            result: Reconstruction SNR in decibels.
        """

        complexMeasured = self.ValidatePreparedSignal(preparedSignal)
        dataSlice = self.waveform.fieldSlices[self.waveform.dataFieldName]
        referenceData = self.referenceSignal[dataSlice]
        measuredData = complexMeasured[dataSlice]
        measuredMatrix = (
            measuredData.reshape(-1, 1)
            if measuredData.ndim == 1
            else measuredData
        )
        errorSignal = measuredData - referenceData
        signalPower = np.mean(np.abs(referenceData) ** 2)
        errorPower = np.mean(np.abs(errorSignal) ** 2)
        return float(
            10.0
            * np.log10(
                max(signalPower, np.finfo(float).tiny)
                / max(errorPower, np.finfo(float).tiny)
            )
        )

    def DemodulateWifiData(self, measuredSignal: np.ndarray) -> np.ndarray:
        """Remove cyclic prefixes and FFT-demodulate Wi-Fi data subcarriers.

        Processing details:
            Algorithm: Carry out the described operation using validated inputs, explicit array-shape handling, and deterministic project conventions.

        Args:
            measuredSignal: Measured or simulated complex samples evaluated against the reference.

        Returns:
            result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
        """

        complexMeasured = self.PrepareMeasuredSignal(measuredSignal)
        return self.DemodulatePreparedWifiData(complexMeasured)

    def DemodulatePreparedWifiData(
        self, preparedSignal: np.ndarray
    ) -> np.ndarray:
        """FFT-demodulate data from an already compensated Wi-Fi signal.

        Processing details:
            Algorithm: Validate the reference-grid signal, remove each cyclic
            prefix, perform a unitary FFT, and select configured data tones.

        Args:
            preparedSignal: Signal returned by ``PrepareMeasuredSignal``.

        Returns:
            result: Matrix indexed by OFDM symbol and data subcarrier.
        """

        complexMeasured = self.ValidatePreparedSignal(preparedSignal)
        demodulatedSymbols = []
        for symbolStart in self.waveform.dataSymbolStarts:
            usefulStart = int(symbolStart) + self.waveform.cpLength
            usefulStop = usefulStart + self.waveform.fftLength
            if usefulStop > complexMeasured.size:
                raise ValueError(
                    "measuredSignal is shorter than the Wi-Fi data field"
                )
            usefulSamples = complexMeasured[usefulStart:usefulStop]
            usefulMatrix = (
                usefulSamples.reshape(-1, 1)
                if usefulSamples.ndim == 1
                else usefulSamples
            )
            frequencyGrid = np.fft.fft(usefulMatrix, axis=0) / np.sqrt(
                self.waveform.fftLength
            )
            antennaData = frequencyGrid[
                np.mod(
                    self.waveform.dataSubcarriers,
                    self.waveform.fftLength,
                )
            ]
            csdPhaseMatrix = BuildCsdPhaseMatrix(
                self.waveform.dataSubcarriers,
                self.waveform.sampleRateHz / self.waveform.fftLength,
                self.waveform.cyclicShiftsSeconds,
            )
            # The transmit mapping is y = s Q^T D_csd. Because Q has
            # orthonormal columns and D_csd is unitary, its left inverse is
            # obtained by conjugating both terms in reverse order.
            spatialStreams = (
                antennaData * np.conj(csdPhaseMatrix)
            ) @ np.conj(self.waveform.spatialMappingMatrix)
            demodulatedSymbols.append(spatialStreams)
        demodulatedArray = np.asarray(demodulatedSymbols)
        if self.waveform.numTransmitAntennas == 1:
            return demodulatedArray[:, :, 0]
        return demodulatedArray

    def CalculateEvm(self, measuredSignal: np.ndarray) -> Tuple[float, float]:
        """Calculate RMS EVM in dB and percent on Wi-Fi data subcarriers.

        Processing details:
            Algorithm: Perform the numerical calculation with explicit power, shape, and normalization handling for comparable results.

        Args:
            measuredSignal: Measured or simulated complex samples evaluated against the reference.

        Returns:
            result: Tuple[float, float]. The computed value described by the summary, with documented units, shape, and normalization.
        """

        preparedSignal = self.PrepareMeasuredSignal(measuredSignal)
        return self.CalculatePreparedEvm(preparedSignal)

    def CalculateEvmAlignedMse(self, measuredSignal: np.ndarray) -> float:
        """Calculate normalized MSE using the exact Wi-Fi EVM signal path.

        Processing details:
            Algorithm: Synchronize and compensate the measured waveform,
            remove cyclic prefixes, demodulate OFDM symbols, undo MIMO spatial
            mapping, retain data subcarriers, and normalize symbol-error power
            by reference-symbol power. The result equals squared RMS EVM.

        Args:
            measuredSignal: Measured or simulated complex samples.

        Returns:
            result: Dimensionless normalized MSE equal to ``EVM_rms**2``.
        """

        preparedSignal = self.PrepareMeasuredSignal(measuredSignal)
        return self.CalculatePreparedEvmAlignedMse(preparedSignal)

    def CalculatePreparedEvmAlignedMse(
        self, preparedSignal: np.ndarray
    ) -> float:
        """Calculate normalized data-subcarrier MSE after compensation.

        Processing details:
            Algorithm: Apply the same OFDM demodulation to reference and
            measurement, then form a symbol-energy-normalized squared error.
            This is the MSE objective whose decibel value exactly equals EVM
            in decibels rather than only approximating it.

        Args:
            preparedSignal: Signal returned by ``PrepareMeasuredSignal``.

        Returns:
            result: Dimensionless EVM-aligned normalized MSE.
        """

        measuredSymbols = self.DemodulatePreparedWifiData(preparedSignal)
        referenceSymbols = self.DemodulatePreparedWifiData(
            self.referenceSignal
        )
        symbolError = measuredSymbols.reshape(-1) - referenceSymbols.reshape(-1)
        return float(
            np.sum(np.abs(symbolError) ** 2)
            / max(
                np.sum(np.abs(referenceSymbols) ** 2),
                np.finfo(float).tiny,
            )
        )

    def CalculatePreparedEvm(
        self, preparedSignal: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate RMS EVM from a compensated reference-grid signal.

        Processing details:
            Algorithm: Demodulate both the stored time-domain reference and
            prepared measurement with identical FFT operations, then compute
            normalized RMS symbol error without performing another gain fit.

        Args:
            preparedSignal: Signal returned by ``PrepareMeasuredSignal``.

        Returns:
            result: Tuple containing EVM in decibels and percent.
        """

        evmAlignedMse = self.CalculatePreparedEvmAlignedMse(preparedSignal)
        evmRatio = np.sqrt(evmAlignedMse)
        evmPercent = 100.0 * evmRatio
        evmDb = 10.0 * np.log10(
            max(evmAlignedMse, np.finfo(float).tiny)
        )
        return float(evmDb), float(evmPercent)

    def CalculatePreparedSnrPerChain(
        self, preparedSignal: np.ndarray
    ) -> Tuple[float, ...]:
        """Calculate data-field reconstruction SNR for each RF chain.

        Processing details:
            Algorithm: Slice the data field and independently divide each
            reference-column power by its compensated error-column power.

        Args:
            preparedSignal: Synchronized samples shaped samples by chains.

        Returns:
            result: Chain-ordered SNR values in decibels.
        """

        complexMeasured = self.ValidatePreparedSignal(preparedSignal)
        referenceMatrix = (
            self.referenceSignal.reshape(-1, 1)
            if self.referenceSignal.ndim == 1
            else self.referenceSignal
        )
        measuredMatrix = (
            complexMeasured.reshape(-1, 1)
            if complexMeasured.ndim == 1
            else complexMeasured
        )
        dataSlice = self.waveform.fieldSlices[self.waveform.dataFieldName]
        snrValues = []
        for chainIndex in range(referenceMatrix.shape[1]):
            referenceData = referenceMatrix[dataSlice, chainIndex]
            errorData = (
                measuredMatrix[dataSlice, chainIndex] - referenceData
            )
            signalPower = max(
                np.mean(np.abs(referenceData) ** 2),
                np.finfo(float).tiny,
            )
            errorPower = max(
                np.mean(np.abs(errorData) ** 2),
                np.finfo(float).tiny,
            )
            snrValues.append(float(10.0 * np.log10(signalPower / errorPower)))
        return tuple(snrValues)

    def CalculatePreparedEvmPerSpatialStream(
        self, preparedSignal: np.ndarray
    ) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """Calculate post-spatial-demapping EVM for every spatial stream.

        Processing details:
            Algorithm: FFT-demodulate the reference and measurement, undo
            cyclic shifts and spatial mapping, then normalize error energy
            independently for each stream across all data tones and symbols.

        Args:
            preparedSignal: Synchronized samples shaped samples by chains.

        Returns:
            result: Tuple of stream EVM-dB tuple and EVM-percent tuple.
        """

        measuredSymbols = self.DemodulatePreparedWifiData(preparedSignal)
        referenceSymbols = self.DemodulatePreparedWifiData(
            self.referenceSignal
        )
        if referenceSymbols.ndim == 2:
            referenceSymbols = referenceSymbols[:, :, np.newaxis]
            measuredSymbols = measuredSymbols[:, :, np.newaxis]
        evmDbValues = []
        evmPercentValues = []
        for streamIndex in range(referenceSymbols.shape[2]):
            referenceStream = referenceSymbols[:, :, streamIndex].reshape(-1)
            measuredStream = measuredSymbols[:, :, streamIndex].reshape(-1)
            errorStream = measuredStream - referenceStream
            evmRatio = np.sqrt(
                np.sum(np.abs(errorStream) ** 2)
                / max(
                    np.sum(np.abs(referenceStream) ** 2),
                    np.finfo(float).tiny,
                )
            )
            evmDbValues.append(
                float(20.0 * np.log10(max(evmRatio, np.finfo(float).tiny)))
            )
            evmPercentValues.append(float(100.0 * evmRatio))
        return tuple(evmDbValues), tuple(evmPercentValues)

    def IntegrateAclr(
        self,
        frequencyBins: np.ndarray,
        powerSpectrum: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Integrate main and adjacent channel PSD regions into ACLR.

        Processing details:
            Algorithm: Use equal-width lower, main, and upper regions centered
            one channel apart, then form wanted-to-adjacent power ratios.

        Args:
            frequencyBins: Centered periodogram frequency coordinates in Hz.
            powerSpectrum: Nonnegative PSD samples corresponding to the bins.

        Returns:
            result: Lower, upper, and worst-case ACLR values in decibels.
        """

        halfBandwidth = self.waveform.bandwidthHz / 2.0
        mainMask = np.abs(frequencyBins) < halfBandwidth
        lowerMask = (frequencyBins >= -3.0 * halfBandwidth) & (
            frequencyBins < -halfBandwidth
        )
        upperMask = (frequencyBins > halfBandwidth) & (
            frequencyBins <= 3.0 * halfBandwidth
        )
        mainPower = max(
            np.sum(powerSpectrum[mainMask]), np.finfo(float).tiny
        )
        lowerPower = max(
            np.sum(powerSpectrum[lowerMask]), np.finfo(float).tiny
        )
        upperPower = max(
            np.sum(powerSpectrum[upperMask]), np.finfo(float).tiny
        )
        lowerAclrDb = 10.0 * np.log10(mainPower / lowerPower)
        upperAclrDb = 10.0 * np.log10(mainPower / upperPower)
        return (
            float(lowerAclrDb),
            float(upperAclrDb),
            float(min(lowerAclrDb, upperAclrDb)),
        )

    def CalculateAclr(
        self, measuredSignal: np.ndarray
    ) -> Tuple[float, float, float]:
        """Calculate lower, upper, and worst adjacent-channel leakage ratios.

        Processing details:
            Algorithm: Perform the numerical calculation with explicit power, shape, and normalization handling for comparable results.

        Args:
            measuredSignal: Measured or simulated complex samples evaluated against the reference.

        Returns:
            result: Tuple[float, float, float]. The computed value described by the summary, with documented units, shape, and normalization.
        """

        complexMeasured = self.PrepareMeasuredSignal(measuredSignal)
        return self.CalculatePreparedAclr(complexMeasured)

    def CalculatePreparedAclr(
        self, preparedSignal: np.ndarray
    ) -> Tuple[float, float, float]:
        """Calculate ACLR from a synchronized and compensated signal.

        Processing details:
            Algorithm: Estimate the data-field PSD and integrate equal-width
            main, lower-adjacent, and upper-adjacent channel regions.

        Args:
            preparedSignal: Signal returned by ``PrepareMeasuredSignal``.

        Returns:
            result: Lower, upper, and worst ACLR values in decibels.
        """

        self.ValidateParameters()
        complexMeasured = self.ValidatePreparedSignal(preparedSignal)
        dataSlice = self.waveform.fieldSlices[self.waveform.dataFieldName]
        measuredData = complexMeasured[dataSlice]
        measuredMatrix = (
            measuredData.reshape(-1, 1)
            if measuredData.ndim == 1
            else measuredData
        )
        sampleRateHz = self.waveform.sampleRateHz
        channelBandwidthHz = self.waveform.bandwidthHz
        minimumAclrOversampling = float(
            self.parameters["minimumAclrOversampling"]
        )
        if sampleRateHz < minimumAclrOversampling * channelBandwidthHz:
            raise ValueError(
                "ACLR analysis requires at least "
                f"{minimumAclrOversampling:g}x oversampling"
            )
        accumulatedSpectrum = None
        frequencyBins = None
        for chainIndex in range(measuredMatrix.shape[1]):
            chainBins, chainSpectrum = AveragePeriodogram(
                measuredMatrix[:, chainIndex],
                sampleRateHz,
                int(self.parameters["maxSegmentLength"]),
            )
            frequencyBins = chainBins
            accumulatedSpectrum = (
                chainSpectrum
                if accumulatedSpectrum is None
                else accumulatedSpectrum + chainSpectrum
            )
        return self.IntegrateAclr(frequencyBins, accumulatedSpectrum)

    def CalculatePreparedAclrPerChain(
        self, preparedSignal: np.ndarray
    ) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]:
        """Calculate conducted ACLR independently for every PA output.

        Processing details:
            Algorithm: Estimate one data-field periodogram per RF chain and
            integrate identical wanted and adjacent frequency regions.

        Args:
            preparedSignal: Synchronized samples shaped samples by chains.

        Returns:
            result: Lower, upper, and worst ACLR tuples ordered by PA chain.
        """

        self.ValidateParameters()
        complexMeasured = self.ValidatePreparedSignal(preparedSignal)
        measuredMatrix = (
            complexMeasured.reshape(-1, 1)
            if complexMeasured.ndim == 1
            else complexMeasured
        )
        minimumAclrOversampling = float(
            self.parameters["minimumAclrOversampling"]
        )
        if (
            self.waveform.sampleRateHz
            < minimumAclrOversampling * self.waveform.bandwidthHz
        ):
            raise ValueError(
                "ACLR analysis requires at least "
                f"{minimumAclrOversampling:g}x oversampling"
            )
        dataSlice = self.waveform.fieldSlices[self.waveform.dataFieldName]
        lowerValues = []
        upperValues = []
        worstValues = []
        for chainIndex in range(measuredMatrix.shape[1]):
            frequencyBins, powerSpectrum = AveragePeriodogram(
                measuredMatrix[dataSlice, chainIndex],
                self.waveform.sampleRateHz,
                int(self.parameters["maxSegmentLength"]),
            )
            lowerAclrDb, upperAclrDb, worstAclrDb = self.IntegrateAclr(
                frequencyBins, powerSpectrum
            )
            lowerValues.append(lowerAclrDb)
            upperValues.append(upperAclrDb)
            worstValues.append(worstAclrDb)
        return tuple(lowerValues), tuple(upperValues), tuple(worstValues)

    def Analyze(self, measuredSignal: np.ndarray) -> SignalMetrics:
        """Calculate SNR, EVM, and ACLR for one PA or DPD output waveform.

        Processing details:
            Algorithm: Perform the numerical calculation with explicit power, shape, and normalization handling for comparable results.

        Args:
            measuredSignal: Measured or simulated complex samples evaluated against the reference.

        Returns:
            result: SignalMetrics. The computed value described by the summary, with documented units, shape, and normalization.
        """

        # Synchronization is intentionally executed once. The same corrected
        # samples feed all metrics so SNR, EVM, and ACLR remain comparable.
        complexMeasured = self.PrepareMeasuredSignal(measuredSignal)
        snrDb = self.CalculatePreparedSnr(complexMeasured)
        evmDb, evmPercent = self.CalculatePreparedEvm(complexMeasured)
        (
            aclrLowerDb,
            aclrUpperDb,
            aclrWorstDb,
        ) = self.CalculatePreparedAclr(complexMeasured)
        if self.waveform.numTransmitAntennas > 1:
            perChainSnrDb = self.CalculatePreparedSnrPerChain(complexMeasured)
            (
                perStreamEvmDb,
                perStreamEvmPercent,
            ) = self.CalculatePreparedEvmPerSpatialStream(complexMeasured)
            (
                perChainAclrLowerDb,
                perChainAclrUpperDb,
                perChainAclrWorstDb,
            ) = self.CalculatePreparedAclrPerChain(complexMeasured)
            self.lastMimoMetrics = MimoSignalMetrics(
                snrDbPerChain=perChainSnrDb,
                evmDbPerSpatialStream=perStreamEvmDb,
                evmPercentPerSpatialStream=perStreamEvmPercent,
                aclrLowerDbPerChain=perChainAclrLowerDb,
                aclrUpperDbPerChain=perChainAclrUpperDb,
                aclrWorstDbPerChain=perChainAclrWorstDb,
            )
        else:
            self.lastMimoMetrics = None
        return SignalMetrics(
            snrDb=snrDb,
            evmDb=evmDb,
            evmPercent=evmPercent,
            aclrLowerDb=aclrLowerDb,
            aclrUpperDb=aclrUpperDb,
            aclrWorstDb=aclrWorstDb,
        )

    def AnalyzeStages(
        self, stageSignals: Mapping[str, np.ndarray]
    ) -> Dict[str, SignalMetrics]:
        """Analyze multiple named stages and retain their result table.

        Processing details:
            Algorithm: Perform the numerical calculation with explicit power, shape, and normalization handling for comparable results.

        Args:
            stageSignals: Mapping from result-stage labels to complex output waveforms.

        Returns:
            result: Dict[str, SignalMetrics]. The computed value described by the summary, with documented units, shape, and normalization.
        """

        self.stageMetrics = {}
        self.stageSignalProcessingResults = {}
        self.stageMimoMetrics = {}
        for stageName, stageSignal in stageSignals.items():
            self.stageMetrics[stageName] = self.Analyze(stageSignal)
            if self.lastSignalProcessingResults:
                self.stageSignalProcessingResults[stageName] = tuple(
                    self.lastSignalProcessingResults
                )
            if self.lastMimoMetrics is not None:
                self.stageMimoMetrics[stageName] = self.lastMimoMetrics
        return dict(self.stageMetrics)

    def AnalyzePowerEvmCurve(
        self,
        driveRmsValues: Sequence[float],
        methodEvaluators: Mapping[
            str, Callable[[np.ndarray, float], np.ndarray]
        ],
    ) -> PowerEvmCurve:
        """Evaluate multiple methods over one common RMS input-power sweep.

        Every evaluator receives the reference waveform scaled to the current
        RMS drive and the numeric drive value. An ILC evaluator may relearn at
        every point, while a deployed DPD evaluator may reuse fixed nominal-
        power coefficients. All methods are scored with identical references.
        """

        driveArray = np.asarray(driveRmsValues, dtype=float).reshape(-1)
        if driveArray.size < 2:
            raise ValueError("driveRmsValues must contain at least two points")
        if not np.all(np.isfinite(driveArray)) or np.any(driveArray <= 0.0):
            raise ValueError("driveRmsValues must contain finite positive values")
        if np.any(np.diff(driveArray) <= 0.0):
            raise ValueError("driveRmsValues must be strictly increasing")
        if not methodEvaluators:
            raise ValueError("methodEvaluators cannot be empty")

        # Forward only values that differ from this instance's internal
        # defaults. Each point Analysis reconstructs its own default layer.
        pointParameterOverrides = {
            parameterName: self.parameters[parameterName]
            for parameterName, defaultValue in self.defaultParameters.items()
            if self.parameters[parameterName] != defaultValue
        }
        evmDbByMethod: Dict[str, np.ndarray] = {}
        evmPercentByMethod: Dict[str, np.ndarray] = {}
        for methodName, methodEvaluator in methodEvaluators.items():
            methodEvmDb = []
            methodEvmPercent = []
            for driveRms in driveArray:
                pointReference = float(driveRms) * self.waveform.samples
                measuredSignal = methodEvaluator(
                    pointReference, float(driveRms)
                )
                pointAnalysis = Analysis(
                    pointReference,
                    self.waveform,
                    parameters=pointParameterOverrides,
                )
                pointMetrics = pointAnalysis.Analyze(measuredSignal)
                methodEvmDb.append(pointMetrics.evmDb)
                methodEvmPercent.append(pointMetrics.evmPercent)
            evmDbByMethod[methodName] = np.asarray(methodEvmDb, dtype=float)
            evmPercentByMethod[methodName] = np.asarray(
                methodEvmPercent, dtype=float
            )

        self.powerEvmCurve = PowerEvmCurve(
            driveRmsValues=driveArray,
            inputPowerDb=20.0 * np.log10(driveArray),
            evmDbByMethod=evmDbByMethod,
            evmPercentByMethod=evmPercentByMethod,
        )
        return self.powerEvmCurve

    def SavePowerEvmCurveData(
        self,
        outputDirectory: Path,
        powerEvmCurve: Optional[PowerEvmCurve] = None,
        fileStem: Optional[str] = None,
    ) -> Tuple[Path, Path]:
        """Save calculated power-EVM samples as CSV and JSON data.

        Processing details:
            Algorithm: Convert validated in-memory results into a stable reporting format without altering later numerical calculations.

        Args:
            outputDirectory: Directory in which result artifacts are written.
            powerEvmCurve: Optional curve; the most recent stored curve is used when omitted.
            fileStem: Optional filename stem overriding the configured default.

        Returns:
            result: Tuple[Path, Path]. Paths to the CSV and JSON data files.
        """

        self.ValidateParameters()
        selectedCurve = (
            self.powerEvmCurve if powerEvmCurve is None else powerEvmCurve
        )
        if selectedCurve is None:
            raise ValueError("no power-EVM curve is available to save")
        selectedFileStem = (
            str(self.parameters["powerEvmFileStem"])
            if fileStem is None
            else fileStem
        )
        if not selectedFileStem or any(
            character in selectedFileStem for character in '<>:"/\\|?*'
        ):
            raise ValueError("fileStem must be a valid simple file name")

        outputPath = Path(outputDirectory)
        outputPath.mkdir(parents=True, exist_ok=True)
        csvPath = outputPath / f"{selectedFileStem}.csv"
        jsonPath = outputPath / f"{selectedFileStem}.json"
        methodNames = list(selectedCurve.evmDbByMethod)

        fieldNames = ["driveRms", "inputPowerDb"]
        for methodName in methodNames:
            fieldNames.extend(
                [f"{methodName} evmDb", f"{methodName} evmPercent"]
            )
        with csvPath.open("w", newline="", encoding="utf-8-sig") as csvFile:
            csvWriter = csv.DictWriter(csvFile, fieldnames=fieldNames)
            csvWriter.writeheader()
            for pointIndex, driveRms in enumerate(
                selectedCurve.driveRmsValues
            ):
                rowData = {
                    "driveRms": float(driveRms),
                    "inputPowerDb": float(
                        selectedCurve.inputPowerDb[pointIndex]
                    ),
                }
                for methodName in methodNames:
                    rowData[f"{methodName} evmDb"] = float(
                        selectedCurve.evmDbByMethod[methodName][pointIndex]
                    )
                    rowData[f"{methodName} evmPercent"] = float(
                        selectedCurve.evmPercentByMethod[methodName][pointIndex]
                    )
                csvWriter.writerow(rowData)

        with jsonPath.open("w", encoding="utf-8") as jsonFile:
            json.dump(
                selectedCurve.ToDict(),
                jsonFile,
                indent=2,
                ensure_ascii=False,
            )

        return csvPath, jsonPath

    def Print(
        self,
        stageMetrics: Optional[Mapping[str, SignalMetrics]] = None,
    ) -> None:
        """Print an aligned table for all selected result stages.

        Processing details:
            Algorithm: Convert validated in-memory results into a stable reporting format without altering later numerical calculations.

        Args:
            stageMetrics: Optional named metrics; stored metrics are used when omitted.

        Returns:
            result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
        """

        selectedMetrics = (
            self.stageMetrics if stageMetrics is None else stageMetrics
        )
        if not selectedMetrics:
            raise ValueError("no stage metrics are available to print")
        header = (
            f"{'Stage':<16} {'SNR(dB)':>10} {'EVM(dB)':>10} "
            f"{'EVM(%)':>10} {'ACLR-L':>10} {'ACLR-U':>10} {'ACLR-W':>10}"
        )
        print(header)
        print("-" * len(header))
        for stageName, metrics in selectedMetrics.items():
            print(
                f"{stageName:<16} {metrics.snrDb:>10.2f} "
                f"{metrics.evmDb:>10.2f} {metrics.evmPercent:>10.3f} "
                f"{metrics.aclrLowerDb:>10.2f} "
                f"{metrics.aclrUpperDb:>10.2f} "
                f"{metrics.aclrWorstDb:>10.2f}"
            )

    def PrintMimo(
        self,
        stageMimoMetrics: Optional[
            Mapping[str, MimoSignalMetrics]
        ] = None,
    ) -> None:
        """Print per-chain SNR/ACLR and per-stream EVM result tables.

        Processing details:
            Algorithm: Expand immutable metric tuples into readable rows,
            preserving one-based PA-chain and spatial-stream labels.

        Args:
            stageMimoMetrics: Optional stage details; stored values are used
                when omitted.

        Returns:
            result: None. Human-readable MIMO detail tables are printed.
        """

        selectedMetrics = (
            self.stageMimoMetrics
            if stageMimoMetrics is None
            else stageMimoMetrics
        )
        if not selectedMetrics:
            raise ValueError("no MIMO stage metrics are available to print")
        for stageName, metrics in selectedMetrics.items():
            print(f"\n{stageName} - conducted PA-chain metrics")
            print(
                f"{'PA':<8} {'SNR(dB)':>10} {'ACLR-L':>10} "
                f"{'ACLR-U':>10} {'ACLR-W':>10}"
            )
            for chainIndex, snrDb in enumerate(metrics.snrDbPerChain):
                print(
                    f"PA {chainIndex + 1:<5} {snrDb:>10.2f} "
                    f"{metrics.aclrLowerDbPerChain[chainIndex]:>10.2f} "
                    f"{metrics.aclrUpperDbPerChain[chainIndex]:>10.2f} "
                    f"{metrics.aclrWorstDbPerChain[chainIndex]:>10.2f}"
                )
            print(f"{stageName} - post-demapping spatial-stream EVM")
            print(f"{'Stream':<8} {'EVM(dB)':>10} {'EVM(%)':>10}")
            for streamIndex, evmDb in enumerate(
                metrics.evmDbPerSpatialStream
            ):
                print(
                    f"SS {streamIndex + 1:<5} {evmDb:>10.2f} "
                    f"{metrics.evmPercentPerSpatialStream[streamIndex]:>10.3f}"
                )

    def Save(
        self,
        outputDirectory: Path,
        runMetadata: Mapping[str, object],
        stageMetrics: Optional[Mapping[str, SignalMetrics]] = None,
    ) -> Tuple[Path, Path]:
        """Save selected metrics as JSON and CSV result files.

        Processing details:
            Algorithm: Convert validated in-memory results into a stable reporting format without altering later numerical calculations.

        Args:
            outputDirectory: Directory in which result artifacts are written.
            runMetadata: Experiment metadata serialized with numerical results.
            stageMetrics: Optional named metrics; stored metrics are used when omitted.

        Returns:
            result: Tuple[Path, Path]. The computed value described by the summary, with documented units, shape, and normalization.
        """

        selectedMetrics = (
            self.stageMetrics if stageMetrics is None else stageMetrics
        )
        if not selectedMetrics:
            raise ValueError("no stage metrics are available to save")
        outputPath = Path(outputDirectory)
        outputPath.mkdir(parents=True, exist_ok=True)
        jsonPath = outputPath / "metrics.json"
        csvPath = outputPath / "metrics.csv"
        serializableMetrics = {
            stageName: metrics.ToDict()
            for stageName, metrics in selectedMetrics.items()
        }
        serializableProcessingResults = {
            stageName: [
                processingResult.ToDict()
                for processingResult in self.stageSignalProcessingResults[
                    stageName
                ]
            ]
            for stageName in selectedMetrics
            if stageName in self.stageSignalProcessingResults
        }
        serializableMimoMetrics = {
            stageName: self.stageMimoMetrics[stageName].ToDict()
            for stageName in selectedMetrics
            if stageName in self.stageMimoMetrics
        }
        jsonPayload = {
            "metadata": dict(runMetadata),
            "metrics": serializableMetrics,
            "signalProcessing": serializableProcessingResults,
            "mimoMetrics": serializableMimoMetrics,
        }
        with jsonPath.open("w", encoding="utf-8") as jsonFile:
            json.dump(jsonPayload, jsonFile, indent=2, ensure_ascii=False)

        fieldNames = ["stage"] + list(
            SignalMetrics.__dataclass_fields__.keys()
        )
        processingFieldNames = []
        for processingChains in serializableProcessingResults.values():
            for chainIndex, processingValues in enumerate(processingChains):
                for fieldName in processingValues:
                    qualifiedName = (
                        f"chain{chainIndex + 1}.{fieldName}"
                    )
                    if qualifiedName not in processingFieldNames:
                        processingFieldNames.append(qualifiedName)
        fieldNames.extend(processingFieldNames)
        mimoFieldNames = []
        for mimoValues in serializableMimoMetrics.values():
            for fieldName in mimoValues:
                qualifiedName = f"mimo.{fieldName}"
                if qualifiedName not in mimoFieldNames:
                    mimoFieldNames.append(qualifiedName)
        fieldNames.extend(mimoFieldNames)
        with csvPath.open("w", newline="", encoding="utf-8-sig") as csvFile:
            csvWriter = csv.DictWriter(csvFile, fieldnames=fieldNames)
            csvWriter.writeheader()
            for stageName, metrics in selectedMetrics.items():
                rowData = {"stage": stageName}
                rowData.update(metrics.ToDict())
                if stageName in serializableProcessingResults:
                    for chainIndex, processingValues in enumerate(
                        serializableProcessingResults[stageName]
                    ):
                        for fieldName, fieldValue in processingValues.items():
                            rowData[
                                f"chain{chainIndex + 1}.{fieldName}"
                            ] = fieldValue
                if stageName in serializableMimoMetrics:
                    for fieldName, fieldValues in serializableMimoMetrics[
                        stageName
                    ].items():
                        rowData[f"mimo.{fieldName}"] = json.dumps(
                            fieldValues
                        )
                csvWriter.writerow(rowData)
        return jsonPath, csvPath

    def SaveConvergence(self, ilcHistory, outputDirectory: Path) -> Path:
        """Save the per-iteration ILC convergence history as a CSV file.

        Processing details:
            Algorithm: Convert validated in-memory results into a stable reporting format without altering later numerical calculations.

        Args:
            ilcHistory: Ordered per-iteration convergence records to serialize.
            outputDirectory: Directory in which result artifacts are written.

        Returns:
            result: Path. The computed value described by the summary, with documented units, shape, and normalization.
        """

        outputPath = Path(outputDirectory)
        outputPath.mkdir(parents=True, exist_ok=True)
        convergencePath = outputPath / "ilc_convergence.csv"
        fieldNames = [
            "iteration",
            "mse",
            "errorRms",
            "nmseDb",
            "linearCompensatedMse",
            "linearCompensatedNmseDb",
            "evmAlignedMse",
            "evmDb",
            "complexGainMagnitudeDb",
            "complexGainPhaseDegrees",
            "inputPeak",
        ]
        with convergencePath.open(
            "w", newline="", encoding="utf-8-sig"
        ) as csvFile:
            csvWriter = csv.DictWriter(csvFile, fieldnames=fieldNames)
            csvWriter.writeheader()
            for iterationRecord in ilcHistory:
                csvWriter.writerow(
                    {
                        "iteration": iterationRecord.iteration,
                        "mse": iterationRecord.mse,
                        "errorRms": iterationRecord.errorRms,
                        "nmseDb": iterationRecord.nmseDb,
                        "linearCompensatedMse": (
                            iterationRecord.linearCompensatedMse
                        ),
                        "linearCompensatedNmseDb": (
                            iterationRecord.linearCompensatedNmseDb
                        ),
                        "evmAlignedMse": iterationRecord.evmAlignedMse,
                        "evmDb": iterationRecord.evmDb,
                        "complexGainMagnitudeDb": (
                            iterationRecord.complexGainMagnitudeDb
                        ),
                        "complexGainPhaseDegrees": (
                            iterationRecord.complexGainPhaseDegrees
                        ),
                        "inputPeak": iterationRecord.inputPeak,
                    }
                )
        return convergencePath

    def PrintConvergence(
        self, ilcHistory, historyName: str = "ILC convergence"
    ) -> None:
        """Print every ILC iteration with raw and EVM-oriented MSE values.

        Processing details:
            Algorithm: Format each immutable history record into aligned
            columns while preserving linear-domain MSE values and their
            normalized decibel forms for direct engineering diagnosis.

        Args:
            ilcHistory: Ordered per-iteration convergence records.
            historyName: Human-readable heading for the selected PA or method.

        Returns:
            result: None. The complete iteration table is written to stdout.
        """

        historyRecords = list(ilcHistory)
        if not historyRecords:
            raise ValueError("ilcHistory cannot be empty")
        print(f"\n{historyName}")
        header = (
            f"{'Iter':>4} {'Raw MSE':>12} {'Raw NMSE':>10} "
            f"{'LC-MSE':>12} {'LC-NMSE':>10} {'EVM-MSE':>12} "
            f"{'EVM(dB)':>9} {'Gain(dB)':>9} {'Phase(deg)':>11} "
            f"{'Peak':>9}"
        )
        print(header)
        print("-" * len(header))
        for iterationRecord in historyRecords:
            evmMseText = (
                "n/a"
                if iterationRecord.evmAlignedMse is None
                else f"{iterationRecord.evmAlignedMse:.5e}"
            )
            evmDbText = (
                "n/a"
                if iterationRecord.evmDb is None
                else f"{iterationRecord.evmDb:.2f}"
            )
            print(
                f"{iterationRecord.iteration:>4d} "
                f"{iterationRecord.mse:>12.5e} "
                f"{iterationRecord.nmseDb:>10.2f} "
                f"{iterationRecord.linearCompensatedMse:>12.5e} "
                f"{iterationRecord.linearCompensatedNmseDb:>10.2f} "
                f"{evmMseText:>12} {evmDbText:>9} "
                f"{iterationRecord.complexGainMagnitudeDb:>9.2f} "
                f"{iterationRecord.complexGainPhaseDegrees:>11.2f} "
                f"{iterationRecord.inputPeak:>9.4f}"
            )
