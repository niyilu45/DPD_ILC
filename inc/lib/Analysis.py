"""Object-oriented SNR, EVM, and ACLR analysis for VHT/HE/EHT simulations."""

import csv
import json
from collections import ChainMap
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np

from ..utils.ConfigUtils import (
    FilterRecognizedParameters,
    RecognizedParameterView,
)
from ..utils.FrameProcess import FrameProcess
from ..utils.ParseWifi import ParsedWifiFrame, ParseWifi
from ..utils.SigProc import PowerCalibration, SigProc, SignalProcessingResult
from ..utils.WifiMetadata import WifiWaveform


class SignalMetrics(TypedDict):
    """Define the keys returned by ``Analysis.Analyze``.

    ``TypedDict`` provides static key and value information only. Every
    runtime result is an ordinary Python ``dict`` and is accessed with
    ``metrics["evmDb"]`` rather than custom-object attributes.
    """

    snrDb: float
    evmDb: float
    evmPercent: float
    aclrLowerDb: float
    aclrUpperDb: float
    aclrWorstDb: float


class MimoSignalMetrics(TypedDict):
    """Define ordinary-dictionary MIMO detail keys."""

    snrDbPerChain: Tuple[float, ...]
    evmDbPerSpatialStream: Tuple[float, ...]
    evmPercentPerSpatialStream: Tuple[float, ...]
    aclrLowerDbPerChain: Tuple[float, ...]
    aclrUpperDbPerChain: Tuple[float, ...]
    aclrWorstDbPerChain: Tuple[float, ...]


@dataclass
class PowerEvmCurve:
    """Store a multi-method EVM sweep over absolute PA output powers."""

    outputPowerDbmValues: np.ndarray
    driveScaleValues: np.ndarray
    targetOutputRmsValues: np.ndarray
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
            "outputPowerDbmValues": self.outputPowerDbmValues
            .astype(float)
            .tolist(),
            "driveScaleValues": self.driveScaleValues
            .astype(float)
            .tolist(),
            "targetOutputRmsValues": self.targetOutputRmsValues
            .astype(float)
            .tolist(),
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


@dataclass(frozen=True)
class ILCPerformanceIteration:
    """Combine native ILC diagnostics with independently analyzed RF metrics."""

    iteration: int
    mse: float
    errorRms: float
    nmseDb: float
    linearCompensatedMse: float
    linearCompensatedNmseDb: float
    complexGainMagnitudeDb: float
    complexGainPhaseDegrees: float
    inputPeak: float
    snrDb: float
    evmAlignedMse: float
    evmDb: float
    evmPercent: float
    aclrLowerDb: float
    aclrUpperDb: float
    aclrWorstDb: float

    def ToDict(self) -> Dict[str, float]:
        """Convert one analyzed iteration to serialization-ready scalars.

        Processing details:
            Algorithm: Flatten the immutable dataclass without recalculating
            either the ILC-native diagnostics or RF performance metrics.

        Returns:
            result: Dictionary preserving every per-iteration scalar.
        """

        return {
            fieldName: float(fieldValue)
            for fieldName, fieldValue in asdict(self).items()
        }


@dataclass(frozen=True)
class ILCAnalysisResult:
    """Store post-ILC performance history and the EVM-best measured candidate."""

    history: Tuple[ILCPerformanceIteration, ...]
    bestIteration: int
    bestInputSignal: np.ndarray
    bestOutputSignal: np.ndarray
    bestMetrics: SignalMetrics


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
    """Analyze PA/DPD results with a supplied or frame-parsed reference.

    The original data-aided path constructs an instance from a transmitted
    reference and ``WifiWaveform`` metadata. The receive-only path omits
    ``waveform``; ``ParseWifi`` then restores the reference and metadata from
    the received frame's protected project signaling descriptor.

    Example:
        ``resultAnalysis = Analysis(referenceSignal, waveform)``
        ``metrics = resultAnalysis.Analyze(paOutput)``
        ``receiveAnalysis = Analysis(receivedWifiFrame)``
        ``metrics = receiveAnalysis.Analyze()``
    """

    def __init__(
        self,
        referenceSignal: Union[np.ndarray, WifiWaveform],
        waveform: Optional[WifiWaveform] = None,
        parameters: Optional[Mapping[str, object]] = None,
        parseParameters: Optional[Mapping[str, object]] = None,
        transmittedSignal: Optional[
            Union[np.ndarray, WifiWaveform]
        ] = None,
        **parameterOverrides: object,
    ) -> None:
        """Initialize a reference-aided or receive-only analysis context.

        Processing details:
            Algorithm: When metadata is supplied, preserve the original
            reference-aided construction. Otherwise parse ``referenceSignal``
            as a received Wi-Fi frame, regenerate its ideal deterministic
            reference, and retain the aligned received packet as the default
            input for a later zero-argument ``Analyze`` call. Analysis defaults
            remain inside this constructor and are resolved through ChainMap.

        Args:
            referenceSignal: Ideal reference samples when ``waveform`` is
                supplied; otherwise a received NumPy array or ``WifiWaveform``
                parsed internally.
            waveform: Optional Wi-Fi metadata for the original data-aided path.
            parameters: Optional external mapping layered ahead of the built-in defaults.
            parseParameters: Optional ``ParseWifi`` parameter mapping used only
                when ``waveform`` is omitted.
            transmittedSignal: Optional known transmit input. ``ParseWifi``
                automatically accepts either a metadata-rich ``WifiWaveform``
                or a NumPy waveform containing samples only.
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
                "loadResistanceOhm": 50.0,
                "maximumOutputPowerDbm": 25.0,
            }
        )
        self.parsedWifiFrame: Optional[ParsedWifiFrame] = None
        self.defaultMeasuredSignal: Optional[np.ndarray] = None
        selectedWaveform = waveform
        selectedReference: Union[np.ndarray, WifiWaveform] = referenceSignal
        if selectedWaveform is None:
            self.parsedWifiFrame = ParseWifi(
                parameters=parseParameters
            ).Parse(
                referenceSignal,
                transmittedSignal=transmittedSignal,
            )
            selectedReference = self.parsedWifiFrame.referenceSignal
            selectedWaveform = self.parsedWifiFrame.waveform
            self.defaultMeasuredSignal = (
                self.parsedWifiFrame.receivedSignal.copy()
            )
        elif parseParameters is not None or transmittedSignal is not None:
            raise ValueError(
                "parseParameters and transmittedSignal are only valid when "
                "waveform is omitted"
            )
        if not isinstance(selectedWaveform, WifiWaveform):
            raise TypeError("waveform must be a WifiWaveform or None")
        if isinstance(selectedReference, WifiWaveform):
            selectedReference = selectedReference.samples
        complexReference = np.asarray(
            selectedReference, dtype=np.complex128
        )
        if complexReference.size == 0:
            raise ValueError("referenceSignal cannot be empty")
        expectedShape = np.asarray(selectedWaveform.samples).shape
        if complexReference.shape != expectedShape:
            raise ValueError(
                "referenceSignal shape must match the Wi-Fi waveform"
            )
        if complexReference.ndim not in (1, 2):
            raise ValueError("referenceSignal must be a vector or matrix")
        if (
            complexReference.ndim == 2
            and complexReference.shape[1]
            != selectedWaveform.numTransmitAntennas
        ):
            raise ValueError(
                "referenceSignal must contain one column per transmit chain"
            )
        if not np.all(np.isfinite(complexReference)):
            raise ValueError("referenceSignal contains NaN or infinite values")
        self.referenceSignal = complexReference
        self.waveform = selectedWaveform
        self.frameProcessor = FrameProcess(selectedWaveform)
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping or None")
        externalParameters: Mapping[str, object] = (
            {}
            if parameters is None
            else RecognizedParameterView(
                parameters,
                self.defaultParameters,
                "Analysis",
            )
        )
        recognizedOverrides = FilterRecognizedParameters(
            parameterOverrides,
            self.defaultParameters,
            "Analysis",
        )
        self.parameters: ChainMap[str, object] = ChainMap(
            recognizedOverrides,
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

    def GetParsedWifiFrame(self) -> Optional[ParsedWifiFrame]:
        """Return receive-only parser output retained by the constructor.

        Processing details:
            Algorithm: Preserve ``None`` for the original reference-aided path;
            otherwise return the immutable parser result containing aligned
            receive samples, recovered parameters, and parse diagnostics.

        Returns:
            result: Parsed frame or ``None`` when explicit metadata was used.
        """

        return self.parsedWifiFrame

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

        recognizedOverrides = FilterRecognizedParameters(
            parameterOverrides,
            self.defaultParameters,
            "Analysis.UpdateParameters",
        )
        previousOverrides = dict(self.parameters.maps[0])
        self.parameters.maps[0].update(recognizedOverrides)
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
        PowerCalibration(
            loadResistanceOhm=self.parameters["loadResistanceOhm"],
            maximumOutputPowerDbm=self.parameters[
                "maximumOutputPowerDbm"
            ],
        )
        # Constructing one temporary processor per conducted chain validates
        # nested settings without duplicating synchronization constraints.
        referenceMatrix = (
            self.referenceSignal.reshape(-1, 1)
            if self.referenceSignal.ndim == 1
            else self.referenceSignal
        )
        for chainIndex in range(referenceMatrix.shape[1]):
            SigProc(
                referenceMatrix[:, chainIndex],
                self.waveform.sampleRateHz,
                parameters=signalProcessingParameters,
            )

    def PrepareMeasuredSignal(self, measuredSignal: np.ndarray) -> np.ndarray:
        """Synchronize and compensate one signal before metric processing.

        Processing details:
            Algorithm: Construct ``SigProc`` with the current nested
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
            signalProcessor = SigProc(
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
            Algorithm: Delegate finite-array conversion, exact reference-grid
            shape checks, and finite-value checks to ``FrameProcess``.

        Args:
            preparedSignal: Synchronized and compensated complex samples.

        Returns:
            result: Valid complex128 vector or samples-by-chains matrix.
        """

        return self.frameProcessor.ValidatePreparedSignal(preparedSignal)

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
            stored reference because ``SigProc`` has already removed the
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
            Algorithm: Delegate cyclic-prefix removal, unitary FFT, data-tone
            selection, CSD removal, and spatial demapping to ``FrameProcess``.

        Args:
            preparedSignal: Signal returned by ``PrepareMeasuredSignal``.

        Returns:
            result: Matrix indexed by OFDM symbol and data subcarrier.
        """

        complexMeasured = self.ValidatePreparedSignal(preparedSignal)
        return self.frameProcessor.DemodulatePreparedWifiData(
            complexMeasured
        )

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
                "sampleRateHz must be at least "
                f"{minimumAclrOversampling:g} times bandwidthHz "
                "for ACLR analysis"
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
                "sampleRateHz must be at least "
                f"{minimumAclrOversampling:g} times bandwidthHz "
                "for ACLR analysis"
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

    def Analyze(
        self, measuredSignal: Optional[np.ndarray] = None
    ) -> SignalMetrics:
        """Calculate SNR, EVM, and ACLR for one received Wi-Fi waveform.

        Processing details:
            Algorithm: Use the explicit measured waveform in the original
            reference-aided path, or the parsed and packet-aligned receive frame
            when this instance was constructed without ``WifiWaveform``.
            Synchronize once and feed the same corrected samples to every
            metric so all results remain comparable.

        Args:
            measuredSignal: Optional measured samples evaluated against the
                stored reference. Omit only for a receive-only parsed instance.

        Returns:
            result: Ordinary dictionary containing SNR, EVM, and ACLR values.
        """

        selectedSignal = measuredSignal
        if selectedSignal is None:
            if self.defaultMeasuredSignal is None:
                raise ValueError(
                    "measuredSignal is required when Analysis was constructed "
                    "with an explicit reference and waveform"
                )
            selectedSignal = self.defaultMeasuredSignal

        # Synchronization is intentionally executed once. The same corrected
        # samples feed all metrics so SNR, EVM, and ACLR remain comparable.
        complexMeasured = self.PrepareMeasuredSignal(selectedSignal)
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
            self.lastMimoMetrics = {
                "snrDbPerChain": perChainSnrDb,
                "evmDbPerSpatialStream": perStreamEvmDb,
                "evmPercentPerSpatialStream": perStreamEvmPercent,
                "aclrLowerDbPerChain": perChainAclrLowerDb,
                "aclrUpperDbPerChain": perChainAclrUpperDb,
                "aclrWorstDbPerChain": perChainAclrWorstDb,
            }
        else:
            self.lastMimoMetrics = None
        return {
            "snrDb": float(snrDb),
            "evmDb": float(evmDb),
            "evmPercent": float(evmPercent),
            "aclrLowerDb": float(aclrLowerDb),
            "aclrUpperDb": float(aclrUpperDb),
            "aclrWorstDb": float(aclrWorstDb),
        }

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
        outputPowerDbmValues: Sequence[float],
        methodEvaluators: Mapping[
            str, Callable[[np.ndarray, float], np.ndarray]
        ],
    ) -> PowerEvmCurve:
        """Evaluate multiple methods over one common absolute-power sweep.

        Every evaluator receives a reference waveform driven according to the
        requested output backoff and the numeric output dBm value. The rated
        maximum output is zero backoff. After nonlinear processing, one
        constant per-chain gain calibrates every method to the requested
        absolute output power. This gain does not alter EVM or ACLR ratios.

        Args:
            outputPowerDbmValues: Strictly increasing absolute per-PA output
                powers in dBm, no greater than ``maximumOutputPowerDbm``.
            methodEvaluators: Mapping from display name to a callable that
                accepts the output-backoff-scaled reference and target output
                dBm, and returns the corresponding PA output waveform.

        Returns:
            result: Power-EVM curve containing output powers, normalized drive
                scales, target RMS voltages, and per-method EVM arrays.
        """

        powerDbmArray = np.asarray(
            outputPowerDbmValues, dtype=float
        ).reshape(-1)
        if powerDbmArray.size < 2:
            raise ValueError(
                "outputPowerDbmValues must contain at least two points"
            )
        if not np.all(np.isfinite(powerDbmArray)):
            raise ValueError(
                "outputPowerDbmValues must contain finite values"
            )
        if np.any(np.diff(powerDbmArray) <= 0.0):
            raise ValueError(
                "outputPowerDbmValues must be strictly increasing"
            )
        if not methodEvaluators:
            raise ValueError("methodEvaluators cannot be empty")
        powerCalibration = PowerCalibration(
            loadResistanceOhm=self.parameters["loadResistanceOhm"],
            maximumOutputPowerDbm=self.parameters[
                "maximumOutputPowerDbm"
            ],
        )
        driveScaleArray = np.asarray(
            [
                powerCalibration.OutputPowerToDriveScale(
                    outputPowerDbm
                )
                for outputPowerDbm in powerDbmArray
            ],
            dtype=float,
        )
        targetOutputRmsArray = np.asarray(
            [
                powerCalibration.DbmToRms(outputPowerDbm)
                for outputPowerDbm in powerDbmArray
            ],
            dtype=float,
        )

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
            for outputPowerDbm, driveScale in zip(
                powerDbmArray, driveScaleArray
            ):
                pointReference = (
                    float(driveScale) * self.waveform.samples
                )
                rawMeasuredSignal = methodEvaluator(
                    pointReference, float(outputPowerDbm)
                )
                measuredSignal = (
                    powerCalibration.ScaleSignalToOutputPower(
                        rawMeasuredSignal,
                        float(outputPowerDbm),
                    )
                )
                pointAnalysis = Analysis(
                    pointReference,
                    self.waveform,
                    parameters=pointParameterOverrides,
                )
                pointMetrics = pointAnalysis.Analyze(measuredSignal)
                methodEvmDb.append(pointMetrics["evmDb"])
                methodEvmPercent.append(pointMetrics["evmPercent"])
            evmDbByMethod[methodName] = np.asarray(methodEvmDb, dtype=float)
            evmPercentByMethod[methodName] = np.asarray(
                methodEvmPercent, dtype=float
            )

        self.powerEvmCurve = PowerEvmCurve(
            outputPowerDbmValues=powerDbmArray,
            driveScaleValues=driveScaleArray,
            targetOutputRmsValues=targetOutputRmsArray,
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

        fieldNames = [
            "outputPowerDbm",
            "normalizedDriveScale",
            "targetOutputRmsVoltage",
        ]
        for methodName in methodNames:
            fieldNames.extend(
                [f"{methodName} evmDb", f"{methodName} evmPercent"]
            )
        with csvPath.open("w", newline="", encoding="utf-8-sig") as csvFile:
            csvWriter = csv.DictWriter(csvFile, fieldnames=fieldNames)
            csvWriter.writeheader()
            for pointIndex, outputPowerDbm in enumerate(
                selectedCurve.outputPowerDbmValues
            ):
                rowData = {
                    "outputPowerDbm": float(outputPowerDbm),
                    "normalizedDriveScale": float(
                        selectedCurve.driveScaleValues[pointIndex]
                    ),
                    "targetOutputRmsVoltage": float(
                        selectedCurve.targetOutputRmsValues[pointIndex]
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
                f"{stageName:<16} {metrics['snrDb']:>10.2f} "
                f"{metrics['evmDb']:>10.2f} "
                f"{metrics['evmPercent']:>10.3f} "
                f"{metrics['aclrLowerDb']:>10.2f} "
                f"{metrics['aclrUpperDb']:>10.2f} "
                f"{metrics['aclrWorstDb']:>10.2f}"
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
            for chainIndex, snrDb in enumerate(
                metrics["snrDbPerChain"]
            ):
                print(
                    f"PA {chainIndex + 1:<5} {snrDb:>10.2f} "
                    f"{metrics['aclrLowerDbPerChain'][chainIndex]:>10.2f} "
                    f"{metrics['aclrUpperDbPerChain'][chainIndex]:>10.2f} "
                    f"{metrics['aclrWorstDbPerChain'][chainIndex]:>10.2f}"
                )
            print(f"{stageName} - post-demapping spatial-stream EVM")
            print(f"{'Stream':<8} {'EVM(dB)':>10} {'EVM(%)':>10}")
            for streamIndex, evmDb in enumerate(
                metrics["evmDbPerSpatialStream"]
            ):
                print(
                    f"SS {streamIndex + 1:<5} {evmDb:>10.2f} "
                    f"{metrics['evmPercentPerSpatialStream'][streamIndex]:>10.3f}"
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
            stageName: dict(metrics)
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
            stageName: dict(self.stageMimoMetrics[stageName])
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

        fieldNames = [
            "stage",
            "snrDb",
            "evmDb",
            "evmPercent",
            "aclrLowerDb",
            "aclrUpperDb",
            "aclrWorstDb",
        ]
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
                rowData.update(metrics)
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

    def AnalyzeIlcHistory(
        self, ilcHistory: Sequence[Any]
    ) -> ILCAnalysisResult:
        """Analyze every stored SISO ILC output after learning has finished.

        Processing details:
            Algorithm: Validate native iteration records, feed each stored PA
            output through the ordinary ``Analyze`` path, combine RF metrics
            with algorithm diagnostics, and select the measured candidate with
            minimum strict Wi-Fi EVM outside the ILC algorithm.

        Args:
            ilcHistory: Native ``DpdIlc.ILCIteration`` records containing each
                measured input/output pair and algorithm MSE diagnostics.

        Returns:
            result: Independent performance history plus the EVM-best measured
                input, output, and complete signal metrics.
        """

        historyRecords = tuple(ilcHistory)
        if not historyRecords:
            raise ValueError("ilcHistory cannot be empty")
        performanceRecords = []
        inputSignals = []
        outputSignals = []
        previousIteration = 0
        for iterationRecord in historyRecords:
            iteration = int(iterationRecord.iteration)
            if iteration <= previousIteration:
                raise ValueError(
                    "ILC iterations must be strictly increasing"
                )
            previousIteration = iteration
            inputSignal = np.asarray(
                iterationRecord.inputSignal, dtype=np.complex128
            )
            outputSignal = np.asarray(
                iterationRecord.outputSignal, dtype=np.complex128
            )
            if (
                inputSignal.shape != self.referenceSignal.shape
                or outputSignal.shape != self.referenceSignal.shape
            ):
                raise ValueError(
                    "ILC iteration signals must match the analysis reference"
                )
            if not np.all(np.isfinite(inputSignal)) or not np.all(
                np.isfinite(outputSignal)
            ):
                raise ValueError(
                    "ILC iteration signals must contain finite samples"
                )
            signalMetrics = self.Analyze(outputSignal)
            evmAlignedMse = float(
                (signalMetrics["evmPercent"] / 100.0) ** 2
            )
            performanceRecords.append(
                ILCPerformanceIteration(
                    iteration=iteration,
                    mse=float(iterationRecord.mse),
                    errorRms=float(iterationRecord.errorRms),
                    nmseDb=float(iterationRecord.nmseDb),
                    linearCompensatedMse=float(
                        iterationRecord.linearCompensatedMse
                    ),
                    linearCompensatedNmseDb=float(
                        iterationRecord.linearCompensatedNmseDb
                    ),
                    complexGainMagnitudeDb=float(
                        iterationRecord.complexGainMagnitudeDb
                    ),
                    complexGainPhaseDegrees=float(
                        iterationRecord.complexGainPhaseDegrees
                    ),
                    inputPeak=float(iterationRecord.inputPeak),
                    snrDb=signalMetrics["snrDb"],
                    evmAlignedMse=evmAlignedMse,
                    evmDb=signalMetrics["evmDb"],
                    evmPercent=signalMetrics["evmPercent"],
                    aclrLowerDb=signalMetrics["aclrLowerDb"],
                    aclrUpperDb=signalMetrics["aclrUpperDb"],
                    aclrWorstDb=signalMetrics["aclrWorstDb"],
                )
            )
            inputSignals.append(inputSignal.copy())
            outputSignals.append(outputSignal.copy())

        performanceTuple = tuple(performanceRecords)
        bestIndex = int(
            np.argmin(
                [record.evmAlignedMse for record in performanceTuple]
            )
        )
        bestOutputSignal = outputSignals[bestIndex].copy()
        bestMetrics = self.Analyze(bestOutputSignal)
        return ILCAnalysisResult(
            history=performanceTuple,
            bestIteration=performanceTuple[bestIndex].iteration,
            bestInputSignal=inputSignals[bestIndex].copy(),
            bestOutputSignal=bestOutputSignal,
            bestMetrics=bestMetrics,
        )

    def AnalyzeMimoIlcHistory(
        self, chainHistories: Sequence[Sequence[Any]]
    ) -> ILCAnalysisResult:
        """Analyze synchronized MIMO candidates assembled from per-PA ILC.

        Processing details:
            Algorithm: Require one equal-length history per transmit chain,
            column-stack the input and PA output stored at each common round,
            aggregate native per-chain MSE diagnostics, evaluate the complete
            matrix through MIMO ``Analysis``, and select the minimum-EVM round.

        Args:
            chainHistories: Ordered native ILC histories, one sequence for each
                physical PA chain in transmit-chain order.

        Returns:
            result: Full-spatial-stream performance history and EVM-best MIMO
                input/output matrices selected outside ``DpdIlc``.
        """

        selectedHistories = tuple(
            tuple(chainHistory) for chainHistory in chainHistories
        )
        if (
            len(selectedHistories)
            != self.waveform.numTransmitAntennas
        ):
            raise ValueError(
                "chainHistories must contain one history per transmit antenna"
            )
        if not selectedHistories or not selectedHistories[0]:
            raise ValueError("chainHistories cannot be empty")
        iterationCount = len(selectedHistories[0])
        if any(
            len(chainHistory) != iterationCount
            for chainHistory in selectedHistories
        ):
            raise ValueError(
                "all MIMO chain histories must have equal iteration counts"
            )

        numericFloor = np.finfo(float).tiny
        targetPower = max(
            float(np.mean(np.abs(self.referenceSignal) ** 2)),
            numericFloor,
        )
        performanceRecords = []
        inputMatrices = []
        outputMatrices = []
        for iterationIndex in range(iterationCount):
            chainRecords = tuple(
                chainHistory[iterationIndex]
                for chainHistory in selectedHistories
            )
            iterationValues = {
                int(chainRecord.iteration)
                for chainRecord in chainRecords
            }
            if len(iterationValues) != 1:
                raise ValueError(
                    "MIMO chain histories must align by iteration number"
                )
            iteration = iterationValues.pop()
            if iteration != iterationIndex + 1:
                raise ValueError(
                    "MIMO ILC iterations must be contiguous and one-based"
                )
            inputMatrix = np.column_stack(
                [
                    np.asarray(
                        chainRecord.inputSignal, dtype=np.complex128
                    ).reshape(-1)
                    for chainRecord in chainRecords
                ]
            )
            outputMatrix = np.column_stack(
                [
                    np.asarray(
                        chainRecord.outputSignal, dtype=np.complex128
                    ).reshape(-1)
                    for chainRecord in chainRecords
                ]
            )
            if (
                inputMatrix.shape != self.referenceSignal.shape
                or outputMatrix.shape != self.referenceSignal.shape
            ):
                raise ValueError(
                    "MIMO iteration matrices must match the analysis reference"
                )
            signalMetrics = self.Analyze(outputMatrix)
            rawMse = float(
                np.mean(
                    [
                        float(chainRecord.mse)
                        for chainRecord in chainRecords
                    ]
                )
            )
            linearCompensatedMse = float(
                np.mean(
                    [
                        float(chainRecord.linearCompensatedMse)
                        for chainRecord in chainRecords
                    ]
                )
            )
            gainPhasors = np.asarray(
                [
                    10.0
                    ** (
                        float(chainRecord.complexGainMagnitudeDb)
                        / 20.0
                    )
                    * np.exp(
                        1j
                        * np.radians(
                            float(
                                chainRecord.complexGainPhaseDegrees
                            )
                        )
                    )
                    for chainRecord in chainRecords
                ],
                dtype=np.complex128,
            )
            averageGain = np.mean(gainPhasors)
            evmAlignedMse = float(
                (signalMetrics["evmPercent"] / 100.0) ** 2
            )
            performanceRecords.append(
                ILCPerformanceIteration(
                    iteration=iteration,
                    mse=rawMse,
                    errorRms=float(np.sqrt(rawMse)),
                    nmseDb=float(
                        10.0
                        * np.log10(
                            max(rawMse, numericFloor) / targetPower
                        )
                    ),
                    linearCompensatedMse=linearCompensatedMse,
                    linearCompensatedNmseDb=float(
                        10.0
                        * np.log10(
                            max(
                                linearCompensatedMse,
                                numericFloor,
                            )
                            / targetPower
                        )
                    ),
                    complexGainMagnitudeDb=float(
                        20.0
                        * np.log10(
                            max(float(np.abs(averageGain)), numericFloor)
                        )
                    ),
                    complexGainPhaseDegrees=float(
                        np.degrees(np.angle(averageGain))
                    ),
                    inputPeak=float(
                        max(
                            float(chainRecord.inputPeak)
                            for chainRecord in chainRecords
                        )
                    ),
                    snrDb=signalMetrics["snrDb"],
                    evmAlignedMse=evmAlignedMse,
                    evmDb=signalMetrics["evmDb"],
                    evmPercent=signalMetrics["evmPercent"],
                    aclrLowerDb=signalMetrics["aclrLowerDb"],
                    aclrUpperDb=signalMetrics["aclrUpperDb"],
                    aclrWorstDb=signalMetrics["aclrWorstDb"],
                )
            )
            inputMatrices.append(inputMatrix.copy())
            outputMatrices.append(outputMatrix.copy())

        performanceTuple = tuple(performanceRecords)
        bestIndex = int(
            np.argmin(
                [record.evmAlignedMse for record in performanceTuple]
            )
        )
        bestOutputSignal = outputMatrices[bestIndex].copy()
        bestMetrics = self.Analyze(bestOutputSignal)
        return ILCAnalysisResult(
            history=performanceTuple,
            bestIteration=performanceTuple[bestIndex].iteration,
            bestInputSignal=inputMatrices[bestIndex].copy(),
            bestOutputSignal=bestOutputSignal,
            bestMetrics=bestMetrics,
        )

    def SaveConvergence(
        self,
        ilcHistory: Sequence[ILCPerformanceIteration],
        outputDirectory: Path,
    ) -> Path:
        """Save independently analyzed per-iteration performance as CSV.

        Processing details:
            Algorithm: Convert validated in-memory results into a stable reporting format without altering later numerical calculations.

        Args:
            ilcHistory: Records returned by ``AnalyzeIlcHistory`` or its MIMO
                counterpart, with algorithm and RF metrics already combined.
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
            "snrDb",
            "evmAlignedMse",
            "evmDb",
            "evmPercent",
            "aclrLowerDb",
            "aclrUpperDb",
            "aclrWorstDb",
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
                        "snrDb": iterationRecord.snrDb,
                        "evmAlignedMse": iterationRecord.evmAlignedMse,
                        "evmDb": iterationRecord.evmDb,
                        "evmPercent": iterationRecord.evmPercent,
                        "aclrLowerDb": iterationRecord.aclrLowerDb,
                        "aclrUpperDb": iterationRecord.aclrUpperDb,
                        "aclrWorstDb": iterationRecord.aclrWorstDb,
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
        self,
        ilcHistory: Sequence[ILCPerformanceIteration],
        historyName: str = "ILC convergence",
    ) -> None:
        """Print every iteration after independent RF performance analysis.

        Processing details:
            Algorithm: Format each immutable history record into aligned
            columns while preserving linear-domain MSE values and their
            normalized decibel forms for direct engineering diagnosis.

        Args:
            ilcHistory: Records returned by ``AnalyzeIlcHistory`` or its MIMO
                counterpart.
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
            f"{'EVM(dB)':>9} {'SNR(dB)':>9} {'ACLR(dB)':>10} "
            f"{'Gain(dB)':>9} {'Phase(deg)':>11} "
            f"{'Peak':>9}"
        )
        print(header)
        print("-" * len(header))
        for iterationRecord in historyRecords:
            print(
                f"{iterationRecord.iteration:>4d} "
                f"{iterationRecord.mse:>12.5e} "
                f"{iterationRecord.nmseDb:>10.2f} "
                f"{iterationRecord.linearCompensatedMse:>12.5e} "
                f"{iterationRecord.linearCompensatedNmseDb:>10.2f} "
                f"{iterationRecord.evmAlignedMse:>12.5e} "
                f"{iterationRecord.evmDb:>9.2f} "
                f"{iterationRecord.snrDb:>9.2f} "
                f"{iterationRecord.aclrWorstDb:>10.2f} "
                f"{iterationRecord.complexGainMagnitudeDb:>9.2f} "
                f"{iterationRecord.complexGainPhaseDegrees:>11.2f} "
                f"{iterationRecord.inputPeak:>9.4f}"
            )
