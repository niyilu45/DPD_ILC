"""Object-oriented Wi-Fi and delegated two-tone RF performance analysis."""

import csv
import json
from collections import ChainMap
from dataclasses import asdict, dataclass, replace
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
    cast,
)

import numpy as np

from .ParseWifi import ParsedWifiFrame, ParseWifi
from .TwoToneAnalysis import TwoToneAnalysis, TwoToneMetrics
from .WaveGenTwoTone import TwoToneWaveform

# Cross-package imports support ``inc.lib`` from the repository root and
# ``lib`` when the caller places the ``inc`` directory on sys.path.
if __package__ and "." in __package__:
    from ..utils.ConfigUtils import (
        FilterRecognizedParameters,
        RecognizedParameterView,
    )
    from ..utils.FixedPoint import FixedPoint
    from ..utils.FrameProcess import FrameProcess
    from ..utils.SigProc import (
        PowerCalibration,
        SignalOverlapResult,
        SignalProcessingResult,
        SigProc,
    )
    from ..utils.WifiMetadata import WifiWaveform
else:
    from utils.ConfigUtils import (
        FilterRecognizedParameters,
        RecognizedParameterView,
    )
    from utils.FixedPoint import FixedPoint
    from utils.FrameProcess import FrameProcess
    from utils.SigProc import (
        PowerCalibration,
        SignalOverlapResult,
        SignalProcessingResult,
        SigProc,
    )
    from utils.WifiMetadata import WifiWaveform


class SignalMetrics(TypedDict):
    """Define the keys returned by ``Analysis.Analyze``.

    ``TypedDict`` provides static key and value information only. Every
    runtime result is an ordinary Python ``dict`` and is accessed with
    ``metrics["evmDb"]`` rather than custom-object attributes.
    """

    snrDb: float
    evmDb: float
    evmPercent: float
    irrDb: float
    aclrLowerDb: float
    aclrUpperDb: float
    aclrWorstDb: float
    outputPowerDbm: float


class IrrMeasurement(TypedDict):
    """Define the ordinary dictionary returned by ``MeasureIrr``.

    Coefficient powers are dimensionless fitted-model quantities rather than
    RF watts. Per-chain tuples preserve the conducted-port reference plane.
    """

    irrDb: float
    irrDbPerChain: Tuple[float, ...]
    desiredCoefficientPower: float
    imageCoefficientPower: float
    imageAmplitudeRatio: float
    directCoefficientRealPerChain: Tuple[float, ...]
    directCoefficientImagPerChain: Tuple[float, ...]
    imageCoefficientRealPerChain: Tuple[float, ...]
    imageCoefficientImagPerChain: Tuple[float, ...]
    residualPowerRatio: float
    regressionConditionNumberPerChain: Tuple[float, ...]


class IntermodulationOrderMetrics(TypedDict):
    """Define one ordinary-dictionary IM3, IM5, or IM7 result."""

    nonlinearOrder: int
    lowerFrequencyHz: float
    upperFrequencyHz: float
    lowerProductDbfs: float
    upperProductDbfs: float
    lowerDbc: float
    upperDbc: float
    worstDbc: float


class MimoSignalMetrics(TypedDict):
    """Define ordinary-dictionary MIMO detail keys."""

    snrDbPerChain: Tuple[float, ...]
    irrDbPerChain: Tuple[float, ...]
    evmDbPerSpatialStream: Tuple[float, ...]
    evmPercentPerSpatialStream: Tuple[float, ...]
    aclrLowerDbPerChain: Tuple[float, ...]
    aclrUpperDbPerChain: Tuple[float, ...]
    aclrWorstDbPerChain: Tuple[float, ...]
    outputPowerDbmPerChain: Tuple[float, ...]


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
    feedbackIntegerDelaySamples: float
    feedbackFractionalDelaySamples: float
    feedbackCarrierFrequencyOffsetHz: float
    feedbackSamplingFrequencyOffsetPpm: float
    feedbackComplexGainMagnitudeDb: float
    feedbackComplexGainPhaseDegrees: float
    outputPowerDbm: float
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
    """Analyze PA/DPD results through three independent reference paths.

    Explicit-reference mode receives ideal samples plus ``WifiWaveform``
    metadata. Transmit-assisted mode receives a measured waveform plus known
    transmitted samples and directly correlates their common interval without
    parsing a descriptor. Blind mode receives no transmitted reference, so
    only that mode invokes ``ParseWifi`` to restore metadata and ideal samples.

    Example:
        ``resultAnalysis = Analysis(referenceSignal, waveform)``
        ``resultAnalysis = Analysis(None, waveform)``
        ``metrics = resultAnalysis.Analyze(paOutput)``
        ``irrMeasurement = resultAnalysis.MeasureIrr(paOutput)``
        ``receiveAnalysis = Analysis(receivedWifiFrame)``
        ``metrics = receiveAnalysis.Analyze()``
        ``imMetrics = Analysis.AnalyzeTwoTone(paOutput, toneWaveform)``
        ``im3Metrics = Analysis.CalculateIm3(paOutput, toneWaveform)``
    """

    def __init__(
        self,
        referenceSignal: Optional[
            Union[np.ndarray, WifiWaveform]
        ] = None,
        waveform: Optional[WifiWaveform] = None,
        parameters: Optional[Mapping[str, object]] = None,
        parseParameters: Optional[Mapping[str, object]] = None,
        transmittedSignal: Optional[
            Union[np.ndarray, WifiWaveform]
        ] = None,
        signalProcessingParameters: Optional[
            Mapping[str, object]
        ] = None,
        sampleRateHz: Optional[float] = None,
        channelBandwidthHz: Optional[float] = None,
        width: Optional[int] = None,
        **parameterOverrides: object,
    ) -> None:
        """Initialize one of the three independent analysis contexts.

        Processing details:
            Algorithm: Select explicit-reference mode when ``waveform`` is
            supplied, use ``waveform.samples`` when that mode receives a
            ``None`` reference, select direct waveform-overlap mode when
            ``transmittedSignal`` is supplied, and use blind descriptor
            parsing only when neither is supplied. Analysis defaults remain
            inside this constructor and are resolved through ChainMap.

        Args:
            referenceSignal: Optional ideal reference samples when
                ``waveform`` is supplied. ``None`` reuses
                ``waveform.samples``. Otherwise this argument is the received
                NumPy array or ``WifiWaveform`` and cannot be ``None``.
            waveform: Optional Wi-Fi metadata selecting explicit-reference
                mode.
            parameters: Optional external mapping layered ahead of the built-in defaults.
            parseParameters: Optional ``ParseWifi`` parameter mapping. Blind
                mode forwards the complete mapping to ``ParseWifi``.
                Transmit-assisted mode accepts ``sampleRateHz`` and
                ``channelBandwidthHz`` as compatibility aliases without
                invoking the parser.
            transmittedSignal: Optional known transmit input selecting direct
                assisted mode. Either a metadata-rich ``WifiWaveform`` or a
                NumPy waveform containing samples only is accepted without
                invoking ``ParseWifi``.
            signalProcessingParameters: Optional explicit ``SigProc``
                configuration mapping. This named argument is preferred over
                nesting the same key inside ``parameters``; the nested form
                remains supported for backward compatibility.
            sampleRateHz: Optional physical sample rate for NumPy-assisted
                waveform-domain analysis.
            channelBandwidthHz: Optional occupied channel bandwidth used to
                enable ACLR in NumPy-assisted waveform-domain analysis.
            width: Optional external I/Q width. None selects the internal
                16-bit default, zero selects floating point, and a positive
                value selects signed integer I/Q codes in complex128.
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
                "activePowerThresholdDb": -60.0,
                "activeGapToleranceSamples": 16,
                "sampleRateHz": None,
                "channelBandwidthHz": None,
                "assistedMaximumOffsetSamples": 2000,
                "assistedReferenceSearchSamples": 32768,
                "assistedMinimumCorrelation": 0.12,
                "width": 16,
            }
        )
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping or None")
        if parseParameters is not None and not isinstance(
            parseParameters, Mapping
        ):
            raise TypeError("parseParameters must be a mapping or None")
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
        if signalProcessingParameters is not None:
            recognizedOverrides["signalProcessingParameters"] = (
                signalProcessingParameters
            )
        if sampleRateHz is not None:
            recognizedOverrides["sampleRateHz"] = sampleRateHz
        if channelBandwidthHz is not None:
            recognizedOverrides["channelBandwidthHz"] = (
                channelBandwidthHz
            )
        if width is not None:
            recognizedOverrides["width"] = width
        self.parameters: ChainMap[str, object] = ChainMap(
            recognizedOverrides,
            externalParameters,
            self.defaultParameters,
        )
        self.interfaceFormat = FixedPoint(
            cast(int, self.parameters["width"])
        )

        self.parsedWifiFrame: Optional[ParsedWifiFrame] = None
        self.signalOverlapResult: Optional[SignalOverlapResult] = None
        self.defaultMeasuredSignal: Optional[np.ndarray] = None
        self.analysisMode = "explicitReference"
        selectedWaveform = waveform
        selectedReference: Optional[
            Union[np.ndarray, WifiWaveform]
        ] = referenceSignal
        if selectedWaveform is not None and not isinstance(
            selectedWaveform, WifiWaveform
        ):
            raise TypeError("waveform must be a WifiWaveform or None")
        if selectedWaveform is not None:
            if parseParameters is not None or transmittedSignal is not None:
                raise ValueError(
                    "parseParameters and transmittedSignal are only valid "
                    "when waveform is omitted"
                )
            if selectedReference is None:
                # WifiWaveform is both the metadata contract and the original
                # generated transmit waveform, so callers do not need to pass
                # the same sample array a second time.
                selectedReference = selectedWaveform.samples
        elif transmittedSignal is not None:
            if referenceSignal is None:
                raise ValueError(
                    "received signal cannot be None in transmit-assisted mode"
                )
            if parseParameters is not None:
                # Older callers often placed the known receiver clock in the
                # parser mapping before the assisted path existed. Preserve
                # that useful configuration without parsing the known
                # transmit waveform. Explicit Analysis settings retain higher
                # priority, and parser-only keys are warned about and ignored.
                assistedParameterNames: Mapping[str, object] = (
                    MappingProxyType(
                        {
                            "sampleRateHz": None,
                            "channelBandwidthHz": None,
                        }
                    )
                )
                assistedCompatibilityParameters = (
                    FilterRecognizedParameters(
                        parseParameters,
                        assistedParameterNames,
                        "Analysis transmit-assisted parseParameters",
                    )
                )
                # Insert the compatibility layer immediately before defaults.
                # Direct arguments and the caller-owned Analysis mapping stay
                # live and retain priority over these migrated parser values.
                self.parameters.maps.insert(
                    len(self.parameters.maps) - 1,
                    assistedCompatibilityParameters,
                )
            self.analysisMode = "transmitAssisted"
            measuredInput = (
                referenceSignal.samples
                if isinstance(referenceSignal, WifiWaveform)
                else referenceSignal
            )
            measuredArray = self.interfaceFormat.QuantizeCodes(
                measuredInput
            )
            floatingMeasuredArray = self.interfaceFormat.DecodeComplex(
                measuredArray
            )
            if isinstance(transmittedSignal, WifiWaveform):
                selectedReference = (
                    self.interfaceFormat.QuantizeCodes(
                        transmittedSignal.samples
                    )
                )
                floatingReference = self.interfaceFormat.DecodeComplex(
                    selectedReference
                )
                selectedWaveform = transmittedSignal
                self.signalOverlapResult = SigProc.EstimateSignalOverlap(
                    floatingMeasuredArray,
                    floatingReference,
                    self.parameters[
                        "assistedMaximumOffsetSamples"
                    ],
                    self.parameters[
                        "assistedReferenceSearchSamples"
                    ],
                    self.parameters[
                        "assistedMinimumCorrelation"
                    ],
                )
                coversCompleteReference = (
                    self.signalOverlapResult.referenceStartSample == 0
                    and self.signalOverlapResult.overlapLength
                    == np.asarray(
                        transmittedSignal.samples
                    ).shape[0]
                )
                if coversCompleteReference:
                    receivedStart = (
                        self.signalOverlapResult.receivedStartSample
                    )
                    receivedStop = (
                        receivedStart
                        + self.signalOverlapResult.overlapLength
                    )
                    self.defaultMeasuredSignal = measuredArray[
                        receivedStart:receivedStop
                    ].copy()
                else:
                    self.defaultMeasuredSignal = measuredArray.copy()
            else:
                transmitArray = np.asarray(
                    self.interfaceFormat.QuantizeCodes(
                        transmittedSignal
                    ),
                    dtype=np.complex128,
                )
                floatingTransmitArray = (
                    self.interfaceFormat.DecodeComplex(transmitArray)
                )
                self.signalOverlapResult = SigProc.EstimateSignalOverlap(
                    floatingMeasuredArray,
                    floatingTransmitArray,
                    self.parameters[
                        "assistedMaximumOffsetSamples"
                    ],
                    self.parameters[
                        "assistedReferenceSearchSamples"
                    ],
                    self.parameters[
                        "assistedMinimumCorrelation"
                    ],
                )
                receivedStart = (
                    self.signalOverlapResult.receivedStartSample
                )
                referenceStart = (
                    self.signalOverlapResult.referenceStartSample
                )
                overlapStop = self.signalOverlapResult.overlapLength
                self.defaultMeasuredSignal = measuredArray[
                    receivedStart:receivedStart + overlapStop
                ].copy()
                selectedReference = transmitArray[
                    referenceStart:referenceStart + overlapStop
                ].copy()
                selectedWaveform = None
        else:
            self.analysisMode = "blind"
            if referenceSignal is None:
                raise ValueError(
                    "received signal cannot be None in blind analysis mode"
                )
            parseConfiguration = (
                {}
                if parseParameters is None
                else dict(parseParameters)
            )
            parseConfiguration["width"] = self.interfaceFormat.width
            parseInput: Union[np.ndarray, WifiWaveform] = (
                replace(
                    referenceSignal,
                    samples=self.interfaceFormat.QuantizeCodes(
                        referenceSignal.samples
                    ),
                )
                if isinstance(referenceSignal, WifiWaveform)
                else self.interfaceFormat.QuantizeCodes(
                    referenceSignal
                )
            )
            self.parsedWifiFrame = ParseWifi(
                parameters=parseConfiguration
            ).Parse(
                parseInput,
            )
            selectedReference = self.parsedWifiFrame.referenceSignal
            selectedWaveform = self.parsedWifiFrame.waveform
            self.defaultMeasuredSignal = (
                self.parsedWifiFrame.receivedSignal.copy()
            )
        if isinstance(selectedReference, WifiWaveform):
            selectedReference = selectedReference.samples
        if selectedReference is None:
            raise RuntimeError(
                "analysis reference resolution produced no signal"
            )
        # Public fixed-point inputs contain raw integer codes. Decode exactly
        # once at the analysis boundary so synchronization and metrics always
        # operate in normalized physical units.
        complexReference = self.interfaceFormat.DecodeComplex(
            selectedReference
        )
        if complexReference.size == 0:
            raise ValueError("referenceSignal cannot be empty")
        if selectedWaveform is not None:
            expectedShape = np.asarray(selectedWaveform.samples).shape
            if complexReference.shape != expectedShape:
                raise ValueError(
                    "referenceSignal shape must match the Wi-Fi waveform"
                )
        if complexReference.ndim not in (1, 2):
            raise ValueError("referenceSignal must be a vector or matrix")
        if (
            complexReference.ndim == 2
            and selectedWaveform is not None
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
        self.sampleRateHz = 1.0
        self.channelBandwidthHz: Optional[float] = None
        self.frameProcessor: Optional[FrameProcess] = (
            None
            if selectedWaveform is None
            else FrameProcess(selectedWaveform)
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
        """Return blind-mode parser output retained by the constructor.

        Processing details:
            Algorithm: Return ``None`` for explicit-reference and
            transmit-assisted modes because neither invokes ``ParseWifi``;
            otherwise return the blind-mode immutable parser result.

        Returns:
            result: Parsed frame only when blind analysis was selected.
        """

        return self.parsedWifiFrame

    def GetAnalysisMode(self) -> str:
        """Return the constructor path selected for this analysis instance.

        Processing details:
            Algorithm: Expose the immutable mode label assigned before any
            parser, overlap estimator, or metric calculation was executed.

        Returns:
            result: ``explicitReference``, ``transmitAssisted``, or ``blind``.
        """

        return self.analysisMode

    @property
    def Width(self) -> int:
        """Return the external I/Q component width.

        Processing details:
            Algorithm: Read the resolved construction-time ChainMap value used
            to quantize every reference and measured signal boundary.

        Returns:
            result: Zero for floating mode or a positive fixed-point width.
        """

        return cast(int, self.parameters["width"])

    width = Width

    def GetSignalOverlapResult(
        self,
    ) -> Optional[SignalOverlapResult]:
        """Return transmit-assisted overlap coordinates.

        Processing details:
            Algorithm: Preserve ``None`` outside transmit-assisted mode and
            otherwise return the immutable NumPy or ``WifiWaveform`` result.

        Returns:
            result: Overlap starts, common length, and confidence, or ``None``.
        """

        return self.signalOverlapResult

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

        self.interfaceFormat = FixedPoint(self.width)
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
        resolvedSampleRate = (
            self.waveform.sampleRateHz
            if self.waveform is not None
            else (
                1.0
                if self.parameters["sampleRateHz"] is None
                else self.parameters["sampleRateHz"]
            )
        )
        if (
            not isinstance(resolvedSampleRate, (int, float))
            or isinstance(resolvedSampleRate, bool)
            or not np.isfinite(resolvedSampleRate)
            or resolvedSampleRate <= 0.0
        ):
            raise ValueError(
                "sampleRateHz must be finite and positive when supplied"
            )
        resolvedBandwidth = (
            self.waveform.bandwidthHz
            if self.waveform is not None
            else self.parameters["channelBandwidthHz"]
        )
        if (
            resolvedBandwidth is not None
            and (
                not isinstance(resolvedBandwidth, (int, float))
                or isinstance(resolvedBandwidth, bool)
                or not np.isfinite(resolvedBandwidth)
                or resolvedBandwidth <= 0.0
            )
        ):
            raise ValueError(
                "channelBandwidthHz must be finite and positive when supplied"
            )
        if (
            resolvedBandwidth is not None
            and float(resolvedSampleRate)
            < 2.0 * float(resolvedBandwidth)
        ):
            raise ValueError(
                "sampleRateHz must be at least twice channelBandwidthHz"
            )
        maximumOffsetSamples = self.parameters[
            "assistedMaximumOffsetSamples"
        ]
        if (
            not isinstance(maximumOffsetSamples, int)
            or isinstance(maximumOffsetSamples, bool)
            or maximumOffsetSamples < 0
        ):
            raise ValueError(
                "assistedMaximumOffsetSamples must be a nonnegative integer"
            )
        referenceSearchSamples = self.parameters[
            "assistedReferenceSearchSamples"
        ]
        if (
            not isinstance(referenceSearchSamples, int)
            or isinstance(referenceSearchSamples, bool)
            or referenceSearchSamples < 16
        ):
            raise ValueError(
                "assistedReferenceSearchSamples must be at least 16"
            )
        minimumCorrelation = self.parameters[
            "assistedMinimumCorrelation"
        ]
        if (
            not isinstance(minimumCorrelation, (int, float))
            or isinstance(minimumCorrelation, bool)
            or not np.isfinite(minimumCorrelation)
            or not 0.0 <= minimumCorrelation <= 1.0
        ):
            raise ValueError(
                "assistedMinimumCorrelation must be between zero and one"
            )
        PowerCalibration(
            parameters={
                "loadResistanceOhm": self.parameters[
                    "loadResistanceOhm"
                ],
                "maximumOutputPowerDbm": self.parameters[
                    "maximumOutputPowerDbm"
                ],
                "activePowerThresholdDb": self.parameters[
                    "activePowerThresholdDb"
                ],
                "activeGapToleranceSamples": self.parameters[
                    "activeGapToleranceSamples"
                ],
                "width": self.width,
            },
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
                float(resolvedSampleRate),
                parameters=signalProcessingParameters,
            )
        self.sampleRateHz = float(resolvedSampleRate)
        self.channelBandwidthHz = (
            None
            if resolvedBandwidth is None
            else float(resolvedBandwidth)
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
        measuredArray = self.interfaceFormat.DecodeComplex(
            measuredSignal
        )
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
        dataSlice = (
            slice(0, referenceMatrix.shape[0])
            if self.waveform is None
            else self.waveform.fieldSlices[self.waveform.dataFieldName]
        )
        processingResults = []
        processedColumns = []
        for chainIndex in range(referenceMatrix.shape[1]):
            signalProcessor = SigProc(
                referenceMatrix[:, chainIndex],
                self.sampleRateHz,
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

        if self.frameProcessor is not None:
            return self.frameProcessor.ValidatePreparedSignal(
                preparedSignal
            )
        complexPrepared = np.asarray(
            preparedSignal, dtype=np.complex128
        )
        if complexPrepared.shape != self.referenceSignal.shape:
            raise ValueError(
                "preparedSignal shape must match the assisted reference"
            )
        if complexPrepared.size == 0 or not np.all(
            np.isfinite(complexPrepared)
        ):
            raise ValueError(
                "preparedSignal must contain finite samples"
            )
        return complexPrepared

    def CalculateOutputPower(
        self, preparedSignal: np.ndarray
    ) -> Tuple[float, Tuple[float, ...]]:
        """Calculate simulated output power before complex-gain removal.

        Processing details:
            Algorithm: Reconstruct each synchronized measured PA output by
            multiplying the compensated signal by the complex gain retained
            by ``SigProc``. Compute one normalized RMS value per conducted
            chain, map normalized RMS equal to one onto
            ``maximumOutputPowerDbm``, and sum independent chain powers for
            the aggregate MIMO result.

        Args:
            preparedSignal: Signal returned by ``PrepareMeasuredSignal``.

        Returns:
            result: Aggregate output power in dBm followed by a chain-ordered
                tuple of conducted output powers in dBm.
        """

        complexPrepared = self.ValidatePreparedSignal(preparedSignal)
        preparedMatrix = (
            complexPrepared.reshape(-1, 1)
            if complexPrepared.ndim == 1
            else complexPrepared
        )
        if len(self.lastSignalProcessingResults) != preparedMatrix.shape[1]:
            raise RuntimeError(
                "PrepareMeasuredSignal must run before output-power analysis"
            )

        # ``processedSignal`` is the aligned measured waveform divided by the
        # estimated common complex gain. Restoring that gain preserves the PA
        # output amplitude while retaining timing/CFO/SFO alignment and the
        # reference-length valid interval.
        alignedMeasuredMatrix = np.column_stack(
            [
                preparedMatrix[:, chainIndex]
                * self.lastSignalProcessingResults[
                    chainIndex
                ].complexGain
                for chainIndex in range(preparedMatrix.shape[1])
            ]
        )
        powerCalibration = PowerCalibration(
            parameters={
                "loadResistanceOhm": self.parameters[
                    "loadResistanceOhm"
                ],
                "maximumOutputPowerDbm": self.parameters[
                    "maximumOutputPowerDbm"
                ],
                "activePowerThresholdDb": self.parameters[
                    "activePowerThresholdDb"
                ],
                "activeGapToleranceSamples": self.parameters[
                    "activeGapToleranceSamples"
                ],
                "width": 0,
            },
        )
        normalizedRmsPerChain = np.asarray(
            powerCalibration.CalculateActiveRmsPerChain(
                alignedMeasuredMatrix
            ),
            dtype=float,
        )
        minimumPositive = np.finfo(float).tiny
        outputPowerDbmPerChain = tuple(
            float("-inf")
            if normalizedRms <= minimumPositive
            else powerCalibration.NormalizedRmsToOutputPowerDbm(
                float(normalizedRms)
            )
            for normalizedRms in normalizedRmsPerChain
        )
        aggregateNormalizedRms = float(
            np.sqrt(np.sum(normalizedRmsPerChain**2))
        )
        aggregateOutputPowerDbm = (
            float("-inf")
            if aggregateNormalizedRms <= minimumPositive
            else powerCalibration.NormalizedRmsToOutputPowerDbm(
                aggregateNormalizedRms
            )
        )
        return aggregateOutputPowerDbm, outputPowerDbmPerChain

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
        dataSlice = (
            slice(0, self.referenceSignal.shape[0])
            if self.waveform is None
            else self.waveform.fieldSlices[self.waveform.dataFieldName]
        )
        referenceData = self.referenceSignal[dataSlice]
        measuredData = complexMeasured[dataSlice]
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
        if self.frameProcessor is None:
            raise ValueError(
                "Wi-Fi demodulation requires WifiWaveform metadata; "
                "NumPy transmit-assisted mode provides waveform-domain EVM"
            )
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

        if self.frameProcessor is None:
            complexMeasured = self.ValidatePreparedSignal(preparedSignal)
            waveformError = (
                complexMeasured.reshape(-1)
                - self.referenceSignal.reshape(-1)
            )
            return float(
                np.sum(np.abs(waveformError) ** 2)
                / max(
                    np.sum(
                        np.abs(self.referenceSignal.reshape(-1)) ** 2
                    ),
                    np.finfo(float).tiny,
                )
            )
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

    def MeasureIrr(
        self,
        measuredSignal: Optional[np.ndarray] = None,
    ) -> IrrMeasurement:
        """Measure IRR and return widely linear fit diagnostics as a dictionary.

        Processing details:
            Algorithm: Select explicit samples or the stored assisted/blind
            capture, run timing, carrier-frequency, sampling-clock, and common
            complex-gain compensation once, then delegate the direct/conjugate
            coefficient fit to ``MeasurePreparedIrr``.

        Args:
            measuredSignal: Optional measured samples. Omit this value in
                transmit-assisted and blind modes; explicit-reference mode
                requires it because no received capture is stored.

        Returns:
            result: Ordinary dictionary containing aggregate and per-chain
                IRR, fitted coefficient powers, image amplitude ratio,
                coefficient components, normalized residual, and condition
                numbers.
        """

        selectedSignal = measuredSignal
        if selectedSignal is None:
            if self.defaultMeasuredSignal is None:
                raise ValueError(
                    "measuredSignal is required when Analysis was constructed "
                    "in explicit-reference mode"
                )
            selectedSignal = self.defaultMeasuredSignal
        preparedSignal = self.PrepareMeasuredSignal(selectedSignal)
        return self.MeasurePreparedIrr(preparedSignal)

    def CalculateIrr(
        self,
        measuredSignal: Optional[np.ndarray] = None,
    ) -> float:
        """Calculate image-rejection ratio after common synchronization.

        Processing details:
            Algorithm: Call ``MeasureIrr`` and return only its aggregate dB
            value. This compact interface is retained for existing callers;
            new measurement code can use ``MeasureIrr`` for diagnostics.

        Args:
            measuredSignal: Optional measured samples. Omit in assisted or
                blind mode and provide it in explicit-reference mode.

        Returns:
            result: Aggregate direct-to-image power ratio in decibels.
        """

        return float(self.MeasureIrr(measuredSignal)["irrDb"])

    def CalculatePreparedIrr(self, preparedSignal: np.ndarray) -> float:
        """Calculate only IRR from an already compensated signal.

        Processing details:
            Algorithm: Call ``MeasurePreparedIrr`` without repeating signal
            synchronization and return its aggregate dB value.

        Args:
            preparedSignal: Signal returned by ``PrepareMeasuredSignal``.

        Returns:
            result: Aggregate image-rejection ratio in dB.
        """

        return float(self.MeasurePreparedIrr(preparedSignal)["irrDb"])

    def MeasurePreparedIrr(
        self,
        preparedSignal: np.ndarray,
    ) -> IrrMeasurement:
        """Estimate IRR and fit quality with regularized widely linear LS.

        Processing details:
            Algorithm: Fit each measured chain to ``a*x + b*conj(x)`` on the
            useful data field, accumulate ``|a|^2`` as desired-path power and
            ``|b|^2`` as image-path power, calculate aggregate and per-chain
            ratios, retain complex coefficient components, and quantify the
            unexplained residual and regression conditioning. A tiny
            scale-relative ridge protects the solve without materially
            biasing ordinary circular Wi-Fi or nonzero complex-tone signals.

        Args:
            preparedSignal: Signal returned by ``PrepareMeasuredSignal``.

        Returns:
            result: Ordinary dictionary containing the IRR result and
                measurement-quality diagnostics.
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
        dataSlice = (
            slice(0, referenceMatrix.shape[0])
            if self.waveform is None
            else self.waveform.fieldSlices[self.waveform.dataFieldName]
        )
        directPower = 0.0
        imagePower = 0.0
        residualPower = 0.0
        measuredPower = 0.0
        perChainIrrDb = []
        directCoefficients = []
        imageCoefficients = []
        conditionNumbers = []
        numericFloor = np.finfo(float).tiny
        for chainIndex in range(referenceMatrix.shape[1]):
            referenceData = referenceMatrix[dataSlice, chainIndex].reshape(-1)
            measuredData = measuredMatrix[dataSlice, chainIndex].reshape(-1)
            regressionMatrix = np.column_stack(
                (referenceData, np.conj(referenceData))
            )
            normalMatrix = (
                regressionMatrix.conj().T @ regressionMatrix
            )
            diagonalScale = max(
                float(np.mean(np.real(np.diag(normalMatrix)))),
                numericFloor,
            )
            regularizedMatrix = normalMatrix + (
                1.0e-12
                * diagonalScale
                * np.eye(2, dtype=np.complex128)
            )
            coefficients = np.linalg.solve(
                regularizedMatrix,
                regressionMatrix.conj().T @ measuredData,
            )
            directCoefficient = complex(coefficients[0])
            imageCoefficient = complex(coefficients[1])
            chainDirectPower = float(np.abs(directCoefficient) ** 2)
            chainImagePower = float(np.abs(imageCoefficient) ** 2)
            fittedSignal = regressionMatrix @ coefficients
            directPower += chainDirectPower
            imagePower += chainImagePower
            residualPower += float(
                np.sum(np.abs(measuredData - fittedSignal) ** 2)
            )
            measuredPower += float(np.sum(np.abs(measuredData) ** 2))
            perChainIrrDb.append(
                float(
                    10.0
                    * np.log10(
                        max(chainDirectPower, numericFloor)
                        / max(chainImagePower, numericFloor)
                    )
                )
            )
            directCoefficients.append(directCoefficient)
            imageCoefficients.append(imageCoefficient)
            conditionNumbers.append(
                float(np.linalg.cond(regressionMatrix))
            )
        aggregateIrrDb = float(
            10.0
            * np.log10(
                max(directPower, numericFloor)
                / max(imagePower, numericFloor)
            )
        )
        imageAmplitudeRatio = float(
            np.sqrt(
                max(imagePower, 0.0)
                / max(directPower, numericFloor)
            )
        )
        return {
            "irrDb": aggregateIrrDb,
            "irrDbPerChain": tuple(perChainIrrDb),
            "desiredCoefficientPower": float(directPower),
            "imageCoefficientPower": float(imagePower),
            "imageAmplitudeRatio": imageAmplitudeRatio,
            "directCoefficientRealPerChain": tuple(
                float(value.real) for value in directCoefficients
            ),
            "directCoefficientImagPerChain": tuple(
                float(value.imag) for value in directCoefficients
            ),
            "imageCoefficientRealPerChain": tuple(
                float(value.real) for value in imageCoefficients
            ),
            "imageCoefficientImagPerChain": tuple(
                float(value.imag) for value in imageCoefficients
            ),
            "residualPowerRatio": float(
                residualPower / max(measuredPower, numericFloor)
            ),
            "regressionConditionNumberPerChain": tuple(conditionNumbers),
        }

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
        dataSlice = (
            slice(0, referenceMatrix.shape[0])
            if self.waveform is None
            else self.waveform.fieldSlices[self.waveform.dataFieldName]
        )
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

        if self.frameProcessor is None:
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
            evmDbValues = []
            evmPercentValues = []
            for chainIndex in range(referenceMatrix.shape[1]):
                referenceStream = referenceMatrix[:, chainIndex]
                errorStream = (
                    measuredMatrix[:, chainIndex] - referenceStream
                )
                evmRatio = np.sqrt(
                    np.sum(np.abs(errorStream) ** 2)
                    / max(
                        np.sum(np.abs(referenceStream) ** 2),
                        np.finfo(float).tiny,
                    )
                )
                evmDbValues.append(
                    float(
                        20.0
                        * np.log10(
                            max(evmRatio, np.finfo(float).tiny)
                        )
                    )
                )
                evmPercentValues.append(float(100.0 * evmRatio))
            return tuple(evmDbValues), tuple(evmPercentValues)
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

        if self.channelBandwidthHz is None:
            raise ValueError(
                "channelBandwidthHz is required for ACLR analysis"
            )
        halfBandwidth = self.channelBandwidthHz / 2.0
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
        if self.channelBandwidthHz is None:
            return float("nan"), float("nan"), float("nan")
        dataSlice = (
            slice(0, complexMeasured.shape[0])
            if self.waveform is None
            else self.waveform.fieldSlices[self.waveform.dataFieldName]
        )
        measuredData = complexMeasured[dataSlice]
        measuredMatrix = (
            measuredData.reshape(-1, 1)
            if measuredData.ndim == 1
            else measuredData
        )
        sampleRateHz = self.sampleRateHz
        channelBandwidthHz = self.channelBandwidthHz
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
        if self.channelBandwidthHz is None:
            missingValues = tuple(
                float("nan")
                for _ in range(measuredMatrix.shape[1])
            )
            return missingValues, missingValues, missingValues
        minimumAclrOversampling = float(
            self.parameters["minimumAclrOversampling"]
        )
        if (
            self.sampleRateHz
            < minimumAclrOversampling * self.channelBandwidthHz
        ):
            raise ValueError(
                "sampleRateHz must be at least "
                f"{minimumAclrOversampling:g} times bandwidthHz "
                "for ACLR analysis"
            )
        dataSlice = (
            slice(0, measuredMatrix.shape[0])
            if self.waveform is None
            else self.waveform.fieldSlices[self.waveform.dataFieldName]
        )
        lowerValues = []
        upperValues = []
        worstValues = []
        for chainIndex in range(measuredMatrix.shape[1]):
            frequencyBins, powerSpectrum = AveragePeriodogram(
                measuredMatrix[dataSlice, chainIndex],
                self.sampleRateHz,
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
        """Calculate power, SNR, EVM, IRR, and ACLR for one measured waveform.

        Processing details:
            Algorithm: Use the explicit measured waveform in reference mode,
            or the stored overlap-aligned waveform in assisted or blind mode.
            Synchronize once and feed the same corrected samples to every
            available metric so all results remain comparable.

        Args:
            measuredSignal: Optional measured samples evaluated against the
                stored reference. Omit in assisted and blind modes.

        Returns:
            result: Ordinary dictionary containing output power, SNR, EVM,
                IRR, and ACLR values.
        """

        selectedSignal = measuredSignal
        if selectedSignal is None:
            if self.defaultMeasuredSignal is None:
                raise ValueError(
                    "measuredSignal is required when Analysis was constructed "
                    "in explicit-reference mode"
                )
            selectedSignal = self.defaultMeasuredSignal

        # Synchronization is intentionally executed once. The same corrected
        # samples feed all metrics so SNR, EVM, and ACLR remain comparable.
        complexMeasured = self.PrepareMeasuredSignal(selectedSignal)
        (
            outputPowerDbm,
            outputPowerDbmPerChain,
        ) = self.CalculateOutputPower(complexMeasured)
        snrDb = self.CalculatePreparedSnr(complexMeasured)
        evmDb, evmPercent = self.CalculatePreparedEvm(complexMeasured)
        irrMeasurement = self.MeasurePreparedIrr(complexMeasured)
        irrDb = irrMeasurement["irrDb"]
        (
            aclrLowerDb,
            aclrUpperDb,
            aclrWorstDb,
        ) = self.CalculatePreparedAclr(complexMeasured)
        transmitChainCount = (
            1
            if self.referenceSignal.ndim == 1
            else self.referenceSignal.shape[1]
        )
        if transmitChainCount > 1:
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
                "irrDbPerChain": irrMeasurement["irrDbPerChain"],
                "evmDbPerSpatialStream": perStreamEvmDb,
                "evmPercentPerSpatialStream": perStreamEvmPercent,
                "aclrLowerDbPerChain": perChainAclrLowerDb,
                "aclrUpperDbPerChain": perChainAclrUpperDb,
                "aclrWorstDbPerChain": perChainAclrWorstDb,
                "outputPowerDbmPerChain": outputPowerDbmPerChain,
            }
        else:
            self.lastMimoMetrics = None
        return {
            "snrDb": float(snrDb),
            "evmDb": float(evmDb),
            "evmPercent": float(evmPercent),
            "irrDb": float(irrDb),
            "aclrLowerDb": float(aclrLowerDb),
            "aclrUpperDb": float(aclrUpperDb),
            "aclrWorstDb": float(aclrWorstDb),
            "outputPowerDbm": float(outputPowerDbm),
        }

    @staticmethod
    def BuildTwoToneWaveform(
        measuredSignal: Union[np.ndarray, Sequence[complex]],
        waveform: Optional[
            Union[TwoToneWaveform, np.ndarray, Sequence[complex]]
        ] = None,
        sampleRateHz: Optional[float] = None,
        toneFrequenciesHz: Optional[Sequence[float]] = None,
        ilcBandwidthHz: Optional[float] = None,
        width: Optional[int] = None,
    ) -> TwoToneWaveform:
        """Resolve metadata-rich or raw two-tone inputs into one waveform.

        Processing details:
            Algorithm: Retain a supplied ``TwoToneWaveform`` unchanged. For
            a NumPy array or Python list, require physical sample rate and two
            distinct tone frequencies, validate finite one-dimensional samples
            and Nyquist-safe IM3/IM5/IM7 locations, estimate descriptive tone
            amplitude and phase fields, and construct the minimal immutable
            metadata object required by ``TwoToneAnalysis``. When no separate
            raw reference is supplied, the measured samples provide only the
            record length and public-format metadata; they are not treated as
            an ideal PA reference.

        Args:
            measuredSignal: Measured floating or fixed-point output samples.
            waveform: Optional ``TwoToneWaveform`` or raw NumPy/list record.
                Omit it to analyze a standalone raw measured record.
            sampleRateHz: Required physical sample rate for raw-record mode.
            toneFrequenciesHz: Required pair of distinct fundamental
                frequencies in hertz for raw-record mode.
            ilcBandwidthHz: Optional descriptive ILC bandwidth for raw mode.
                ``None`` derives a conservative bandwidth from IM7 locations.
            width: Optional public I/Q width for raw mode. ``None`` selects
                floating-point samples; metadata-rich mode inherits its width.

        Returns:
            result: Existing or newly constructed immutable two-tone metadata.
        """

        if isinstance(waveform, TwoToneWaveform):
            if sampleRateHz is not None and not np.isclose(
                float(sampleRateHz), waveform.sampleRateHz
            ):
                raise ValueError(
                    "sampleRateHz must match the supplied TwoToneWaveform"
                )
            if toneFrequenciesHz is not None:
                suppliedFrequencies = tuple(toneFrequenciesHz)
                if len(suppliedFrequencies) != 2 or not np.allclose(
                    suppliedFrequencies,
                    waveform.toneFrequenciesHz,
                ):
                    raise ValueError(
                        "toneFrequenciesHz must match the supplied "
                        "TwoToneWaveform"
                    )
            if ilcBandwidthHz is not None and not np.isclose(
                float(ilcBandwidthHz), waveform.ilcBandwidthHz
            ):
                raise ValueError(
                    "ilcBandwidthHz must match the supplied TwoToneWaveform"
                )
            if width is not None and FixedPoint(width).width != waveform.width:
                raise ValueError(
                    "width must match the supplied TwoToneWaveform"
                )
            return waveform

        rawMetadataSignal = measuredSignal if waveform is None else waveform
        if sampleRateHz is None or toneFrequenciesHz is None:
            raise ValueError(
                "raw NumPy/list two-tone input requires sampleRateHz and "
                "toneFrequenciesHz"
            )
        if (
            not isinstance(sampleRateHz, (int, float))
            or isinstance(sampleRateHz, bool)
            or not np.isfinite(sampleRateHz)
            or float(sampleRateHz) <= 0.0
        ):
            raise ValueError("sampleRateHz must be finite and positive")
        frequencyValues = tuple(toneFrequenciesHz)
        if (
            len(frequencyValues) != 2
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isfinite(value)
                for value in frequencyValues
            )
            or frequencyValues[0] == frequencyValues[1]
        ):
            raise ValueError(
                "toneFrequenciesHz must contain two distinct finite values"
            )
        interfaceFormat = FixedPoint(0 if width is None else width)
        resolvedWidth = interfaceFormat.width
        try:
            rawSamples = np.asarray(
                rawMetadataSignal,
                dtype=np.complex128,
            )
        except (TypeError, ValueError) as error:
            raise TypeError(
                "raw two-tone waveform must be a NumPy array or Python list"
            ) from error
        if rawSamples.ndim != 1 or rawSamples.size < 64:
            raise ValueError(
                "raw two-tone waveform must be a one-dimensional record with "
                "at least 64 samples"
            )
        if not np.all(np.isfinite(rawSamples)):
            raise ValueError("raw two-tone waveform must contain finite samples")
        decodedSamples = interfaceFormat.DecodeComplex(rawSamples).reshape(-1)
        sortedFrequenciesHz = tuple(
            float(value) for value in sorted(frequencyValues)
        )
        nyquistHz = 0.5 * float(sampleRateHz)
        for nonlinearOrder in (3, 5, 7):
            outerCoefficient = (nonlinearOrder + 1) // 2
            innerCoefficient = (nonlinearOrder - 1) // 2
            productFrequenciesHz = (
                outerCoefficient * sortedFrequenciesHz[0]
                - innerCoefficient * sortedFrequenciesHz[1],
                outerCoefficient * sortedFrequenciesHz[1]
                - innerCoefficient * sortedFrequenciesHz[0],
            )
            if any(
                abs(frequencyHz) >= nyquistHz
                for frequencyHz in productFrequenciesHz
            ):
                raise ValueError(
                    f"IM{nonlinearOrder} products must lie inside complex Nyquist"
                )
        sampleIndices = np.arange(decodedSamples.size, dtype=float)
        toneCoefficients = tuple(
            complex(
                np.mean(
                    decodedSamples
                    * np.exp(
                        -1j
                        * 2.0
                        * np.pi
                        * frequencyHz
                        * sampleIndices
                        / float(sampleRateHz)
                    )
                )
            )
            for frequencyHz in sortedFrequenciesHz
        )
        derivedBandwidthHz = 2.2 * max(
            abs(4.0 * sortedFrequenciesHz[0] - 3.0 * sortedFrequenciesHz[1]),
            abs(4.0 * sortedFrequenciesHz[1] - 3.0 * sortedFrequenciesHz[0]),
        )
        resolvedIlcBandwidthHz = (
            derivedBandwidthHz
            if ilcBandwidthHz is None
            else float(ilcBandwidthHz)
        )
        if (
            not np.isfinite(resolvedIlcBandwidthHz)
            or resolvedIlcBandwidthHz <= 0.0
            or resolvedIlcBandwidthHz >= float(sampleRateHz)
        ):
            raise ValueError(
                "ilcBandwidthHz must be finite, positive, and smaller than "
                "sampleRateHz"
            )
        waveformRms = float(np.sqrt(np.mean(np.abs(decodedSamples) ** 2)))
        return TwoToneWaveform(
            samples=rawSamples.copy(),
            sampleRateHz=float(sampleRateHz),
            toneFrequenciesHz=sortedFrequenciesHz,
            toneAmplitudes=tuple(
                float(abs(value)) for value in toneCoefficients
            ),
            tonePhasesDegrees=tuple(
                float(np.rad2deg(np.angle(value)))
                for value in toneCoefficients
            ),
            numSamples=int(rawSamples.size),
            rmsLevel=waveformRms,
            width=resolvedWidth,
            ilcBandwidthHz=resolvedIlcBandwidthHz,
        )

    @staticmethod
    def AnalyzeTwoTone(
        measuredSignal: Union[np.ndarray, Sequence[complex]],
        waveform: Optional[
            Union[TwoToneWaveform, np.ndarray, Sequence[complex]]
        ] = None,
        parameters: Optional[Mapping[str, object]] = None,
        width: Optional[int] = None,
        sampleRateHz: Optional[float] = None,
        toneFrequenciesHz: Optional[Sequence[float]] = None,
        ilcBandwidthHz: Optional[float] = None,
        **parameterOverrides: object,
    ) -> TwoToneMetrics:
        """Calculate fundamentals, IM3, IM5, IM7, and output power.

        Processing details:
            Algorithm: Resolve metadata-rich or raw NumPy/list input into a
            ``TwoToneWaveform``, construct the dedicated metadata-aware
            ``TwoToneAnalysis`` implementation, and delegate exact-frequency
            windowed projection without coupling the ordinary Wi-Fi ``Analyze``
            path to two-tone metadata.

        Args:
            measuredSignal: Floating or fixed-point PA output NumPy/list data.
            waveform: Optional metadata-rich waveform or raw NumPy/list record.
            parameters: Optional TwoToneAnalysis parameter mapping.
            width: Optional public I/Q width override for raw records.
            sampleRateHz: Required sample rate when ``waveform`` is raw or
                omitted; it must match metadata-rich input when supplied.
            toneFrequenciesHz: Required two-tone frequencies for raw mode.
            ilcBandwidthHz: Optional raw-mode descriptive bandwidth.
            parameterOverrides: Highest-priority TwoToneAnalysis settings.

        Returns:
            result: Ordinary dictionary containing both fundamentals, paired
                IM3, IM5, IM7 values in dBc, worst IM, and output power.
        """

        metadataWidth = width
        if metadataWidth is None and parameters is not None:
            metadataWidth = parameters.get("width")
        resolvedWaveform = Analysis.BuildTwoToneWaveform(
            measuredSignal,
            waveform=waveform,
            sampleRateHz=sampleRateHz,
            toneFrequenciesHz=toneFrequenciesHz,
            ilcBandwidthHz=ilcBandwidthHz,
            width=metadataWidth,
        )
        toneAnalysis = TwoToneAnalysis(
            resolvedWaveform,
            parameters=parameters,
            width=width,
            **parameterOverrides,
        )
        return toneAnalysis.Analyze(measuredSignal)

    @staticmethod
    def CalculateIntermodulationOrder(
        measuredSignal: Union[np.ndarray, Sequence[complex]],
        waveform: Optional[
            Union[TwoToneWaveform, np.ndarray, Sequence[complex]]
        ] = None,
        nonlinearOrder: int = 3,
        parameters: Optional[Mapping[str, object]] = None,
        width: Optional[int] = None,
        sampleRateHz: Optional[float] = None,
        toneFrequenciesHz: Optional[Sequence[float]] = None,
        ilcBandwidthHz: Optional[float] = None,
        **parameterOverrides: object,
    ) -> IntermodulationOrderMetrics:
        """Calculate one paired IM3, IM5, or IM7 measurement.

        Processing details:
            Algorithm: Resolve metadata-rich or raw NumPy/list input once,
            reuse ``AnalyzeTwoTone`` for one consistent spectral pass, select
            the requested lower, upper, and worse-side dBc fields, recover
            absolute product dBFS by adding each dBc result to its same-side
            fundamental dBFS, and attach exact product frequencies.

        Args:
            measuredSignal: Floating or fixed-point PA output NumPy/list data.
            waveform: Optional metadata-rich waveform or raw NumPy/list record.
            nonlinearOrder: Supported odd order 3, 5, or 7.
            parameters: Optional TwoToneAnalysis parameter mapping.
            width: Optional public I/Q width override.
            sampleRateHz: Required sample rate for raw or omitted waveform.
            toneFrequenciesHz: Required fundamental pair for raw mode.
            ilcBandwidthHz: Optional raw-mode descriptive bandwidth.
            parameterOverrides: Highest-priority TwoToneAnalysis settings.

        Returns:
            result: Ordinary dictionary with order, product frequencies,
                absolute dBFS levels, same-side dBc values, and worst dBc.
        """

        if (
            not isinstance(nonlinearOrder, int)
            or isinstance(nonlinearOrder, bool)
            or nonlinearOrder not in (3, 5, 7)
        ):
            raise ValueError(
                "nonlinearOrder has an invalid value. Allowed values: "
                "3, 5, or 7."
            )
        metadataWidth = width
        if metadataWidth is None and parameters is not None:
            metadataWidth = parameters.get("width")
        resolvedWaveform = Analysis.BuildTwoToneWaveform(
            measuredSignal,
            waveform=waveform,
            sampleRateHz=sampleRateHz,
            toneFrequenciesHz=toneFrequenciesHz,
            ilcBandwidthHz=ilcBandwidthHz,
            width=metadataWidth,
        )
        completeMetrics: Mapping[str, float] = Analysis.AnalyzeTwoTone(
            measuredSignal,
            resolvedWaveform,
            parameters=parameters,
            width=width,
            **parameterOverrides,
        )
        metricPrefix = f"im{nonlinearOrder}"
        lowerDbc = float(completeMetrics[f"{metricPrefix}LowerDbc"])
        upperDbc = float(completeMetrics[f"{metricPrefix}UpperDbc"])
        lowerFrequencyHz, upperFrequencyHz = (
            resolvedWaveform.IntermodulationFrequencies(nonlinearOrder)
        )
        return {
            "nonlinearOrder": nonlinearOrder,
            "lowerFrequencyHz": float(lowerFrequencyHz),
            "upperFrequencyHz": float(upperFrequencyHz),
            "lowerProductDbfs": float(
                completeMetrics["fundamentalLowerDbfs"] + lowerDbc
            ),
            "upperProductDbfs": float(
                completeMetrics["fundamentalUpperDbfs"] + upperDbc
            ),
            "lowerDbc": lowerDbc,
            "upperDbc": upperDbc,
            "worstDbc": float(
                completeMetrics[f"{metricPrefix}WorstDbc"]
            ),
        }

    @staticmethod
    def CalculateIm3(
        measuredSignal: Union[np.ndarray, Sequence[complex]],
        waveform: Optional[
            Union[TwoToneWaveform, np.ndarray, Sequence[complex]]
        ] = None,
        parameters: Optional[Mapping[str, object]] = None,
        width: Optional[int] = None,
        sampleRateHz: Optional[float] = None,
        toneFrequenciesHz: Optional[Sequence[float]] = None,
        ilcBandwidthHz: Optional[float] = None,
        **parameterOverrides: object,
    ) -> IntermodulationOrderMetrics:
        """Calculate lower, upper, and worse-side third-order IM metrics.

        Processing details:
            Algorithm: Delegate to ``CalculateIntermodulationOrder`` with
            nonlinear order three so frequency, dBFS, and dBc conventions
            remain identical to the combined two-tone result.

        Args:
            measuredSignal: Floating or fixed-point PA output NumPy/list data.
            waveform: Optional metadata-rich waveform or raw NumPy/list record.
            parameters: Optional TwoToneAnalysis parameter mapping.
            width: Optional public I/Q width override.
            sampleRateHz: Required sample rate for raw or omitted waveform.
            toneFrequenciesHz: Required fundamental pair for raw mode.
            ilcBandwidthHz: Optional raw-mode descriptive bandwidth.
            parameterOverrides: Highest-priority TwoToneAnalysis settings.

        Returns:
            result: Ordinary dictionary containing the paired IM3 result.
        """

        return Analysis.CalculateIntermodulationOrder(
            measuredSignal,
            waveform,
            3,
            parameters=parameters,
            width=width,
            sampleRateHz=sampleRateHz,
            toneFrequenciesHz=toneFrequenciesHz,
            ilcBandwidthHz=ilcBandwidthHz,
            **parameterOverrides,
        )

    @staticmethod
    def CalculateIm5(
        measuredSignal: Union[np.ndarray, Sequence[complex]],
        waveform: Optional[
            Union[TwoToneWaveform, np.ndarray, Sequence[complex]]
        ] = None,
        parameters: Optional[Mapping[str, object]] = None,
        width: Optional[int] = None,
        sampleRateHz: Optional[float] = None,
        toneFrequenciesHz: Optional[Sequence[float]] = None,
        ilcBandwidthHz: Optional[float] = None,
        **parameterOverrides: object,
    ) -> IntermodulationOrderMetrics:
        """Calculate lower, upper, and worse-side fifth-order IM metrics.

        Processing details:
            Algorithm: Delegate to ``CalculateIntermodulationOrder`` with
            nonlinear order five so frequency, dBFS, and dBc conventions
            remain identical to the combined two-tone result.

        Args:
            measuredSignal: Floating or fixed-point PA output NumPy/list data.
            waveform: Optional metadata-rich waveform or raw NumPy/list record.
            parameters: Optional TwoToneAnalysis parameter mapping.
            width: Optional public I/Q width override.
            sampleRateHz: Required sample rate for raw or omitted waveform.
            toneFrequenciesHz: Required fundamental pair for raw mode.
            ilcBandwidthHz: Optional raw-mode descriptive bandwidth.
            parameterOverrides: Highest-priority TwoToneAnalysis settings.

        Returns:
            result: Ordinary dictionary containing the paired IM5 result.
        """

        return Analysis.CalculateIntermodulationOrder(
            measuredSignal,
            waveform,
            5,
            parameters=parameters,
            width=width,
            sampleRateHz=sampleRateHz,
            toneFrequenciesHz=toneFrequenciesHz,
            ilcBandwidthHz=ilcBandwidthHz,
            **parameterOverrides,
        )

    @staticmethod
    def CalculateIm7(
        measuredSignal: Union[np.ndarray, Sequence[complex]],
        waveform: Optional[
            Union[TwoToneWaveform, np.ndarray, Sequence[complex]]
        ] = None,
        parameters: Optional[Mapping[str, object]] = None,
        width: Optional[int] = None,
        sampleRateHz: Optional[float] = None,
        toneFrequenciesHz: Optional[Sequence[float]] = None,
        ilcBandwidthHz: Optional[float] = None,
        **parameterOverrides: object,
    ) -> IntermodulationOrderMetrics:
        """Calculate lower, upper, and worse-side seventh-order IM metrics.

        Processing details:
            Algorithm: Delegate to ``CalculateIntermodulationOrder`` with
            nonlinear order seven so frequency, dBFS, and dBc conventions
            remain identical to the combined two-tone result.

        Args:
            measuredSignal: Floating or fixed-point PA output NumPy/list data.
            waveform: Optional metadata-rich waveform or raw NumPy/list record.
            parameters: Optional TwoToneAnalysis parameter mapping.
            width: Optional public I/Q width override.
            sampleRateHz: Required sample rate for raw or omitted waveform.
            toneFrequenciesHz: Required fundamental pair for raw mode.
            ilcBandwidthHz: Optional raw-mode descriptive bandwidth.
            parameterOverrides: Highest-priority TwoToneAnalysis settings.

        Returns:
            result: Ordinary dictionary containing the paired IM7 result.
        """

        return Analysis.CalculateIntermodulationOrder(
            measuredSignal,
            waveform,
            7,
            parameters=parameters,
            width=width,
            sampleRateHz=sampleRateHz,
            toneFrequenciesHz=toneFrequenciesHz,
            ilcBandwidthHz=ilcBandwidthHz,
            **parameterOverrides,
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
        outputPowerDbmValues: Sequence[float],
        methodEvaluators: Mapping[
            str, Callable[[np.ndarray, float], np.ndarray]
        ],
    ) -> PowerEvmCurve:
        """Evaluate multiple methods over one common absolute-power sweep.

        Every evaluator is treated as a complete DPD-plus-PA plant. For each
        requested dBm point, ``PowerCalibration`` iteratively changes the
        plant input, observes actual active-burst output power, and stops only
        inside the configured tolerance. No post-PA amplitude normalization
        is applied, so EVM reflects the PA's true compression operating point.

        Args:
            outputPowerDbmValues: Strictly increasing absolute per-PA output
                powers in dBm, no greater than ``maximumOutputPowerDbm``.
            methodEvaluators: Mapping from display name to a callable that
                accepts the trial reference and target output dBm, and returns
                the corresponding measured PA output waveform.

        Returns:
            result: Power-EVM curve containing output powers, nominal initial
                drive scales, target RMS voltages, and per-method EVM arrays.
        """

        if self.waveform is None:
            raise ValueError(
                "AnalyzePowerEvmCurve requires WifiWaveform metadata"
            )
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
            parameters={
                "loadResistanceOhm": self.parameters[
                    "loadResistanceOhm"
                ],
                "maximumOutputPowerDbm": self.parameters[
                    "maximumOutputPowerDbm"
                ],
                "activePowerThresholdDb": self.parameters[
                    "activePowerThresholdDb"
                ],
                "activeGapToleranceSamples": self.parameters[
                    "activeGapToleranceSamples"
                ],
                "width": self.width,
            },
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
            currentOutputPowerDbm = [float(powerDbmArray[0])]
            # The mutable one-element list lets one bound callable retain the
            # calibrator's hidden preset while the requested sweep dBm changes.
            powerCalibration.SetPaModel(
                lambda trialInput: methodEvaluator(
                    trialInput, currentOutputPowerDbm[0]
                )
            )
            for outputPowerDbm in powerDbmArray:
                currentOutputPowerDbm[0] = float(outputPowerDbm)
                powerCalibration.UpdateParameters(
                    outputPowerDbm=float(outputPowerDbm),
                    outputPowerDbmPerChain=None,
                )
                pointReference = powerCalibration.Calibrate(
                    self.waveform.samples
                )
                measuredSignal = powerCalibration.GetLastPaOutput()
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
            f"{'Stage':<16} {'Pout(dBm)':>10} {'SNR(dB)':>10} "
            f"{'EVM(dB)':>10} {'IRR(dB)':>10} "
            f"{'EVM(%)':>10} {'ACLR-L':>10} {'ACLR-U':>10} {'ACLR-W':>10}"
        )
        print(header)
        print("-" * len(header))
        for stageName, metrics in selectedMetrics.items():
            print(
                f"{stageName:<16} {metrics['outputPowerDbm']:>10.2f} "
                f"{metrics['snrDb']:>10.2f} "
                f"{metrics['evmDb']:>10.2f} "
                f"{metrics['irrDb']:>10.2f} "
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
        """Print per-chain SNR/IRR/ACLR and per-stream EVM result tables.

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
                f"{'PA':<8} {'Pout(dBm)':>10} {'SNR(dB)':>10} "
                f"{'IRR(dB)':>10} {'ACLR-L':>10} "
                f"{'ACLR-U':>10} {'ACLR-W':>10}"
            )
            for chainIndex, snrDb in enumerate(
                metrics["snrDbPerChain"]
            ):
                print(
                    f"PA {chainIndex + 1:<5} "
                    f"{metrics['outputPowerDbmPerChain'][chainIndex]:>10.2f} "
                    f"{snrDb:>10.2f} "
                    f"{metrics['irrDbPerChain'][chainIndex]:>10.2f} "
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
            "irrDb",
            "aclrLowerDb",
            "aclrUpperDb",
            "aclrWorstDb",
            "outputPowerDbm",
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
                    feedbackIntegerDelaySamples=float(
                        getattr(
                            iterationRecord,
                            "integerDelaySamples",
                            0,
                        )
                    ),
                    feedbackFractionalDelaySamples=float(
                        getattr(
                            iterationRecord,
                            "fractionalDelaySamples",
                            0.0,
                        )
                    ),
                    feedbackCarrierFrequencyOffsetHz=float(
                        getattr(
                            iterationRecord,
                            "carrierFrequencyOffsetHz",
                            0.0,
                        )
                    ),
                    feedbackSamplingFrequencyOffsetPpm=float(
                        getattr(
                            iterationRecord,
                            "samplingFrequencyOffsetPpm",
                            0.0,
                        )
                    ),
                    feedbackComplexGainMagnitudeDb=float(
                        20.0
                        * np.log10(
                            max(
                                float(
                                    np.abs(
                                        getattr(
                                            iterationRecord,
                                            "feedbackComplexGain",
                                            1.0 + 0.0j,
                                        )
                                    )
                                ),
                                np.finfo(float).tiny,
                            )
                        )
                    ),
                    feedbackComplexGainPhaseDegrees=float(
                        np.degrees(
                            np.angle(
                                getattr(
                                    iterationRecord,
                                    "feedbackComplexGain",
                                    1.0 + 0.0j,
                                )
                            )
                        )
                    ),
                    outputPowerDbm=signalMetrics["outputPowerDbm"],
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
        expectedChainCount = (
            1
            if self.referenceSignal.ndim == 1
            else self.referenceSignal.shape[1]
        )
        if (
            len(selectedHistories)
            != expectedChainCount
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
            feedbackGainPhasors = np.asarray(
                [
                    complex(
                        getattr(
                            chainRecord,
                            "feedbackComplexGain",
                            1.0 + 0.0j,
                        )
                    )
                    for chainRecord in chainRecords
                ],
                dtype=np.complex128,
            )
            averageFeedbackGain = np.mean(feedbackGainPhasors)
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
                    feedbackIntegerDelaySamples=float(
                        np.mean(
                            [
                                float(
                                    getattr(
                                        chainRecord,
                                        "integerDelaySamples",
                                        0,
                                    )
                                )
                                for chainRecord in chainRecords
                            ]
                        )
                    ),
                    feedbackFractionalDelaySamples=float(
                        np.mean(
                            [
                                float(
                                    getattr(
                                        chainRecord,
                                        "fractionalDelaySamples",
                                        0.0,
                                    )
                                )
                                for chainRecord in chainRecords
                            ]
                        )
                    ),
                    feedbackCarrierFrequencyOffsetHz=float(
                        np.mean(
                            [
                                float(
                                    getattr(
                                        chainRecord,
                                        "carrierFrequencyOffsetHz",
                                        0.0,
                                    )
                                )
                                for chainRecord in chainRecords
                            ]
                        )
                    ),
                    feedbackSamplingFrequencyOffsetPpm=float(
                        np.mean(
                            [
                                float(
                                    getattr(
                                        chainRecord,
                                        "samplingFrequencyOffsetPpm",
                                        0.0,
                                    )
                                )
                                for chainRecord in chainRecords
                            ]
                        )
                    ),
                    feedbackComplexGainMagnitudeDb=float(
                        20.0
                        * np.log10(
                            max(
                                float(np.abs(averageFeedbackGain)),
                                numericFloor,
                            )
                        )
                    ),
                    feedbackComplexGainPhaseDegrees=float(
                        np.degrees(np.angle(averageFeedbackGain))
                    ),
                    outputPowerDbm=signalMetrics["outputPowerDbm"],
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
            "outputPowerDbm",
            "snrDb",
            "evmAlignedMse",
            "evmDb",
            "evmPercent",
            "aclrLowerDb",
            "aclrUpperDb",
            "aclrWorstDb",
            "complexGainMagnitudeDb",
            "complexGainPhaseDegrees",
            "feedbackIntegerDelaySamples",
            "feedbackFractionalDelaySamples",
            "feedbackCarrierFrequencyOffsetHz",
            "feedbackSamplingFrequencyOffsetPpm",
            "feedbackComplexGainMagnitudeDb",
            "feedbackComplexGainPhaseDegrees",
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
                        "outputPowerDbm": (
                            iterationRecord.outputPowerDbm
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
                        "feedbackIntegerDelaySamples": (
                            iterationRecord.feedbackIntegerDelaySamples
                        ),
                        "feedbackFractionalDelaySamples": (
                            iterationRecord.feedbackFractionalDelaySamples
                        ),
                        "feedbackCarrierFrequencyOffsetHz": (
                            iterationRecord.feedbackCarrierFrequencyOffsetHz
                        ),
                        "feedbackSamplingFrequencyOffsetPpm": (
                            iterationRecord.feedbackSamplingFrequencyOffsetPpm
                        ),
                        "feedbackComplexGainMagnitudeDb": (
                            iterationRecord.feedbackComplexGainMagnitudeDb
                        ),
                        "feedbackComplexGainPhaseDegrees": (
                            iterationRecord.feedbackComplexGainPhaseDegrees
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
            f"{'Pout':>9} {'EVM(dB)':>9} {'SNR(dB)':>9} "
            f"{'ACLR(dB)':>10} "
            f"{'Delay':>9} {'CFO(Hz)':>10} "
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
                f"{iterationRecord.outputPowerDbm:>9.2f} "
                f"{iterationRecord.evmDb:>9.2f} "
                f"{iterationRecord.snrDb:>9.2f} "
                f"{iterationRecord.aclrWorstDb:>10.2f} "
                f"{iterationRecord.feedbackIntegerDelaySamples:>9.2f} "
                f"{iterationRecord.feedbackCarrierFrequencyOffsetHz:>10.1f} "
                f"{iterationRecord.feedbackComplexGainMagnitudeDb:>9.2f} "
                f"{iterationRecord.feedbackComplexGainPhaseDegrees:>11.2f} "
                f"{iterationRecord.inputPeak:>9.4f}"
            )
