"""Measure MIMO channel flatness, coupling, delay, and conditioning.

The analyzer treats a channel as a linear multiple-input multiple-output
network during its low-power characterization interval.  It excites one
source chain at a time, reconstructs the complete causal impulse-response
matrix, and derives frequency-domain path metrics without depending on the
nonlinear PA or DPD implementation.
"""

from collections import ChainMap
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    Callable,
    Dict,
    Mapping,
    Optional,
    Tuple,
    cast,
)

import numpy as np

# Support both ``inc.lib`` and the compatibility ``lib`` package entry points.
if __package__ and "." in __package__:
    from ..utils.ConfigUtils import (
        FilterRecognizedParameters,
        RecognizedParameterView,
    )
    from ..utils.FixedPoint import FixedPoint
else:
    from utils.ConfigUtils import (
        FilterRecognizedParameters,
        RecognizedParameterView,
    )
    from utils.FixedPoint import FixedPoint


@dataclass(frozen=True)
class ChannelPathMeasurement:
    """Store one directed source-to-destination transfer-path result."""

    sourceChain: int
    destinationChain: int
    isDirectPath: bool
    detected: bool
    gainDb: float
    phaseDegrees: float
    flatnessDb: float
    groupDelaySamples: float
    groupDelayNs: float

    def ToDict(self) -> Dict[str, object]:
        """Convert one immutable path result to a serialization-ready mapping.

        Processing details:
            Algorithm: Copy the directed path identity, detection flag, and
            all scalar magnitude, phase, flatness, and delay estimates without
            recalculating or rounding the measurement.

        Returns:
            result: Ordinary dictionary suitable for JSON and CSV output.
        """

        return {
            "sourceChain": self.sourceChain,
            "destinationChain": self.destinationChain,
            "isDirectPath": self.isDirectPath,
            "detected": self.detected,
            "gainDb": self.gainDb,
            "phaseDegrees": self.phaseDegrees,
            "flatnessDb": self.flatnessDb,
            "groupDelaySamples": self.groupDelaySamples,
            "groupDelayNs": self.groupDelayNs,
        }


@dataclass(frozen=True)
class ChannelMeasurementResult:
    """Store a complete measured MIMO channel and its scalar summaries."""

    stageName: str
    sampleRateHz: float
    channelBandwidthHz: float
    fftLength: int
    impulseResponses: np.ndarray
    frequencyResponse: np.ndarray
    frequencyBinsHz: np.ndarray
    paths: Tuple[ChannelPathMeasurement, ...]
    worstDirectFlatnessDb: float
    worstDetectedPathFlatnessDb: float
    worstCouplingDb: float
    medianConditionNumber: float
    worstConditionNumber: float

    def ToDict(self) -> Dict[str, object]:
        """Convert channel summaries and path measurements to plain values.

        Processing details:
            Algorithm: Preserve the measured configuration and scalar
            summaries, then delegate each directed path conversion.  Large
            complex impulse/frequency arrays intentionally remain in memory
            and are saved separately by the benchmark as compact tables.

        Returns:
            result: JSON-compatible mapping excluding complex array payloads.
        """

        return {
            "stageName": self.stageName,
            "sampleRateHz": self.sampleRateHz,
            "channelBandwidthHz": self.channelBandwidthHz,
            "fftLength": self.fftLength,
            "chainCount": int(self.impulseResponses.shape[1]),
            "impulseLength": int(self.impulseResponses.shape[0]),
            "worstDirectFlatnessDb": self.worstDirectFlatnessDb,
            "worstDetectedPathFlatnessDb": (
                self.worstDetectedPathFlatnessDb
            ),
            "worstCouplingDb": self.worstCouplingDb,
            "medianConditionNumber": self.medianConditionNumber,
            "worstConditionNumber": self.worstConditionNumber,
            "paths": [path.ToDict() for path in self.paths],
        }

    def GetPath(
        self, sourceChain: int, destinationChain: int
    ) -> ChannelPathMeasurement:
        """Return one directed path identified by zero-based chain indices.

        Processing details:
            Algorithm: Search the immutable path tuple for the exact source
            and destination pair and fail explicitly if the requested channel
            was not part of the measurement.

        Args:
            sourceChain: Zero-based excited input-chain index.
            destinationChain: Zero-based observed output-chain index.

        Returns:
            result: Matching direct or coupling-path measurement.
        """

        for pathMeasurement in self.paths:
            if (
                pathMeasurement.sourceChain == sourceChain
                and pathMeasurement.destinationChain == destinationChain
            ):
                return pathMeasurement
        raise IndexError("requested channel path was not measured")


class ChannelAnalyse:
    """Estimate linear channel properties from orthogonal impulse captures."""

    def __init__(
        self,
        parameters: Optional[Mapping[str, object]] = None,
        width: Optional[int] = None,
        **parameterOverrides: object,
    ) -> None:
        """Initialize live measurement parameters and fixed-point boundaries.

        Processing details:
            Algorithm: Define all defaults inside the constructor, layer the
            caller mapping and direct overrides with ChainMap precedence,
            filter unknown names with warnings, and validate the complete
            physical and numerical measurement setup.

        Args:
            parameters: Optional caller-owned live parameter mapping.
            width: Optional public I/Q component width override.
            parameterOverrides: Highest-priority recognized local settings.

        Returns:
            result: None. The analyzer is ready to create channel probes.
        """

        self.defaultParameters: Mapping[str, object] = MappingProxyType(
            {
                "sampleRateHz": 80.0e6,
                "channelBandwidthHz": 20.0e6,
                "fftLength": 2048,
                "impulseLength": 64,
                "probeDelaySamples": 8,
                "couplingDetectionThresholdDb": -70.0,
                "magnitudeFloorDb": -180.0,
                "groupDelayMagnitudeRangeDb": 35.0,
                "width": 16,
            }
        )
        directOverrides = dict(parameterOverrides)
        if width is not None:
            directOverrides["width"] = width
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping or None")
        externalParameters: Mapping[str, object] = (
            {}
            if parameters is None
            else RecognizedParameterView(
                parameters,
                self.defaultParameters,
                "ChannelAnalyse",
            )
        )
        recognizedOverrides = FilterRecognizedParameters(
            directOverrides,
            self.defaultParameters,
            "ChannelAnalyse",
        )
        self.parameters: ChainMap[str, object] = ChainMap(
            recognizedOverrides,
            externalParameters,
            self.defaultParameters,
        )
        self.ValidateParameters()

    @property
    def Width(self) -> int:
        """Return the public signed I/Q component width.

        Processing details:
            Algorithm: Read the validated live ChainMap value so external
            parameter changes affect the next probe and capture conversion.

        Returns:
            result: Zero for floating data or a positive fixed-point width.
        """

        return cast(int, self.parameters["width"])

    width = Width

    def GetParameters(self) -> Dict[str, object]:
        """Return a detached snapshot of all resolved measurement settings.

        Processing details:
            Algorithm: Flatten the current ChainMap without exposing any
            mutable internal layer to the caller.

        Returns:
            result: Ordinary dictionary of effective parameters.
        """

        return dict(self.parameters)

    def UpdateParameters(self, **parameterOverrides: object) -> None:
        """Apply recognized parameter changes transactionally.

        Processing details:
            Algorithm: Filter unsupported keys with warnings, update the
            highest-priority mapping, validate the complete state, and restore
            the previous mapping if any recognized value is invalid.

        Args:
            parameterOverrides: Supported settings to replace locally.

        Returns:
            result: None. Valid values affect subsequent measurements.
        """

        recognizedOverrides = FilterRecognizedParameters(
            parameterOverrides,
            self.defaultParameters,
            "ChannelAnalyse.UpdateParameters",
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
        """Validate bandwidth, FFT, probe, detection, and width controls.

        Processing details:
            Algorithm: Enforce a positive Nyquist-valid occupied bandwidth,
            a sufficiently long FFT and impulse record, a nonnegative probe
            guard, ordered negative dB thresholds, and a valid public format.

        Returns:
            result: None. Invalid recognized settings raise an exception.
        """

        sampleRateHz = self.parameters["sampleRateHz"]
        channelBandwidthHz = self.parameters["channelBandwidthHz"]
        if (
            not isinstance(sampleRateHz, (int, float))
            or isinstance(sampleRateHz, bool)
            or not np.isfinite(sampleRateHz)
            or float(sampleRateHz) <= 0.0
        ):
            raise ValueError("sampleRateHz must be finite and positive")
        if (
            not isinstance(channelBandwidthHz, (int, float))
            or isinstance(channelBandwidthHz, bool)
            or not np.isfinite(channelBandwidthHz)
            or float(channelBandwidthHz) <= 0.0
            or float(channelBandwidthHz) > float(sampleRateHz)
        ):
            raise ValueError(
                "channelBandwidthHz must be positive and no greater than "
                "sampleRateHz"
            )
        fftLength = self.parameters["fftLength"]
        impulseLength = self.parameters["impulseLength"]
        probeDelaySamples = self.parameters["probeDelaySamples"]
        if (
            not isinstance(fftLength, int)
            or isinstance(fftLength, bool)
            or fftLength < 64
        ):
            raise ValueError("fftLength must be an integer no smaller than 64")
        if (
            not isinstance(impulseLength, int)
            or isinstance(impulseLength, bool)
            or impulseLength < 2
            or impulseLength > fftLength
        ):
            raise ValueError(
                "impulseLength must be an integer from 2 through fftLength"
            )
        if (
            not isinstance(probeDelaySamples, int)
            or isinstance(probeDelaySamples, bool)
            or probeDelaySamples < 0
        ):
            raise ValueError(
                "probeDelaySamples must be a nonnegative integer"
            )
        couplingThresholdDb = self.parameters[
            "couplingDetectionThresholdDb"
        ]
        magnitudeFloorDb = self.parameters["magnitudeFloorDb"]
        delayRangeDb = self.parameters["groupDelayMagnitudeRangeDb"]
        for parameterName, parameterValue in (
            ("couplingDetectionThresholdDb", couplingThresholdDb),
            ("magnitudeFloorDb", magnitudeFloorDb),
            ("groupDelayMagnitudeRangeDb", delayRangeDb),
        ):
            if (
                not isinstance(parameterValue, (int, float))
                or isinstance(parameterValue, bool)
                or not np.isfinite(parameterValue)
            ):
                raise ValueError(f"{parameterName} must be finite")
        if float(couplingThresholdDb) >= 0.0:
            raise ValueError(
                "couplingDetectionThresholdDb must be negative"
            )
        if float(magnitudeFloorDb) >= float(couplingThresholdDb):
            raise ValueError(
                "magnitudeFloorDb must be below the coupling threshold"
            )
        if float(delayRangeDb) <= 0.0:
            raise ValueError(
                "groupDelayMagnitudeRangeDb must be positive"
            )
        FixedPoint(self.width)

    def BuildImpulseProbe(
        self, chainCount: int, sourceChain: int
    ) -> np.ndarray:
        """Build one isolated broadband probe in the public data convention.

        Processing details:
            Algorithm: Place a unit complex impulse after a leading guard on
            only the selected source column, then encode the matrix once at
            the configured floating or fixed-point boundary.

        Args:
            chainCount: Number of physical input and output chains.
            sourceChain: Zero-based chain excited by this probe.

        Returns:
            result: Samples-by-chains impulse matrix for the channel processor.
        """

        if (
            not isinstance(chainCount, int)
            or isinstance(chainCount, bool)
            or chainCount < 1
        ):
            raise ValueError("chainCount must be a positive integer")
        if (
            not isinstance(sourceChain, int)
            or isinstance(sourceChain, bool)
            or sourceChain < 0
            or sourceChain >= chainCount
        ):
            raise IndexError("sourceChain is outside the probe matrix")
        probeDelaySamples = cast(
            int, self.parameters["probeDelaySamples"]
        )
        impulseLength = cast(int, self.parameters["impulseLength"])
        probeMatrix = np.zeros(
            (probeDelaySamples + impulseLength, chainCount),
            dtype=np.complex128,
        )
        probeMatrix[probeDelaySamples, sourceChain] = 1.0 + 0.0j
        return FixedPoint(self.width).EncodeComplex(probeMatrix)

    def Measure(
        self,
        channelProcessor: Callable[[np.ndarray], np.ndarray],
        chainCount: int,
        stageName: str,
    ) -> ChannelMeasurementResult:
        """Measure a square MIMO channel through sequential orthogonal probes.

        Processing details:
            Algorithm: Excite one input at a time, decode each complete
            output capture, extract the causal response relative to the probe
            position, assemble ``h[delay, destination, source]``, and derive
            all frequency-domain summaries from that measured tensor.

        Args:
            channelProcessor: Callable accepting and returning waveform arrays.
            chainCount: Equal number of measured source and destination chains.
            stageName: Human-readable label such as ``"pre-PA"``.

        Returns:
            result: Complete impulse/frequency response and path summaries.
        """

        self.ValidateParameters()
        if not callable(channelProcessor):
            raise TypeError("channelProcessor must be callable")
        if (
            not isinstance(chainCount, int)
            or isinstance(chainCount, bool)
            or chainCount < 1
        ):
            raise ValueError("chainCount must be a positive integer")
        if not isinstance(stageName, str) or not stageName.strip():
            raise ValueError("stageName must be a nonempty string")
        impulseLength = cast(int, self.parameters["impulseLength"])
        probeDelaySamples = cast(
            int, self.parameters["probeDelaySamples"]
        )
        impulseResponses = np.zeros(
            (impulseLength, chainCount, chainCount),
            dtype=np.complex128,
        )
        interfaceFormat = FixedPoint(self.width)
        for sourceChain in range(chainCount):
            probeSignal = self.BuildImpulseProbe(
                chainCount, sourceChain
            )
            outputSignal = interfaceFormat.DecodeComplex(
                channelProcessor(probeSignal)
            )
            outputMatrix = np.asarray(
                outputSignal, dtype=np.complex128
            )
            if outputMatrix.ndim == 1 and chainCount == 1:
                outputMatrix = outputMatrix.reshape(-1, 1)
            if (
                outputMatrix.ndim != 2
                or outputMatrix.shape[1] != chainCount
                or outputMatrix.shape[0]
                < probeDelaySamples + impulseLength
                or not np.all(np.isfinite(outputMatrix))
            ):
                raise ValueError(
                    "channelProcessor must return a finite "
                    "samples-by-chain capture covering the probe record"
                )
            impulseResponses[:, :, sourceChain] = outputMatrix[
                probeDelaySamples : probeDelaySamples + impulseLength,
                :,
            ]
        return self.AnalyzeImpulseResponses(
            impulseResponses, stageName
        )

    def AnalyzeImpulseResponses(
        self,
        impulseResponses: np.ndarray,
        stageName: str,
    ) -> ChannelMeasurementResult:
        """Derive path and MIMO conditioning metrics from measured responses.

        Processing details:
            Algorithm: FFT every directed impulse response, restrict metrics
            to the configured occupied band, normalize off-diagonal paths to
            their source-chain direct path, estimate delay from unwrapped
            phase slope, and calculate singular-value condition numbers for
            the full transfer matrix at every occupied frequency.

        Args:
            impulseResponses: Tensor shaped delay-by-destination-by-source.
            stageName: Human-readable measurement-stage label.

        Returns:
            result: Frequency response, per-path values, and worst summaries.
        """

        self.ValidateParameters()
        responseTensor = np.asarray(
            impulseResponses, dtype=np.complex128
        )
        if (
            responseTensor.ndim != 3
            or responseTensor.shape[0] < 2
            or responseTensor.shape[1] != responseTensor.shape[2]
            or responseTensor.shape[0]
            > cast(int, self.parameters["fftLength"])
            or not np.all(np.isfinite(responseTensor))
        ):
            raise ValueError(
                "impulseResponses must be a finite "
                "delay-by-destination-by-source square tensor"
            )
        if not isinstance(stageName, str) or not stageName.strip():
            raise ValueError("stageName must be a nonempty string")
        fftLength = cast(int, self.parameters["fftLength"])
        sampleRateHz = float(self.parameters["sampleRateHz"])
        channelBandwidthHz = float(
            self.parameters["channelBandwidthHz"]
        )
        frequencyResponse = np.fft.fftshift(
            np.fft.fft(responseTensor, n=fftLength, axis=0),
            axes=0,
        )
        frequencyBinsHz = np.fft.fftshift(
            np.fft.fftfreq(fftLength, d=1.0 / sampleRateHz)
        )
        occupiedMask = (
            np.abs(frequencyBinsHz) <= channelBandwidthHz / 2.0
        )
        pathMeasurements = []
        chainCount = responseTensor.shape[1]
        for sourceChain in range(chainCount):
            directResponse = frequencyResponse[
                :, sourceChain, sourceChain
            ]
            for destinationChain in range(chainCount):
                pathResponse = frequencyResponse[
                    :, destinationChain, sourceChain
                ]
                isDirectPath = sourceChain == destinationChain
                relativeResponse = (
                    pathResponse
                    if isDirectPath
                    else pathResponse
                    / self.ProtectMagnitude(directResponse)
                )
                pathMeasurements.append(
                    self.MeasurePath(
                        relativeResponse,
                        frequencyBinsHz,
                        occupiedMask,
                        sourceChain,
                        destinationChain,
                        isDirectPath,
                    )
                )
        conditionNumbers = []
        for frequencyIndex in np.flatnonzero(occupiedMask):
            conditionNumbers.append(
                float(
                    np.linalg.cond(
                        frequencyResponse[frequencyIndex, :, :]
                    )
                )
            )
        finiteConditionNumbers = np.asarray(
            conditionNumbers, dtype=float
        )
        finiteConditionNumbers = finiteConditionNumbers[
            np.isfinite(finiteConditionNumbers)
        ]
        if finiteConditionNumbers.size == 0:
            medianConditionNumber = float("inf")
            worstConditionNumber = float("inf")
        else:
            medianConditionNumber = float(
                np.median(finiteConditionNumbers)
            )
            worstConditionNumber = float(
                np.max(finiteConditionNumbers)
            )
        directFlatnessValues = [
            path.flatnessDb
            for path in pathMeasurements
            if path.isDirectPath
        ]
        detectedCouplingValues = [
            path.gainDb
            for path in pathMeasurements
            if not path.isDirectPath and path.detected
        ]
        detectedFlatnessValues = [
            path.flatnessDb
            for path in pathMeasurements
            if path.detected
        ]
        return ChannelMeasurementResult(
            stageName=stageName.strip(),
            sampleRateHz=sampleRateHz,
            channelBandwidthHz=channelBandwidthHz,
            fftLength=fftLength,
            impulseResponses=responseTensor.copy(),
            frequencyResponse=np.asarray(
                frequencyResponse, dtype=np.complex128
            ),
            frequencyBinsHz=np.asarray(frequencyBinsHz, dtype=float),
            paths=tuple(pathMeasurements),
            worstDirectFlatnessDb=float(max(directFlatnessValues)),
            worstDetectedPathFlatnessDb=float(
                max(detectedFlatnessValues)
            ),
            worstCouplingDb=(
                float(max(detectedCouplingValues))
                if detectedCouplingValues
                else float(self.parameters["magnitudeFloorDb"])
            ),
            medianConditionNumber=medianConditionNumber,
            worstConditionNumber=worstConditionNumber,
        )

    def ProtectMagnitude(self, inputSignal: np.ndarray) -> np.ndarray:
        """Replace only unsafe complex magnitudes while preserving phase.

        Processing details:
            Algorithm: Convert the configured dB floor to linear voltage,
            retain values above it, and replace smaller values by the same
            phase at the floor magnitude so path division cannot overflow.

        Args:
            inputSignal: Complex frequency-response vector.

        Returns:
            result: Finite complex vector with magnitude no smaller than floor.
        """

        complexSignal = np.asarray(
            inputSignal, dtype=np.complex128
        )
        magnitudeFloor = np.power(
            10.0, float(self.parameters["magnitudeFloorDb"]) / 20.0
        )
        magnitude = np.abs(complexSignal)
        phase = np.exp(1j * np.angle(complexSignal))
        return np.where(
            magnitude >= magnitudeFloor,
            complexSignal,
            magnitudeFloor * phase,
        )

    def MeasurePath(
        self,
        relativeResponse: np.ndarray,
        frequencyBinsHz: np.ndarray,
        occupiedMask: np.ndarray,
        sourceChain: int,
        destinationChain: int,
        isDirectPath: bool,
    ) -> ChannelPathMeasurement:
        """Estimate gain, phase, flatness, and group delay for one path.

        Processing details:
            Algorithm: Evaluate voltage magnitude over the occupied band,
            classify off-diagonal paths against the configured threshold,
            take gain and phase at the bin closest to DC, and fit unwrapped
            phase versus frequency only where path energy remains near its
            occupied-band peak.

        Args:
            relativeResponse: Direct response or coupling/direct ratio.
            frequencyBinsHz: Shifted FFT frequency grid.
            occupiedMask: Boolean selection of the analyzed channel band.
            sourceChain: Zero-based excited chain.
            destinationChain: Zero-based observed chain.
            isDirectPath: Whether the path lies on the transfer diagonal.

        Returns:
            result: Scalar path measurement with samples and nanoseconds delay.
        """

        occupiedResponse = relativeResponse[occupiedMask]
        occupiedFrequencyHz = frequencyBinsHz[occupiedMask]
        magnitudeFloorDb = float(self.parameters["magnitudeFloorDb"])
        magnitudeDb = 20.0 * np.log10(
            np.maximum(
                np.abs(occupiedResponse),
                np.power(10.0, magnitudeFloorDb / 20.0),
            )
        )
        peakMagnitudeDb = float(np.max(magnitudeDb))
        detected = isDirectPath or (
            peakMagnitudeDb
            >= float(self.parameters["couplingDetectionThresholdDb"])
        )
        centerIndex = int(np.argmin(np.abs(occupiedFrequencyHz)))
        gainDb = float(magnitudeDb[centerIndex])
        phaseDegrees = float(
            np.degrees(np.angle(occupiedResponse[centerIndex]))
        )
        flatnessDb = float(np.max(magnitudeDb) - np.min(magnitudeDb))
        delayMagnitudeMask = magnitudeDb >= (
            peakMagnitudeDb
            - float(self.parameters["groupDelayMagnitudeRangeDb"])
        )
        if detected and np.count_nonzero(delayMagnitudeMask) >= 3:
            selectedFrequencyHz = occupiedFrequencyHz[
                delayMagnitudeMask
            ]
            selectedPhase = np.unwrap(
                np.angle(occupiedResponse[delayMagnitudeMask])
            )
            phaseSlope = float(
                np.polyfit(selectedFrequencyHz, selectedPhase, 1)[0]
            )
            groupDelaySeconds = -phaseSlope / (2.0 * np.pi)
        else:
            groupDelaySeconds = 0.0
        return ChannelPathMeasurement(
            sourceChain=sourceChain,
            destinationChain=destinationChain,
            isDirectPath=isDirectPath,
            detected=bool(detected),
            gainDb=gainDb,
            phaseDegrees=phaseDegrees,
            flatnessDb=flatnessDb,
            groupDelaySamples=float(
                groupDelaySeconds
                * float(self.parameters["sampleRateHz"])
            ),
            groupDelayNs=float(groupDelaySeconds * 1.0e9),
        )
