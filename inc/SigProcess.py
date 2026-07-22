"""Reusable synchronization and impairment-compensation utilities."""

from collections import ChainMap
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SignalProcessingResult:
    """Store one signal-processing pass and all estimated impairments."""

    processedSignal: np.ndarray
    integerDelaySamples: int
    fractionalDelaySamples: float
    carrierFrequencyOffsetHz: float
    samplingFrequencyOffsetPpm: float
    complexGain: complex

    def ToDict(self) -> Dict[str, float]:
        """Return synchronization estimates in a serialization-ready form.

        Processing details:
            Algorithm: Split the complex gain into real and imaginary parts
            while preserving every timing and frequency estimate numerically.

        Returns:
            result: Dictionary containing scalar impairment estimates. The
                processed sample array is intentionally excluded.
        """

        return {
            "integerDelaySamples": float(self.integerDelaySamples),
            "fractionalDelaySamples": float(self.fractionalDelaySamples),
            "carrierFrequencyOffsetHz": float(
                self.carrierFrequencyOffsetHz
            ),
            "samplingFrequencyOffsetPpm": float(
                self.samplingFrequencyOffsetPpm
            ),
            "complexGainReal": float(np.real(self.complexGain)),
            "complexGainImag": float(np.imag(self.complexGain)),
            "complexGainMagnitude": float(np.abs(self.complexGain)),
            "complexGainPhaseDegrees": float(
                np.degrees(np.angle(self.complexGain))
            ),
        }


class SigProcess:
    """Estimate and compensate deterministic baseband signal impairments.

    The processor uses a known complex reference waveform. A data-aided
    approach is appropriate for this DPD-ILC project because every measured
    PA output is generated from, or captured in response to, that reference.
    Timing and frequency synchronization therefore remain independent from
    the later SNR, EVM, and ACLR definitions.
    """

    def __init__(
        self,
        referenceSignal: np.ndarray,
        sampleRateHz: float,
        parameters: Optional[Mapping[str, object]] = None,
        **parameterOverrides: object,
    ) -> None:
        """Initialize the processor with constructor-internal defaults.

        Processing details:
            Algorithm: Validate the reference, define immutable default
            synchronization settings locally, and layer caller overrides in
            front of those defaults with ``ChainMap`` precedence.

        Args:
            referenceSignal: Known finite one-dimensional complex waveform.
            sampleRateHz: Nominal complex sample rate in samples per second.
            parameters: Optional caller-owned mapping containing only values
                that differ from the internal defaults.
            parameterOverrides: Highest-priority per-instance overrides.

        Returns:
            result: None. Validated reference and configuration state are
                retained for subsequent ``Process`` calls.
        """

        self.defaultParameters: Mapping[str, object] = MappingProxyType(
            {
                "enableIntegerDelayCompensation": True,
                "enableFractionalDelayCompensation": True,
                "enableCarrierFrequencyOffsetCompensation": True,
                "enableSamplingFrequencyOffsetCompensation": True,
                "enableComplexGainCompensation": True,
                "maxIntegerDelaySamples": None,
                "maxCarrierFrequencyOffsetHz": None,
                "maxSamplingFrequencyOffsetPpm": 200.0,
                "timingWindowCount": 9,
                "timingWindowLength": 2048,
                "interpolationHalfLength": 12,
            }
        )
        self.referenceSignal = self.ValidateSignal(
            referenceSignal, "referenceSignal"
        )
        if (
            not isinstance(sampleRateHz, (int, float))
            or isinstance(sampleRateHz, bool)
            or not np.isfinite(sampleRateHz)
            or sampleRateHz <= 0.0
        ):
            raise ValueError("sampleRateHz must be finite and positive")
        self.sampleRateHz = float(sampleRateHz)
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping or None")
        externalParameters = {} if parameters is None else parameters
        self.parameters: ChainMap[str, object] = ChainMap(
            dict(parameterOverrides),
            externalParameters,
            self.defaultParameters,
        )
        self.ValidateParameters()
        self.lastResult: Optional[SignalProcessingResult] = None

    @staticmethod
    def ValidateSignal(inputSignal: np.ndarray, signalName: str) -> np.ndarray:
        """Convert and validate one finite complex baseband signal.

        Processing details:
            Algorithm: Flatten the input into a deterministic one-dimensional
            complex128 array and reject empty or non-finite data.

        Args:
            inputSignal: Array-like complex samples to validate.
            signalName: Human-readable name used in validation messages.

        Returns:
            result: One-dimensional finite complex128 sample array.
        """

        complexSignal = np.asarray(
            inputSignal, dtype=np.complex128
        ).reshape(-1)
        if complexSignal.size == 0:
            raise ValueError(f"{signalName} cannot be empty")
        if not np.all(np.isfinite(complexSignal)):
            raise ValueError(f"{signalName} contains NaN or infinite values")
        return complexSignal

    def GetParameters(self) -> Dict[str, object]:
        """Return a flattened snapshot of effective processing parameters.

        Processing details:
            Algorithm: Resolve all ``ChainMap`` layers and copy the result so
            the returned dictionary cannot mutate processor state.

        Returns:
            result: Dictionary containing every effective setting.
        """

        return dict(self.parameters)

    def UpdateParameters(self, **parameterOverrides: object) -> None:
        """Apply validated highest-priority processing overrides.

        Processing details:
            Algorithm: Update the local ``ChainMap`` layer transactionally and
            restore its prior contents when validation fails.

        Args:
            parameterOverrides: Supported settings to update.

        Returns:
            result: None. Valid values affect subsequent processing calls.
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
        """Validate all resolved synchronization parameters.

        Processing details:
            Algorithm: Reject unknown keys first, then validate switches,
            physical search limits, estimator window sizes, and interpolation
            support in a deterministic order.

        Returns:
            result: None. Invalid settings raise a descriptive exception.
        """

        unknownParameters = set(self.parameters).difference(
            self.defaultParameters
        )
        if unknownParameters:
            unknownNames = ", ".join(
                sorted(str(parameterName) for parameterName in unknownParameters)
            )
            raise TypeError(f"unknown SigProcess parameters: {unknownNames}")

        switchNames = (
            "enableIntegerDelayCompensation",
            "enableFractionalDelayCompensation",
            "enableCarrierFrequencyOffsetCompensation",
            "enableSamplingFrequencyOffsetCompensation",
            "enableComplexGainCompensation",
        )
        for switchName in switchNames:
            if not isinstance(self.parameters[switchName], bool):
                raise TypeError(f"{switchName} must be boolean")

        maxIntegerDelaySamples = self.parameters["maxIntegerDelaySamples"]
        if maxIntegerDelaySamples is not None and (
            not isinstance(maxIntegerDelaySamples, int)
            or isinstance(maxIntegerDelaySamples, bool)
            or maxIntegerDelaySamples < 0
        ):
            raise ValueError(
                "maxIntegerDelaySamples must be a nonnegative integer or None"
            )

        maxCarrierFrequencyOffsetHz = self.parameters[
            "maxCarrierFrequencyOffsetHz"
        ]
        if maxCarrierFrequencyOffsetHz is not None and (
            not isinstance(maxCarrierFrequencyOffsetHz, (int, float))
            or isinstance(maxCarrierFrequencyOffsetHz, bool)
            or not np.isfinite(maxCarrierFrequencyOffsetHz)
            or maxCarrierFrequencyOffsetHz <= 0.0
            or maxCarrierFrequencyOffsetHz >= self.sampleRateHz / 2.0
        ):
            raise ValueError(
                "maxCarrierFrequencyOffsetHz must be positive, below Nyquist, or None"
            )

        maxSamplingFrequencyOffsetPpm = self.parameters[
            "maxSamplingFrequencyOffsetPpm"
        ]
        if (
            not isinstance(maxSamplingFrequencyOffsetPpm, (int, float))
            or isinstance(maxSamplingFrequencyOffsetPpm, bool)
            or not np.isfinite(maxSamplingFrequencyOffsetPpm)
            or maxSamplingFrequencyOffsetPpm < 0.0
        ):
            raise ValueError(
                "maxSamplingFrequencyOffsetPpm must be finite and nonnegative"
            )

        integerMinimums = {
            "timingWindowCount": 3,
            "timingWindowLength": 32,
            "interpolationHalfLength": 2,
        }
        for parameterName, minimumValue in integerMinimums.items():
            parameterValue = self.parameters[parameterName]
            if (
                not isinstance(parameterValue, int)
                or isinstance(parameterValue, bool)
                or parameterValue < minimumValue
            ):
                raise ValueError(
                    f"{parameterName} must be an integer of at least {minimumValue}"
                )

    def ResolveMaximumIntegerDelay(self) -> int:
        """Resolve the configured or automatic integer-delay search radius.

        Processing details:
            Algorithm: Use the explicit caller limit when provided; otherwise
            search up to one quarter of the reference with a 4096-sample cap.

        Returns:
            result: Nonnegative integer lag radius in samples.
        """

        configuredMaximum = self.parameters["maxIntegerDelaySamples"]
        if configuredMaximum is not None:
            return min(int(configuredMaximum), self.referenceSignal.size - 1)
        automaticMaximum = max(32, self.referenceSignal.size // 4)
        return min(4096, automaticMaximum, self.referenceSignal.size - 1)

    def EstimateIntegerDelay(self, measuredSignal: np.ndarray) -> int:
        """Estimate the signed integer delay of measurement versus reference.

        A positive result means that the measured waveform occurs later and
        must be sampled at ``n + delay`` to align it with reference sample
        ``n``.

        Processing details:
            Algorithm: Compute linear cross-correlation with an FFT, restrict
            the lag search, normalize every candidate by overlap energy, and
            select the maximum normalized magnitude.

        Args:
            measuredSignal: Finite measured complex waveform.

        Returns:
            result: Signed integer delay in nominal samples.
        """

        complexMeasured = self.ValidateSignal(
            measuredSignal, "measuredSignal"
        )
        referenceLength = self.referenceSignal.size
        measuredLength = complexMeasured.size
        fullLength = referenceLength + measuredLength - 1
        fftLength = 1 << int(np.ceil(np.log2(max(fullLength, 2))))
        correlation = np.fft.ifft(
            np.fft.fft(complexMeasured, fftLength)
            * np.fft.fft(np.conj(self.referenceSignal[::-1]), fftLength)
        )[:fullLength]
        lags = np.arange(fullLength, dtype=int) - (referenceLength - 1)
        maximumDelay = self.ResolveMaximumIntegerDelay()
        minimumOverlap = max(16, min(referenceLength, measuredLength) // 4)
        referenceEnergyPrefix = np.r_[
            0.0, np.cumsum(np.abs(self.referenceSignal) ** 2)
        ]
        measuredEnergyPrefix = np.r_[
            0.0, np.cumsum(np.abs(complexMeasured) ** 2)
        ]
        bestScore = -np.inf
        bestLag = 0

        for correlationIndex, lagValue in enumerate(lags):
            if abs(int(lagValue)) > maximumDelay:
                continue
            referenceStart = max(0, -int(lagValue))
            referenceStop = min(
                referenceLength, measuredLength - int(lagValue)
            )
            overlapLength = referenceStop - referenceStart
            if overlapLength < minimumOverlap:
                continue
            measuredStart = referenceStart + int(lagValue)
            measuredStop = measuredStart + overlapLength
            referenceEnergy = (
                referenceEnergyPrefix[referenceStop]
                - referenceEnergyPrefix[referenceStart]
            )
            measuredEnergy = (
                measuredEnergyPrefix[measuredStop]
                - measuredEnergyPrefix[measuredStart]
            )
            normalization = np.sqrt(
                max(referenceEnergy * measuredEnergy, np.finfo(float).tiny)
            )
            candidateScore = abs(correlation[correlationIndex]) / normalization
            if candidateScore > bestScore:
                bestScore = float(candidateScore)
                bestLag = int(lagValue)

        if not np.isfinite(bestScore):
            raise RuntimeError("unable to estimate integer delay")
        return bestLag

    def ExtractIntegerAligned(
        self, measuredSignal: np.ndarray, integerDelaySamples: int
    ) -> np.ndarray:
        """Extract a reference-length signal using one signed integer delay.

        Processing details:
            Algorithm: Map reference index ``n`` to measured index
            ``n + integerDelaySamples`` and zero-fill unavailable boundaries.

        Args:
            measuredSignal: Finite measured complex waveform.
            integerDelaySamples: Signed delay estimate in samples.

        Returns:
            result: Reference-length integer-aligned complex array.
        """

        complexMeasured = self.ValidateSignal(
            measuredSignal, "measuredSignal"
        )
        outputSignal = np.zeros_like(self.referenceSignal)
        referenceStart = max(0, -integerDelaySamples)
        referenceStop = min(
            self.referenceSignal.size,
            complexMeasured.size - integerDelaySamples,
        )
        if referenceStop <= referenceStart:
            raise ValueError("integer delay leaves no overlapping samples")
        measuredStart = referenceStart + integerDelaySamples
        measuredStop = measuredStart + (referenceStop - referenceStart)
        outputSignal[referenceStart:referenceStop] = complexMeasured[
            measuredStart:measuredStop
        ]
        return outputSignal

    def EstimateCarrierFrequencyOffset(
        self,
        integerAlignedSignal: np.ndarray,
    ) -> float:
        """Estimate data-aided carrier-frequency offset from block gains.

        Processing details:
            Algorithm: Estimate one least-squares complex gain in each of
            several time windows, unwrap the gain phases, fit their weighted
            linear slope versus sample index, convert radians per sample to
            hertz, and apply both a resolution deadband and the search bound.
            Block gains suppress nonlinear sample-to-sample PA phase changes
            that would otherwise resemble a false carrier offset.

        Args:
            integerAlignedSignal: Reference-length measurement after coarse
                integer-delay alignment.

        Returns:
            result: Estimated carrier-frequency offset in hertz.
        """

        complexAligned = self.ValidateSignal(
            integerAlignedSignal, "integerAlignedSignal"
        )
        if complexAligned.size != self.referenceSignal.size:
            raise ValueError(
                "integerAlignedSignal must match the reference length"
            )
        signalLength = self.referenceSignal.size
        requestedWindowLength = int(self.parameters["timingWindowLength"])
        windowLength = min(
            requestedWindowLength,
            max(64, signalLength // 12),
        )
        if windowLength >= signalLength:
            windowLength = max(16, signalLength // 3)
        halfWindow = windowLength // 2
        centerCount = min(
            max(int(self.parameters["timingWindowCount"]), 5),
            max(5, signalLength // max(windowLength, 1)),
        )
        firstCenter = halfWindow
        lastCenter = signalLength - halfWindow - 1
        if lastCenter <= firstCenter:
            return 0.0
        centerIndices = np.linspace(
            firstCenter, lastCenter, centerCount
        ).round().astype(int)
        validCenters = []
        blockPhases = []
        blockWeights = []
        for centerIndex in centerIndices:
            startIndex = int(centerIndex) - halfWindow
            stopIndex = startIndex + windowLength
            referenceWindow = self.referenceSignal[startIndex:stopIndex]
            measuredWindow = complexAligned[startIndex:stopIndex]
            referenceEnergy = float(
                np.vdot(referenceWindow, referenceWindow).real
            )
            measuredEnergy = float(
                np.vdot(measuredWindow, measuredWindow).real
            )
            if (
                referenceEnergy <= np.finfo(float).tiny
                or measuredEnergy <= np.finfo(float).tiny
            ):
                continue
            gainNumerator = np.vdot(referenceWindow, measuredWindow)
            normalizedCorrelation = abs(gainNumerator) / np.sqrt(
                referenceEnergy * measuredEnergy
            )
            validCenters.append(float(centerIndex))
            blockPhases.append(float(np.angle(gainNumerator)))
            blockWeights.append(max(float(normalizedCorrelation), 1.0e-6))

        if len(validCenters) < 3:
            return 0.0
        centerArray = np.asarray(validCenters, dtype=float)
        phaseArray = np.unwrap(np.asarray(blockPhases, dtype=float))
        weightArray = np.asarray(blockWeights, dtype=float) ** 2
        weightedCenter = np.average(centerArray, weights=weightArray)
        weightedPhase = np.average(phaseArray, weights=weightArray)
        centeredCoordinates = centerArray - weightedCenter
        slopeDenominator = np.sum(weightArray * centeredCoordinates**2)
        if slopeDenominator <= np.finfo(float).tiny:
            return 0.0
        radiansPerSample = np.sum(
            weightArray
            * centeredCoordinates
            * (phaseArray - weightedPhase)
        ) / slopeDenominator
        frequencyOffsetHz = (
            radiansPerSample * self.sampleRateHz / (2.0 * np.pi)
        )
        frequencyResolutionHz = self.sampleRateHz / max(signalLength, 1)
        if abs(frequencyOffsetHz) < 0.1 * frequencyResolutionHz:
            frequencyOffsetHz = 0.0
        configuredMaximum = self.parameters["maxCarrierFrequencyOffsetHz"]
        maximumOffsetHz = (
            self.sampleRateHz / 4.0
            if configuredMaximum is None
            else float(configuredMaximum)
        )
        return float(
            np.clip(frequencyOffsetHz, -maximumOffsetHz, maximumOffsetHz)
        )

    def CompensateCarrierFrequencyOffset(
        self, measuredSignal: np.ndarray, frequencyOffsetHz: float
    ) -> np.ndarray:
        """Remove a carrier-frequency offset from measured samples.

        Processing details:
            Algorithm: Multiply measured sample ``m`` by
            ``exp(-j*2*pi*frequencyOffsetHz*m/sampleRateHz)``.

        Args:
            measuredSignal: Finite measured complex waveform.
            frequencyOffsetHz: Offset estimate in hertz.

        Returns:
            result: Frequency-corrected complex waveform of unchanged length.
        """

        complexMeasured = self.ValidateSignal(
            measuredSignal, "measuredSignal"
        )
        sampleIndices = np.arange(complexMeasured.size, dtype=float)
        correction = np.exp(
            -1j
            * 2.0
            * np.pi
            * float(frequencyOffsetHz)
            * sampleIndices
            / self.sampleRateHz
        )
        return complexMeasured * correction

    @staticmethod
    def RefineCorrelationPeak(
        lowerScore: float, centerScore: float, upperScore: float
    ) -> float:
        """Refine one discrete correlation maximum with a parabola.

        Processing details:
            Algorithm: Fit a three-point quadratic around the integer maximum
            and bound the vertex to half a sample for numerical robustness.

        Args:
            lowerScore: Correlation magnitude at lag ``k-1``.
            centerScore: Correlation magnitude at lag ``k``.
            upperScore: Correlation magnitude at lag ``k+1``.

        Returns:
            result: Fractional correction in the interval ``[-0.5, 0.5]``.
        """

        denominator = lowerScore - 2.0 * centerScore + upperScore
        if abs(denominator) <= np.finfo(float).eps:
            return 0.0
        peakOffset = 0.5 * (lowerScore - upperScore) / denominator
        return float(np.clip(peakOffset, -0.5, 0.5))

    def EstimateTimingOffsets(
        self,
        frequencyCorrectedSignal: np.ndarray,
        integerDelaySamples: int,
    ) -> Tuple[int, float, float]:
        """Estimate residual fractional delay and sampling-frequency offset.

        Processing details:
            Algorithm: Correlate several reference windows against local
            measured windows, refine each timing peak to sub-sample accuracy,
            and fit ``delay(n) = fractionalDelay + slope*n``. The slope is
            reported in parts per million and the intercept is normalized to
            a signed half-sample interval by adjusting the integer delay.

        Args:
            frequencyCorrectedSignal: Measured waveform after CFO removal.
            integerDelaySamples: Coarse signed integer-delay estimate.

        Returns:
            result: Tuple containing adjusted integer delay, fractional delay
                in samples, and sampling-frequency offset in ppm.
        """

        complexMeasured = self.ValidateSignal(
            frequencyCorrectedSignal, "frequencyCorrectedSignal"
        )
        referenceLength = self.referenceSignal.size
        requestedWindowLength = int(self.parameters["timingWindowLength"])
        windowLength = min(requestedWindowLength, max(32, referenceLength // 4))
        if windowLength >= referenceLength:
            windowLength = max(16, referenceLength // 2)
        windowCount = min(
            int(self.parameters["timingWindowCount"]),
            max(3, referenceLength // max(windowLength, 1)),
        )
        maximumSamplingOffsetPpm = float(
            self.parameters["maxSamplingFrequencyOffsetPpm"]
        )
        maximumDriftSamples = (
            maximumSamplingOffsetPpm * referenceLength / 1.0e6
        )
        localSearchRadius = max(2, int(np.ceil(maximumDriftSamples)) + 2)
        halfWindow = windowLength // 2
        firstCenter = halfWindow + localSearchRadius
        lastCenter = referenceLength - halfWindow - localSearchRadius - 1
        if lastCenter <= firstCenter:
            return integerDelaySamples, 0.0, 0.0
        centerIndices = np.linspace(
            firstCenter,
            lastCenter,
            windowCount,
        ).round().astype(int)
        timingCenters = []
        timingDelays = []
        timingWeights = []

        for centerIndex in centerIndices:
            referenceStart = int(centerIndex) - halfWindow
            referenceStop = referenceStart + windowLength
            referenceWindow = self.referenceSignal[
                referenceStart:referenceStop
            ]
            referenceEnergy = max(
                float(np.vdot(referenceWindow, referenceWindow).real),
                np.finfo(float).tiny,
            )
            lagValues = np.arange(
                -localSearchRadius,
                localSearchRadius + 1,
                dtype=int,
            )
            lagScores = np.full(lagValues.size, -np.inf, dtype=float)
            for lagIndex, localLag in enumerate(lagValues):
                measuredStart = (
                    referenceStart + integerDelaySamples + int(localLag)
                )
                measuredStop = measuredStart + windowLength
                if measuredStart < 0 or measuredStop > complexMeasured.size:
                    continue
                measuredWindow = complexMeasured[measuredStart:measuredStop]
                measuredEnergy = max(
                    float(np.vdot(measuredWindow, measuredWindow).real),
                    np.finfo(float).tiny,
                )
                lagScores[lagIndex] = abs(
                    np.vdot(referenceWindow, measuredWindow)
                ) / np.sqrt(referenceEnergy * measuredEnergy)

            peakIndex = int(np.argmax(lagScores))
            if not np.isfinite(lagScores[peakIndex]):
                continue
            fractionalPeak = 0.0
            if 0 < peakIndex < lagScores.size - 1:
                fractionalPeak = self.RefineCorrelationPeak(
                    float(lagScores[peakIndex - 1]),
                    float(lagScores[peakIndex]),
                    float(lagScores[peakIndex + 1]),
                )
            timingCenters.append(float(centerIndex))
            timingDelays.append(
                float(lagValues[peakIndex]) + fractionalPeak
            )
            timingWeights.append(max(float(lagScores[peakIndex]), 1.0e-6))

        if not timingCenters:
            return integerDelaySamples, 0.0, 0.0
        centerArray = np.asarray(timingCenters, dtype=float)
        delayArray = np.asarray(timingDelays, dtype=float)
        weightArray = np.asarray(timingWeights, dtype=float) ** 2
        enableSamplingOffset = bool(
            self.parameters["enableSamplingFrequencyOffsetCompensation"]
        )
        if enableSamplingOffset and centerArray.size >= 3:
            weightedCenter = np.average(centerArray, weights=weightArray)
            weightedDelay = np.average(delayArray, weights=weightArray)
            centeredCoordinates = centerArray - weightedCenter
            slopeDenominator = np.sum(
                weightArray * centeredCoordinates**2
            )
            if slopeDenominator > np.finfo(float).tiny:
                timingSlope = np.sum(
                    weightArray
                    * centeredCoordinates
                    * (delayArray - weightedDelay)
                ) / slopeDenominator
            else:
                timingSlope = 0.0
            maximumSlope = maximumSamplingOffsetPpm / 1.0e6
            timingSlope = float(
                np.clip(timingSlope, -maximumSlope, maximumSlope)
            )
            timingIntercept = weightedDelay - timingSlope * weightedCenter
        else:
            timingSlope = 0.0
            timingIntercept = float(np.median(delayArray))

        enableFractionalDelay = bool(
            self.parameters["enableFractionalDelayCompensation"]
        )
        if enableFractionalDelay:
            integerAdjustment = int(np.floor(timingIntercept + 0.5))
            fractionalDelaySamples = timingIntercept - integerAdjustment
        else:
            integerAdjustment = int(np.round(timingIntercept))
            fractionalDelaySamples = 0.0
        adjustedIntegerDelay = integerDelaySamples + integerAdjustment
        samplingFrequencyOffsetPpm = timingSlope * 1.0e6
        return (
            adjustedIntegerDelay,
            float(fractionalDelaySamples),
            float(samplingFrequencyOffsetPpm),
        )

    def InterpolateSignal(
        self,
        inputSignal: np.ndarray,
        samplePositions: np.ndarray,
    ) -> np.ndarray:
        """Evaluate a complex signal at arbitrary fractional sample positions.

        Processing details:
            Algorithm: Return exact indexed values for an all-integer grid;
            otherwise apply a normalized finite-support Lanczos sinc kernel in
            bounded chunks and zero-fill positions outside the input record.

        Args:
            inputSignal: Finite one-dimensional complex samples.
            samplePositions: Floating-point source indices for each output.

        Returns:
            result: Complex samples evaluated at the requested positions.
        """

        complexInput = self.ValidateSignal(inputSignal, "inputSignal")
        positionArray = np.asarray(samplePositions, dtype=float).reshape(-1)
        if not np.all(np.isfinite(positionArray)):
            raise ValueError("samplePositions contains NaN or infinite values")
        roundedPositions = np.rint(positionArray).astype(np.int64)
        if np.all(np.abs(positionArray - roundedPositions) < 1.0e-12):
            outputSignal = np.zeros(positionArray.size, dtype=np.complex128)
            validMask = (
                (roundedPositions >= 0)
                & (roundedPositions < complexInput.size)
            )
            outputSignal[validMask] = complexInput[
                roundedPositions[validMask]
            ]
            return outputSignal

        halfLength = int(self.parameters["interpolationHalfLength"])
        tapOffsets = np.arange(-halfLength + 1, halfLength + 1)
        outputSignal = np.zeros(positionArray.size, dtype=np.complex128)
        chunkLength = 32768
        for startIndex in range(0, positionArray.size, chunkLength):
            stopIndex = min(startIndex + chunkLength, positionArray.size)
            chunkPositions = positionArray[startIndex:stopIndex]
            centerIndices = np.floor(chunkPositions).astype(np.int64)
            sourceIndices = centerIndices[:, None] + tapOffsets[None, :]
            distances = chunkPositions[:, None] - sourceIndices
            interpolationWeights = (
                np.sinc(distances)
                * np.sinc(distances / float(halfLength))
            )
            validMask = (
                (sourceIndices >= 0)
                & (sourceIndices < complexInput.size)
                & (np.abs(distances) < halfLength)
            )
            interpolationWeights *= validMask
            weightSums = np.sum(interpolationWeights, axis=1)
            safeWeightSums = np.where(
                np.abs(weightSums) > np.finfo(float).eps,
                weightSums,
                1.0,
            )
            clippedIndices = np.clip(
                sourceIndices, 0, complexInput.size - 1
            )
            outputSignal[startIndex:stopIndex] = np.sum(
                complexInput[clippedIndices] * interpolationWeights,
                axis=1,
            ) / safeWeightSums
        return outputSignal

    @staticmethod
    def EstimateComplexGain(
        referenceSignal: np.ndarray, measuredSignal: np.ndarray
    ) -> complex:
        """Estimate the least-squares gain mapping reference to measurement.

        Processing details:
            Algorithm: Evaluate ``reference^H * measured`` divided by
            ``reference^H * reference`` with a positive numerical floor.

        Args:
            referenceSignal: Known finite complex samples.
            measuredSignal: Aligned finite complex samples of equal length.

        Returns:
            result: Least-squares complex gain applied by the measured path.
        """

        complexReference = SigProcess.ValidateSignal(
            referenceSignal, "referenceSignal"
        )
        complexMeasured = SigProcess.ValidateSignal(
            measuredSignal, "measuredSignal"
        )
        if complexReference.size != complexMeasured.size:
            raise ValueError(
                "referenceSignal and measuredSignal must have equal length"
            )
        denominator = max(
            float(np.vdot(complexReference, complexReference).real),
            np.finfo(float).tiny,
        )
        return complex(
            np.vdot(complexReference, complexMeasured) / denominator
        )

    @staticmethod
    def ResolveEstimationSlice(
        estimationSlice: Optional[slice], signalLength: int
    ) -> slice:
        """Normalize an optional gain-estimation slice to valid boundaries.

        Processing details:
            Algorithm: Expand ``None`` to the complete signal, apply standard
            slice boundary normalization, and reject empty or strided regions.

        Args:
            estimationSlice: Optional contiguous region used for gain fitting.
            signalLength: Positive available reference length.

        Returns:
            result: Valid contiguous unit-step slice.
        """

        if estimationSlice is None:
            return slice(0, signalLength, 1)
        if not isinstance(estimationSlice, slice):
            raise TypeError("estimationSlice must be a slice or None")
        startIndex, stopIndex, stepSize = estimationSlice.indices(signalLength)
        if stepSize != 1:
            raise ValueError("estimationSlice must use a unit step")
        if stopIndex <= startIndex:
            raise ValueError("estimationSlice cannot be empty")
        return slice(startIndex, stopIndex, 1)

    def Process(
        self,
        measuredSignal: np.ndarray,
        estimationSlice: Optional[slice] = None,
    ) -> SignalProcessingResult:
        """Estimate and compensate all enabled signal impairments.

        Processing details:
            Algorithm: Estimate coarse integer timing, remove CFO, estimate
            residual fractional timing and sample-rate drift, interpolate the
            measured record onto the reference grid, estimate complex gain on
            the requested region, and divide out that gain. Every impairment
            switch can disable its corresponding estimate and correction.

        Args:
            measuredSignal: Captured or simulated complex waveform. Its length
                may differ from the reference because alignment performs the
                final extraction.
            estimationSlice: Optional reference-grid region used to estimate
                complex gain, normally the Wi-Fi data field.

        Returns:
            result: ``SignalProcessingResult`` containing the compensated
                reference-length signal and all scalar estimates.
        """

        self.ValidateParameters()
        complexMeasured = self.ValidateSignal(
            measuredSignal, "measuredSignal"
        )
        enableIntegerDelay = bool(
            self.parameters["enableIntegerDelayCompensation"]
        )
        integerDelaySamples = (
            self.EstimateIntegerDelay(complexMeasured)
            if enableIntegerDelay
            else 0
        )
        integerAlignedSignal = self.ExtractIntegerAligned(
            complexMeasured, integerDelaySamples
        )

        enableCarrierFrequencyOffset = bool(
            self.parameters["enableCarrierFrequencyOffsetCompensation"]
        )
        carrierFrequencyOffsetHz = (
            self.EstimateCarrierFrequencyOffset(integerAlignedSignal)
            if enableCarrierFrequencyOffset
            else 0.0
        )
        frequencyCorrectedSignal = self.CompensateCarrierFrequencyOffset(
            complexMeasured, carrierFrequencyOffsetHz
        )

        enableFractionalDelay = bool(
            self.parameters["enableFractionalDelayCompensation"]
        )
        enableSamplingOffset = bool(
            self.parameters["enableSamplingFrequencyOffsetCompensation"]
        )
        coarseAlignedSignal = self.ExtractIntegerAligned(
            frequencyCorrectedSignal, integerDelaySamples
        )
        coarseGain = self.EstimateComplexGain(
            self.referenceSignal, coarseAlignedSignal
        )
        coarseError = coarseAlignedSignal - coarseGain * self.referenceSignal
        coarseErrorRatio = np.sum(np.abs(coarseError) ** 2) / max(
            np.sum(np.abs(coarseGain * self.referenceSignal) ** 2),
            np.finfo(float).tiny,
        )
        coarseAlignmentIsExact = coarseErrorRatio < 1.0e-24
        if (
            (enableFractionalDelay or enableSamplingOffset)
            and not coarseAlignmentIsExact
        ):
            (
                integerDelaySamples,
                fractionalDelaySamples,
                samplingFrequencyOffsetPpm,
            ) = self.EstimateTimingOffsets(
                frequencyCorrectedSignal,
                integerDelaySamples,
            )
            # Correlation interpolation has a finite numerical floor even for
            # an exactly aligned record. Deadbands preserve bit-identical
            # ideal paths without masking physically meaningful impairments.
            if abs(fractionalDelaySamples) < 1.0e-3:
                fractionalDelaySamples = 0.0
            if abs(samplingFrequencyOffsetPpm) < 5.0e-2:
                samplingFrequencyOffsetPpm = 0.0
        else:
            fractionalDelaySamples = 0.0
            samplingFrequencyOffsetPpm = 0.0

        referenceIndices = np.arange(
            self.referenceSignal.size, dtype=float
        )
        sourcePositions = (
            float(integerDelaySamples)
            + float(fractionalDelaySamples)
            + referenceIndices
            * (1.0 + float(samplingFrequencyOffsetPpm) / 1.0e6)
        )
        alignedSignal = self.InterpolateSignal(
            frequencyCorrectedSignal, sourcePositions
        )

        gainSlice = self.ResolveEstimationSlice(
            estimationSlice, self.referenceSignal.size
        )
        enableComplexGain = bool(
            self.parameters["enableComplexGainCompensation"]
        )
        if enableComplexGain:
            complexGain = self.EstimateComplexGain(
                self.referenceSignal[gainSlice],
                alignedSignal[gainSlice],
            )
            if abs(complexGain) <= np.finfo(float).tiny:
                raise ValueError("estimated complex gain is numerically zero")
            processedSignal = alignedSignal / complexGain
        else:
            complexGain = 1.0 + 0.0j
            processedSignal = alignedSignal

        self.lastResult = SignalProcessingResult(
            processedSignal=processedSignal,
            integerDelaySamples=int(integerDelaySamples),
            fractionalDelaySamples=float(fractionalDelaySamples),
            carrierFrequencyOffsetHz=float(carrierFrequencyOffsetHz),
            samplingFrequencyOffsetPpm=float(samplingFrequencyOffsetPpm),
            complexGain=complex(complexGain),
        )
        return self.lastResult
