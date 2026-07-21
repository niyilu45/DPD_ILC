"""Frequency-domain ILC and GMP predistorter fitting.

The waveform ILC follows the regularized update from the theory document:

``U[k+1] = Q(U[k] + L[k] * E[k])``

``L[k] = mu * conj(H[k]) / (abs(H[k])**2 + lambda)``

After convergence, a generalized memory polynomial is fitted to map the
original Wi-Fi waveform onto the learned PA input. This converts waveform-
specific ILC labels into a reusable deployable predistorter.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .PaModel import AddAwgn


@dataclass(frozen=True)
class ILCConfig:
    """Configure regularized frequency-domain iterative learning control."""

    numIterations: int = 8
    learningRate: float = 0.15
    regularization: float = 1e-3
    maxAmplitude: float = 2.0
    feedbackSnrDb: Optional[float] = None
    feedbackAverages: int = 1
    projectionBandwidthFactor: float = 1.6
    responseFloorDb: float = -45.0
    randomSeed: int = 19

    def Validate(self) -> None:
        """Validate convergence, regularization, and feedback parameters."""

        if self.numIterations < 1:
            raise ValueError("numIterations must be positive")
        if not 0.0 < self.learningRate < 2.0:
            raise ValueError("learningRate must be between 0 and 2")
        if self.regularization <= 0.0:
            raise ValueError("regularization must be positive")
        if self.maxAmplitude <= 0.0:
            raise ValueError("maxAmplitude must be positive")
        if self.feedbackAverages < 1:
            raise ValueError("feedbackAverages must be positive")
        if self.projectionBandwidthFactor <= 1.0:
            raise ValueError("projectionBandwidthFactor must exceed 1")


@dataclass(frozen=True)
class ILCIteration:
    """Store one iteration of ILC convergence diagnostics."""

    iteration: int
    errorRms: float
    nmseDb: float
    inputPeak: float


@dataclass
class ILCResult:
    """Return the learned predistorted waveform and convergence history."""

    learnedInput: np.ndarray
    outputSignal: np.ndarray
    history: List[ILCIteration]


def _NextPowerOfTwo(value: int) -> int:
    """Return the smallest power of two greater than or equal to value."""

    return 1 << max(0, int(value - 1).bit_length())


def _LimitAmplitude(inputSignal: np.ndarray, maxAmplitude: float) -> np.ndarray:
    """Project each complex sample onto the requested peak-amplitude disk."""

    sampleMagnitude = np.abs(inputSignal)
    limitedSignal = inputSignal.copy()
    overLimit = sampleMagnitude > maxAmplitude
    if np.any(overLimit):
        limitedSignal[overLimit] *= maxAmplitude / sampleMagnitude[overLimit]
    return limitedSignal


def _MeasurePaOutput(
    paModel,
    inputSignal: np.ndarray,
    config: ILCConfig,
    randomGenerator: np.random.Generator,
) -> np.ndarray:
    """Average repeated noisy feedback captures of the same PA waveform."""

    accumulatedOutput = np.zeros_like(inputSignal, dtype=np.complex128)
    for _ in range(config.feedbackAverages):
        noiselessOutput = paModel.Process(inputSignal)
        accumulatedOutput += AddAwgn(
            noiselessOutput, config.feedbackSnrDb, randomGenerator
        )
    return accumulatedOutput / float(config.feedbackAverages)


def RunFrequencyDomainIlc(
    referenceSignal: np.ndarray,
    paModel,
    sampleRateHz: float,
    channelBandwidthHz: float,
    config: ILCConfig = ILCConfig(),
) -> ILCResult:
    """Learn a PA input waveform using regularized frequency-domain ILC.

    The local transfer response is estimated from the current PA input and
    feedback spectra. Low-excitation FFT bins smoothly fall back to a scalar
    least-squares gain, preventing division by spectral nulls. The update is
    projected onto a bandwidth wider than the wanted channel so the ILC can
    synthesize out-of-band cancellation components needed to reduce ACLR.
    """

    config.Validate()
    targetSignal = np.asarray(referenceSignal, dtype=np.complex128).reshape(-1)
    if targetSignal.size == 0:
        raise ValueError("referenceSignal cannot be empty")
    if sampleRateHz <= 0.0 or channelBandwidthHz <= 0.0:
        raise ValueError("sampleRateHz and channelBandwidthHz must be positive")
    if sampleRateHz < 2.0 * channelBandwidthHz:
        raise ValueError("ILC requires at least 2x waveform oversampling")

    randomGenerator = np.random.default_rng(config.randomSeed)
    inputSignal = _LimitAmplitude(targetSignal.copy(), config.maxAmplitude)
    fftLength = _NextPowerOfTwo(targetSignal.size)
    frequencyBins = np.fft.fftfreq(fftLength, d=1.0 / sampleRateHz)

    # A raised-cosine-like transition avoids sharp frequency truncation that
    # would otherwise create long time-domain ringing around packet edges.
    passbandEdge = 0.45 * channelBandwidthHz * config.projectionBandwidthFactor
    stopbandEdge = 0.50 * channelBandwidthHz * config.projectionBandwidthFactor
    absoluteFrequency = np.abs(frequencyBins)
    projectionMask = np.ones(fftLength, dtype=float)
    projectionMask[absoluteFrequency >= stopbandEdge] = 0.0
    transitionBins = (absoluteFrequency > passbandEdge) & (
        absoluteFrequency < stopbandEdge
    )
    transitionPhase = (
        absoluteFrequency[transitionBins] - passbandEdge
    ) / max(stopbandEdge - passbandEdge, np.finfo(float).tiny)
    projectionMask[transitionBins] = 0.5 * (
        1.0 + np.cos(np.pi * transitionPhase)
    )

    history: List[ILCIteration] = []
    targetPower = max(
        np.mean(np.abs(targetSignal) ** 2), np.finfo(float).tiny
    )

    # Estimate H(f) once at a low-power operating point. A low-level probe
    # captures the PA memory response without allowing nonlinear spectral
    # regrowth to bias Y(f)/U(f). In a laboratory workflow this corresponds
    # to the small-signal frequency-response measurement performed before ILC.
    targetRms = np.sqrt(targetPower)
    probeRms = min(0.05, 0.25 * targetRms)
    probeSignal = targetSignal * (probeRms / targetRms)
    probeOutput = paModel.Process(probeSignal)
    probeSpectrum = np.fft.fft(probeSignal, fftLength)
    probeOutputSpectrum = np.fft.fft(probeOutput, fftLength)
    probeSpectrumPower = np.abs(probeSpectrum) ** 2
    responseFloor = max(probeSpectrumPower.max(), np.finfo(float).tiny) * (
        10.0 ** (config.responseFloorDb / 10.0)
    )
    scalarDenominator = max(
        np.vdot(probeSignal, probeSignal).real, np.finfo(float).tiny
    )
    scalarGain = np.vdot(probeSignal, probeOutput) / scalarDenominator
    spectralResponse = (
        probeOutputSpectrum * np.conj(probeSpectrum)
    ) / (probeSpectrumPower + responseFloor)
    responseConfidence = probeSpectrumPower / (
        probeSpectrumPower + responseFloor
    )
    localResponse = (
        responseConfidence * spectralResponse
        + (1.0 - responseConfidence) * scalarGain
    )
    responseScale = max(np.abs(scalarGain) ** 2, np.finfo(float).tiny)
    learningFilter = (
        config.learningRate
        * np.conj(localResponse)
        / (
            np.abs(localResponse) ** 2
            + config.regularization * responseScale
        )
    )

    bestSelectionError = np.inf
    bestInput = inputSignal.copy()
    for iteration in range(config.numIterations):
        measuredOutput = _MeasurePaOutput(
            paModel, inputSignal, config, randomGenerator
        )
        errorSignal = targetSignal - measuredOutput
        errorPower = np.mean(np.abs(errorSignal) ** 2)
        currentInputPeak = float(np.max(np.abs(inputSignal)))

        # Fixed-gain ILC can become nonmonotonic in strong compression. Select
        # the best waveform using the gain-normalized residual that underlies
        # reconstruction SNR and EVM. The first candidate is the unmodified
        # PA input, so this rule cannot select a later, poorer calibration.
        selectionDenominator = max(
            np.vdot(targetSignal, targetSignal).real,
            np.finfo(float).tiny,
        )
        selectionGain = np.vdot(targetSignal, measuredOutput) / selectionDenominator
        fittedTarget = selectionGain * targetSignal
        selectionResidual = measuredOutput - fittedTarget
        selectionError = np.mean(np.abs(selectionResidual) ** 2) / max(
            np.mean(np.abs(fittedTarget) ** 2), np.finfo(float).tiny
        )
        if selectionError < bestSelectionError:
            bestSelectionError = selectionError
            bestInput = inputSignal.copy()

        errorSpectrum = np.fft.fft(errorSignal, fftLength)
        updateSpectrum = projectionMask * learningFilter * errorSpectrum
        updateSignal = np.fft.ifft(updateSpectrum)[: targetSignal.size]
        inputSignal = _LimitAmplitude(
            inputSignal + updateSignal, config.maxAmplitude
        )

        nmseDb = 10.0 * np.log10(
            max(errorPower, np.finfo(float).tiny) / targetPower
        )
        history.append(
            ILCIteration(
                iteration=iteration + 1,
                errorRms=float(np.sqrt(errorPower)),
                nmseDb=float(nmseDb),
                inputPeak=currentInputPeak,
            )
        )

    finalOutput = paModel.Process(bestInput)
    return ILCResult(
        learnedInput=bestInput,
        outputSignal=finalOutput,
        history=history,
    )


FeatureSpec = Tuple[str, int, int, int]


def _BuildFeatureSpecs(
    nonlinearOrders: Sequence[int],
    memoryDepth: int,
    crossMemoryDepth: int,
) -> List[FeatureSpec]:
    """Enumerate GMP main, lagging, and leading basis-function indices."""

    featureSpecs: List[FeatureSpec] = []
    for nonlinearOrder in nonlinearOrders:
        for memoryIndex in range(memoryDepth):
            featureSpecs.append(("main", nonlinearOrder, memoryIndex, 0))
    for nonlinearOrder in nonlinearOrders:
        if nonlinearOrder == 1:
            continue
        for memoryIndex in range(memoryDepth):
            for crossIndex in range(1, crossMemoryDepth + 1):
                featureSpecs.append(
                    ("lagging", nonlinearOrder, memoryIndex, crossIndex)
                )
                featureSpecs.append(
                    ("leading", nonlinearOrder, memoryIndex, crossIndex)
                )
    return featureSpecs


def _DelayedSlice(
    inputSignal: np.ndarray, sampleDelay: int, startIndex: int, stopIndex: int
) -> np.ndarray:
    """Return a delayed signal slice while padding unavailable history with zero."""

    outputLength = stopIndex - startIndex
    delayedValues = np.zeros(outputLength, dtype=np.complex128)
    sourceStart = max(0, startIndex - sampleDelay)
    sourceStop = max(0, stopIndex - sampleDelay)
    destinationStart = max(0, sampleDelay - startIndex)
    availableLength = max(0, sourceStop - sourceStart)
    if availableLength:
        delayedValues[
            destinationStart : destinationStart + availableLength
        ] = inputSignal[sourceStart:sourceStop]
    return delayedValues


def _BuildGmpBasisChunk(
    inputSignal: np.ndarray,
    featureSpecs: Sequence[FeatureSpec],
    startIndex: int,
    stopIndex: int,
) -> np.ndarray:
    """Build one bounded-memory block of the complex GMP design matrix."""

    chunkLength = stopIndex - startIndex
    basisMatrix = np.empty(
        (chunkLength, len(featureSpecs)), dtype=np.complex128
    )
    delayCache = {}

    def GetDelayed(sampleDelay: int) -> np.ndarray:
        """Cache delayed slices shared by many polynomial terms."""

        if sampleDelay not in delayCache:
            delayCache[sampleDelay] = _DelayedSlice(
                inputSignal, sampleDelay, startIndex, stopIndex
            )
        return delayCache[sampleDelay]

    for featureIndex, (
        branchName,
        nonlinearOrder,
        memoryIndex,
        crossIndex,
    ) in enumerate(featureSpecs):
        if branchName == "main":
            carrierSignal = GetDelayed(memoryIndex)
            envelopeSignal = carrierSignal
        elif branchName == "lagging":
            carrierSignal = GetDelayed(memoryIndex)
            envelopeSignal = GetDelayed(memoryIndex + crossIndex)
        else:
            carrierSignal = GetDelayed(memoryIndex + crossIndex)
            envelopeSignal = GetDelayed(memoryIndex)
        basisMatrix[:, featureIndex] = (
            carrierSignal * np.abs(envelopeSignal) ** (nonlinearOrder - 1)
        )
    return basisMatrix


@dataclass
class GMPPredistorter:
    """Store and evaluate a fitted generalized memory polynomial DPD."""

    featureSpecs: List[FeatureSpec]
    coefficients: np.ndarray

    def Process(self, inputSignal: np.ndarray, chunkSize: int = 16384) -> np.ndarray:
        """Apply the fitted GMP in chunks to limit temporary memory usage."""

        complexInput = np.asarray(inputSignal, dtype=np.complex128).reshape(-1)
        outputSignal = np.zeros_like(complexInput)
        for startIndex in range(0, complexInput.size, chunkSize):
            stopIndex = min(startIndex + chunkSize, complexInput.size)
            basisChunk = _BuildGmpBasisChunk(
                complexInput, self.featureSpecs, startIndex, stopIndex
            )
            outputSignal[startIndex:stopIndex] = basisChunk @ self.coefficients
        return outputSignal


def FitGmpPredistorter(
    referenceSignal: np.ndarray,
    learnedInput: np.ndarray,
    nonlinearOrders: Sequence[int] = (1, 3, 5, 7),
    memoryDepth: int = 3,
    crossMemoryDepth: int = 2,
    ridgeFactor: float = 1e-6,
    chunkSize: int = 8192,
) -> GMPPredistorter:
    """Fit a reusable GMP DPD to ILC labels using regularized least squares.

    Column normalization prevents high-order low-amplitude terms from being
    numerically discarded. Normal equations are accumulated in chunks, so
    fitting a wideband packet does not require retaining the full design
    matrix in memory.
    """

    complexReference = np.asarray(referenceSignal, dtype=np.complex128).reshape(-1)
    complexLabels = np.asarray(learnedInput, dtype=np.complex128).reshape(-1)
    if complexReference.size != complexLabels.size:
        raise ValueError("referenceSignal and learnedInput must have equal length")
    if complexReference.size == 0:
        raise ValueError("training signals cannot be empty")
    if ridgeFactor <= 0.0:
        raise ValueError("ridgeFactor must be positive")

    featureSpecs = _BuildFeatureSpecs(
        nonlinearOrders, memoryDepth, crossMemoryDepth
    )
    featureCount = len(featureSpecs)
    featureEnergy = np.zeros(featureCount, dtype=float)

    # First pass: estimate the RMS scale of every basis column.
    for startIndex in range(0, complexReference.size, chunkSize):
        stopIndex = min(startIndex + chunkSize, complexReference.size)
        basisChunk = _BuildGmpBasisChunk(
            complexReference, featureSpecs, startIndex, stopIndex
        )
        featureEnergy += np.sum(np.abs(basisChunk) ** 2, axis=0)
    featureScale = np.sqrt(featureEnergy / complexReference.size)
    featureScale = np.maximum(featureScale, 1e-12)

    normalMatrix = np.zeros((featureCount, featureCount), dtype=np.complex128)
    targetProjection = np.zeros(featureCount, dtype=np.complex128)
    for startIndex in range(0, complexReference.size, chunkSize):
        stopIndex = min(startIndex + chunkSize, complexReference.size)
        basisChunk = _BuildGmpBasisChunk(
            complexReference, featureSpecs, startIndex, stopIndex
        )
        normalizedBasis = basisChunk / featureScale
        labelChunk = complexLabels[startIndex:stopIndex]
        normalMatrix += normalizedBasis.conj().T @ normalizedBasis
        targetProjection += normalizedBasis.conj().T @ labelChunk

    diagonalScale = max(
        float(np.mean(np.real(np.diag(normalMatrix)))), np.finfo(float).tiny
    )
    regularizedMatrix = normalMatrix + (
        ridgeFactor * diagonalScale * np.eye(featureCount)
    )
    normalizedCoefficients = np.linalg.solve(
        regularizedMatrix, targetProjection
    )
    coefficients = normalizedCoefficients / featureScale
    return GMPPredistorter(featureSpecs, coefficients)
