"""Unified iterative learning control and deployable DPD implementations.

This module is the single home of all ILC-related code in the project. It
contains common convergence records, every waveform and parameter update law,
SISO and MIMO execution, ILC-label deployment models, and the all-method
benchmark. Keeping these cooperating algorithms together makes their shared
measurement, projection, and selection rules explicit.

The frequency-domain waveform ILC follows the regularized update:

``U[k+1] = Q(U[k] + L[k] * E[k])``

``L[k] = mu * conj(H[k]) / (abs(H[k])**2 + lambda)``

After convergence, a generalized memory polynomial is fitted to map the
original Wi-Fi waveform onto the learned PA input. This converts waveform-
specific ILC labels into reusable GMP, Volterra, LUT, or neural
predistorters.
"""

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import (
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from .Analysis import Analysis, SignalMetrics
from .Draw import Draw
from .PaModel import AddAwgn, IQImbalancePA, MimoPaModel, PaModel
from .waveGen import GenWifi


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
    evmMseEvaluator: Optional[Callable[[np.ndarray], float]] = None

    def Validate(self) -> None:
        """Validate convergence, regularization, and feedback parameters.

        Processing details:
            Algorithm: Evaluate every documented constraint in deterministic order and stop at the first invalid condition without changing valid state.

        Returns:
            result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
        """

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
        if self.evmMseEvaluator is not None and not callable(
            self.evmMseEvaluator
        ):
            raise TypeError("evmMseEvaluator must be callable or None")


@dataclass(frozen=True)
class ILCIteration:
    """Store one iteration of ILC convergence diagnostics."""

    iteration: int
    mse: float
    errorRms: float
    nmseDb: float
    linearCompensatedMse: float
    linearCompensatedNmseDb: float
    evmAlignedMse: Optional[float]
    evmDb: Optional[float]
    complexGainMagnitudeDb: float
    complexGainPhaseDegrees: float
    inputPeak: float


@dataclass
class ILCResult:
    """Return the learned predistorted waveform and convergence history."""

    learnedInput: np.ndarray
    outputSignal: np.ndarray
    history: List[ILCIteration]


def CalculateIterationMetrics(
    iteration: int,
    targetSignal: np.ndarray,
    measuredOutput: np.ndarray,
    inputPeak: float,
    evmMseEvaluator: Optional[Callable[[np.ndarray], float]] = None,
) -> ILCIteration:
    """Calculate raw, linear-compensated, and EVM-aligned MSE metrics.

    The raw MSE preserves absolute amplitude and phase errors. The
    linear-compensated MSE removes the least-squares common complex gain and
    is therefore a useful EVM proxy when Wi-Fi frame metadata is unavailable.
    An optional evaluator can add the exact normalized data-subcarrier MSE;
    that value is mathematically equal to squared RMS EVM.

    Args:
        iteration: One-based ILC iteration index.
        targetSignal: Ideal time-domain output target.
        measuredOutput: PA feedback captured for the current input waveform.
        inputPeak: Peak magnitude of the current PA input waveform.
        evmMseEvaluator: Optional callable returning normalized EVM-aligned MSE.

    Returns:
        result: Complete immutable diagnostics for one ILC iteration.
    """

    complexTarget = np.asarray(targetSignal, dtype=np.complex128).reshape(-1)
    complexMeasured = np.asarray(
        measuredOutput, dtype=np.complex128
    ).reshape(-1)
    if complexTarget.size == 0 or complexTarget.shape != complexMeasured.shape:
        raise ValueError("targetSignal and measuredOutput must have equal length")
    if not np.all(np.isfinite(complexTarget)) or not np.all(
        np.isfinite(complexMeasured)
    ):
        raise ValueError("iteration metric signals must contain finite samples")

    numericFloor = np.finfo(float).tiny
    targetPower = max(
        float(np.mean(np.abs(complexTarget) ** 2)), numericFloor
    )
    rawError = complexTarget - complexMeasured
    rawMse = float(np.mean(np.abs(rawError) ** 2))

    # Least-squares projection separates a common gain/phase term from the
    # waveform-shape residual. Dividing residual power by |g|^2 expresses the
    # error back in reference-signal units before normalizing by target power.
    gainDenominator = max(
        float(np.vdot(complexTarget, complexTarget).real), numericFloor
    )
    complexGain = np.vdot(complexTarget, complexMeasured) / gainDenominator
    fittedTarget = complexGain * complexTarget
    orthogonalResidual = complexMeasured - fittedTarget
    gainPower = max(float(np.abs(complexGain) ** 2), numericFloor)
    linearCompensatedMse = float(
        np.mean(np.abs(orthogonalResidual) ** 2) / gainPower
    )
    linearCompensatedNmse = linearCompensatedMse / targetPower

    evmAlignedMse = None
    evmDb = None
    if evmMseEvaluator is not None:
        evaluatedMse = float(evmMseEvaluator(complexMeasured))
        if not np.isfinite(evaluatedMse) or evaluatedMse < 0.0:
            raise ValueError(
                "evmMseEvaluator must return a finite nonnegative value"
            )
        evmAlignedMse = evaluatedMse
        evmDb = float(
            10.0 * np.log10(max(evaluatedMse, numericFloor))
        )

    return ILCIteration(
        iteration=int(iteration),
        mse=rawMse,
        errorRms=float(np.sqrt(rawMse)),
        nmseDb=float(
            10.0 * np.log10(max(rawMse, numericFloor) / targetPower)
        ),
        linearCompensatedMse=linearCompensatedMse,
        linearCompensatedNmseDb=float(
            10.0 * np.log10(max(linearCompensatedNmse, numericFloor))
        ),
        evmAlignedMse=evmAlignedMse,
        evmDb=evmDb,
        complexGainMagnitudeDb=float(
            20.0 * np.log10(max(float(np.abs(complexGain)), numericFloor))
        ),
        complexGainPhaseDegrees=float(np.degrees(np.angle(complexGain))),
        inputPeak=float(inputPeak),
    )


def NextPowerOfTwo(value: int) -> int:
    """Return the smallest power of two greater than or equal to value.

    Processing details:
        Algorithm: Carry out the described operation using validated inputs, explicit array-shape handling, and deterministic project conventions.

    Args:
        value: Integer value whose representation or size is being calculated.

    Returns:
        result: int. The computed value described by the summary, with documented units, shape, and normalization.
    """

    return 1 << max(0, int(value - 1).bit_length())


def LimitAmplitude(inputSignal: np.ndarray, maxAmplitude: float) -> np.ndarray:
    """Project each complex sample onto the requested peak-amplitude disk.

    Processing details:
        Algorithm: Apply the bounded sample-domain transformation without changing array length or causal indexing conventions.

    Args:
        inputSignal: One-dimensional complex baseband samples supplied to the operation.
        maxAmplitude: Maximum allowed complex-envelope magnitude.

    Returns:
        result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
    """

    sampleMagnitude = np.abs(inputSignal)
    limitedSignal = inputSignal.copy()
    overLimit = sampleMagnitude > maxAmplitude
    if np.any(overLimit):
        limitedSignal[overLimit] *= maxAmplitude / sampleMagnitude[overLimit]
    return limitedSignal


def MeasurePaOutput(
    paModel,
    inputSignal: np.ndarray,
    config: ILCConfig,
    randomGenerator: np.random.Generator,
) -> np.ndarray:
    """Average repeated noisy feedback captures of the same PA waveform.

    Processing details:
        Algorithm: Execute the configured signal-processing path, preserve sample alignment, and return the complete downstream result.

    Args:
        paModel: PA object exposing Process and SmallSignalGain operations.
        inputSignal: One-dimensional complex baseband samples supplied to the operation.
        config: Validated configuration object controlling this operation.
        randomGenerator: NumPy random generator that makes results reproducible.

    Returns:
        result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
    """

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
    inputSignal = LimitAmplitude(targetSignal.copy(), config.maxAmplitude)
    fftLength = NextPowerOfTwo(targetSignal.size)
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
        float(np.mean(np.abs(targetSignal) ** 2)), np.finfo(float).tiny
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
        measuredOutput = MeasurePaOutput(
            paModel, inputSignal, config, randomGenerator
        )
        errorSignal = targetSignal - measuredOutput
        currentInputPeak = float(np.max(np.abs(inputSignal)))

        # Retain all three MSE views before updating the waveform. When exact
        # Wi-Fi EVM evaluation is available, best-iteration selection follows
        # the same data-subcarrier objective reported to the user.
        iterationMetrics = CalculateIterationMetrics(
            iteration + 1,
            targetSignal,
            measuredOutput,
            currentInputPeak,
            config.evmMseEvaluator,
        )
        selectionError = (
            iterationMetrics.evmAlignedMse
            if iterationMetrics.evmAlignedMse is not None
            else 10.0 ** (iterationMetrics.linearCompensatedNmseDb / 10.0)
        )
        if selectionError < bestSelectionError:
            bestSelectionError = selectionError
            bestInput = inputSignal.copy()
        history.append(iterationMetrics)

        errorSpectrum = np.fft.fft(errorSignal, fftLength)
        updateSpectrum = projectionMask * learningFilter * errorSpectrum
        updateSignal = np.fft.ifft(updateSpectrum)[: targetSignal.size]
        inputSignal = LimitAmplitude(
            inputSignal + updateSignal, config.maxAmplitude
        )

    finalOutput = paModel.Process(bestInput)
    return ILCResult(
        learnedInput=bestInput,
        outputSignal=finalOutput,
        history=history,
    )


FeatureSpec = Tuple[str, int, int, int]


def BuildFeatureSpecs(
    nonlinearOrders: Sequence[int],
    memoryDepth: int,
    crossMemoryDepth: int,
) -> List[FeatureSpec]:
    """Enumerate GMP main, lagging, and leading basis-function indices.

    Processing details:
        Algorithm: Construct the requested model structure in deterministic order so coefficient indices and delayed samples remain reproducible.

    Args:
        nonlinearOrders: Positive odd polynomial orders included in the model.
        memoryDepth: Number of causal sample delays included in the model.
        crossMemoryDepth: Number of envelope cross-delays included in the GMP model.

    Returns:
        result: List[FeatureSpec]. The computed value described by the summary, with documented units, shape, and normalization.
    """

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


def DelayedSlice(
    inputSignal: np.ndarray, sampleDelay: int, startIndex: int, stopIndex: int
) -> np.ndarray:
    """Return a delayed signal slice while padding unavailable history with zero.

    Processing details:
        Algorithm: Apply the bounded sample-domain transformation without changing array length or causal indexing conventions.

    Args:
        inputSignal: One-dimensional complex baseband samples supplied to the operation.
        sampleDelay: Nonnegative causal delay measured in complex samples.
        startIndex: Caller-supplied value consumed according to the function contract.
        stopIndex: Caller-supplied value consumed according to the function contract.

    Returns:
        result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
    """

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


def BuildGmpBasisChunk(
    inputSignal: np.ndarray,
    featureSpecs: Sequence[FeatureSpec],
    startIndex: int,
    stopIndex: int,
) -> np.ndarray:
    """Build one bounded-memory block of the complex GMP design matrix.

    Processing details:
        Algorithm: Construct the requested model structure in deterministic order so coefficient indices and delayed samples remain reproducible.

    Args:
        inputSignal: One-dimensional complex baseband samples supplied to the operation.
        featureSpecs: Caller-supplied value consumed according to the function contract.
        startIndex: Caller-supplied value consumed according to the function contract.
        stopIndex: Caller-supplied value consumed according to the function contract.

    Returns:
        result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
    """

    chunkLength = stopIndex - startIndex
    basisMatrix = np.empty(
        (chunkLength, len(featureSpecs)), dtype=np.complex128
    )
    delayCache = {}

    def GetDelayed(sampleDelay: int) -> np.ndarray:
        """Cache delayed slices shared by many polynomial terms.

        Processing details:
            Algorithm: Build each requested causal delay once per chunk and
            reuse the cached slice across every polynomial basis term.

        Args:
            sampleDelay: Nonnegative causal delay measured in complex samples.

        Returns:
            result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
        """

        if sampleDelay not in delayCache:
            delayCache[sampleDelay] = DelayedSlice(
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
        """Apply the fitted GMP in chunks to limit temporary memory usage.

        Processing details:
            Algorithm: Execute the configured signal-processing path, preserve sample alignment, and return the complete downstream result.

        Args:
            inputSignal: One-dimensional complex baseband samples supplied to the operation.
            chunkSize: Maximum samples processed per temporary basis-matrix chunk.

        Returns:
            result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
        """

        complexInput = np.asarray(inputSignal, dtype=np.complex128).reshape(-1)
        outputSignal = np.zeros_like(complexInput)
        for startIndex in range(0, complexInput.size, chunkSize):
            stopIndex = min(startIndex + chunkSize, complexInput.size)
            basisChunk = BuildGmpBasisChunk(
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

    featureSpecs = BuildFeatureSpecs(
        nonlinearOrders, memoryDepth, crossMemoryDepth
    )
    featureCount = len(featureSpecs)
    featureEnergy = np.zeros(featureCount, dtype=float)

    # First pass: estimate the RMS scale of every basis column.
    for startIndex in range(0, complexReference.size, chunkSize):
        stopIndex = min(startIndex + chunkSize, complexReference.size)
        basisChunk = BuildGmpBasisChunk(
            complexReference, featureSpecs, startIndex, stopIndex
        )
        featureEnergy += np.sum(np.abs(basisChunk) ** 2, axis=0)
    featureScale = np.sqrt(featureEnergy / complexReference.size)
    featureScale = np.maximum(featureScale, 1e-12)

    normalMatrix = np.zeros((featureCount, featureCount), dtype=np.complex128)
    targetProjection = np.zeros(featureCount, dtype=np.complex128)
    for startIndex in range(0, complexReference.size, chunkSize):
        stopIndex = min(startIndex + chunkSize, complexReference.size)
        basisChunk = BuildGmpBasisChunk(
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


# =============================================================================
# ILC label deployment models
# =============================================================================

def DelaySignal(inputSignal: np.ndarray, sampleDelay: int) -> np.ndarray:
    """Return a causal integer-delayed copy with zero initial conditions.

    Processing details:
        Algorithm: Apply the bounded sample-domain transformation without changing array length or causal indexing conventions.

    Args:
        inputSignal: One-dimensional complex baseband samples supplied to the operation.
        sampleDelay: Nonnegative causal delay measured in complex samples.

    Returns:
        result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
    """

    delayedSignal = np.zeros_like(inputSignal, dtype=np.complex128)
    if sampleDelay == 0:
        delayedSignal[:] = inputSignal
    elif sampleDelay < inputSignal.size:
        delayedSignal[sampleDelay:] = inputSignal[:-sampleDelay]
    return delayedSignal

VolterraSpec = Tuple[str, int, int, int]

def BuildVolterraSpecs(memoryDepth: int) -> List[VolterraSpec]:
    """Enumerate first- and third-order complex-baseband Volterra terms.

    Processing details:
        Algorithm: Construct the requested model structure in deterministic order so coefficient indices and delayed samples remain reproducible.

    Args:
        memoryDepth: Number of causal sample delays included in the model.

    Returns:
        result: List[VolterraSpec]. The computed value described by the summary, with documented units, shape, and normalization.
    """

    featureSpecs: List[VolterraSpec] = []
    for firstDelay in range(memoryDepth):
        featureSpecs.append(("linear", firstDelay, 0, 0))
    for firstDelay in range(memoryDepth):
        for secondDelay in range(firstDelay, memoryDepth):
            for conjugateDelay in range(memoryDepth):
                featureSpecs.append(
                    (
                        "third",
                        firstDelay,
                        secondDelay,
                        conjugateDelay,
                    )
                )
    return featureSpecs

def BuildVolterraBasis(
    inputSignal: np.ndarray, featureSpecs: List[VolterraSpec]
) -> np.ndarray:
    """Build a simplified complex Volterra basis containing x*x*conj(x).

    Processing details:
        Algorithm: Construct the requested model structure in deterministic order so coefficient indices and delayed samples remain reproducible.

    Args:
        inputSignal: One-dimensional complex baseband samples supplied to the operation.
        featureSpecs: Caller-supplied value consumed according to the function contract.

    Returns:
        result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
    """

    complexInput = np.asarray(inputSignal, dtype=np.complex128).reshape(-1)
    maximumDelay = max(
        max(firstDelay, secondDelay, conjugateDelay)
        for _, firstDelay, secondDelay, conjugateDelay in featureSpecs
    )
    delayCache = {
        sampleDelay: DelaySignal(complexInput, sampleDelay)
        for sampleDelay in range(maximumDelay + 1)
    }
    basisMatrix = np.empty(
        (complexInput.size, len(featureSpecs)), dtype=np.complex128
    )
    for featureIndex, (
        branchName,
        firstDelay,
        secondDelay,
        conjugateDelay,
    ) in enumerate(featureSpecs):
        if branchName == "linear":
            basisMatrix[:, featureIndex] = delayCache[firstDelay]
        else:
            basisMatrix[:, featureIndex] = (
                delayCache[firstDelay]
                * delayCache[secondDelay]
                * np.conj(delayCache[conjugateDelay])
            )
    return basisMatrix

@dataclass
class VolterraPredistorter:
    """Store a fitted simplified complex third-order Volterra DPD."""

    featureSpecs: List[VolterraSpec]
    coefficients: np.ndarray

    def Process(self, inputSignal: np.ndarray) -> np.ndarray:
        """Evaluate the fitted complex Volterra expansion.

        Processing details:
            Algorithm: Execute the configured signal-processing path, preserve sample alignment, and return the complete downstream result.

        Args:
            inputSignal: One-dimensional complex baseband samples supplied to the operation.

        Returns:
            result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
        """

        basisMatrix = BuildVolterraBasis(inputSignal, self.featureSpecs)
        return basisMatrix @ self.coefficients

def FitVolterraPredistorter(
    referenceSignal: np.ndarray,
    learnedInput: np.ndarray,
    memoryDepth: int = 3,
    ridgeFactor: float = 1e-6,
) -> VolterraPredistorter:
    """Fit a simplified complex Volterra DPD by normalized ridge regression.

    Processing details:
        Algorithm: Form the required numerical representation, apply documented regularization or normalization, and return the fitted result.

    Args:
        referenceSignal: Ideal complex baseband samples used as the target or regression input.
        learnedInput: Waveform-specific ILC labels used for deployment fitting.
        memoryDepth: Number of causal sample delays included in the model.
        ridgeFactor: Nonnegative diagonal regularization for least-squares fitting.

    Returns:
        result: VolterraPredistorter. The computed value described by the summary, with documented units, shape, and normalization.
    """

    complexReference = np.asarray(referenceSignal, dtype=np.complex128).reshape(-1)
    complexLabels = np.asarray(learnedInput, dtype=np.complex128).reshape(-1)
    if complexReference.size != complexLabels.size:
        raise ValueError("referenceSignal and learnedInput must have equal length")
    featureSpecs = BuildVolterraSpecs(memoryDepth)
    basisMatrix = BuildVolterraBasis(complexReference, featureSpecs)
    featureScale = np.sqrt(np.mean(np.abs(basisMatrix) ** 2, axis=0))
    featureScale = np.maximum(featureScale, 1e-12)
    normalizedBasis = basisMatrix / featureScale
    normalMatrix = normalizedBasis.conj().T @ normalizedBasis
    diagonalScale = max(
        np.mean(np.real(np.diag(normalMatrix))), np.finfo(float).tiny
    )
    coefficientsNormalized = np.linalg.solve(
        normalMatrix
        + ridgeFactor
        * diagonalScale
        * np.eye(normalizedBasis.shape[1]),
        normalizedBasis.conj().T @ complexLabels,
    )
    return VolterraPredistorter(
        featureSpecs=featureSpecs,
        coefficients=coefficientsNormalized / featureScale,
    )

@dataclass
class LUTPredistorter:
    """Store a memoryless complex-gain amplitude lookup table."""

    binEdges: np.ndarray
    binCoefficients: np.ndarray

    def Process(self, inputSignal: np.ndarray) -> np.ndarray:
        """Select the amplitude bin and apply its fitted complex gain.

        Processing details:
            Algorithm: Execute the configured signal-processing path, preserve sample alignment, and return the complete downstream result.

        Args:
            inputSignal: One-dimensional complex baseband samples supplied to the operation.

        Returns:
            result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
        """

        complexInput = np.asarray(inputSignal, dtype=np.complex128).reshape(-1)
        binIndices = np.searchsorted(
            self.binEdges, np.abs(complexInput), side="right"
        ) - 1
        binIndices = np.clip(binIndices, 0, self.binCoefficients.size - 1)
        return self.binCoefficients[binIndices] * complexInput

def FitLutPredistorter(
    referenceSignal: np.ndarray,
    learnedInput: np.ndarray,
    binCount: int = 64,
    ridgeFactor: float = 1e-8,
) -> LUTPredistorter:
    """Fit the regularized complex gain of every input-amplitude bin.

    Processing details:
        Algorithm: Form the required numerical representation, apply documented regularization or normalization, and return the fitted result.

    Args:
        referenceSignal: Ideal complex baseband samples used as the target or regression input.
        learnedInput: Waveform-specific ILC labels used for deployment fitting.
        binCount: Caller-supplied value consumed according to the function contract.
        ridgeFactor: Nonnegative diagonal regularization for least-squares fitting.

    Returns:
        result: LUTPredistorter. The computed value described by the summary, with documented units, shape, and normalization.
    """

    complexReference = np.asarray(referenceSignal, dtype=np.complex128).reshape(-1)
    complexLabels = np.asarray(learnedInput, dtype=np.complex128).reshape(-1)
    if complexReference.size != complexLabels.size:
        raise ValueError("referenceSignal and learnedInput must have equal length")
    maximumAmplitude = max(np.max(np.abs(complexReference)), 1e-12)
    binEdges = np.linspace(0.0, maximumAmplitude * (1.0 + 1e-9), binCount + 1)
    binIndices = np.searchsorted(
        binEdges, np.abs(complexReference), side="right"
    ) - 1
    binIndices = np.clip(binIndices, 0, binCount - 1)
    binCoefficients = np.ones(binCount, dtype=np.complex128)
    populatedBins = np.zeros(binCount, dtype=bool)

    for binIndex in range(binCount):
        sampleMask = binIndices == binIndex
        if not np.any(sampleMask):
            continue
        inputSamples = complexReference[sampleMask]
        labelSamples = complexLabels[sampleMask]
        denominator = np.sum(np.abs(inputSamples) ** 2) + ridgeFactor
        binCoefficients[binIndex] = (
            np.vdot(inputSamples, labelSamples) / denominator
        )
        populatedBins[binIndex] = True

    # Empty high-amplitude bins are common in finite training packets. Fill
    # them with the nearest populated coefficient to avoid discontinuities.
    populatedIndices = np.flatnonzero(populatedBins)
    if populatedIndices.size:
        for binIndex in np.flatnonzero(~populatedBins):
            nearestPosition = np.argmin(np.abs(populatedIndices - binIndex))
            binCoefficients[binIndex] = binCoefficients[
                populatedIndices[nearestPosition]
            ]
    return LUTPredistorter(binEdges, binCoefficients)

def BuildNeuralInputs(
    inputSignal: np.ndarray, memoryDepth: int
) -> np.ndarray:
    """Build real I/Q/envelope time-delay inputs for a lightweight DPD NN.

    Processing details:
        Algorithm: Construct the requested model structure in deterministic order so coefficient indices and delayed samples remain reproducible.

    Args:
        inputSignal: One-dimensional complex baseband samples supplied to the operation.
        memoryDepth: Number of causal sample delays included in the model.

    Returns:
        result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
    """

    complexInput = np.asarray(inputSignal, dtype=np.complex128).reshape(-1)
    inputColumns = []
    for memoryIndex in range(memoryDepth):
        delayedSignal = DelaySignal(complexInput, memoryIndex)
        inputColumns.extend(
            (delayedSignal.real, delayedSignal.imag, np.abs(delayedSignal))
        )
    return np.column_stack(inputColumns)

@dataclass
class NeuralPredistorter:
    """Store a single-hidden-layer time-delay neural predistorter."""

    memoryDepth: int
    inputMean: np.ndarray
    inputScale: np.ndarray
    hiddenWeights: np.ndarray
    hiddenBias: np.ndarray
    outputWeights: np.ndarray

    def Process(self, inputSignal: np.ndarray) -> np.ndarray:
        """Evaluate standardized linear skip features and tanh hidden units.

        Processing details:
            Algorithm: Execute the configured signal-processing path, preserve sample alignment, and return the complete downstream result.

        Args:
            inputSignal: One-dimensional complex baseband samples supplied to the operation.

        Returns:
            result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
        """

        neuralInputs = BuildNeuralInputs(inputSignal, self.memoryDepth)
        standardizedInputs = (
            neuralInputs - self.inputMean
        ) / self.inputScale
        hiddenFeatures = np.tanh(
            standardizedInputs @ self.hiddenWeights + self.hiddenBias
        )
        designMatrix = np.column_stack(
            (
                np.ones(standardizedInputs.shape[0]),
                standardizedInputs,
                hiddenFeatures,
            )
        )
        return designMatrix @ self.outputWeights

def FitNeuralPredistorter(
    referenceSignal: np.ndarray,
    learnedInput: np.ndarray,
    memoryDepth: int = 4,
    hiddenUnitCount: int = 32,
    ridgeFactor: float = 1e-5,
    randomSeed: int = 71,
) -> NeuralPredistorter:
    """Fit a deterministic ELM-style time-delay neural DPD.

    Hidden tanh weights are fixed after seeded random initialization, and the
    complex output layer is solved by ridge regression. This provides a true
    nonlinear neural basis without adding a framework dependency.
    """

    complexLabels = np.asarray(learnedInput, dtype=np.complex128).reshape(-1)
    neuralInputs = BuildNeuralInputs(referenceSignal, memoryDepth)
    if neuralInputs.shape[0] != complexLabels.size:
        raise ValueError("referenceSignal and learnedInput must have equal length")
    inputMean = np.mean(neuralInputs, axis=0)
    inputScale = np.std(neuralInputs, axis=0)
    inputScale = np.maximum(inputScale, 1e-9)
    standardizedInputs = (neuralInputs - inputMean) / inputScale
    randomGenerator = np.random.default_rng(randomSeed)
    hiddenWeights = randomGenerator.standard_normal(
        (standardizedInputs.shape[1], hiddenUnitCount)
    ) / np.sqrt(standardizedInputs.shape[1])
    hiddenBias = randomGenerator.uniform(-0.5, 0.5, hiddenUnitCount)
    hiddenFeatures = np.tanh(
        standardizedInputs @ hiddenWeights + hiddenBias
    )
    designMatrix = np.column_stack(
        (
            np.ones(standardizedInputs.shape[0]),
            standardizedInputs,
            hiddenFeatures,
        )
    )
    normalMatrix = designMatrix.T @ designMatrix
    diagonalScale = max(
        np.mean(np.diag(normalMatrix)), np.finfo(float).tiny
    )
    outputWeights = np.linalg.solve(
        normalMatrix
        + ridgeFactor * diagonalScale * np.eye(normalMatrix.shape[0]),
        designMatrix.T @ complexLabels,
    )
    return NeuralPredistorter(
        memoryDepth=memoryDepth,
        inputMean=inputMean,
        inputScale=inputScale,
        hiddenWeights=hiddenWeights,
        hiddenBias=hiddenBias,
        outputWeights=outputWeights,
    )


# =============================================================================
# Additional ILC update laws
# =============================================================================

UpdateFunction = Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]

def MeasureOutput(
    paModel,
    inputSignal: np.ndarray,
    config: ILCConfig,
    randomGenerator: np.random.Generator,
) -> np.ndarray:
    """Average noisy PA feedback captures according to the ILC configuration.

    Processing details:
        Algorithm: Execute the configured signal-processing path, preserve sample alignment, and return the complete downstream result.

    Args:
        paModel: PA object exposing Process and SmallSignalGain operations.
        inputSignal: One-dimensional complex baseband samples supplied to the operation.
        config: Validated configuration object controlling this operation.
        randomGenerator: NumPy random generator that makes results reproducible.

    Returns:
        result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
    """

    averagedOutput = np.zeros_like(inputSignal, dtype=np.complex128)
    for _ in range(config.feedbackAverages):
        paOutput = paModel.Process(inputSignal)
        averagedOutput += AddAwgn(
            paOutput, config.feedbackSnrDb, randomGenerator
        )
    return averagedOutput / float(config.feedbackAverages)

def SelectionError(
    targetSignal: np.ndarray, measuredOutput: np.ndarray
) -> float:
    """Return a gain-normalized residual compatible with SNR and EVM.

    Processing details:
        Algorithm: Apply the bounded sample-domain transformation without changing array length or causal indexing conventions.

    Args:
        targetSignal: Caller-supplied value consumed according to the function contract.
        measuredOutput: Caller-supplied value consumed according to the function contract.

    Returns:
        result: float. The computed value described by the summary, with documented units, shape, and normalization.
    """

    denominator = max(
        np.vdot(targetSignal, targetSignal).real, np.finfo(float).tiny
    )
    complexGain = np.vdot(targetSignal, measuredOutput) / denominator
    fittedTarget = complexGain * targetSignal
    residualSignal = measuredOutput - fittedTarget
    return float(
        np.mean(np.abs(residualSignal) ** 2)
        / max(np.mean(np.abs(fittedTarget) ** 2), np.finfo(float).tiny)
    )

def RunWaveformUpdate(
    referenceSignal: np.ndarray,
    paModel,
    config: ILCConfig,
    updateFunction: UpdateFunction,
) -> ILCResult:
    """Run a generic waveform ILC loop with best-iteration retention.

    Processing details:
        Algorithm: Execute the configured signal-processing path, preserve sample alignment, and return the complete downstream result.

    Args:
        referenceSignal: Ideal complex baseband samples used as the target or regression input.
        paModel: PA object exposing Process and SmallSignalGain operations.
        config: Validated configuration object controlling this operation.
        updateFunction: Caller-supplied value consumed according to the function contract.

    Returns:
        result: ILCResult. The computed value described by the summary, with documented units, shape, and normalization.
    """

    config.Validate()
    targetSignal = np.asarray(referenceSignal, dtype=np.complex128).reshape(-1)
    if targetSignal.size == 0:
        raise ValueError("referenceSignal cannot be empty")
    randomGenerator = np.random.default_rng(config.randomSeed)
    inputSignal = LimitAmplitude(targetSignal, config.maxAmplitude)
    bestInput = inputSignal.copy()
    bestError = np.inf
    history: List[ILCIteration] = []

    for iteration in range(config.numIterations):
        measuredOutput = MeasureOutput(
            paModel, inputSignal, config, randomGenerator
        )
        errorSignal = targetSignal - measuredOutput
        iterationMetrics = CalculateIterationMetrics(
            iteration + 1,
            targetSignal,
            measuredOutput,
            float(np.max(np.abs(inputSignal))),
            config.evmMseEvaluator,
        )
        selectionError = (
            iterationMetrics.evmAlignedMse
            if iterationMetrics.evmAlignedMse is not None
            else 10.0 ** (iterationMetrics.linearCompensatedNmseDb / 10.0)
        )
        if selectionError < bestError:
            bestError = selectionError
            bestInput = inputSignal.copy()
        history.append(iterationMetrics)
        updateSignal = updateFunction(
            inputSignal, measuredOutput, errorSignal, iteration
        )
        inputSignal = LimitAmplitude(
            inputSignal + updateSignal, config.maxAmplitude
        )

    return ILCResult(
        learnedInput=bestInput,
        outputSignal=paModel.Process(bestInput),
        history=history,
    )

def EstimateComplexGain(referenceSignal: np.ndarray, paModel) -> complex:
    """Measure the low-power least-squares complex gain of a PA.

    Processing details:
        Algorithm: Form the required numerical representation, apply documented regularization or normalization, and return the fitted result.

    Args:
        referenceSignal: Ideal complex baseband samples used as the target or regression input.
        paModel: PA object exposing Process and SmallSignalGain operations.

    Returns:
        result: complex. The computed value described by the summary, with documented units, shape, and normalization.
    """

    referenceRms = max(
        np.sqrt(np.mean(np.abs(referenceSignal) ** 2)),
        np.finfo(float).tiny,
    )
    probeSignal = referenceSignal * (min(0.04, 0.2 * referenceRms) / referenceRms)
    probeOutput = paModel.Process(probeSignal)
    denominator = max(
        np.vdot(probeSignal, probeSignal).real, np.finfo(float).tiny
    )
    return np.vdot(probeSignal, probeOutput) / denominator

def RunScalarPIlc(
    referenceSignal: np.ndarray,
    paModel,
    config: ILCConfig = ILCConfig(),
) -> ILCResult:
    """Run scalar P-type ILC using the nominal target gain of one.

    Processing details:
        Algorithm: Execute the configured signal-processing path, preserve sample alignment, and return the complete downstream result.

    Args:
        referenceSignal: Ideal complex baseband samples used as the target or regression input.
        paModel: PA object exposing Process and SmallSignalGain operations.
        config: Validated configuration object controlling this operation.

    Returns:
        result: ILCResult. The computed value described by the summary, with documented units, shape, and normalization.
    """

    def BuildUpdate(
        inputSignal: np.ndarray,
        measuredOutput: np.ndarray,
        errorSignal: np.ndarray,
        iteration: int,
    ) -> np.ndarray:
        """Apply the scalar equation u[k+1] = u[k] + mu*e[k].

        Processing details:
            Algorithm: Construct the requested model structure in deterministic order so coefficient indices and delayed samples remain reproducible.

        Args:
            inputSignal: One-dimensional complex baseband samples supplied to the operation.
            measuredOutput: Caller-supplied value consumed according to the function contract.
            errorSignal: Caller-supplied value consumed according to the function contract.
            iteration: Caller-supplied value consumed according to the function contract.

        Returns:
            result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
        """

        return config.learningRate * errorSignal

    return RunWaveformUpdate(referenceSignal, paModel, config, BuildUpdate)

def RunComplexGainIlc(
    referenceSignal: np.ndarray,
    paModel,
    config: ILCConfig = ILCConfig(),
) -> ILCResult:
    """Run regularized complex-gain-normalized ILC.

    Processing details:
        Algorithm: Execute the configured signal-processing path, preserve sample alignment, and return the complete downstream result.

    Args:
        referenceSignal: Ideal complex baseband samples used as the target or regression input.
        paModel: PA object exposing Process and SmallSignalGain operations.
        config: Validated configuration object controlling this operation.

    Returns:
        result: ILCResult. The computed value described by the summary, with documented units, shape, and normalization.
    """

    complexGain = EstimateComplexGain(referenceSignal, paModel)
    gainScale = max(np.abs(complexGain) ** 2, np.finfo(float).tiny)
    learningGain = (
        config.learningRate
        * np.conj(complexGain)
        / (gainScale + config.regularization * gainScale)
    )

    def BuildUpdate(
        inputSignal: np.ndarray,
        measuredOutput: np.ndarray,
        errorSignal: np.ndarray,
        iteration: int,
    ) -> np.ndarray:
        """Apply the regularized scalar complex inverse to every sample.

        Processing details:
            Algorithm: Construct the requested model structure in deterministic order so coefficient indices and delayed samples remain reproducible.

        Args:
            inputSignal: One-dimensional complex baseband samples supplied to the operation.
            measuredOutput: Caller-supplied value consumed according to the function contract.
            errorSignal: Caller-supplied value consumed according to the function contract.
            iteration: Caller-supplied value consumed according to the function contract.

        Returns:
            result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
        """

        return learningGain * errorSignal

    return RunWaveformUpdate(referenceSignal, paModel, config, BuildUpdate)

def EstimateFrequencyResponse(
    referenceSignal: np.ndarray,
    paModel,
    fftLength: int,
    responseFloorDb: float,
) -> np.ndarray:
    """Estimate a regularized small-signal PA frequency response.

    Processing details:
        Algorithm: Form the required numerical representation, apply documented regularization or normalization, and return the fitted result.

    Args:
        referenceSignal: Ideal complex baseband samples used as the target or regression input.
        paModel: PA object exposing Process and SmallSignalGain operations.
        fftLength: Number of time samples and frequency bins in the OFDM transform.
        responseFloorDb: Caller-supplied value consumed according to the function contract.

    Returns:
        result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
    """

    referenceRms = max(
        np.sqrt(np.mean(np.abs(referenceSignal) ** 2)),
        np.finfo(float).tiny,
    )
    probeSignal = referenceSignal * (min(0.04, 0.2 * referenceRms) / referenceRms)
    probeOutput = paModel.Process(probeSignal)
    probeSpectrum = np.fft.fft(probeSignal, fftLength)
    outputSpectrum = np.fft.fft(probeOutput, fftLength)
    probePower = np.abs(probeSpectrum) ** 2
    powerFloor = max(probePower.max(), np.finfo(float).tiny) * (
        10.0 ** (responseFloorDb / 10.0)
    )
    complexGain = EstimateComplexGain(referenceSignal, paModel)
    confidence = probePower / (probePower + powerFloor)
    spectralRatio = outputSpectrum * np.conj(probeSpectrum) / (
        probePower + powerFloor
    )
    return confidence * spectralRatio + (1.0 - confidence) * complexGain

def RunFirIlc(
    referenceSignal: np.ndarray,
    paModel,
    config: ILCConfig = ILCConfig(),
    firLength: int = 17,
) -> ILCResult:
    """Run FIR-filtered ILC using a truncated regularized inverse response.

    Processing details:
        Algorithm: Execute the configured signal-processing path, preserve sample alignment, and return the complete downstream result.

    Args:
        referenceSignal: Ideal complex baseband samples used as the target or regression input.
        paModel: PA object exposing Process and SmallSignalGain operations.
        config: Validated configuration object controlling this operation.
        firLength: Number of taps in the learned FIR update filter.

    Returns:
        result: ILCResult. The computed value described by the summary, with documented units, shape, and normalization.
    """

    targetSignal = np.asarray(referenceSignal, dtype=np.complex128).reshape(-1)
    fftLength = NextPowerOfTwo(targetSignal.size)
    frequencyResponse = EstimateFrequencyResponse(
        targetSignal, paModel, fftLength, config.responseFloorDb
    )
    responseScale = max(
        np.median(np.abs(frequencyResponse) ** 2), np.finfo(float).tiny
    )
    inverseResponse = (
        config.learningRate
        * np.conj(frequencyResponse)
        / (
            np.abs(frequencyResponse) ** 2
            + config.regularization * responseScale
        )
    )

    # Retain causal and anticausal taps around zero delay. The anticausal part
    # is valid for offline ILC because the complete repeated waveform is known.
    inverseImpulse = np.fft.ifft(inverseResponse)
    truncatedImpulse = np.zeros_like(inverseImpulse)
    halfLength = max(1, firLength // 2)
    truncatedImpulse[: halfLength + 1] = inverseImpulse[: halfLength + 1]
    truncatedImpulse[-halfLength:] = inverseImpulse[-halfLength:]
    firResponse = np.fft.fft(truncatedImpulse)

    def BuildUpdate(
        inputSignal: np.ndarray,
        measuredOutput: np.ndarray,
        errorSignal: np.ndarray,
        iteration: int,
    ) -> np.ndarray:
        """Filter the output error with the finite-length learning filter.

        Processing details:
            Algorithm: Construct the requested model structure in deterministic order so coefficient indices and delayed samples remain reproducible.

        Args:
            inputSignal: One-dimensional complex baseband samples supplied to the operation.
            measuredOutput: Caller-supplied value consumed according to the function contract.
            errorSignal: Caller-supplied value consumed according to the function contract.
            iteration: Caller-supplied value consumed according to the function contract.

        Returns:
            result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
        """

        errorSpectrum = np.fft.fft(errorSignal, fftLength)
        return np.fft.ifft(firResponse * errorSpectrum)[: errorSignal.size]

    return RunWaveformUpdate(targetSignal, paModel, config, BuildUpdate)

def RunDirectionalGaussNewtonIlc(
    referenceSignal: np.ndarray,
    paModel,
    config: ILCConfig = ILCConfig(),
    finiteDifferenceRms: float = 1e-3,
) -> ILCResult:
    """Run a model-aware Gauss-Newton step projected onto the error direction.

    A finite-difference PA call evaluates ``J*e``. Solving the regularized
    one-dimensional least-squares problem gives the best complex step in the
    current error direction while retaining PA memory effects.
    """

    def BuildUpdate(
        inputSignal: np.ndarray,
        measuredOutput: np.ndarray,
        errorSignal: np.ndarray,
        iteration: int,
    ) -> np.ndarray:
        """Calculate a directional finite-difference Gauss-Newton update.

        Processing details:
            Algorithm: Construct the requested model structure in deterministic order so coefficient indices and delayed samples remain reproducible.

        Args:
            inputSignal: One-dimensional complex baseband samples supplied to the operation.
            measuredOutput: Caller-supplied value consumed according to the function contract.
            errorSignal: Caller-supplied value consumed according to the function contract.
            iteration: Caller-supplied value consumed according to the function contract.

        Returns:
            result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
        """

        directionRms = max(
            np.sqrt(np.mean(np.abs(errorSignal) ** 2)),
            np.finfo(float).tiny,
        )
        differenceScale = finiteDifferenceRms / directionRms
        trialOutput = paModel.Process(
            inputSignal + differenceScale * errorSignal
        )
        cleanOutput = paModel.Process(inputSignal)
        jacobianDirection = (trialOutput - cleanOutput) / differenceScale
        jacobianPower = max(
            np.vdot(jacobianDirection, jacobianDirection).real,
            np.finfo(float).tiny,
        )
        stepGain = np.vdot(jacobianDirection, errorSignal) / (
            jacobianPower * (1.0 + config.regularization)
        )
        return config.learningRate * stepGain * errorSignal

    return RunWaveformUpdate(referenceSignal, paModel, config, BuildUpdate)

def MemoryPolynomialBasis(
    inputSignal: np.ndarray,
    nonlinearOrders: tuple,
    memoryDepth: int,
) -> np.ndarray:
    """Build the memory-polynomial basis used by parameter-domain ILC.

    Processing details:
        Algorithm: Carry out the described operation using validated inputs, explicit array-shape handling, and deterministic project conventions.

    Args:
        inputSignal: One-dimensional complex baseband samples supplied to the operation.
        nonlinearOrders: Positive odd polynomial orders included in the model.
        memoryDepth: Number of causal sample delays included in the model.

    Returns:
        result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
    """

    complexInput = np.asarray(inputSignal, dtype=np.complex128).reshape(-1)
    basisColumns = []
    for nonlinearOrder in nonlinearOrders:
        for memoryIndex in range(memoryDepth):
            delayedSignal = np.zeros_like(complexInput)
            if memoryIndex == 0:
                delayedSignal = complexInput
            elif memoryIndex < complexInput.size:
                delayedSignal[memoryIndex:] = complexInput[:-memoryIndex]
            basisColumns.append(
                delayedSignal * np.abs(delayedSignal) ** (nonlinearOrder - 1)
            )
    return np.column_stack(basisColumns)

def RunParameterDomainIlc(
    referenceSignal: np.ndarray,
    paModel,
    config: ILCConfig = ILCConfig(),
    nonlinearOrders: tuple = (1, 3, 5, 7),
    memoryDepth: int = 3,
) -> ILCResult:
    """Update memory-polynomial DPD coefficients directly in each ILC round.

    Processing details:
        Algorithm: Execute the configured signal-processing path, preserve sample alignment, and return the complete downstream result.

    Args:
        referenceSignal: Ideal complex baseband samples used as the target or regression input.
        paModel: PA object exposing Process and SmallSignalGain operations.
        config: Validated configuration object controlling this operation.
        nonlinearOrders: Positive odd polynomial orders included in the model.
        memoryDepth: Number of causal sample delays included in the model.

    Returns:
        result: ILCResult. The computed value described by the summary, with documented units, shape, and normalization.
    """

    config.Validate()
    targetSignal = np.asarray(referenceSignal, dtype=np.complex128).reshape(-1)
    basisMatrix = MemoryPolynomialBasis(
        targetSignal, nonlinearOrders, memoryDepth
    )
    featureScale = np.sqrt(np.mean(np.abs(basisMatrix) ** 2, axis=0))
    featureScale = np.maximum(featureScale, 1e-12)
    normalizedBasis = basisMatrix / featureScale
    normalMatrix = normalizedBasis.conj().T @ normalizedBasis
    diagonalScale = max(
        np.mean(np.real(np.diag(normalMatrix))), np.finfo(float).tiny
    )
    inverseNormal = np.linalg.inv(
        normalMatrix
        + config.regularization
        * diagonalScale
        * np.eye(normalizedBasis.shape[1])
    )

    # Initialize the DPD with the identity term x[n]. Coefficients are stored
    # in the normalized basis coordinate system.
    normalizedCoefficients = np.zeros(
        normalizedBasis.shape[1], dtype=np.complex128
    )
    normalizedCoefficients[0] = featureScale[0]
    complexGain = EstimateComplexGain(targetSignal, paModel)
    bestInput = targetSignal.copy()
    bestError = np.inf
    history: List[ILCIteration] = []
    randomGenerator = np.random.default_rng(config.randomSeed)
    for iteration in range(config.numIterations):
        inputSignal = normalizedBasis @ normalizedCoefficients
        inputSignal = LimitAmplitude(inputSignal, config.maxAmplitude)
        measuredOutput = MeasureOutput(
            paModel, inputSignal, config, randomGenerator
        )
        errorSignal = targetSignal - measuredOutput
        iterationMetrics = CalculateIterationMetrics(
            iteration + 1,
            targetSignal,
            measuredOutput,
            float(np.max(np.abs(inputSignal))),
            config.evmMseEvaluator,
        )
        selectionError = (
            iterationMetrics.evmAlignedMse
            if iterationMetrics.evmAlignedMse is not None
            else 10.0 ** (iterationMetrics.linearCompensatedNmseDb / 10.0)
        )
        if selectionError < bestError:
            bestError = selectionError
            bestInput = inputSignal.copy()
        history.append(iterationMetrics)
        coefficientUpdate = inverseNormal @ (
            normalizedBasis.conj().T @ errorSignal
        )
        normalizedCoefficients += (
            config.learningRate * coefficientUpdate / complexGain
        )

    return ILCResult(
        learnedInput=bestInput,
        outputSignal=paModel.Process(bestInput),
        history=history,
    )

def RunAugmentedIqIlc(
    referenceSignal: np.ndarray,
    paModel,
    config: ILCConfig = ILCConfig(),
) -> ILCResult:
    """Run widely-linear augmented ILC with error and conjugate-error paths.

    Processing details:
        Algorithm: Execute the configured signal-processing path, preserve sample alignment, and return the complete downstream result.

    Args:
        referenceSignal: Ideal complex baseband samples used as the target or regression input.
        paModel: PA object exposing Process and SmallSignalGain operations.
        config: Validated configuration object controlling this operation.

    Returns:
        result: ILCResult. The computed value described by the summary, with documented units, shape, and normalization.
    """

    targetSignal = np.asarray(referenceSignal, dtype=np.complex128).reshape(-1)
    referenceRms = max(
        np.sqrt(np.mean(np.abs(targetSignal) ** 2)),
        np.finfo(float).tiny,
    )
    probeSignal = targetSignal * (min(0.04, 0.2 * referenceRms) / referenceRms)
    probeOutput = paModel.Process(probeSignal)
    regressionMatrix = np.column_stack((probeSignal, np.conj(probeSignal)))
    forwardCoefficients = np.linalg.lstsq(
        regressionMatrix, probeOutput, rcond=None
    )[0]
    directGain, imageGain = forwardCoefficients
    augmentedMatrix = np.array(
        [
            [directGain, imageGain],
            [np.conj(imageGain), np.conj(directGain)],
        ],
        dtype=np.complex128,
    )
    inverseMatrix = np.linalg.solve(
        augmentedMatrix.conj().T @ augmentedMatrix
        + config.regularization * np.eye(2),
        augmentedMatrix.conj().T,
    )
    directLearning = config.learningRate * inverseMatrix[0, 0]
    imageLearning = config.learningRate * inverseMatrix[0, 1]

    def BuildUpdate(
        inputSignal: np.ndarray,
        measuredOutput: np.ndarray,
        errorSignal: np.ndarray,
        iteration: int,
    ) -> np.ndarray:
        """Apply both direct and conjugate branches of the augmented inverse.

        Processing details:
            Algorithm: Construct the requested model structure in deterministic order so coefficient indices and delayed samples remain reproducible.

        Args:
            inputSignal: One-dimensional complex baseband samples supplied to the operation.
            measuredOutput: Caller-supplied value consumed according to the function contract.
            errorSignal: Caller-supplied value consumed according to the function contract.
            iteration: Caller-supplied value consumed according to the function contract.

        Returns:
            result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
        """

        return directLearning * errorSignal + imageLearning * np.conj(errorSignal)

    return RunWaveformUpdate(targetSignal, paModel, config, BuildUpdate)


# =============================================================================
# Independent per-PA MIMO ILC and deployment
# =============================================================================

class MimoPaChain:
    """Expose one chain of ``MimoPaModel`` as an ordinary ILC plant."""

    def __init__(self, mimoPaModel: MimoPaModel, chainIndex: int) -> None:
        """Store the parent PA bank and selected zero-based chain index.

        Processing details:
            Algorithm: Validate the index against the live MIMO PA count and
            retain references without copying nonlinear model state.

        Args:
            mimoPaModel: Parent object owning all independent PA models.
            chainIndex: Zero-based chain processed by this view.

        Returns:
            result: None. A lightweight chain adapter is initialized.
        """

        if chainIndex < 0 or chainIndex >= mimoPaModel.numTransmitChains:
            raise IndexError("chainIndex is outside the configured chain range")
        self.mimoPaModel = mimoPaModel
        self.chainIndex = chainIndex

    def Process(self, inputSignal: np.ndarray) -> np.ndarray:
        """Process one vector through the selected physical PA path.

        Processing details:
            Algorithm: Delegate nonlinear and power-calibration operations to
            ``MimoPaModel.ProcessChain`` so ILC observes the deployed plant.

        Args:
            inputSignal: One-dimensional complex samples for this chain.

        Returns:
            result: Processed one-dimensional PA output samples.
        """

        return self.mimoPaModel.ProcessChain(inputSignal, self.chainIndex)

@dataclass(frozen=True)
class MimoIlcResult:
    """Store matrix-valued MIMO ILC results and each chain history."""

    learnedInput: np.ndarray
    outputSignal: np.ndarray
    chainResults: Tuple[ILCResult, ...]

class MimoGmpPredistorter:
    """Apply one independently fitted GMP predistorter to each PA input."""

    def __init__(
        self, chainPredistorters: Tuple[GMPPredistorter, ...]
    ) -> None:
        """Store one fitted GMP model per transmit chain.

        Processing details:
            Algorithm: Require at least one model and preserve tuple ordering
            as the physical RF-chain ordering used throughout the project.

        Args:
            chainPredistorters: Chain-ordered fitted GMP models.

        Returns:
            result: None. Predistorters are ready for matrix processing.
        """

        if not chainPredistorters:
            raise ValueError("chainPredistorters cannot be empty")
        self.chainPredistorters = tuple(chainPredistorters)

    def Process(self, inputSignal: np.ndarray) -> np.ndarray:
        """Predistort every column of a samples-by-chains matrix.

        Processing details:
            Algorithm: Validate exact chain count, evaluate each GMP on its
            corresponding column, and restore the common matrix orientation.

        Args:
            inputSignal: Complex matrix shaped samples by transmit chains.

        Returns:
            result: Predistorted matrix with the same shape as the input.
        """

        complexInput = np.asarray(inputSignal, dtype=np.complex128)
        if (
            complexInput.ndim != 2
            or complexInput.shape[1] != len(self.chainPredistorters)
            or complexInput.shape[0] == 0
        ):
            raise ValueError(
                "inputSignal must have one column per chain predistorter"
            )
        if not np.all(np.isfinite(complexInput)):
            raise ValueError("inputSignal must contain finite samples")
        return np.column_stack(
            [
                predistorter.Process(complexInput[:, chainIndex])
                for chainIndex, predistorter in enumerate(
                    self.chainPredistorters
                )
            ]
        )

def RunMimoFrequencyDomainIlc(
    referenceSignal: np.ndarray,
    mimoPaModel: MimoPaModel,
    sampleRateHz: float,
    channelBandwidthHz: float,
    config: ILCConfig = ILCConfig(),
) -> MimoIlcResult:
    """Run waveform ILC independently on every conducted transmit path.

    Processing details:
        Algorithm: Treat each PA as a separate repeatable SISO plant, offset
        feedback random seeds by chain, run identical regularized ILC, and
        stack learned inputs and outputs into the original MIMO orientation.

    Args:
        referenceSignal: Complex samples-by-transmit-chains target matrix.
        mimoPaModel: Independent nonlinear PA bank and power settings.
        sampleRateHz: Complex sampling rate in samples per second.
        channelBandwidthHz: Wanted Wi-Fi channel bandwidth in hertz.
        config: Shared ILC convergence and feedback configuration.

    Returns:
        result: Matrix ILC result plus all chain-specific histories.
    """

    complexReference = np.asarray(referenceSignal, dtype=np.complex128)
    if (
        complexReference.ndim != 2
        or complexReference.shape[1] != mimoPaModel.numTransmitChains
        or complexReference.shape[0] == 0
    ):
        raise ValueError(
            "referenceSignal must have one column per MIMO PA chain"
        )
    chainResults = []
    for chainIndex in range(mimoPaModel.numTransmitChains):
        chainConfig = replace(
            config,
            randomSeed=config.randomSeed + chainIndex,
            # A per-chain waveform cannot be evaluated by the full MIMO
            # post-spatial-demapping EVM callback. The common complex-gain
            # compensated MSE remains available for every physical PA.
            evmMseEvaluator=None,
        )
        chainResults.append(
            RunFrequencyDomainIlc(
                complexReference[:, chainIndex],
                MimoPaChain(mimoPaModel, chainIndex),
                sampleRateHz,
                channelBandwidthHz,
                chainConfig,
            )
        )
    resultTuple = tuple(chainResults)
    return MimoIlcResult(
        learnedInput=np.column_stack(
            [chainResult.learnedInput for chainResult in resultTuple]
        ),
        outputSignal=np.column_stack(
            [chainResult.outputSignal for chainResult in resultTuple]
        ),
        chainResults=resultTuple,
    )

def FitMimoGmpPredistorter(
    referenceSignal: np.ndarray,
    learnedInput: np.ndarray,
    nonlinearOrders: Tuple[int, ...] = (1, 3, 5, 7),
    memoryDepth: int = 3,
    crossMemoryDepth: int = 2,
    ridgeFactor: float = 1e-6,
) -> MimoGmpPredistorter:
    """Fit one deployable GMP to the ILC labels of every transmit chain.

    Processing details:
        Algorithm: Validate equal samples-by-chains matrices, independently
        regress each learned PA input on its original chain waveform, and
        assemble the fitted models in physical-chain order.

    Args:
        referenceSignal: Original MIMO waveform used as regression input.
        learnedInput: MIMO PA drive matrix learned by per-chain ILC.
        nonlinearOrders: Odd GMP main and cross nonlinear orders.
        memoryDepth: Main-branch sample memory depth.
        crossMemoryDepth: Lagging/leading envelope memory depth.
        ridgeFactor: Positive coefficient regularization strength.

    Returns:
        result: Matrix-capable collection of chain GMP predistorters.
    """

    complexReference = np.asarray(referenceSignal, dtype=np.complex128)
    complexLearned = np.asarray(learnedInput, dtype=np.complex128)
    if (
        complexReference.ndim != 2
        or complexReference.shape != complexLearned.shape
        or complexReference.shape[0] == 0
    ):
        raise ValueError(
            "referenceSignal and learnedInput must be equal nonempty matrices"
        )
    chainPredistorters = tuple(
        FitGmpPredistorter(
            complexReference[:, chainIndex],
            complexLearned[:, chainIndex],
            nonlinearOrders=nonlinearOrders,
            memoryDepth=memoryDepth,
            crossMemoryDepth=crossMemoryDepth,
            ridgeFactor=ridgeFactor,
        )
        for chainIndex in range(complexReference.shape[1])
    )
    return MimoGmpPredistorter(chainPredistorters)


# =============================================================================
# All-method ILC benchmark and reporting
# =============================================================================

@dataclass(frozen=True)
class BenchmarkConfig:
    """Configure a compact but representative all-method comparison."""

    frameFormat: str = "EHT"
    bandwidthMhz: int = 20
    mcs: int = 7
    numDataSymbols: int = 10
    oversampling: int = 4
    guardIntervalUs: float = 0.8
    driveRms: float = 0.24
    numIterations: int = 10
    paModelName: str = "wiener"
    seed: int = 101
    powerStartRms: float = 0.08
    powerStopRms: float = 0.40
    powerPointCount: int = 5
    generatePowerEvmCurve: bool = True
    outputDirectory: Path = Path("results/all_ilc_benchmark")

@dataclass(frozen=True)
class BenchmarkRow:
    """Store one method result and its improvement over scenario baseline."""

    methodName: str
    category: str
    scenario: str
    metrics: SignalMetrics
    snrImprovementDb: float
    evmImprovementDb: float
    aclrImprovementDb: float

    def ToDict(self) -> Dict[str, object]:
        """Convert a row to flat JSON/CSV-compatible values.

        Processing details:
            Algorithm: Convert validated in-memory results into a stable reporting format without altering later numerical calculations.

        Returns:
            result: Dict[str, object]. The computed value described by the summary, with documented units, shape, and normalization.
        """

        rowData: Dict[str, object] = {
            "methodName": self.methodName,
            "category": self.category,
            "scenario": self.scenario,
            "snrImprovementDb": self.snrImprovementDb,
            "evmImprovementDb": self.evmImprovementDb,
            "aclrImprovementDb": self.aclrImprovementDb,
        }
        rowData.update(self.metrics.ToDict())
        return rowData

def AddRow(
    rows: List[BenchmarkRow],
    methodName: str,
    category: str,
    scenario: str,
    metrics: SignalMetrics,
    baselineMetrics: SignalMetrics,
) -> None:
    """Append metrics and consistently signed improvements to the result table.

    Processing details:
        Algorithm: Carry out the described operation using validated inputs, explicit array-shape handling, and deterministic project conventions.

    Args:
        rows: Benchmark rows accumulated or emitted by the reporting operation.
        methodName: Human-readable algorithm or deployment-model label.
        category: Caller-supplied value consumed according to the function contract.
        scenario: Description of the impairment or validation scenario.
        metrics: Signal-quality metrics calculated for the selected output.
        baselineMetrics: Reference metrics used to calculate improvements.

    Returns:
        result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
    """

    rows.append(
        BenchmarkRow(
            methodName=methodName,
            category=category,
            scenario=scenario,
            metrics=metrics,
            snrImprovementDb=metrics.snrDb - baselineMetrics.snrDb,
            # More-negative EVM dB is better, so baseline minus result is a
            # positive improvement.
            evmImprovementDb=baselineMetrics.evmDb - metrics.evmDb,
            aclrImprovementDb=metrics.aclrWorstDb
            - baselineMetrics.aclrWorstDb,
        )
    )

def SaveHistory(
    methodName: str, ilcResult: ILCResult, outputDirectory: Path
) -> None:
    """Save a separate convergence CSV without overwriting other methods.

    Processing details:
        Algorithm: Convert validated in-memory results into a stable reporting format without altering later numerical calculations.

    Args:
        methodName: Human-readable algorithm or deployment-model label.
        ilcResult: Caller-supplied value consumed according to the function contract.
        outputDirectory: Directory in which result artifacts are written.

    Returns:
        result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
    """

    safeName = "".join(
        character.lower() if character.isalnum() else "_"
        for character in methodName
    ).strip("_")
    historyPath = outputDirectory / f"convergence_{safeName}.csv"
    with historyPath.open("w", newline="", encoding="utf-8-sig") as csvFile:
        csvWriter = csv.DictWriter(
            csvFile,
            fieldnames=(
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
            ),
        )
        csvWriter.writeheader()
        for iterationRecord in ilcResult.history:
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
    Draw(convergenceFileStem=f"convergence_{safeName}").SaveConvergenceCurve(
        ilcResult.history, outputDirectory
    )

def ReportHistory(
    methodName: str,
    ilcResult: ILCResult,
    resultAnalysis: Analysis,
    outputDirectory: Path,
) -> None:
    """Print and save one method's complete per-iteration MSE history.

    Processing details:
        Algorithm: Use ``Analysis`` for the console table, then serialize the
        same immutable records and render their convergence figure without
        recalculating any metric.

    Args:
        methodName: Human-readable ILC method label.
        ilcResult: Completed ILC result containing ordered history records.
        resultAnalysis: Analysis instance used for consistent presentation.
        outputDirectory: Destination for CSV and PNG result artifacts.

    Returns:
        result: None. Console and file outputs are produced as side effects.
    """

    resultAnalysis.PrintConvergence(
        ilcResult.history, f"{methodName} iteration metrics"
    )
    SaveHistory(methodName, ilcResult, outputDirectory)

def EvaluateDeployment(
    predistorter,
    validationSignal: np.ndarray,
    paModel,
    resultAnalysis: Analysis,
    maxAmplitude: float,
) -> SignalMetrics:
    """Evaluate one fitted DPD on a held-out Wi-Fi packet.

    Processing details:
        Algorithm: Execute the configured signal-processing path, preserve sample alignment, and return the complete downstream result.

    Args:
        predistorter: Caller-supplied value consumed according to the function contract.
        validationSignal: Independent complex waveform used to evaluate generalization.
        paModel: PA object exposing Process and SmallSignalGain operations.
        resultAnalysis: Caller-supplied value consumed according to the function contract.
        maxAmplitude: Maximum allowed complex-envelope magnitude.

    Returns:
        result: SignalMetrics. The computed value described by the summary, with documented units, shape, and normalization.
    """

    predistortedInput = LimitAmplitude(
        predistorter.Process(validationSignal), maxAmplitude
    )
    paOutput = paModel.Process(predistortedInput)
    return resultAnalysis.Analyze(paOutput)

def RunIlcCurvePoint(
    referenceSignal: np.ndarray,
    driveRms: float,
    paModel,
    waveform,
    methodName: str,
    methodFunction,
    methodConfig: ILCConfig,
) -> np.ndarray:
    """Run one selected ILC method at one power-EVM sweep point.

    Processing details:
        Algorithm: Execute the configured signal-processing path, preserve sample alignment, and return the complete downstream result.

    Args:
        referenceSignal: Ideal complex baseband samples used as the target or regression input.
        driveRms: Current RMS drive value in the power sweep.
        paModel: PA object exposing Process and SmallSignalGain operations.
        waveform: Wi-Fi metadata defining field locations, FFT sizes, and subcarriers.
        methodName: Human-readable algorithm or deployment-model label.
        methodFunction: Selected ILC update-law callable.
        methodConfig: Validated ILC configuration for the selected update law.

    Returns:
        result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
    """

    del driveRms
    pointAnalysis = Analysis(referenceSignal, waveform)
    pointConfig = replace(
        methodConfig,
        evmMseEvaluator=pointAnalysis.CalculateEvmAlignedMse,
    )
    if methodName == "Frequency-domain ILC":
        return RunFrequencyDomainIlc(
            referenceSignal,
            paModel,
            waveform.sampleRateHz,
            waveform.bandwidthHz,
            pointConfig,
        ).outputSignal
    return methodFunction(
        referenceSignal, paModel, pointConfig
    ).outputSignal

def RunAllIlcBenchmark(config: BenchmarkConfig = BenchmarkConfig()) -> List[BenchmarkRow]:
    """Run every update law and every ILC-label deployment model.

    Waveform update laws use one repeated training packet. Deployment models
    are fitted to frequency-domain ILC labels and evaluated on an independent
    payload generated with a different seed. Augmented ILC uses an IQ-image
    scenario, and noise-aware ILC learns from averaged noisy feedback while
    final metrics are measured on the clean PA output.
    """

    outputDirectory = Path(config.outputDirectory)
    outputDirectory.mkdir(parents=True, exist_ok=True)
    sharedWifiParameters = {
        "frameFormat": config.frameFormat,
        "bandwidthMhz": config.bandwidthMhz,
        "mcs": config.mcs,
        "numDataSymbols": config.numDataSymbols,
        "guardIntervalUs": config.guardIntervalUs,
        "oversampling": config.oversampling,
    }
    trainingParameters = dict(sharedWifiParameters)
    trainingParameters["seed"] = config.seed
    validationParameters = dict(sharedWifiParameters)
    validationParameters["seed"] = config.seed + 97
    trainingGenerator = GenWifi(parameters=trainingParameters)
    validationGenerator = GenWifi(parameters=validationParameters)
    trainingWaveform = trainingGenerator.Generate()
    validationWaveform = validationGenerator.Generate()
    trainingSignal = config.driveRms * trainingWaveform.samples
    validationSignal = config.driveRms * validationWaveform.samples
    paParameters = {"modelName": config.paModelName}
    paModel = PaModel(parameters=paParameters)
    trainingAnalysis = Analysis(trainingSignal, trainingWaveform)
    validationAnalysis = Analysis(validationSignal, validationWaveform)
    maxAmplitude = max(2.0, 1.6 * np.max(np.abs(trainingSignal)))

    baselineOutput = paModel.Process(trainingSignal)
    baselineMetrics = trainingAnalysis.Analyze(baselineOutput)
    powerEvaluators = {
        "PA baseline": lambda pointReference, _: paModel.Process(
            pointReference
        )
    }
    rows: List[BenchmarkRow] = []
    AddRow(
        rows,
        "PA baseline",
        "baseline",
        "nominal repeated waveform",
        baselineMetrics,
        baselineMetrics,
    )

    # Each algorithm receives tuned but conservative learning parameters. The
    # waveform, PA, iteration budget, and metrics remain identical.
    scalarConfig = ILCConfig(
        numIterations=config.numIterations,
        learningRate=0.10,
        maxAmplitude=maxAmplitude,
        randomSeed=config.seed + 1,
        evmMseEvaluator=trainingAnalysis.CalculateEvmAlignedMse,
    )
    complexConfig = ILCConfig(
        numIterations=config.numIterations,
        learningRate=0.15,
        maxAmplitude=maxAmplitude,
        randomSeed=config.seed + 2,
        evmMseEvaluator=trainingAnalysis.CalculateEvmAlignedMse,
    )
    firConfig = ILCConfig(
        numIterations=config.numIterations,
        learningRate=0.15,
        maxAmplitude=maxAmplitude,
        randomSeed=config.seed + 3,
        evmMseEvaluator=trainingAnalysis.CalculateEvmAlignedMse,
    )
    frequencyConfig = ILCConfig(
        numIterations=config.numIterations,
        learningRate=0.15,
        maxAmplitude=maxAmplitude,
        randomSeed=config.seed + 4,
        evmMseEvaluator=trainingAnalysis.CalculateEvmAlignedMse,
    )
    gaussNewtonConfig = ILCConfig(
        numIterations=config.numIterations,
        learningRate=0.65,
        maxAmplitude=maxAmplitude,
        randomSeed=config.seed + 5,
        evmMseEvaluator=trainingAnalysis.CalculateEvmAlignedMse,
    )
    parameterConfig = ILCConfig(
        numIterations=config.numIterations,
        learningRate=0.20,
        maxAmplitude=maxAmplitude,
        randomSeed=config.seed + 6,
        evmMseEvaluator=trainingAnalysis.CalculateEvmAlignedMse,
    )

    methodRuns = (
        ("Scalar P ILC", RunScalarPIlc, scalarConfig),
        ("Complex-gain ILC", RunComplexGainIlc, complexConfig),
        ("FIR ILC", RunFirIlc, firConfig),
        ("Frequency-domain ILC", None, frequencyConfig),
        (
            "Directional Gauss-Newton ILC",
            RunDirectionalGaussNewtonIlc,
            gaussNewtonConfig,
        ),
        ("Parameter-domain MP ILC", RunParameterDomainIlc, parameterConfig),
    )
    frequencyResult = None
    for methodName, methodFunction, methodConfig in methodRuns:
        if methodName == "Frequency-domain ILC":
            methodResult = RunFrequencyDomainIlc(
                trainingSignal,
                paModel,
                trainingWaveform.sampleRateHz,
                trainingWaveform.bandwidthHz,
                methodConfig,
            )
            frequencyResult = methodResult
        else:
            methodResult = methodFunction(
                trainingSignal, paModel, methodConfig
            )
        methodMetrics = trainingAnalysis.Analyze(methodResult.outputSignal)
        AddRow(
            rows,
            methodName,
            "ILC update law",
            "nominal repeated waveform",
            methodMetrics,
            baselineMetrics,
        )
        ReportHistory(
            methodName, methodResult, trainingAnalysis, outputDirectory
        )
        powerEvaluators[methodName] = (
            lambda pointReference,
            pointDrive,
            selectedName=methodName,
            selectedFunction=methodFunction,
            selectedConfig=methodConfig: RunIlcCurvePoint(
                pointReference,
                pointDrive,
                paModel,
                trainingWaveform,
                selectedName,
                selectedFunction,
                selectedConfig,
            )
        )

    if frequencyResult is None:
        raise RuntimeError("frequency-domain ILC result was not generated")

    # Constrained ILC uses a peak only 5 percent above the original waveform.
    constrainedPeak = 1.05 * np.max(np.abs(trainingSignal))
    constrainedResult = RunFrequencyDomainIlc(
        trainingSignal,
        paModel,
        trainingWaveform.sampleRateHz,
        trainingWaveform.bandwidthHz,
        ILCConfig(
            numIterations=config.numIterations,
            learningRate=0.12,
            maxAmplitude=constrainedPeak,
            randomSeed=config.seed + 7,
            evmMseEvaluator=trainingAnalysis.CalculateEvmAlignedMse,
        ),
    )
    constrainedMetrics = trainingAnalysis.Analyze(
        constrainedResult.outputSignal
    )
    AddRow(
        rows,
        "Constrained CFR-ILC",
        "ILC update law",
        "peak-constrained waveform",
        constrainedMetrics,
        baselineMetrics,
    )
    ReportHistory(
        "Constrained CFR-ILC",
        constrainedResult,
        trainingAnalysis,
        outputDirectory,
    )
    powerEvaluators["Constrained CFR-ILC"] = (
        lambda pointReference, _: RunFrequencyDomainIlc(
            pointReference,
            paModel,
            trainingWaveform.sampleRateHz,
            trainingWaveform.bandwidthHz,
            ILCConfig(
                numIterations=config.numIterations,
                learningRate=0.12,
                maxAmplitude=1.05 * np.max(np.abs(pointReference)),
                randomSeed=config.seed + 7,
            ),
        ).outputSignal
    )

    # Noise-aware learning uses a higher regularization and four averaged
    # feedback captures at 32 dB feedback SNR.
    noisyBaselineMetrics = baselineMetrics
    AddRow(
        rows,
        "Noisy-feedback baseline",
        "baseline",
        "32 dB averaged feedback",
        noisyBaselineMetrics,
        noisyBaselineMetrics,
    )
    noiseAwareResult = RunFrequencyDomainIlc(
        trainingSignal,
        paModel,
        trainingWaveform.sampleRateHz,
        trainingWaveform.bandwidthHz,
        ILCConfig(
            numIterations=config.numIterations,
            learningRate=0.10,
            regularization=1e-2,
            maxAmplitude=maxAmplitude,
            feedbackSnrDb=32.0,
            feedbackAverages=4,
            randomSeed=config.seed + 8,
            evmMseEvaluator=trainingAnalysis.CalculateEvmAlignedMse,
        ),
    )
    noiseAwareMetrics = trainingAnalysis.Analyze(
        noiseAwareResult.outputSignal
    )
    AddRow(
        rows,
        "Noise-aware ILC",
        "ILC update law",
        "32 dB averaged feedback",
        noiseAwareMetrics,
        noisyBaselineMetrics,
    )
    ReportHistory(
        "Noise-aware ILC",
        noiseAwareResult,
        trainingAnalysis,
        outputDirectory,
    )
    powerEvaluators["Noise-aware ILC"] = (
        lambda pointReference, _: RunFrequencyDomainIlc(
            pointReference,
            paModel,
            trainingWaveform.sampleRateHz,
            trainingWaveform.bandwidthHz,
            ILCConfig(
                numIterations=config.numIterations,
                learningRate=0.10,
                regularization=1e-2,
                maxAmplitude=maxAmplitude,
                feedbackSnrDb=32.0,
                feedbackAverages=4,
                randomSeed=config.seed + 8,
            ),
        ).outputSignal
    )

    # Augmented ILC is evaluated in the IQ-image scenario for which its
    # conjugate branch is designed.
    iqPaModel = IQImbalancePA(PaModel(parameters=paParameters))
    iqBaselineOutput = iqPaModel.Process(trainingSignal)
    iqBaselineMetrics = trainingAnalysis.Analyze(iqBaselineOutput)
    AddRow(
        rows,
        "IQ-imbalance baseline",
        "baseline",
        "IQ image impairment",
        iqBaselineMetrics,
        iqBaselineMetrics,
    )
    augmentedResult = RunAugmentedIqIlc(
        trainingSignal,
        iqPaModel,
        ILCConfig(
            numIterations=config.numIterations,
            learningRate=0.18,
            maxAmplitude=maxAmplitude,
            randomSeed=config.seed + 9,
            evmMseEvaluator=trainingAnalysis.CalculateEvmAlignedMse,
        ),
    )
    augmentedMetrics = trainingAnalysis.Analyze(
        augmentedResult.outputSignal
    )
    AddRow(
        rows,
        "Augmented IQ ILC",
        "ILC update law",
        "IQ image impairment",
        augmentedMetrics,
        iqBaselineMetrics,
    )
    ReportHistory(
        "Augmented IQ ILC",
        augmentedResult,
        trainingAnalysis,
        outputDirectory,
    )
    powerEvaluators["IQ-imbalance baseline"] = (
        lambda pointReference, _: iqPaModel.Process(pointReference)
    )
    powerEvaluators["Augmented IQ ILC"] = (
        lambda pointReference, _: RunAugmentedIqIlc(
            pointReference,
            iqPaModel,
            ILCConfig(
                numIterations=config.numIterations,
                learningRate=0.18,
                maxAmplitude=maxAmplitude,
                randomSeed=config.seed + 9,
            ),
        ).outputSignal
    )

    # Fit every deployable model to the same converged ILC labels, then test
    # on a held-out Wi-Fi payload to measure generalization rather than recall.
    validationBaselineOutput = paModel.Process(validationSignal)
    validationBaselineMetrics = validationAnalysis.Analyze(
        validationBaselineOutput
    )
    AddRow(
        rows,
        "Validation baseline",
        "baseline",
        "held-out Wi-Fi packet",
        validationBaselineMetrics,
        validationBaselineMetrics,
    )
    deploymentModels = (
        (
            "ILC label + MP",
            FitGmpPredistorter(
                trainingSignal,
                frequencyResult.learnedInput,
                nonlinearOrders=(1, 3, 5, 7),
                memoryDepth=3,
                crossMemoryDepth=0,
            ),
        ),
        (
            "ILC label + GMP",
            FitGmpPredistorter(
                trainingSignal,
                frequencyResult.learnedInput,
                nonlinearOrders=(1, 3, 5, 7),
                memoryDepth=3,
                crossMemoryDepth=2,
            ),
        ),
        (
            "ILC label + Volterra",
            FitVolterraPredistorter(
                trainingSignal,
                frequencyResult.learnedInput,
                memoryDepth=3,
            ),
        ),
        (
            "ILC label + LUT",
            FitLutPredistorter(
                trainingSignal,
                frequencyResult.learnedInput,
                binCount=64,
            ),
        ),
        (
            "ILC label + NN",
            FitNeuralPredistorter(
                trainingSignal,
                frequencyResult.learnedInput,
                memoryDepth=4,
                hiddenUnitCount=32,
                randomSeed=config.seed + 10,
            ),
        ),
    )
    for methodName, predistorter in deploymentModels:
        methodMetrics = EvaluateDeployment(
            predistorter,
            validationSignal,
            paModel,
            validationAnalysis,
            maxAmplitude,
        )
        AddRow(
            rows,
            methodName,
            "ILC label deployment",
            "held-out Wi-Fi packet",
            methodMetrics,
            validationBaselineMetrics,
        )
        powerEvaluators[methodName] = (
            lambda pointReference,
            _,
            selectedPredistorter=predistorter: paModel.Process(
                LimitAmplitude(
                    selectedPredistorter.Process(pointReference),
                    maxAmplitude,
                )
            )
        )
    metadata: Mapping[str, object] = {
        "frameFormat": trainingWaveform.frameFormat,
        "bandwidthMhz": config.bandwidthMhz,
        "mcs": config.mcs,
        "numDataSymbols": config.numDataSymbols,
        "oversampling": config.oversampling,
        "guardIntervalUs": config.guardIntervalUs,
        "driveRms": config.driveRms,
        "numIterations": config.numIterations,
        "paModel": config.paModelName,
        "trainingSeed": config.seed,
        "validationSeed": config.seed + 97,
        "powerStartRms": config.powerStartRms,
        "powerStopRms": config.powerStopRms,
        "powerPointCount": config.powerPointCount,
        "generatePowerEvmCurve": config.generatePowerEvmCurve,
    }
    SaveBenchmarkResults(rows, outputDirectory, metadata)
    powerCurvePaths = None
    if config.generatePowerEvmCurve:
        powerDriveValues = np.geomspace(
            config.powerStartRms,
            config.powerStopRms,
            config.powerPointCount,
        )
        powerEvmCurve = trainingAnalysis.AnalyzePowerEvmCurve(
            powerDriveValues, powerEvaluators
        )
        powerDataPaths = trainingAnalysis.SavePowerEvmCurveData(
            outputDirectory,
            fileStem="all_ilc_power_evm_curve",
        )
        powerFigurePath = Draw(
            powerEvmFileStem="all_ilc_power_evm_curve"
        ).SavePowerEvmCurve(powerEvmCurve, outputDirectory)
        powerCurvePaths = (*powerDataPaths, powerFigurePath)
    PrintBenchmarkResults(rows)
    if powerCurvePaths is not None:
        powerCsvPath, powerJsonPath, powerFigurePath = powerCurvePaths
        print(f"\nPower-EVM CSV:  {powerCsvPath.resolve()}")
        print(f"Power-EVM JSON: {powerJsonPath.resolve()}")
        print(f"Power-EVM plot: {powerFigurePath.resolve()}")
    return rows

def SaveBenchmarkResults(
    rows: List[BenchmarkRow],
    outputDirectory: Path,
    metadata: Mapping[str, object],
) -> None:
    """Save the complete all-method benchmark as flat CSV and structured JSON.

    Processing details:
        Algorithm: Convert validated in-memory results into a stable reporting format without altering later numerical calculations.

    Args:
        rows: Benchmark rows accumulated or emitted by the reporting operation.
        outputDirectory: Directory in which result artifacts are written.
        metadata: Caller-supplied value consumed according to the function contract.

    Returns:
        result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
    """

    outputDirectory.mkdir(parents=True, exist_ok=True)
    csvPath = outputDirectory / "all_ilc_metrics.csv"
    jsonPath = outputDirectory / "all_ilc_metrics.json"
    flatRows = [row.ToDict() for row in rows]
    with csvPath.open("w", newline="", encoding="utf-8-sig") as csvFile:
        csvWriter = csv.DictWriter(csvFile, fieldnames=list(flatRows[0].keys()))
        csvWriter.writeheader()
        csvWriter.writerows(flatRows)
    with jsonPath.open("w", encoding="utf-8") as jsonFile:
        json.dump(
            {"metadata": dict(metadata), "results": flatRows},
            jsonFile,
            ensure_ascii=False,
            indent=2,
        )

def PrintBenchmarkResults(rows: List[BenchmarkRow]) -> None:
    """Print a compact all-method SNR, EVM, and worst-ACLR table.

    Processing details:
        Algorithm: Convert validated in-memory results into a stable reporting format without altering later numerical calculations.

    Args:
        rows: Benchmark rows accumulated or emitted by the reporting operation.

    Returns:
        result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
    """

    header = (
        f"{'Method':<32} {'Scenario':<25} {'SNR':>8} "
        f"{'EVM%':>9} {'ACLR-W':>9} {'dEVM':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.methodName:<32} {row.scenario:<25} "
            f"{row.metrics.snrDb:>8.2f} {row.metrics.evmPercent:>9.3f} "
            f"{row.metrics.aclrWorstDb:>9.2f} "
            f"{row.evmImprovementDb:>8.2f}"
        )
