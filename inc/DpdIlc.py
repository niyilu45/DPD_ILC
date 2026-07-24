"""Unified iterative learning control and deployable DPD implementations.

This module is the single home of reusable ILC-related algorithms in the
project. It contains common convergence records, every waveform and parameter
update law, SISO and MIMO execution, and ILC-label deployment models. Scenario
construction and performance comparison belong to ``tests/BenchMark.py`` so
production algorithms remain independent of the benchmark harness.

The frequency-domain waveform ILC follows the regularized update:

``U[k+1] = Q(U[k] + L[k] * E[k])``

``L[k] = mu * conj(H[k]) / (abs(H[k])**2 + lambda)``

After convergence, a generalized memory polynomial is fitted to map the
original Wi-Fi waveform onto the learned PA input. This converts waveform-
specific ILC labels into reusable GMP, Volterra, LUT, or neural
predistorters.
"""

from dataclasses import dataclass, replace
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from .PaModel import AddAwgn, MimoPaModel


@dataclass(frozen=True)
class ILCConfig:
    """Configure ILC updates, constraints, and feedback acquisition.

    Signal-quality evaluators are intentionally excluded. Callers pass an
    optional iteration evaluator directly to a SISO ``Run...Ilc`` entry and
    send the returned PA output to ``Analysis`` for final SNR, EVM, and ACLR.
    """

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
        if not callable(evmMseEvaluator):
            raise TypeError("evmMseEvaluator must be callable or None")
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
    paModel: Any,
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
    paModel: Any,
    sampleRateHz: float,
    channelBandwidthHz: float,
    config: ILCConfig = ILCConfig(),
    evmMseEvaluator: Optional[Callable[[np.ndarray], float]] = None,
) -> ILCResult:
    """Learn a PA input waveform using regularized frequency-domain ILC.

    The local transfer response is estimated from the current PA input and
    feedback spectra. Low-excitation FFT bins smoothly fall back to a scalar
    least-squares gain, preventing division by spectral nulls. The update is
    projected onto a bandwidth wider than the wanted channel so the ILC can
    synthesize out-of-band cancellation components needed to reduce ACLR.

    Performance reporting remains outside ``ILCConfig``. The optional
    ``evmMseEvaluator`` is supplied independently by ``Analysis`` only when
    exact Wi-Fi EVM-MSE is needed for per-iteration diagnostics and
    best-iteration selection.

    Args:
        referenceSignal: Ideal complex baseband PA output target.
        paModel: Repeatable PA object exposing ``Process``.
        sampleRateHz: Complex sampling rate in samples per second.
        channelBandwidthHz: Wanted channel bandwidth in hertz.
        config: Algorithm, constraint, and feedback-measurement settings.
        evmMseEvaluator: Independent optional callable that accepts the
            current PA output and returns normalized EVM-aligned MSE.

    Returns:
        result: Best measured PA input, its clean PA output, and iteration
        diagnostics.
    """

    config.Validate()
    targetSignal = np.asarray(referenceSignal, dtype=np.complex128).reshape(-1)
    if targetSignal.size == 0:
        raise ValueError("referenceSignal cannot be empty")
    if sampleRateHz <= 0.0 or channelBandwidthHz <= 0.0:
        raise ValueError("sampleRateHz and channelBandwidthHz must be positive")
    if sampleRateHz < 2.0 * channelBandwidthHz:
        raise ValueError(
            "sampleRateHz must be at least twice channelBandwidthHz"
        )

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
            evmMseEvaluator,
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


def BuildFeatureSpecs(
    nonlinearOrders: Sequence[int],
    memoryDepth: int,
    crossMemoryDepth: int,
) -> List[Tuple[str, int, int, int]]:
    """Enumerate GMP main, lagging, and leading basis-function indices.

    Processing details:
        Algorithm: Construct the requested model structure in deterministic order so coefficient indices and delayed samples remain reproducible.

    Args:
        nonlinearOrders: Positive odd polynomial orders included in the model.
        memoryDepth: Number of causal sample delays included in the model.
        crossMemoryDepth: Number of envelope cross-delays included in the GMP model.

    Returns:
        result: List[Tuple[str, int, int, int]]. The computed value described by the summary, with documented units, shape, and normalization.
    """

    featureSpecs: List[Tuple[str, int, int, int]] = []
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
    featureSpecs: Sequence[Tuple[str, int, int, int]],
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

    featureSpecs: List[Tuple[str, int, int, int]]
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

def BuildVolterraSpecs(
    memoryDepth: int,
) -> List[Tuple[str, int, int, int]]:
    """Enumerate first- and third-order complex-baseband Volterra terms.

    Processing details:
        Algorithm: Construct the requested model structure in deterministic order so coefficient indices and delayed samples remain reproducible.

    Args:
        memoryDepth: Number of causal sample delays included in the model.

    Returns:
        result: List[Tuple[str, int, int, int]]. The computed value described by the summary, with documented units, shape, and normalization.
    """

    featureSpecs: List[Tuple[str, int, int, int]] = []
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
    inputSignal: np.ndarray,
    featureSpecs: List[Tuple[str, int, int, int]],
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

    featureSpecs: List[Tuple[str, int, int, int]]
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

def MeasureOutput(
    paModel: Any,
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
    paModel: Any,
    config: ILCConfig,
    updateFunction: Callable[
        [np.ndarray, np.ndarray, np.ndarray, int],
        np.ndarray,
    ],
    evmMseEvaluator: Optional[Callable[[np.ndarray], float]] = None,
) -> ILCResult:
    """Run a generic waveform ILC loop with best-iteration retention.

    Processing details:
        Algorithm: Execute the configured signal-processing path, preserve sample alignment, and return the complete downstream result.

    Args:
        referenceSignal: Ideal complex baseband samples used as the target or regression input.
        paModel: PA object exposing Process and SmallSignalGain operations.
        config: Validated configuration object controlling this operation.
        updateFunction: Caller-supplied value consumed according to the function contract.
        evmMseEvaluator: Independent optional EVM-MSE evaluator used for
            diagnostics and best-iteration selection.

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
            evmMseEvaluator,
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

def EstimateComplexGain(
    referenceSignal: np.ndarray, paModel: Any
) -> complex:
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
    paModel: Any,
    config: ILCConfig = ILCConfig(),
    evmMseEvaluator: Optional[Callable[[np.ndarray], float]] = None,
) -> ILCResult:
    """Run scalar P-type ILC using the nominal target gain of one.

    Processing details:
        Algorithm: Execute the configured signal-processing path, preserve sample alignment, and return the complete downstream result.

    Args:
        referenceSignal: Ideal complex baseband samples used as the target or regression input.
        paModel: PA object exposing Process and SmallSignalGain operations.
        config: Validated configuration object controlling this operation.
        evmMseEvaluator: Independent optional EVM-MSE evaluator.

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

    return RunWaveformUpdate(
        referenceSignal,
        paModel,
        config,
        BuildUpdate,
        evmMseEvaluator,
    )

def RunComplexGainIlc(
    referenceSignal: np.ndarray,
    paModel: Any,
    config: ILCConfig = ILCConfig(),
    evmMseEvaluator: Optional[Callable[[np.ndarray], float]] = None,
) -> ILCResult:
    """Run regularized complex-gain-normalized ILC.

    Processing details:
        Algorithm: Execute the configured signal-processing path, preserve sample alignment, and return the complete downstream result.

    Args:
        referenceSignal: Ideal complex baseband samples used as the target or regression input.
        paModel: PA object exposing Process and SmallSignalGain operations.
        config: Validated configuration object controlling this operation.
        evmMseEvaluator: Independent optional EVM-MSE evaluator.

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

    return RunWaveformUpdate(
        referenceSignal,
        paModel,
        config,
        BuildUpdate,
        evmMseEvaluator,
    )

def EstimateFrequencyResponse(
    referenceSignal: np.ndarray,
    paModel: Any,
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
    paModel: Any,
    config: ILCConfig = ILCConfig(),
    firLength: int = 17,
    evmMseEvaluator: Optional[Callable[[np.ndarray], float]] = None,
) -> ILCResult:
    """Run FIR-filtered ILC using a truncated regularized inverse response.

    Processing details:
        Algorithm: Execute the configured signal-processing path, preserve sample alignment, and return the complete downstream result.

    Args:
        referenceSignal: Ideal complex baseband samples used as the target or regression input.
        paModel: PA object exposing Process and SmallSignalGain operations.
        config: Validated configuration object controlling this operation.
        firLength: Number of taps in the learned FIR update filter.
        evmMseEvaluator: Independent optional EVM-MSE evaluator.

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

    return RunWaveformUpdate(
        targetSignal,
        paModel,
        config,
        BuildUpdate,
        evmMseEvaluator,
    )

def RunDirectionalGaussNewtonIlc(
    referenceSignal: np.ndarray,
    paModel: Any,
    config: ILCConfig = ILCConfig(),
    finiteDifferenceRms: float = 1e-3,
    evmMseEvaluator: Optional[Callable[[np.ndarray], float]] = None,
) -> ILCResult:
    """Run a model-aware Gauss-Newton step projected onto the error direction.

    A finite-difference PA call evaluates ``J*e``. Solving the regularized
    one-dimensional least-squares problem gives the best complex step in the
    current error direction while retaining PA memory effects.

    The optional evaluator is an analysis dependency supplied separately from
    the immutable ILC algorithm configuration.

    Args:
        referenceSignal: Ideal complex baseband PA output target.
        paModel: Repeatable PA object exposing ``Process``.
        config: Algorithm, constraint, and feedback-measurement settings.
        finiteDifferenceRms: RMS size of the Jacobian-direction probe.
        evmMseEvaluator: Independent optional EVM-MSE evaluator.

    Returns:
        result: Best measured PA input, its clean PA output, and iteration
        diagnostics.
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

    return RunWaveformUpdate(
        referenceSignal,
        paModel,
        config,
        BuildUpdate,
        evmMseEvaluator,
    )

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
    paModel: Any,
    config: ILCConfig = ILCConfig(),
    nonlinearOrders: tuple = (1, 3, 5, 7),
    memoryDepth: int = 3,
    evmMseEvaluator: Optional[Callable[[np.ndarray], float]] = None,
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
        evmMseEvaluator: Independent optional EVM-MSE evaluator.

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
            evmMseEvaluator,
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
    paModel: Any,
    config: ILCConfig = ILCConfig(),
    evmMseEvaluator: Optional[Callable[[np.ndarray], float]] = None,
) -> ILCResult:
    """Run widely-linear augmented ILC with error and conjugate-error paths.

    Processing details:
        Algorithm: Execute the configured signal-processing path, preserve sample alignment, and return the complete downstream result.

    Args:
        referenceSignal: Ideal complex baseband samples used as the target or regression input.
        paModel: PA object exposing Process and SmallSignalGain operations.
        config: Validated configuration object controlling this operation.
        evmMseEvaluator: Independent optional EVM-MSE evaluator.

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

    return RunWaveformUpdate(
        targetSignal,
        paModel,
        config,
        BuildUpdate,
        evmMseEvaluator,
    )


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
