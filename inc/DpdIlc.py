"""Frequency-domain ILC and GMP predistorter fitting.

The waveform ILC follows the regularized update from the theory document:

``U[k+1] = Q(U[k] + L[k] * E[k])``

``L[k] = mu * conj(H[k]) / (abs(H[k])**2 + lambda)``

After convergence, a generalized memory polynomial is fitted to map the
original Wi-Fi waveform onto the learned PA input. This converts waveform-
specific ILC labels into a reusable deployable predistorter.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

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
