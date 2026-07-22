"""Additional ILC update laws described by the DPD-ILC theory document.

The implementations share the same result and configuration objects as the
frequency-domain implementation in ``DpdIlc.py``. This makes convergence and
signal-quality comparisons use identical waveform, PA, and stopping rules.
"""

from typing import Callable, List

import numpy as np

from .DpdIlc import ILCConfig, ILCIteration, ILCResult
from .PaModel import AddAwgn


UpdateFunction = Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]


def LimitAmplitude(inputSignal: np.ndarray, maxAmplitude: float) -> np.ndarray:
    """Apply the pointwise peak constraint used by constrained ILC.

    Processing details:
        Algorithm: Apply the bounded sample-domain transformation without changing array length or causal indexing conventions.

    Args:
        inputSignal: One-dimensional complex baseband samples supplied to the operation.
        maxAmplitude: Maximum allowed complex-envelope magnitude.

    Returns:
        result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
    """

    limitedSignal = np.asarray(inputSignal, dtype=np.complex128).copy()
    sampleMagnitude = np.abs(limitedSignal)
    overLimit = sampleMagnitude > maxAmplitude
    if np.any(overLimit):
        limitedSignal[overLimit] *= maxAmplitude / sampleMagnitude[overLimit]
    return limitedSignal


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
    targetPower = max(
        np.mean(np.abs(targetSignal) ** 2), np.finfo(float).tiny
    )
    bestInput = inputSignal.copy()
    bestError = np.inf
    history: List[ILCIteration] = []

    for iteration in range(config.numIterations):
        measuredOutput = MeasureOutput(
            paModel, inputSignal, config, randomGenerator
        )
        errorSignal = targetSignal - measuredOutput
        selectionError = SelectionError(targetSignal, measuredOutput)
        if selectionError < bestError:
            bestError = selectionError
            bestInput = inputSignal.copy()

        errorPower = np.mean(np.abs(errorSignal) ** 2)
        history.append(
            ILCIteration(
                iteration=iteration + 1,
                errorRms=float(np.sqrt(errorPower)),
                nmseDb=float(
                    10.0
                    * np.log10(
                        max(errorPower, np.finfo(float).tiny) / targetPower
                    )
                ),
                inputPeak=float(np.max(np.abs(inputSignal))),
            )
        )
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


def NextPowerOfTwo(value: int) -> int:
    """Return the smallest radix-two FFT length that contains value samples.

    Processing details:
        Algorithm: Carry out the described operation using validated inputs, explicit array-shape handling, and deterministic project conventions.

    Args:
        value: Integer value whose representation or size is being calculated.

    Returns:
        result: int. The computed value described by the summary, with documented units, shape, and normalization.
    """

    return 1 << max(0, int(value - 1).bit_length())


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
    targetPower = max(
        np.mean(np.abs(targetSignal) ** 2), np.finfo(float).tiny
    )

    for iteration in range(config.numIterations):
        inputSignal = normalizedBasis @ normalizedCoefficients
        inputSignal = LimitAmplitude(inputSignal, config.maxAmplitude)
        measuredOutput = MeasureOutput(
            paModel, inputSignal, config, randomGenerator
        )
        errorSignal = targetSignal - measuredOutput
        selectionError = SelectionError(targetSignal, measuredOutput)
        if selectionError < bestError:
            bestError = selectionError
            bestInput = inputSignal.copy()

        errorPower = np.mean(np.abs(errorSignal) ** 2)
        history.append(
            ILCIteration(
                iteration=iteration + 1,
                errorRms=float(np.sqrt(errorPower)),
                nmseDb=float(
                    10.0
                    * np.log10(
                        max(errorPower, np.finfo(float).tiny) / targetPower
                    )
                ),
                inputPeak=float(np.max(np.abs(inputSignal))),
            )
        )
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
