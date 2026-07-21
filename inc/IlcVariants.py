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


def _LimitAmplitude(inputSignal: np.ndarray, maxAmplitude: float) -> np.ndarray:
    """Apply the pointwise peak constraint used by constrained ILC."""

    limitedSignal = np.asarray(inputSignal, dtype=np.complex128).copy()
    sampleMagnitude = np.abs(limitedSignal)
    overLimit = sampleMagnitude > maxAmplitude
    if np.any(overLimit):
        limitedSignal[overLimit] *= maxAmplitude / sampleMagnitude[overLimit]
    return limitedSignal


def _MeasureOutput(
    paModel,
    inputSignal: np.ndarray,
    config: ILCConfig,
    randomGenerator: np.random.Generator,
) -> np.ndarray:
    """Average noisy PA feedback captures according to the ILC configuration."""

    averagedOutput = np.zeros_like(inputSignal, dtype=np.complex128)
    for _ in range(config.feedbackAverages):
        paOutput = paModel.Process(inputSignal)
        averagedOutput += AddAwgn(
            paOutput, config.feedbackSnrDb, randomGenerator
        )
    return averagedOutput / float(config.feedbackAverages)


def _SelectionError(
    targetSignal: np.ndarray, measuredOutput: np.ndarray
) -> float:
    """Return a gain-normalized residual compatible with SNR and EVM."""

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

def _RunWaveformUpdate(
    referenceSignal: np.ndarray,
    paModel,
    config: ILCConfig,
    updateFunction: UpdateFunction,
) -> ILCResult:
    """Run a generic waveform ILC loop with best-iteration retention."""

    config.Validate()
    targetSignal = np.asarray(referenceSignal, dtype=np.complex128).reshape(-1)
    if targetSignal.size == 0:
        raise ValueError("referenceSignal cannot be empty")
    randomGenerator = np.random.default_rng(config.randomSeed)
    inputSignal = _LimitAmplitude(targetSignal, config.maxAmplitude)
    targetPower = max(
        np.mean(np.abs(targetSignal) ** 2), np.finfo(float).tiny
    )
    bestInput = inputSignal.copy()
    bestError = np.inf
    history: List[ILCIteration] = []

    for iteration in range(config.numIterations):
        measuredOutput = _MeasureOutput(
            paModel, inputSignal, config, randomGenerator
        )
        errorSignal = targetSignal - measuredOutput
        selectionError = _SelectionError(targetSignal, measuredOutput)
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
        inputSignal = _LimitAmplitude(
            inputSignal + updateSignal, config.maxAmplitude
        )

    return ILCResult(
        learnedInput=bestInput,
        outputSignal=paModel.Process(bestInput),
        history=history,
    )


def _EstimateComplexGain(referenceSignal: np.ndarray, paModel) -> complex:
    """Measure the low-power least-squares complex gain of a PA."""

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
    """Run scalar P-type ILC using the nominal target gain of one."""

    def BuildUpdate(
        inputSignal: np.ndarray,
        measuredOutput: np.ndarray,
        errorSignal: np.ndarray,
        iteration: int,
    ) -> np.ndarray:
        """Apply the scalar equation u[k+1] = u[k] + mu*e[k]."""

        return config.learningRate * errorSignal

    return _RunWaveformUpdate(referenceSignal, paModel, config, BuildUpdate)


def RunComplexGainIlc(
    referenceSignal: np.ndarray,
    paModel,
    config: ILCConfig = ILCConfig(),
) -> ILCResult:
    """Run regularized complex-gain-normalized ILC."""

    complexGain = _EstimateComplexGain(referenceSignal, paModel)
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
        """Apply the regularized scalar complex inverse to every sample."""

        return learningGain * errorSignal

    return _RunWaveformUpdate(referenceSignal, paModel, config, BuildUpdate)


def _NextPowerOfTwo(value: int) -> int:
    """Return the smallest radix-two FFT length that contains value samples."""

    return 1 << max(0, int(value - 1).bit_length())


def _EstimateFrequencyResponse(
    referenceSignal: np.ndarray,
    paModel,
    fftLength: int,
    responseFloorDb: float,
) -> np.ndarray:
    """Estimate a regularized small-signal PA frequency response."""

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
    complexGain = _EstimateComplexGain(referenceSignal, paModel)
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
    """Run FIR-filtered ILC using a truncated regularized inverse response."""

    targetSignal = np.asarray(referenceSignal, dtype=np.complex128).reshape(-1)
    fftLength = _NextPowerOfTwo(targetSignal.size)
    frequencyResponse = _EstimateFrequencyResponse(
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
        """Filter the output error with the finite-length learning filter."""

        errorSpectrum = np.fft.fft(errorSignal, fftLength)
        return np.fft.ifft(firResponse * errorSpectrum)[: errorSignal.size]

    return _RunWaveformUpdate(targetSignal, paModel, config, BuildUpdate)


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
        """Calculate a directional finite-difference Gauss-Newton update."""

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

    return _RunWaveformUpdate(referenceSignal, paModel, config, BuildUpdate)


def _MemoryPolynomialBasis(
    inputSignal: np.ndarray,
    nonlinearOrders: tuple,
    memoryDepth: int,
) -> np.ndarray:
    """Build the memory-polynomial basis used by parameter-domain ILC."""

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
    """Update memory-polynomial DPD coefficients directly in each ILC round."""

    config.Validate()
    targetSignal = np.asarray(referenceSignal, dtype=np.complex128).reshape(-1)
    basisMatrix = _MemoryPolynomialBasis(
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
    complexGain = _EstimateComplexGain(targetSignal, paModel)
    bestInput = targetSignal.copy()
    bestError = np.inf
    history: List[ILCIteration] = []
    randomGenerator = np.random.default_rng(config.randomSeed)
    targetPower = max(
        np.mean(np.abs(targetSignal) ** 2), np.finfo(float).tiny
    )

    for iteration in range(config.numIterations):
        inputSignal = normalizedBasis @ normalizedCoefficients
        inputSignal = _LimitAmplitude(inputSignal, config.maxAmplitude)
        measuredOutput = _MeasureOutput(
            paModel, inputSignal, config, randomGenerator
        )
        errorSignal = targetSignal - measuredOutput
        selectionError = _SelectionError(targetSignal, measuredOutput)
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
    """Run widely-linear augmented ILC with error and conjugate-error paths."""

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
        """Apply both direct and conjugate branches of the augmented inverse."""

        return directLearning * errorSignal + imageLearning * np.conj(errorSignal)

    return _RunWaveformUpdate(targetSignal, paModel, config, BuildUpdate)
