"""Deployable DPD models fitted from converged ILC waveform labels.

Memory polynomial and GMP fitting are implemented in ``DpdIlc.py``. This
module adds simplified complex Volterra, amplitude LUT, and lightweight
time-delay neural-network alternatives so every label-fitting route in the
theory document can be benchmarked with the same validation waveform.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


def _DelaySignal(inputSignal: np.ndarray, sampleDelay: int) -> np.ndarray:
    """Return a causal integer-delayed copy with zero initial conditions."""

    delayedSignal = np.zeros_like(inputSignal, dtype=np.complex128)
    if sampleDelay == 0:
        delayedSignal[:] = inputSignal
    elif sampleDelay < inputSignal.size:
        delayedSignal[sampleDelay:] = inputSignal[:-sampleDelay]
    return delayedSignal


VolterraSpec = Tuple[str, int, int, int]


def _BuildVolterraSpecs(memoryDepth: int) -> List[VolterraSpec]:
    """Enumerate first- and third-order complex-baseband Volterra terms."""

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


def _BuildVolterraBasis(
    inputSignal: np.ndarray, featureSpecs: List[VolterraSpec]
) -> np.ndarray:
    """Build a simplified complex Volterra basis containing x*x*conj(x)."""

    complexInput = np.asarray(inputSignal, dtype=np.complex128).reshape(-1)
    maximumDelay = max(
        max(firstDelay, secondDelay, conjugateDelay)
        for _, firstDelay, secondDelay, conjugateDelay in featureSpecs
    )
    delayCache = {
        sampleDelay: _DelaySignal(complexInput, sampleDelay)
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
        """Evaluate the fitted complex Volterra expansion."""

        basisMatrix = _BuildVolterraBasis(inputSignal, self.featureSpecs)
        return basisMatrix @ self.coefficients


def FitVolterraPredistorter(
    referenceSignal: np.ndarray,
    learnedInput: np.ndarray,
    memoryDepth: int = 3,
    ridgeFactor: float = 1e-6,
) -> VolterraPredistorter:
    """Fit a simplified complex Volterra DPD by normalized ridge regression."""

    complexReference = np.asarray(referenceSignal, dtype=np.complex128).reshape(-1)
    complexLabels = np.asarray(learnedInput, dtype=np.complex128).reshape(-1)
    if complexReference.size != complexLabels.size:
        raise ValueError("referenceSignal and learnedInput must have equal length")
    featureSpecs = _BuildVolterraSpecs(memoryDepth)
    basisMatrix = _BuildVolterraBasis(complexReference, featureSpecs)
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
        """Select the amplitude bin and apply its fitted complex gain."""

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
    """Fit the regularized complex gain of every input-amplitude bin."""

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


def _BuildNeuralInputs(
    inputSignal: np.ndarray, memoryDepth: int
) -> np.ndarray:
    """Build real I/Q/envelope time-delay inputs for a lightweight DPD NN."""

    complexInput = np.asarray(inputSignal, dtype=np.complex128).reshape(-1)
    inputColumns = []
    for memoryIndex in range(memoryDepth):
        delayedSignal = _DelaySignal(complexInput, memoryIndex)
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
        """Evaluate standardized linear skip features and tanh hidden units."""

        neuralInputs = _BuildNeuralInputs(inputSignal, self.memoryDepth)
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
    neuralInputs = _BuildNeuralInputs(referenceSignal, memoryDepth)
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
