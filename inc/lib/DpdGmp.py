"""Object-oriented generalized-memory-polynomial DPD implementation."""

from collections import ChainMap
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np

from .DpdIlc import BuildFeatureSpecs, BuildGmpBasisChunk

# Support both ``inc.lib`` and the compatibility ``lib`` package.
if __package__ and "." in __package__:
    from ..utils.ConfigUtils import (
        FilterRecognizedParameters,
        RecognizedParameterView,
    )
    from ..utils.FixedPoint import FixedPoint
    from ..utils.SigProc import SigProc
else:
    from utils.ConfigUtils import (
        FilterRecognizedParameters,
        RecognizedParameterView,
    )
    from utils.FixedPoint import FixedPoint
    from utils.SigProc import SigProc


@dataclass(frozen=True)
class DpdGmpTrainingResult:
    """Store coefficient-update diagnostics without RF performance metrics."""

    sampleCount: int
    segmentCount: int
    featureCount: int
    beforeNmseDb: float
    afterNmseDb: float
    regularizedConditionNumber: float
    normalizedCoefficientUpdateNorm: float
    regionSmoothnessPenalty: float = 0.0

    def ToDict(self) -> Dict[str, object]:
        """Return a stable dictionary containing all training diagnostics.

        Processing details:
            Algorithm: Copy immutable counts, normalized errors, matrix
            conditioning, and update magnitude without recalculating them.

        Returns:
            result: Serialization-ready training diagnostic dictionary.
        """

        return {
            "sampleCount": self.sampleCount,
            "segmentCount": self.segmentCount,
            "featureCount": self.featureCount,
            "beforeNmseDb": self.beforeNmseDb,
            "afterNmseDb": self.afterNmseDb,
            "regularizedConditionNumber": (
                self.regularizedConditionNumber
            ),
            "normalizedCoefficientUpdateNorm": (
                self.normalizedCoefficientUpdateNorm
            ),
            "regionSmoothnessPenalty": self.regionSmoothnessPenalty,
        }


class DpdGmp:
    """Train, update, and apply one SISO GMP digital predistorter.

    The model uses the same deterministic main, lagging-envelope, and
    leading-envelope feature ordering as the ILC deployment fitter. Public
    floating/fixed-point conversion occurs only at method boundaries; basis
    construction and coefficient estimation remain normalized floating point.
    """

    def __init__(
        self,
        parameters: Optional[Mapping[str, object]] = None,
        width: Optional[int] = None,
        **parameterOverrides: object,
    ) -> None:
        """Initialize internal defaults, live overrides, and identity DPD.

        Processing details:
            Algorithm: Define all default parameters inside the constructor,
            layer recognized caller overrides with ChainMap precedence,
            validate the resolved GMP structure, construct deterministic
            feature specifications, and initialize the zero-delay first-order
            coefficient to one.

        Args:
            parameters: Optional caller-owned live mapping of changed values.
            width: Optional external I/Q width; None uses the internal default.
            parameterOverrides: Highest-priority local configuration values.

        Returns:
            result: None. An identity-initialized DPD is ready for training.
        """

        self.defaultParameters: Mapping[str, object] = MappingProxyType(
            {
                "nonlinearOrders": (1, 3, 5, 7),
                "memoryDepth": 3,
                "crossMemoryDepth": 2,
                "ridgeFactor": 1.0e-6,
                "coefficientLearningRate": 1.0,
                "chunkSize": 8192,
                "peakWeightExponent": 0.0,
                "maximumOutputMagnitude": 2.0,
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
                "DpdGmp",
            )
        )
        recognizedOverrides = FilterRecognizedParameters(
            directOverrides,
            self.defaultParameters,
            "DpdGmp",
        )
        self.parameters: ChainMap[str, object] = ChainMap(
            recognizedOverrides,
            externalParameters,
            self.defaultParameters,
        )
        self.featureSpecs: List[Tuple[str, int, int, int]] = []
        self.coefficients = np.zeros(0, dtype=np.complex128)
        self.lastTrainingResult: Optional[DpdGmpTrainingResult] = None
        self.activeStructure: Tuple[
            Tuple[int, ...], int, int
        ] = (tuple(), 0, 0)
        self.ValidateParameters()
        self.RebuildStructure(resetCoefficients=True)

    @property
    def Width(self) -> int:
        """Return the public signed I/Q component width.

        Processing details:
            Algorithm: Read the validated ChainMap value without changing the
            caller-owned mapping.

        Returns:
            result: Zero for floating mode or a positive fixed-point width.
        """

        return cast(int, self.parameters["width"])

    width = Width

    def GetParameters(self) -> Dict[str, object]:
        """Return a flattened snapshot of the resolved DPD configuration.

        Processing details:
            Algorithm: Resolve every ChainMap layer into a detached ordinary
            dictionary so caller mutation cannot change DPD state.

        Returns:
            result: All active GMP, fitting, clipping, and width settings.
        """

        self.SynchronizeStructure()
        return dict(self.parameters)

    def UpdateParameters(self, **parameterOverrides: object) -> None:
        """Apply validated parameter changes and rebuild when structure changes.

        Processing details:
            Algorithm: Filter unknown keys with warnings, update the local
            ChainMap layer transactionally, validate all values, and reset the
            identity coefficients only when order or memory dimensions change.

        Args:
            parameterOverrides: Supported values to place in the local layer.

        Returns:
            result: None. Valid settings affect subsequent training/inference.
        """

        recognizedOverrides = FilterRecognizedParameters(
            parameterOverrides,
            self.defaultParameters,
            "DpdGmp.UpdateParameters",
        )
        previousOverrides = dict(self.parameters.maps[0])
        previousStructure = self.activeStructure
        self.parameters.maps[0].update(recognizedOverrides)
        try:
            self.ValidateParameters()
            structureChanged = self.ResolveStructure() != previousStructure
            if structureChanged:
                self.RebuildStructure(resetCoefficients=True)
        except (TypeError, ValueError):
            self.parameters.maps[0].clear()
            self.parameters.maps[0].update(previousOverrides)
            raise

    def ValidateParameters(self) -> None:
        """Validate GMP structure, solver controls, clipping, and public width.

        Processing details:
            Algorithm: Require unique increasing positive odd orders, valid
            memory dimensions, positive ridge/chunk controls, a convex update
            mixing factor, nonnegative peak weighting, optional positive
            output clipping, and a supported FixedPoint width.

        Returns:
            result: None. Invalid configuration raises a descriptive error.
        """

        nonlinearOrders = self.parameters["nonlinearOrders"]
        if (
            not isinstance(nonlinearOrders, tuple)
            or not nonlinearOrders
            or any(
                not isinstance(nonlinearOrder, int)
                or isinstance(nonlinearOrder, bool)
                or nonlinearOrder < 1
                or nonlinearOrder % 2 == 0
                for nonlinearOrder in nonlinearOrders
            )
            or tuple(sorted(set(nonlinearOrders))) != nonlinearOrders
        ):
            raise ValueError(
                "nonlinearOrders must be a strictly increasing tuple "
                "of positive odd integers"
            )
        if 1 not in nonlinearOrders:
            raise ValueError(
                "nonlinearOrders must include one for the identity path"
            )
        for parameterName, minimumValue in (
            ("memoryDepth", 1),
            ("crossMemoryDepth", 0),
            ("chunkSize", 64),
        ):
            parameterValue = self.parameters[parameterName]
            if (
                not isinstance(parameterValue, int)
                or isinstance(parameterValue, bool)
                or parameterValue < minimumValue
            ):
                raise ValueError(
                    f"{parameterName} must be an integer no smaller "
                    f"than {minimumValue}"
                )
        for parameterName in ("ridgeFactor",):
            parameterValue = self.parameters[parameterName]
            if (
                not isinstance(parameterValue, (int, float))
                or isinstance(parameterValue, bool)
                or not np.isfinite(parameterValue)
                or float(parameterValue) <= 0.0
            ):
                raise ValueError(
                    f"{parameterName} must be finite and positive"
                )
        coefficientLearningRate = self.parameters[
            "coefficientLearningRate"
        ]
        if (
            not isinstance(coefficientLearningRate, (int, float))
            or isinstance(coefficientLearningRate, bool)
            or not np.isfinite(coefficientLearningRate)
            or not 0.0 < float(coefficientLearningRate) <= 1.0
        ):
            raise ValueError(
                "coefficientLearningRate must be in the interval (0, 1]"
            )
        peakWeightExponent = self.parameters["peakWeightExponent"]
        if (
            not isinstance(peakWeightExponent, (int, float))
            or isinstance(peakWeightExponent, bool)
            or not np.isfinite(peakWeightExponent)
            or float(peakWeightExponent) < 0.0
        ):
            raise ValueError(
                "peakWeightExponent must be finite and nonnegative"
            )
        maximumOutputMagnitude = self.parameters[
            "maximumOutputMagnitude"
        ]
        if maximumOutputMagnitude is not None and (
            not isinstance(maximumOutputMagnitude, (int, float))
            or isinstance(maximumOutputMagnitude, bool)
            or not np.isfinite(maximumOutputMagnitude)
            or float(maximumOutputMagnitude) <= 0.0
        ):
            raise ValueError(
                "maximumOutputMagnitude must be positive finite or None"
            )
        FixedPoint(self.width)
        BuildFeatureSpecs(*self.ResolveStructure())

    def ResolveStructure(self) -> Tuple[Tuple[int, ...], int, int]:
        """Return the validated GMP structure in canonical tuple form.

        Processing details:
            Algorithm: Cast the ChainMap values into immutable order and
            memory dimensions used by basis enumeration.

        Returns:
            result: Nonlinear orders, main memory depth, and cross depth.
        """

        return (
            cast(Tuple[int, ...], self.parameters["nonlinearOrders"]),
            cast(int, self.parameters["memoryDepth"]),
            cast(int, self.parameters["crossMemoryDepth"]),
        )

    def SynchronizeStructure(self) -> None:
        """Synchronize live external structural parameters with coefficients.

        Processing details:
            Algorithm: Validate the current ChainMap, compare its order and
            memory dimensions with the structure that owns the coefficient
            vector, and rebuild an identity model when a caller-owned live
            mapping changed those dimensions between method calls.

        Returns:
            result: None. Nonstructural live updates preserve coefficients;
            structural live updates safely reset them.
        """

        self.ValidateParameters()
        if self.ResolveStructure() != self.activeStructure:
            self.RebuildStructure(resetCoefficients=True)

    def RebuildStructure(self, resetCoefficients: bool) -> None:
        """Rebuild feature indices and optionally restore identity coefficients.

        Processing details:
            Algorithm: Enumerate the canonical GMP main/lagging/leading order,
            resize coefficient storage, and place one on the zero-delay
            first-order main term whenever a reset is requested.

        Args:
            resetCoefficients: Whether to discard coefficients and use identity.

        Returns:
            result: None. Feature and coefficient state are synchronized.
        """

        featureSpecs = BuildFeatureSpecs(*self.ResolveStructure())
        if not resetCoefficients and len(featureSpecs) != len(
            self.coefficients
        ):
            raise ValueError(
                "coefficient count cannot be preserved across structure change"
            )
        self.featureSpecs = list(featureSpecs)
        self.activeStructure = self.ResolveStructure()
        if resetCoefficients:
            self.coefficients = np.zeros(
                len(self.featureSpecs), dtype=np.complex128
            )
            identityIndex = self.featureSpecs.index(
                ("main", 1, 0, 0)
            )
            self.coefficients[identityIndex] = 1.0 + 0.0j
            self.lastTrainingResult = None

    def ResetCoefficients(self) -> None:
        """Restore the exact identity DPD for the active feature structure.

        Processing details:
            Algorithm: Reuse canonical structure rebuilding and set only the
            zero-delay first-order main coefficient to one.

        Returns:
            result: None. Previously fitted coefficients are discarded.
        """

        self.ValidateParameters()
        self.RebuildStructure(resetCoefficients=True)

    def GetFeatureSpecs(self) -> Tuple[Tuple[str, int, int, int], ...]:
        """Return immutable GMP coefficient-to-feature index metadata.

        Processing details:
            Algorithm: Convert internal deterministic feature ordering to a
            tuple so callers can inspect but not mutate it.

        Returns:
            result: Main, lagging, and leading feature specification tuple.
        """

        self.SynchronizeStructure()
        return tuple(self.featureSpecs)

    def GetCoefficients(self) -> np.ndarray:
        """Return a detached copy of the current complex GMP coefficients.

        Processing details:
            Algorithm: Copy coefficient storage so external mutation cannot
            silently change inference.

        Returns:
            result: Complex vector in GetFeatureSpecs ordering.
        """

        self.SynchronizeStructure()
        return self.coefficients.copy()

    def SetCoefficients(self, coefficients: np.ndarray) -> None:
        """Replace coefficients after exact size and finiteness validation.

        Processing details:
            Algorithm: Flatten the supplied complex vector, require one value
            per feature and finite real/imaginary parts, then copy it into
            owned storage.

        Args:
            coefficients: Complex coefficients in canonical feature order.

        Returns:
            result: None. Valid coefficients become active for inference.
        """

        self.SynchronizeStructure()
        complexCoefficients = np.asarray(
            coefficients, dtype=np.complex128
        ).reshape(-1)
        if complexCoefficients.size != len(self.featureSpecs):
            raise ValueError(
                "coefficients must contain one value per GMP feature"
            )
        if not np.all(np.isfinite(complexCoefficients)):
            raise ValueError("coefficients must be finite")
        self.coefficients = complexCoefficients.copy()
        self.lastTrainingResult = None

    def PreparePublicSignal(
        self,
        inputSignal: np.ndarray,
        signalName: str,
        fullScaleAmplitude: float = 1.0,
    ) -> np.ndarray:
        """Decode and validate one public signal as a floating-point vector.

        Processing details:
            Algorithm: Use the configured FixedPoint boundary exactly once,
            flatten to one complex stream, and reject empty or nonfinite data.

        Args:
            inputSignal: Floating samples or public fixed-point complex codes.
            signalName: Name included in validation errors.
            fullScaleAmplitude: Physical component magnitude represented by
                the fixed-point code rail. The default is the normalized DPD
                input scale of one.

        Returns:
            result: Normalized finite complex training/inference vector.
        """

        decodedSignal = FixedPoint(
            self.width, fullScaleAmplitude
        ).DecodeComplex(inputSignal).reshape(-1)
        if decodedSignal.size == 0:
            raise ValueError(f"{signalName} cannot be empty")
        if not np.all(np.isfinite(decodedSignal)):
            raise ValueError(
                f"{signalName} contains NaN or infinite values"
            )
        return decodedSignal

    def LimitMagnitude(self, inputSignal: np.ndarray) -> np.ndarray:
        """Apply the configured smooth-free hard envelope safety limit.

        Processing details:
            Algorithm: Preserve phase and scale only samples exceeding the
            configured maximum magnitude; None disables normalized clipping.

        Args:
            inputSignal: Normalized floating DPD output.

        Returns:
            result: Magnitude-limited complex vector.
        """

        maximumOutputMagnitude = self.parameters[
            "maximumOutputMagnitude"
        ]
        if maximumOutputMagnitude is None:
            return inputSignal.copy()
        outputSignal = inputSignal.copy()
        magnitude = np.abs(outputSignal)
        limitValue = float(maximumOutputMagnitude)
        excessiveMask = magnitude > limitValue
        outputSignal[excessiveMask] *= (
            limitValue / magnitude[excessiveMask]
        )
        return outputSignal

    def ProcessFloating(self, inputSignal: np.ndarray) -> np.ndarray:
        """Apply GMP coefficients to normalized floating-point samples.

        Processing details:
            Algorithm: Validate the vector, construct bounded basis chunks,
            multiply each chunk by current coefficients, and enforce the
            configured output-envelope limit.

        Args:
            inputSignal: Normalized finite complex desired waveform.

        Returns:
            result: Normalized floating predistorted PA input.
        """

        self.SynchronizeStructure()
        complexInput = np.asarray(
            inputSignal, dtype=np.complex128
        ).reshape(-1)
        if complexInput.size == 0 or not np.all(np.isfinite(complexInput)):
            raise ValueError(
                "inputSignal must be a finite nonempty complex vector"
            )
        outputSignal = np.zeros_like(complexInput)
        chunkSize = cast(int, self.parameters["chunkSize"])
        for startIndex in range(0, complexInput.size, chunkSize):
            stopIndex = min(startIndex + chunkSize, complexInput.size)
            basisChunk = self.BuildBasisChunk(
                complexInput,
                startIndex,
                stopIndex,
            )
            outputSignal[startIndex:stopIndex] = (
                basisChunk @ self.coefficients
            )
        return self.LimitMagnitude(outputSignal)

    def BuildBasisChunk(
        self,
        inputSignal: np.ndarray,
        startIndex: int,
        stopIndex: int,
    ) -> np.ndarray:
        """Build one deterministic GMP regression-basis chunk.

        Processing details:
            Algorithm: Evaluate the active main, lagging-envelope, and
            leading-envelope feature specifications on the requested sample
            interval. Subclasses may override this method to add physically
            motivated branches while reusing the common training solver.

        Args:
            inputSignal: Normalized finite complex desired waveform.
            startIndex: Inclusive output-sample index of the basis chunk.
            stopIndex: Exclusive output-sample index of the basis chunk.

        Returns:
            result: Complex matrix with one column per active coefficient.
        """

        return BuildGmpBasisChunk(
            inputSignal,
            self.featureSpecs,
            startIndex,
            stopIndex,
        )

    def BuildAdditionalRegularizationMatrix(
        self,
        featureScale: np.ndarray,
        diagonalScale: float,
    ) -> np.ndarray:
        """Return an optional normalized-coordinate coefficient penalty.

        Processing details:
            Algorithm: Validate the supplied feature scales and return a zero
            Hermitian matrix for the ordinary unconstrained GMP. Subclasses
            may override this hook to add positive-semidefinite structure to
            the common joint normal equations without duplicating the solver.

        Args:
            featureScale: Positive RMS normalization for every basis column.
            diagonalScale: Positive mean diagonal of the data normal matrix.

        Returns:
            result: Feature-by-feature complex regularization matrix.
        """

        resolvedFeatureScale = np.asarray(
            featureScale, dtype=float
        ).reshape(-1)
        if (
            resolvedFeatureScale.size != len(self.featureSpecs)
            or not np.all(np.isfinite(resolvedFeatureScale))
            or np.any(resolvedFeatureScale <= 0.0)
            or not np.isfinite(diagonalScale)
            or diagonalScale <= 0.0
        ):
            raise ValueError(
                "featureScale and diagonalScale must be finite and positive"
            )
        return np.zeros(
            (len(self.featureSpecs), len(self.featureSpecs)),
            dtype=np.complex128,
        )

    def CalculateRegionSmoothnessPenalty(
        self, coefficients: np.ndarray
    ) -> float:
        """Return the regional coefficient-difference diagnostic.

        Processing details:
            Algorithm: Validate one coefficient vector and return zero for an
            ordinary GMP, which has no independent envelope-region blocks.
            Piecewise subclasses override this metric using adjacent regions.

        Args:
            coefficients: Complex coefficient vector in active feature order.

        Returns:
            result: Nonnegative adjacent-region squared-difference sum.
        """

        complexCoefficients = np.asarray(
            coefficients, dtype=np.complex128
        ).reshape(-1)
        if (
            complexCoefficients.size != len(self.featureSpecs)
            or not np.all(np.isfinite(complexCoefficients))
        ):
            raise ValueError(
                "coefficients must contain one finite value per feature"
            )
        return 0.0

    def Process(self, inputSignal: np.ndarray) -> np.ndarray:
        """Apply DPD while preserving the configured public data convention.

        Processing details:
            Algorithm: Decode public samples once, process the normalized
            vector, and encode the result once so floating and fixed modes
            expose identical shapes and complex128 containers.

        Args:
            inputSignal: Public floating waveform or integer-valued I/Q codes.

        Returns:
            result: Predistorted waveform in the same public convention.
        """

        interfaceFormat = FixedPoint(self.width)
        floatingInput = self.PreparePublicSignal(
            inputSignal, "inputSignal"
        )
        floatingOutput = self.ProcessFloating(floatingInput)
        return interfaceFormat.EncodeComplex(floatingOutput)

    def BuildSampleWeights(
        self,
        referenceSignal: np.ndarray,
        sampleWeights: Optional[np.ndarray],
        segmentWeight: float,
    ) -> np.ndarray:
        """Combine explicit, segment, and envelope-peak training weights.

        Processing details:
            Algorithm: Validate optional nonnegative sample weights, multiply
            by a positive segment weight, apply normalized envelope weighting
            when requested, and normalize mean weight to one so ridge strength
            remains comparable across datasets.

        Args:
            referenceSignal: Normalized regression input for one segment.
            sampleWeights: Optional per-sample importance values.
            segmentWeight: Positive relative importance of this segment.

        Returns:
            result: Positive finite mean-one sample weight vector.
        """

        if (
            not isinstance(segmentWeight, (int, float))
            or isinstance(segmentWeight, bool)
            or not np.isfinite(segmentWeight)
            or float(segmentWeight) <= 0.0
        ):
            raise ValueError("every segmentWeight must be finite and positive")
        if sampleWeights is None:
            resolvedWeights = np.ones(
                referenceSignal.size, dtype=float
            )
        else:
            resolvedWeights = np.asarray(
                sampleWeights, dtype=float
            ).reshape(-1)
            if resolvedWeights.size != referenceSignal.size:
                raise ValueError(
                    "sampleWeights must match its training segment length"
                )
            if (
                not np.all(np.isfinite(resolvedWeights))
                or np.any(resolvedWeights < 0.0)
                or not np.any(resolvedWeights > 0.0)
            ):
                raise ValueError(
                    "sampleWeights must be finite, nonnegative, and nonzero"
                )
        peakWeightExponent = float(
            self.parameters["peakWeightExponent"]
        )
        if peakWeightExponent > 0.0:
            peakMagnitude = max(
                float(np.max(np.abs(referenceSignal))),
                np.finfo(float).tiny,
            )
            normalizedMagnitude = np.abs(referenceSignal) / peakMagnitude
            resolvedWeights = resolvedWeights * np.maximum(
                normalizedMagnitude, 0.05
            ) ** peakWeightExponent
        meanWeight = float(np.mean(resolvedWeights))
        if meanWeight <= np.finfo(float).tiny:
            raise ValueError("resolved sample weights have zero mean")
        # Normalize the within-segment shape first, then apply the segment
        # weight. Reversing this order would cancel segmentWeight completely
        # when dividing by the mean and would make multi-power weighting inert.
        return (
            resolvedWeights
            / meanWeight
            * float(segmentWeight)
        )

    def CalculateNmse(
        self,
        referenceSignal: np.ndarray,
        targetSignal: np.ndarray,
        sampleWeights: Optional[np.ndarray] = None,
    ) -> float:
        """Calculate optional-weight GMP label-modeling NMSE in decibels.

        Processing details:
            Algorithm: Decode public reference and target samples, apply the
            current DPD, validate optional nonnegative weights without adding
            the configured training peak exponent, and calculate weighted
            residual power divided by weighted target power.

        Args:
            referenceSignal: Desired waveform at the DPD input.
            targetSignal: PA-input or ILC-label waveform to predict.
            sampleWeights: Optional explicit evaluation weights.

        Returns:
            result: Weighted label-modeling NMSE in dB; lower is better.
        """

        preparedReference = self.PreparePublicSignal(
            referenceSignal, "referenceSignal"
        )
        preparedTarget = self.PreparePublicSignal(
            targetSignal, "targetSignal"
        )
        if preparedReference.size != preparedTarget.size:
            raise ValueError(
                "referenceSignal and targetSignal must have equal length"
            )
        if sampleWeights is None:
            resolvedWeights = np.ones(
                preparedReference.size, dtype=float
            )
        else:
            resolvedWeights = np.asarray(
                sampleWeights, dtype=float
            ).reshape(-1)
            if resolvedWeights.size != preparedReference.size:
                raise ValueError(
                    "sampleWeights must match the signal length"
                )
            if (
                not np.all(np.isfinite(resolvedWeights))
                or np.any(resolvedWeights < 0.0)
                or not np.any(resolvedWeights > 0.0)
            ):
                raise ValueError(
                    "sampleWeights must be finite, nonnegative, and nonzero"
                )
        predictedTarget = self.ProcessFloating(preparedReference)
        residual = preparedTarget - predictedTarget
        residualPower = float(
            np.sum(resolvedWeights * np.abs(residual) ** 2)
        )
        targetPower = float(
            np.sum(resolvedWeights * np.abs(preparedTarget) ** 2)
        )
        numericFloor = np.finfo(float).tiny
        return float(
            10.0
            * np.log10(
                max(residualPower, numericFloor)
                / max(targetPower, numericFloor)
            )
        )

    def Fit(
        self,
        referenceSignal: np.ndarray,
        targetSignal: np.ndarray,
        sampleWeights: Optional[np.ndarray] = None,
    ) -> DpdGmpTrainingResult:
        """Reset and fit one GMP coefficient set to a waveform pair.

        Processing details:
            Algorithm: Restore the identity prior and perform one regularized
            coefficient update using desired waveform features to predict the
            target PA-input labels.

        Args:
            referenceSignal: Desired waveform presented to the DPD input.
            targetSignal: Predistorted PA-input labels to reproduce.
            sampleWeights: Optional per-sample regression importance.

        Returns:
            result: Before/after normalized fit and solver diagnostics.
        """

        self.ResetCoefficients()
        return self.UpdateCoefficients(
            referenceSignal,
            targetSignal,
            sampleWeights,
        )

    def FitSegments(
        self,
        referenceSignals: Sequence[np.ndarray],
        targetSignals: Sequence[np.ndarray],
        segmentWeights: Optional[Sequence[float]] = None,
        sampleWeights: Optional[Sequence[Optional[np.ndarray]]] = None,
    ) -> DpdGmpTrainingResult:
        """Reset and jointly fit independent waveform or power segments.

        Processing details:
            Algorithm: Restore identity and accumulate each segment's normal
            equations independently so zero padding, delays, and memory never
            leak across artificial concatenation boundaries.

        Args:
            referenceSignals: Desired DPD-input waveform segments.
            targetSignals: Matching predistorted PA-input label segments.
            segmentWeights: Optional positive relative weights per segment.
            sampleWeights: Optional per-segment arrays of sample weights.

        Returns:
            result: Joint regularized fit diagnostics.
        """

        self.ResetCoefficients()
        return self.UpdateCoefficientSegments(
            referenceSignals,
            targetSignals,
            segmentWeights,
            sampleWeights,
        )

    def UpdateCoefficients(
        self,
        referenceSignal: np.ndarray,
        targetSignal: np.ndarray,
        sampleWeights: Optional[np.ndarray] = None,
    ) -> DpdGmpTrainingResult:
        """Update coefficients from one new waveform pair.

        Processing details:
            Algorithm: Treat current coefficients as a ridge prior, solve the
            weighted regularized least-squares target for one segment, and
            blend toward it by coefficientLearningRate.

        Args:
            referenceSignal: Desired waveform defining GMP basis functions.
            targetSignal: Desired PA input or ILC-learned label waveform.
            sampleWeights: Optional per-sample importance values.

        Returns:
            result: Training error, conditioning, and update diagnostics.
        """

        return self.UpdateCoefficientSegments(
            (referenceSignal,),
            (targetSignal,),
            (1.0,),
            (sampleWeights,),
        )

    def UpdateCoefficientSegments(
        self,
        referenceSignals: Sequence[np.ndarray],
        targetSignals: Sequence[np.ndarray],
        segmentWeights: Optional[Sequence[float]] = None,
        sampleWeights: Optional[Sequence[Optional[np.ndarray]]] = None,
    ) -> DpdGmpTrainingResult:
        """Update one coefficient set from independent weighted segments.

        Processing details:
            Algorithm: Decode and validate every segment, estimate weighted
            column RMS scales in a first pass, accumulate normalized complex
            normal equations in a second pass, solve a ridge problem centered
            on current coefficients, blend the solution, and evaluate
            weighted before/after NMSE in a final pass.

        Args:
            referenceSignals: Desired waveform segments defining GMP bases.
            targetSignals: Matching DPD target-label segments.
            segmentWeights: Optional positive relative weights.
            sampleWeights: Optional per-segment sample-weight arrays.

        Returns:
            result: Immutable coefficient-update diagnostics.
        """

        self.SynchronizeStructure()
        if (
            isinstance(referenceSignals, (str, bytes))
            or isinstance(targetSignals, (str, bytes))
        ):
            raise TypeError("training segments must be signal sequences")
        referenceSequence = tuple(referenceSignals)
        targetSequence = tuple(targetSignals)
        if (
            not referenceSequence
            or len(referenceSequence) != len(targetSequence)
        ):
            raise ValueError(
                "referenceSignals and targetSignals must have equal "
                "nonzero segment counts"
            )
        resolvedSegmentWeights = (
            tuple(1.0 for _ in referenceSequence)
            if segmentWeights is None
            else tuple(segmentWeights)
        )
        resolvedSampleWeights = (
            tuple(None for _ in referenceSequence)
            if sampleWeights is None
            else tuple(sampleWeights)
        )
        if (
            len(resolvedSegmentWeights) != len(referenceSequence)
            or len(resolvedSampleWeights) != len(referenceSequence)
        ):
            raise ValueError(
                "segmentWeights and sampleWeights must match segment count"
            )
        preparedReferences = []
        preparedTargets = []
        preparedWeights = []
        for segmentIndex, (
            referenceSignal,
            targetSignal,
            segmentWeight,
            segmentSampleWeights,
        ) in enumerate(
            zip(
                referenceSequence,
                targetSequence,
                resolvedSegmentWeights,
                resolvedSampleWeights,
            )
        ):
            preparedReference = self.PreparePublicSignal(
                referenceSignal,
                f"referenceSignals[{segmentIndex}]",
            )
            preparedTarget = self.PreparePublicSignal(
                targetSignal,
                f"targetSignals[{segmentIndex}]",
            )
            if preparedReference.size != preparedTarget.size:
                raise ValueError(
                    "every reference and target segment pair must "
                    "have equal length"
                )
            preparedReferences.append(preparedReference)
            preparedTargets.append(preparedTarget)
            preparedWeights.append(
                self.BuildSampleWeights(
                    preparedReference,
                    segmentSampleWeights,
                    segmentWeight,
                )
            )
        featureCount = len(self.featureSpecs)
        featureEnergy = np.zeros(featureCount, dtype=float)
        totalWeight = 0.0
        chunkSize = cast(int, self.parameters["chunkSize"])
        for referenceSignal, weightSignal in zip(
            preparedReferences, preparedWeights
        ):
            for startIndex in range(0, referenceSignal.size, chunkSize):
                stopIndex = min(
                    startIndex + chunkSize, referenceSignal.size
                )
                basisChunk = self.BuildBasisChunk(
                    referenceSignal,
                    startIndex,
                    stopIndex,
                )
                weightChunk = weightSignal[startIndex:stopIndex]
                featureEnergy += np.sum(
                    np.abs(basisChunk) ** 2
                    * weightChunk.reshape(-1, 1),
                    axis=0,
                )
                totalWeight += float(np.sum(weightChunk))
        featureScale = np.sqrt(
            featureEnergy / max(totalWeight, np.finfo(float).tiny)
        )
        featureScale = np.maximum(featureScale, 1.0e-12)
        normalMatrix = np.zeros(
            (featureCount, featureCount), dtype=np.complex128
        )
        targetProjection = np.zeros(
            featureCount, dtype=np.complex128
        )
        for referenceSignal, targetSignal, weightSignal in zip(
            preparedReferences,
            preparedTargets,
            preparedWeights,
        ):
            for startIndex in range(0, referenceSignal.size, chunkSize):
                stopIndex = min(
                    startIndex + chunkSize, referenceSignal.size
                )
                basisChunk = self.BuildBasisChunk(
                    referenceSignal,
                    startIndex,
                    stopIndex,
                )
                normalizedBasis = basisChunk / featureScale
                squareRootWeight = np.sqrt(
                    weightSignal[startIndex:stopIndex]
                ).reshape(-1, 1)
                weightedBasis = normalizedBasis * squareRootWeight
                weightedTarget = (
                    targetSignal[startIndex:stopIndex]
                    * squareRootWeight[:, 0]
                )
                normalMatrix += (
                    weightedBasis.conj().T @ weightedBasis
                )
                targetProjection += (
                    weightedBasis.conj().T @ weightedTarget
                )
        diagonalScale = max(
            float(np.mean(np.real(np.diag(normalMatrix)))),
            np.finfo(float).tiny,
        )
        ridgeScale = (
            float(self.parameters["ridgeFactor"]) * diagonalScale
        )
        additionalRegularization = (
            self.BuildAdditionalRegularizationMatrix(
                featureScale,
                diagonalScale,
            )
        )
        if (
            additionalRegularization.shape
            != (featureCount, featureCount)
            or not np.all(np.isfinite(additionalRegularization))
        ):
            raise RuntimeError(
                "additional GMP regularization returned an invalid matrix"
            )
        regularizedMatrix = normalMatrix + (
            ridgeScale * np.eye(featureCount)
            + additionalRegularization
        )
        priorNormalizedCoefficients = (
            self.coefficients * featureScale
        )
        solvedNormalizedCoefficients = np.linalg.solve(
            regularizedMatrix,
            targetProjection
            + ridgeScale * priorNormalizedCoefficients,
        )
        coefficientLearningRate = float(
            self.parameters["coefficientLearningRate"]
        )
        updatedNormalizedCoefficients = (
            priorNormalizedCoefficients
            + coefficientLearningRate
            * (
                solvedNormalizedCoefficients
                - priorNormalizedCoefficients
            )
        )
        updatedCoefficients = (
            updatedNormalizedCoefficients / featureScale
        )
        if not np.all(np.isfinite(updatedCoefficients)):
            raise RuntimeError(
                "regularized GMP coefficient update produced nonfinite values"
            )
        beforeErrorPower = 0.0
        afterErrorPower = 0.0
        targetPower = 0.0
        for referenceSignal, targetSignal, weightSignal in zip(
            preparedReferences,
            preparedTargets,
            preparedWeights,
        ):
            for startIndex in range(0, referenceSignal.size, chunkSize):
                stopIndex = min(
                    startIndex + chunkSize, referenceSignal.size
                )
                basisChunk = self.BuildBasisChunk(
                    referenceSignal,
                    startIndex,
                    stopIndex,
                )
                targetChunk = targetSignal[startIndex:stopIndex]
                weightChunk = weightSignal[startIndex:stopIndex]
                beforePrediction = self.LimitMagnitude(
                    basisChunk @ self.coefficients
                )
                afterPrediction = self.LimitMagnitude(
                    basisChunk @ updatedCoefficients
                )
                beforeResidual = (
                    targetChunk
                    - beforePrediction
                )
                afterResidual = (
                    targetChunk
                    - afterPrediction
                )
                beforeErrorPower += float(
                    np.sum(weightChunk * np.abs(beforeResidual) ** 2)
                )
                afterErrorPower += float(
                    np.sum(weightChunk * np.abs(afterResidual) ** 2)
                )
                targetPower += float(
                    np.sum(weightChunk * np.abs(targetChunk) ** 2)
                )
        numericFloor = np.finfo(float).tiny
        beforeNmseDb = float(
            10.0
            * np.log10(
                max(beforeErrorPower, numericFloor)
                / max(targetPower, numericFloor)
            )
        )
        afterNmseDb = float(
            10.0
            * np.log10(
                max(afterErrorPower, numericFloor)
                / max(targetPower, numericFloor)
            )
        )
        updateNorm = float(
            np.linalg.norm(
                updatedNormalizedCoefficients
                - priorNormalizedCoefficients
            )
        )
        self.coefficients = updatedCoefficients
        trainingResult = DpdGmpTrainingResult(
            sampleCount=sum(
                referenceSignal.size
                for referenceSignal in preparedReferences
            ),
            segmentCount=len(preparedReferences),
            featureCount=featureCount,
            beforeNmseDb=beforeNmseDb,
            afterNmseDb=afterNmseDb,
            regularizedConditionNumber=float(
                np.linalg.cond(regularizedMatrix)
            ),
            normalizedCoefficientUpdateNorm=updateNorm,
            regionSmoothnessPenalty=(
                self.CalculateRegionSmoothnessPenalty(
                    updatedCoefficients
                )
            ),
        )
        self.lastTrainingResult = trainingResult
        return trainingResult

    def FitFromIlc(
        self,
        referenceSignal: np.ndarray,
        learnedInput: np.ndarray,
        sampleWeights: Optional[np.ndarray] = None,
    ) -> DpdGmpTrainingResult:
        """Fit reusable GMP coefficients from converged ILC waveform labels.

        Processing details:
            Algorithm: Use the original desired waveform as regression input
            and the ILC-learned PA input as target, then run the same
            identity-prior weighted ridge fit as Fit.

        Args:
            referenceSignal: Original desired waveform used by ILC.
            learnedInput: Converged predistorted PA input learned by ILC.
            sampleWeights: Optional per-sample regression importance.

        Returns:
            result: Coefficient-fit diagnostics.
        """

        return self.Fit(
            referenceSignal,
            learnedInput,
            sampleWeights,
        )

    def FitIndirect(
        self,
        paInputSignal: np.ndarray,
        paOutputSignal: np.ndarray,
        sampleRateHz: float,
        signalProcessingParameters: Optional[
            Mapping[str, object]
        ] = None,
        sampleWeights: Optional[np.ndarray] = None,
        paOutputFullScaleAmplitude: float = 1.0,
    ) -> DpdGmpTrainingResult:
        """Fit an indirect-learning postinverse and copy it into the DPD.

        Processing details:
            Algorithm: Synchronize and common-gain-normalize measured PA
            output against its input, use the corrected output as postinverse
            regression input and the actual PA input as target, then copy the
            fitted postinverse coefficients into the predistorter.

        Args:
            paInputSignal: Public waveform actually presented to the PA.
            paOutputSignal: Matching public PA output capture.
            sampleRateHz: Complex sample rate used for synchronization.
            signalProcessingParameters: Optional SigProc overrides.
            sampleWeights: Optional per-sample regression importance.
            paOutputFullScaleAmplitude: Physical component magnitude
                represented by the PA/feedback output code rail. The default
                preserves historical Q1 captures.

        Returns:
            result: Indirect-learning coefficient-fit diagnostics.
        """

        floatingInput = self.PreparePublicSignal(
            paInputSignal, "paInputSignal"
        )
        floatingOutput = self.PreparePublicSignal(
            paOutputSignal,
            "paOutputSignal",
            paOutputFullScaleAmplitude,
        )
        if floatingInput.size != floatingOutput.size:
            raise ValueError(
                "paInputSignal and paOutputSignal must have equal length"
            )
        signalProcessor = SigProc(
            floatingInput,
            sampleRateHz,
            parameters=signalProcessingParameters,
        )
        processedOutput = signalProcessor.Process(
            floatingOutput
        ).processedSignal
        interfaceFormat = FixedPoint(self.width)
        return self.Fit(
            interfaceFormat.EncodeComplex(processedOutput),
            interfaceFormat.EncodeComplex(floatingInput),
            sampleWeights,
        )

    def GetLastTrainingResult(
        self,
    ) -> Optional[DpdGmpTrainingResult]:
        """Return the most recent immutable coefficient-update diagnostics.

        Processing details:
            Algorithm: Return the frozen diagnostic record directly because
            it contains only immutable scalars.

        Returns:
            result: Last result or None before coefficient training.
        """

        self.SynchronizeStructure()
        return self.lastTrainingResult


class PiecewiseDpdGmp(DpdGmp):
    """Fit one jointly optimized GMP model across smooth envelope regions.

    The low, middle, and high regions each own a complete GMP coefficient
    vector.  Two C2-continuous envelope transitions blend their basis columns
    before both regression and inference.  Consequently the ordinary
    ``DpdGmp`` solver can estimate every region jointly, while the deployed
    mapping remains continuous through both configured boundaries.
    """

    regionNames: Tuple[str, ...] = ("low", "middle", "high")

    def __init__(
        self,
        parameters: Optional[Mapping[str, object]] = None,
        width: Optional[int] = None,
        **parameterOverrides: object,
    ) -> None:
        """Initialize a three-region GMP with smooth envelope transitions.

        Processing details:
            Algorithm: Define the ordinary GMP controls together with two
            normalized-envelope boundaries and transition widths, layer live
            caller parameters using the project ChainMap convention, validate
            the complete configuration, and create an exact identity mapping
            in all three regions.

        Args:
            parameters: Optional caller-owned live mapping of configuration
                values, including ``envelopeBoundaries`` and
                ``transitionWidths``.
            width: Optional public I/Q component width. None selects the
                internal 16-bit default and zero selects floating point.
            parameterOverrides: Highest-priority recognized configuration
                values.

        Returns:
            result: None. A validated identity-initialized piecewise DPD is
                ready for joint training or inference.
        """

        self.defaultParameters: Mapping[str, object] = MappingProxyType(
            {
                "nonlinearOrders": (1, 3, 5, 7),
                "memoryDepth": 3,
                "crossMemoryDepth": 2,
                "ridgeFactor": 1.0e-6,
                "coefficientLearningRate": 1.0,
                "chunkSize": 8192,
                "peakWeightExponent": 0.0,
                "maximumOutputMagnitude": 2.0,
                "envelopeBoundaries": (0.25, 0.60),
                "transitionWidths": (0.12, 0.18),
                "regionSmoothnessFactor": 1.0e-4,
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
                "PiecewiseDpdGmp",
            )
        )
        recognizedOverrides = FilterRecognizedParameters(
            directOverrides,
            self.defaultParameters,
            "PiecewiseDpdGmp",
        )
        self.parameters: ChainMap[str, object] = ChainMap(
            recognizedOverrides,
            externalParameters,
            self.defaultParameters,
        )
        self.featureSpecs: List[Tuple[str, int, int, int]] = []
        self.baseFeatureSpecs: List[
            Tuple[str, int, int, int]
        ] = []
        self.coefficients = np.zeros(0, dtype=np.complex128)
        self.lastTrainingResult: Optional[DpdGmpTrainingResult] = None
        self.activeStructure: Tuple[
            Tuple[int, ...], int, int
        ] = (tuple(), 0, 0)
        self.activeEnvelopeConfiguration: Tuple[
            Tuple[float, float], Tuple[float, float]
        ] = ((0.0, 0.0), (0.0, 0.0))
        self.ValidateParameters()
        self.RebuildStructure(resetCoefficients=True)

    def ResolveEnvelopeConfiguration(
        self,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Resolve the two boundaries and widths as immutable float pairs.

        Processing details:
            Algorithm: Require each configured value to be a non-text
            two-element sequence, convert finite real scalars to floats,
            enforce positive increasing boundaries and positive widths, and
            return tuples suitable for structural change detection.

        Returns:
            result: ``(envelopeBoundaries, transitionWidths)`` in normalized
                complex-envelope amplitude units.
        """

        resolvedPairs = []
        for parameterName in (
            "envelopeBoundaries",
            "transitionWidths",
        ):
            rawValues = self.parameters[parameterName]
            if (
                isinstance(rawValues, (str, bytes))
                or not isinstance(rawValues, (Sequence, np.ndarray))
                or len(rawValues) != 2
            ):
                raise ValueError(
                    f"{parameterName} must contain exactly two values"
                )
            numericValues = []
            for rawValue in rawValues:
                if (
                    not isinstance(
                        rawValue,
                        (int, float, np.integer, np.floating),
                    )
                    or isinstance(rawValue, (bool, np.bool_))
                    or not np.isfinite(rawValue)
                    or float(rawValue) <= 0.0
                ):
                    raise ValueError(
                        f"every {parameterName} value must be finite and "
                        "positive"
                    )
                numericValues.append(float(rawValue))
            resolvedPairs.append(
                (numericValues[0], numericValues[1])
            )
        envelopeBoundaries = resolvedPairs[0]
        transitionWidths = resolvedPairs[1]
        if envelopeBoundaries[0] >= envelopeBoundaries[1]:
            raise ValueError(
                "envelopeBoundaries must be strictly increasing"
            )
        return (envelopeBoundaries, transitionWidths)

    def ValidateParameters(self) -> None:
        """Validate ordinary GMP and piecewise envelope configuration.

        Processing details:
            Algorithm: Delegate all common order, memory, solver, clipping,
            and fixed-point checks to ``DpdGmp``, then validate both envelope
            boundary and transition-width pairs used by the piecewise basis.

        Returns:
            result: None. Invalid configuration raises a descriptive error.
        """

        super().ValidateParameters()
        self.ResolveEnvelopeConfiguration()
        regionSmoothnessFactor = self.parameters[
            "regionSmoothnessFactor"
        ]
        if (
            not isinstance(
                regionSmoothnessFactor,
                (int, float, np.integer, np.floating),
            )
            or isinstance(regionSmoothnessFactor, (bool, np.bool_))
            or not np.isfinite(regionSmoothnessFactor)
            or float(regionSmoothnessFactor) < 0.0
        ):
            raise ValueError(
                "regionSmoothnessFactor must be finite and nonnegative"
            )

    def UpdateParameters(self, **parameterOverrides: object) -> None:
        """Apply piecewise or ordinary settings transactionally.

        Processing details:
            Algorithm: Filter recognized values, validate the updated live
            configuration, reset coefficients when either GMP dimensions or
            envelope transition settings change, and restore the prior local
            layer and model state after any invalid update.

        Args:
            parameterOverrides: Supported values placed in the highest
                priority local ChainMap layer.

        Returns:
            result: None. Valid changes affect subsequent calls.
        """

        recognizedOverrides = FilterRecognizedParameters(
            parameterOverrides,
            self.defaultParameters,
            "PiecewiseDpdGmp.UpdateParameters",
        )
        previousOverrides = dict(self.parameters.maps[0])
        previousSignature = (
            self.activeStructure,
            self.activeEnvelopeConfiguration,
        )
        self.parameters.maps[0].update(recognizedOverrides)
        try:
            self.ValidateParameters()
            currentSignature = (
                self.ResolveStructure(),
                self.ResolveEnvelopeConfiguration(),
            )
            if currentSignature != previousSignature:
                self.RebuildStructure(resetCoefficients=True)
        except (TypeError, ValueError):
            self.parameters.maps[0].clear()
            self.parameters.maps[0].update(previousOverrides)
            raise

    def SynchronizeStructure(self) -> None:
        """Synchronize live GMP and envelope settings with coefficients.

        Processing details:
            Algorithm: Validate the complete current ChainMap, compare its
            base structure and smooth-region controls with the signatures that
            own the coefficient vector, and restore piecewise identity if a
            caller changed either group between public method calls.

        Returns:
            result: None. Coefficients always match the active piecewise basis.
        """

        self.ValidateParameters()
        if (
            self.ResolveStructure() != self.activeStructure
            or self.ResolveEnvelopeConfiguration()
            != self.activeEnvelopeConfiguration
        ):
            self.RebuildStructure(resetCoefficients=True)

    def RebuildStructure(self, resetCoefficients: bool) -> None:
        """Build low, middle, and high copies of the canonical GMP basis.

        Processing details:
            Algorithm: Enumerate the ordinary GMP features once, create three
            region-prefixed copies in low-to-high order, preserve coefficients
            only when their size remains compatible, and initialize every
            region's zero-delay first-order term to one so partition-of-unity
            weighting implements exact identity before training.

        Args:
            resetCoefficients: Whether to discard coefficients and initialize
                the piecewise identity mapping.

        Returns:
            result: None. Feature metadata and coefficient storage agree.
        """

        baseFeatureSpecs = list(
            BuildFeatureSpecs(*self.ResolveStructure())
        )
        featureSpecs = [
            (
                f"{regionName}_{branchName}",
                nonlinearOrder,
                signalDelay,
                envelopeDelay,
            )
            for regionName in self.regionNames
            for (
                branchName,
                nonlinearOrder,
                signalDelay,
                envelopeDelay,
            ) in baseFeatureSpecs
        ]
        if not resetCoefficients and len(featureSpecs) != len(
            self.coefficients
        ):
            raise ValueError(
                "coefficient count cannot be preserved across structure change"
            )
        self.baseFeatureSpecs = baseFeatureSpecs
        self.featureSpecs = featureSpecs
        self.activeStructure = self.ResolveStructure()
        self.activeEnvelopeConfiguration = (
            self.ResolveEnvelopeConfiguration()
        )
        if resetCoefficients:
            self.coefficients = np.zeros(
                len(self.featureSpecs), dtype=np.complex128
            )
            for regionName in self.regionNames:
                identityIndex = self.featureSpecs.index(
                    (f"{regionName}_main", 1, 0, 0)
                )
                self.coefficients[identityIndex] = 1.0 + 0.0j
            self.lastTrainingResult = None

    def CalculateEnvelopeWeights(
        self, inputSignal: np.ndarray
    ) -> np.ndarray:
        """Return smooth low, middle, and high weights for every sample.

        Processing details:
            Algorithm: Center one finite-width transition on each configured
            envelope boundary, map its clipped position through the C2
            smootherstep ``6z^5 - 15z^4 + 10z^3``, then form
            ``low=1-S1``, ``middle=S1*(1-S2)``, and ``high=S1*S2``. The three
            nonnegative weights sum to one even when transitions overlap.

        Args:
            inputSignal: Normalized finite complex waveform whose instantaneous
                envelope selects the regional mixture.

        Returns:
            result: Samples-by-three real matrix ordered low, middle, high.
        """

        self.SynchronizeStructure()
        complexInput = np.asarray(
            inputSignal, dtype=np.complex128
        ).reshape(-1)
        if complexInput.size == 0 or not np.all(np.isfinite(complexInput)):
            raise ValueError(
                "inputSignal must be a finite nonempty complex vector"
            )
        envelopeMagnitude = np.abs(complexInput)
        (
            envelopeBoundaries,
            transitionWidths,
        ) = self.activeEnvelopeConfiguration
        transitionValues = []
        for boundaryValue, transitionWidth in zip(
            envelopeBoundaries, transitionWidths
        ):
            transitionStart = boundaryValue - 0.5 * transitionWidth
            transitionPosition = np.clip(
                (envelopeMagnitude - transitionStart)
                / transitionWidth,
                0.0,
                1.0,
            )
            smootherStep = np.clip(
                transitionPosition**3
                * (
                    transitionPosition
                    * (6.0 * transitionPosition - 15.0)
                    + 10.0
                ),
                0.0,
                1.0,
            )
            transitionValues.append(smootherStep)
        lowerTransition, upperTransition = transitionValues
        lowWeight = 1.0 - lowerTransition
        middleWeight = lowerTransition * (1.0 - upperTransition)
        highWeight = lowerTransition * upperTransition
        return np.column_stack(
            (lowWeight, middleWeight, highWeight)
        )

    def BuildBasisChunk(
        self,
        inputSignal: np.ndarray,
        startIndex: int,
        stopIndex: int,
    ) -> np.ndarray:
        """Build the joint smoothly weighted three-region GMP basis.

        Processing details:
            Algorithm: Evaluate the canonical GMP columns once for the chunk,
            calculate the same envelope weights used during deployment, and
            concatenate low-, middle-, and high-weighted copies in the exact
            region-major order exposed by ``GetFeatureSpecs``.

        Args:
            inputSignal: Normalized finite desired waveform.
            startIndex: Inclusive output-sample index of the basis chunk.
            stopIndex: Exclusive output-sample index of the basis chunk.

        Returns:
            result: Complex joint-regression matrix with three columns per
                ordinary GMP feature.
        """

        baseBasis = BuildGmpBasisChunk(
            inputSignal,
            self.baseFeatureSpecs,
            startIndex,
            stopIndex,
        )
        envelopeWeights = self.CalculateEnvelopeWeights(
            inputSignal[startIndex:stopIndex]
        )
        return np.column_stack(
            tuple(
                baseBasis
                * envelopeWeights[:, regionIndex].reshape(-1, 1)
                for regionIndex in range(len(self.regionNames))
            )
        )

    def BuildAdditionalRegularizationMatrix(
        self,
        featureScale: np.ndarray,
        diagonalScale: float,
    ) -> np.ndarray:
        """Penalize differences between adjacent regional coefficients.

        Processing details:
            Algorithm: Build the block first-difference operator for
            low-to-middle and middle-to-high raw coefficient vectors, map it
            into the solver's normalized coefficient coordinates, normalize
            its mean diagonal so ``regionSmoothnessFactor`` is dimensionless,
            and return the resulting positive-semidefinite matrix scaled
            relative to the data normal matrix. A zero factor disables the
            penalty exactly.

        Args:
            featureScale: Positive RMS normalization for every joint basis
                column.
            diagonalScale: Positive mean diagonal of the data normal matrix.

        Returns:
            result: Hermitian matrix implementing
                ``lambda_s*(||c_middle-c_low||^2 +
                ||c_high-c_middle||^2)`` up to a common unit normalization.
        """

        baseMatrix = super().BuildAdditionalRegularizationMatrix(
            featureScale,
            diagonalScale,
        )
        smoothnessFactor = float(
            self.parameters["regionSmoothnessFactor"]
        )
        if smoothnessFactor == 0.0:
            return baseMatrix
        featureScaleVector = np.asarray(
            featureScale, dtype=float
        ).reshape(-1)
        baseFeatureCount = len(self.baseFeatureSpecs)
        differenceMatrix = np.zeros(
            (
                2 * baseFeatureCount,
                len(self.featureSpecs),
            ),
            dtype=float,
        )
        featureIndices = np.arange(baseFeatureCount)
        differenceMatrix[
            featureIndices, featureIndices
        ] = -1.0
        differenceMatrix[
            featureIndices,
            baseFeatureCount + featureIndices,
        ] = 1.0
        secondRows = baseFeatureCount + featureIndices
        differenceMatrix[
            secondRows,
            baseFeatureCount + featureIndices,
        ] = -1.0
        differenceMatrix[
            secondRows,
            2 * baseFeatureCount + featureIndices,
        ] = 1.0
        normalizedDifferenceMatrix = (
            differenceMatrix / featureScaleVector.reshape(1, -1)
        )
        rawPenaltyMatrix = (
            normalizedDifferenceMatrix.conj().T
            @ normalizedDifferenceMatrix
        )
        penaltyDiagonalMean = max(
            float(np.mean(np.real(np.diag(rawPenaltyMatrix)))),
            np.finfo(float).tiny,
        )
        return np.asarray(
            baseMatrix
            + smoothnessFactor
            * diagonalScale
            * rawPenaltyMatrix
            / penaltyDiagonalMean,
            dtype=np.complex128,
        )

    def CalculateRegionSmoothnessPenalty(
        self, coefficients: np.ndarray
    ) -> float:
        """Measure adjacent low/middle/high coefficient disagreement.

        Processing details:
            Algorithm: Validate the complete piecewise coefficient vector,
            split it into equal region-major blocks, and sum the squared
            complex Euclidean distances from low to middle and middle to high
            without imposing a component-wise monotonic constraint.

        Args:
            coefficients: Joint piecewise coefficient vector.

        Returns:
            result: Nonnegative raw coefficient smoothness penalty.
        """

        complexCoefficients = np.asarray(
            coefficients, dtype=np.complex128
        ).reshape(-1)
        if (
            complexCoefficients.size != len(self.featureSpecs)
            or not np.all(np.isfinite(complexCoefficients))
        ):
            raise ValueError(
                "coefficients must contain one finite value per feature"
            )
        baseFeatureCount = len(self.baseFeatureSpecs)
        lowCoefficients = complexCoefficients[:baseFeatureCount]
        middleCoefficients = complexCoefficients[
            baseFeatureCount : 2 * baseFeatureCount
        ]
        highCoefficients = complexCoefficients[
            2 * baseFeatureCount : 3 * baseFeatureCount
        ]
        return float(
            np.sum(np.abs(middleCoefficients - lowCoefficients) ** 2)
            + np.sum(np.abs(highCoefficients - middleCoefficients) ** 2)
        )

    def GetRegionCoefficients(self, regionName: str) -> np.ndarray:
        """Return one detached regional GMP coefficient vector.

        Processing details:
            Algorithm: Normalize the requested low, middle, or high name,
            synchronize live configuration, locate its deterministic
            region-major block, and return an owned copy without exposing
            mutable model state.

        Args:
            regionName: Case-insensitive ``low``, ``middle``, or ``high``.

        Returns:
            result: Complex coefficient vector in ordinary GMP feature order.
        """

        if not isinstance(regionName, str):
            raise TypeError("regionName must be a string")
        normalizedRegionName = regionName.strip().lower()
        if normalizedRegionName not in self.regionNames:
            raise ValueError(
                "regionName must be 'low', 'middle', or 'high'"
            )
        self.SynchronizeStructure()
        regionIndex = self.regionNames.index(normalizedRegionName)
        featureCount = len(self.baseFeatureSpecs)
        blockStart = regionIndex * featureCount
        return self.coefficients[
            blockStart : blockStart + featureCount
        ].copy()


class AugmentedDpdGmp(DpdGmp):
    """Model direct and conjugate nonlinear paths with one joint GMP fit.

    The direct branch represents ordinary PA AM/AM, AM/PM, and memory
    behavior. The conjugate branch represents IQ image leakage and its
    nonlinear memory products. Both branches share the proven normalized
    ridge solver implemented by ``DpdGmp``.
    """

    def RebuildStructure(self, resetCoefficients: bool) -> None:
        """Build paired direct/conjugate feature indices and identity state.

        Processing details:
            Algorithm: Enumerate the ordinary GMP feature list first, append
            an ``image_`` copy in identical order, preserve an existing
            coefficient vector only when its size remains valid, and
            initialize only the direct zero-delay linear feature to one.

        Args:
            resetCoefficients: Whether to discard coefficients and use identity.

        Returns:
            result: None. Augmented feature and coefficient state are synchronized.
        """

        directFeatureSpecs = BuildFeatureSpecs(*self.ResolveStructure())
        imageFeatureSpecs = [
            (
                f"image_{branchName}",
                nonlinearOrder,
                signalDelay,
                envelopeDelay,
            )
            for (
                branchName,
                nonlinearOrder,
                signalDelay,
                envelopeDelay,
            ) in directFeatureSpecs
        ]
        featureSpecs = [
            *directFeatureSpecs,
            *imageFeatureSpecs,
        ]
        if not resetCoefficients and len(featureSpecs) != len(
            self.coefficients
        ):
            raise ValueError(
                "coefficient count cannot be preserved across structure change"
            )
        self.featureSpecs = list(featureSpecs)
        self.activeStructure = self.ResolveStructure()
        if resetCoefficients:
            self.coefficients = np.zeros(
                len(self.featureSpecs), dtype=np.complex128
            )
            identityIndex = self.featureSpecs.index(
                ("main", 1, 0, 0)
            )
            self.coefficients[identityIndex] = 1.0 + 0.0j
            self.lastTrainingResult = None

    def BuildBasisChunk(
        self,
        inputSignal: np.ndarray,
        startIndex: int,
        stopIndex: int,
    ) -> np.ndarray:
        """Build paired GMP and conjugate-GMP basis columns.

        Processing details:
            Algorithm: Evaluate the canonical direct GMP basis once and
            concatenate its complex conjugate. Because envelope factors are
            real magnitudes, conjugating a GMP column changes the carrier
            factor from ``x`` to ``conj(x)`` while preserving every nonlinear
            order, signal delay, and envelope cross-memory delay.

        Args:
            inputSignal: Normalized finite complex desired waveform.
            startIndex: Inclusive output-sample index of the basis chunk.
            stopIndex: Exclusive output-sample index of the basis chunk.

        Returns:
            result: Direct columns followed by conjugate-image columns.
        """

        directFeatureCount = len(self.featureSpecs) // 2
        directBasis = BuildGmpBasisChunk(
            inputSignal,
            self.featureSpecs[:directFeatureCount],
            startIndex,
            stopIndex,
        )
        return np.column_stack(
            (directBasis, np.conj(directBasis))
        )

    def GetDirectCoefficients(self) -> np.ndarray:
        """Return a detached copy of ordinary GMP branch coefficients.

        Processing details:
            Algorithm: Synchronize live structure settings and copy the first
            half of the deterministic coefficient vector.

        Returns:
            result: Direct main/lagging/leading GMP coefficients.
        """

        self.SynchronizeStructure()
        directFeatureCount = len(self.featureSpecs) // 2
        return self.coefficients[:directFeatureCount].copy()

    def GetImageCoefficients(self) -> np.ndarray:
        """Return a detached copy of conjugate-image branch coefficients.

        Processing details:
            Algorithm: Synchronize live structure settings and copy the
            second half of the deterministic coefficient vector.

        Returns:
            result: Conjugate main/lagging/leading GMP coefficients.
        """

        self.SynchronizeStructure()
        directFeatureCount = len(self.featureSpecs) // 2
        return self.coefficients[directFeatureCount:].copy()


@dataclass(frozen=True)
class CouplingAwareDpdGmpTrainingResult:
    """Store per-chain GMP fitting diagnostics for one coupled MIMO update."""

    chainResults: Tuple[DpdGmpTrainingResult, ...]
    segmentCount: int
    compensatePrePaCoupling: bool
    compensatePostPaCoupling: bool

    def ToDict(self) -> Dict[str, object]:
        """Convert the coupled training result to ordinary nested mappings.

        Processing details:
            Algorithm: Copy segment and compensation-mode information, then
            serialize every immutable SISO training result in physical-chain
            order without recomputing coefficient diagnostics.

        Returns:
            result: JSON-compatible coupled-DPD training summary.
        """

        return {
            "segmentCount": self.segmentCount,
            "compensatePrePaCoupling": self.compensatePrePaCoupling,
            "compensatePostPaCoupling": self.compensatePostPaCoupling,
            "chainResults": [
                chainResult.ToDict()
                for chainResult in self.chainResults
            ],
        }


class CouplingAwareDpdGmp:
    """Deploy per-PA GMP models around measured pre/post coupling networks.

    The class implements the cascade

    ``reference -> inverse(post) -> per-PA DPD -> inverse(pre) -> DAC``.

    Consequently, the physical pre-PA network recreates the intended
    predistorted PA drives and the physical post-PA network recombines the
    independently linearized PA outputs into the requested port waveforms.
    Coupling inverses are obtained from measured causal impulse-response
    matrices rather than from private ``Channel`` configuration values.
    """

    def __init__(
        self,
        dpdModels: Sequence[DpdGmp],
        preChannelMeasurement: Optional[Any] = None,
        postChannelMeasurement: Optional[Any] = None,
        parameters: Optional[Mapping[str, object]] = None,
        width: Optional[int] = None,
        **parameterOverrides: object,
    ) -> None:
        """Initialize a coupling-aware MIMO wrapper around SISO GMP models.

        Processing details:
            Algorithm: Define all defaults locally, layer recognized settings
            with ChainMap precedence, retain one independent DpdGmp per PA,
            validate chain count and inverse controls, and copy measured
            pre/post impulse tensors after shape and finiteness checks.

        Args:
            dpdModels: Ordered sequence containing one trained model per PA.
            preChannelMeasurement: Optional result exposing
                ``impulseResponses`` or a raw response tensor.
            postChannelMeasurement: Optional post-PA result or raw tensor.
            parameters: Optional caller-owned live parameter mapping.
            width: Optional public I/Q component width override.
            parameterOverrides: Highest-priority recognized local settings.

        Returns:
            result: None. Identity coupling is used for omitted measurements.
        """

        self.defaultParameters: Mapping[str, object] = MappingProxyType(
            {
                "compensatePrePaCoupling": True,
                "compensatePostPaCoupling": True,
                "inverseRegularization": 1.0e-8,
                "maximumInverseGainDb": 18.0,
                "impulseTruncationDb": -100.0,
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
                "CouplingAwareDpdGmp",
            )
        )
        recognizedOverrides = FilterRecognizedParameters(
            directOverrides,
            self.defaultParameters,
            "CouplingAwareDpdGmp",
        )
        self.parameters: ChainMap[str, object] = ChainMap(
            recognizedOverrides,
            externalParameters,
            self.defaultParameters,
        )
        if isinstance(dpdModels, (str, bytes)):
            raise TypeError("dpdModels must be a sequence of DpdGmp objects")
        self.dpdModels = tuple(dpdModels)
        self.preImpulseResponses: Optional[np.ndarray] = None
        self.postImpulseResponses: Optional[np.ndarray] = None
        self.lastTrainingResult: Optional[
            CouplingAwareDpdGmpTrainingResult
        ] = None
        self.ValidateParameters()
        self.ConfigureChannelMeasurements(
            preChannelMeasurement,
            postChannelMeasurement,
        )

    @property
    def Width(self) -> int:
        """Return the public signed I/Q component width.

        Processing details:
            Algorithm: Read the validated ChainMap value so live external
            parameter changes affect the next matrix conversion.

        Returns:
            result: Zero for floating mode or a positive fixed-point width.
        """

        return cast(int, self.parameters["width"])

    width = Width

    @property
    def ChainCount(self) -> int:
        """Return the number of independently modeled PA branches.

        Processing details:
            Algorithm: Derive chain count directly from the immutable DPD
            tuple so it cannot disagree with coefficient storage.

        Returns:
            result: Positive number of input/output columns.
        """

        return len(self.dpdModels)

    chainCount = ChainCount

    def GetParameters(self) -> Dict[str, object]:
        """Return a flattened snapshot of all coupled-DPD settings.

        Processing details:
            Algorithm: Validate live values and copy the resolved ChainMap
            without exposing caller or default mapping layers.

        Returns:
            result: Ordinary dictionary of inverse and boundary controls.
        """

        self.ValidateParameters()
        return dict(self.parameters)

    def UpdateParameters(self, **parameterOverrides: object) -> None:
        """Apply recognized coupled-DPD parameter changes transactionally.

        Processing details:
            Algorithm: Filter unknown keys with warnings, update the local
            ChainMap layer, validate the entire new state, and restore prior
            values if any recognized setting is invalid.

        Args:
            parameterOverrides: Supported highest-priority replacements.

        Returns:
            result: None. Valid changes affect future fitting and inference.
        """

        recognizedOverrides = FilterRecognizedParameters(
            parameterOverrides,
            self.defaultParameters,
            "CouplingAwareDpdGmp.UpdateParameters",
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
        """Validate model count, inverse controls, booleans, and public width.

        Processing details:
            Algorithm: Require at least two genuine DpdGmp branches, exact
            booleans for each coupling stage, positive regularization and
            inverse-gain limits, a negative impulse truncation level, and a
            supported floating or signed fixed-point boundary.

        Returns:
            result: None. Invalid values raise descriptive exceptions.
        """

        if len(self.dpdModels) < 2 or any(
            not isinstance(dpdModel, DpdGmp)
            for dpdModel in self.dpdModels
        ):
            raise ValueError(
                "dpdModels must contain at least two DpdGmp objects"
            )
        for parameterName in (
            "compensatePrePaCoupling",
            "compensatePostPaCoupling",
        ):
            if not isinstance(self.parameters[parameterName], bool):
                raise TypeError(f"{parameterName} must be a bool")
        for parameterName in (
            "inverseRegularization",
            "maximumInverseGainDb",
        ):
            parameterValue = self.parameters[parameterName]
            if (
                not isinstance(parameterValue, (int, float))
                or isinstance(parameterValue, bool)
                or not np.isfinite(parameterValue)
                or float(parameterValue) <= 0.0
            ):
                raise ValueError(
                    f"{parameterName} must be finite and positive"
                )
        impulseTruncationDb = self.parameters["impulseTruncationDb"]
        if (
            not isinstance(impulseTruncationDb, (int, float))
            or isinstance(impulseTruncationDb, bool)
            or not np.isfinite(impulseTruncationDb)
            or float(impulseTruncationDb) >= 0.0
        ):
            raise ValueError(
                "impulseTruncationDb must be finite and negative"
            )
        FixedPoint(self.width)

    def ConfigureChannelMeasurements(
        self,
        preChannelMeasurement: Optional[Any],
        postChannelMeasurement: Optional[Any],
    ) -> None:
        """Copy measured channel tensors used by training and compensation.

        Processing details:
            Algorithm: Accept either ChannelMeasurementResult-like objects or
            raw delay-by-destination-by-source arrays, validate their chain
            dimensions, remove only a trailing region below the configured
            relative energy threshold, and use identity tensors for None.

        Args:
            preChannelMeasurement: PA-input network measurement or None.
            postChannelMeasurement: PA-output network measurement or None.

        Returns:
            result: None. New responses affect subsequent operations.
        """

        self.ValidateParameters()
        self.preImpulseResponses = self.ResolveImpulseResponses(
            preChannelMeasurement,
            "preChannelMeasurement",
        )
        self.postImpulseResponses = self.ResolveImpulseResponses(
            postChannelMeasurement,
            "postChannelMeasurement",
        )

    def ResolveImpulseResponses(
        self,
        channelMeasurement: Optional[Any],
        measurementName: str,
    ) -> np.ndarray:
        """Resolve one measurement object or array to a compact owned tensor.

        Processing details:
            Algorithm: Substitute an identity impulse for None, retrieve the
            public ``impulseResponses`` attribute when present, validate a
            finite square tensor matching the DPD chain count, and retain all
            taps through the last sample above the relative truncation floor.

        Args:
            channelMeasurement: Measurement result, raw tensor, or None.
            measurementName: Name included in validation errors.

        Returns:
            result: Owned causal complex response tensor.
        """

        if channelMeasurement is None:
            identityResponse = np.zeros(
                (1, self.chainCount, self.chainCount),
                dtype=np.complex128,
            )
            identityResponse[0, :, :] = np.eye(
                self.chainCount, dtype=np.complex128
            )
            return identityResponse
        rawImpulseResponses = getattr(
            channelMeasurement,
            "impulseResponses",
            channelMeasurement,
        )
        responseTensor = np.asarray(
            rawImpulseResponses, dtype=np.complex128
        )
        if (
            responseTensor.ndim != 3
            or responseTensor.shape[0] < 1
            or responseTensor.shape[1]
            != responseTensor.shape[2]
            or responseTensor.shape[1] != self.chainCount
            or not np.all(np.isfinite(responseTensor))
        ):
            raise ValueError(
                f"{measurementName} must contain a finite "
                "delay-by-destination-by-source tensor matching chain count"
            )
        tapMagnitudes = np.max(
            np.abs(responseTensor), axis=(1, 2)
        )
        peakMagnitude = max(
            float(np.max(tapMagnitudes)),
            np.finfo(float).tiny,
        )
        truncationMagnitude = peakMagnitude * np.power(
            10.0,
            float(self.parameters["impulseTruncationDb"]) / 20.0,
        )
        significantTapIndices = np.flatnonzero(
            tapMagnitudes >= truncationMagnitude
        )
        lastTapIndex = (
            int(significantTapIndices[-1])
            if significantTapIndices.size
            else 0
        )
        return responseTensor[: lastTapIndex + 1, :, :].copy()

    def PreparePublicMatrix(
        self, inputSignal: np.ndarray, signalName: str
    ) -> np.ndarray:
        """Decode and validate one public samples-by-chain waveform matrix.

        Processing details:
            Algorithm: Decode the configured interface once, permit a vector
            only for an impossible single-chain wrapper, and require a finite
            nonempty matrix with exactly one column per physical PA.

        Args:
            inputSignal: Public floating samples or fixed I/Q codes.
            signalName: Name included in validation errors.

        Returns:
            result: Normalized finite complex matrix.
        """

        decodedSignal = FixedPoint(self.width).DecodeComplex(
            inputSignal
        )
        signalMatrix = np.asarray(
            decodedSignal, dtype=np.complex128
        )
        if signalMatrix.ndim == 1 and self.chainCount == 1:
            signalMatrix = signalMatrix.reshape(-1, 1)
        if (
            signalMatrix.ndim != 2
            or signalMatrix.shape[0] == 0
            or signalMatrix.shape[1] != self.chainCount
            or not np.all(np.isfinite(signalMatrix))
        ):
            raise ValueError(
                f"{signalName} must be a finite nonempty "
                "samples-by-chain matrix"
            )
        return signalMatrix

    def ApplyMeasuredResponse(
        self,
        inputSignal: np.ndarray,
        impulseResponses: np.ndarray,
    ) -> np.ndarray:
        """Apply a measured causal MIMO response without circular convolution.

        Processing details:
            Algorithm: Convolve every source column with each directed FIR,
            retain the original record length, and sum contributions at each
            destination exactly according to
            ``y[n] = sum_l H[l] x[n-l]``.

        Args:
            inputSignal: Normalized samples-by-source matrix.
            impulseResponses: Delay-by-destination-by-source tensor.

        Returns:
            result: Same-length normalized destination matrix.
        """

        inputMatrix = np.asarray(
            inputSignal, dtype=np.complex128
        )
        responseTensor = np.asarray(
            impulseResponses, dtype=np.complex128
        )
        if (
            inputMatrix.ndim != 2
            or inputMatrix.shape[1] != self.chainCount
            or responseTensor.ndim != 3
            or responseTensor.shape[1:]
            != (self.chainCount, self.chainCount)
        ):
            raise ValueError(
                "inputSignal and impulseResponses have incompatible shapes"
            )
        outputMatrix = np.zeros_like(inputMatrix)
        for destinationChain in range(self.chainCount):
            for sourceChain in range(self.chainCount):
                outputMatrix[:, destinationChain] += np.convolve(
                    inputMatrix[:, sourceChain],
                    responseTensor[
                        :, destinationChain, sourceChain
                    ],
                    mode="full",
                )[: inputMatrix.shape[0]]
        if not np.all(np.isfinite(outputMatrix)):
            raise ValueError(
                "measured channel response exceeded numeric range"
            )
        return outputMatrix

    def InvertMeasuredResponse(
        self,
        targetSignal: np.ndarray,
        impulseResponses: np.ndarray,
    ) -> np.ndarray:
        """Solve a regularized causal MIMO deconvolution sample by sample.

        Processing details:
            Algorithm: Build a Tikhonov-regularized SVD inverse of the
            zero-delay transfer matrix, cap every inverse singular gain, then
            recursively subtract all already-known delayed contributions
            before solving the current source vector.  This avoids circular
            FFT wraparound and remains stable for weak measured coupling.

        Args:
            targetSignal: Desired samples-by-destination matrix.
            impulseResponses: Measured causal response tensor.

        Returns:
            result: Source waveform whose measured response follows target.
        """

        targetMatrix = np.asarray(
            targetSignal, dtype=np.complex128
        )
        responseTensor = np.asarray(
            impulseResponses, dtype=np.complex128
        )
        if (
            targetMatrix.ndim != 2
            or targetMatrix.shape[1] != self.chainCount
            or responseTensor.ndim != 3
            or responseTensor.shape[1:]
            != (self.chainCount, self.chainCount)
            or not np.all(np.isfinite(targetMatrix))
            or not np.all(np.isfinite(responseTensor))
        ):
            raise ValueError(
                "targetSignal and impulseResponses have incompatible shapes"
            )
        zeroDelayMatrix = responseTensor[0, :, :]
        leftVectors, singularValues, rightVectorsHermitian = (
            np.linalg.svd(zeroDelayMatrix)
        )
        regularization = float(
            self.parameters["inverseRegularization"]
        )
        maximumInverseGain = np.power(
            10.0,
            float(self.parameters["maximumInverseGainDb"]) / 20.0,
        )
        inverseSingularValues = np.minimum(
            singularValues
            / (singularValues**2 + regularization),
            maximumInverseGain,
        )
        zeroDelayInverse = (
            rightVectorsHermitian.conj().T
            @ np.diag(inverseSingularValues)
            @ leftVectors.conj().T
        )
        sourceMatrix = np.zeros_like(targetMatrix)
        tapCount = responseTensor.shape[0]
        for sampleIndex in range(targetMatrix.shape[0]):
            delayedContribution = np.zeros(
                self.chainCount, dtype=np.complex128
            )
            maximumDelay = min(tapCount - 1, sampleIndex)
            for delayIndex in range(1, maximumDelay + 1):
                delayedContribution += (
                    responseTensor[delayIndex, :, :]
                    @ sourceMatrix[sampleIndex - delayIndex, :]
                )
            sourceMatrix[sampleIndex, :] = (
                zeroDelayInverse
                @ (
                    targetMatrix[sampleIndex, :]
                    - delayedContribution
                )
            )
        if not np.all(np.isfinite(sourceMatrix)):
            raise ValueError(
                "measured coupling inverse exceeded numeric range"
            )
        return sourceMatrix

    def BuildPaOutputTargets(
        self, referenceSignal: np.ndarray
    ) -> np.ndarray:
        """De-embed post-PA coupling from requested observation waveforms.

        Processing details:
            Algorithm: Validate a normalized reference matrix and, when
            enabled, solve the measured post-network inverse so each SISO GMP
            is trained and evaluated against the PA output required before
            physical output coupling.

        Args:
            referenceSignal: Desired final observed waveform matrix.

        Returns:
            result: Desired individual PA-output matrix.
        """

        referenceMatrix = np.asarray(
            referenceSignal, dtype=np.complex128
        )
        if (
            referenceMatrix.ndim != 2
            or referenceMatrix.shape[1] != self.chainCount
            or referenceMatrix.shape[0] == 0
            or not np.all(np.isfinite(referenceMatrix))
        ):
            raise ValueError(
                "referenceSignal must be a finite samples-by-chain matrix"
            )
        if not cast(
            bool, self.parameters["compensatePostPaCoupling"]
        ):
            return referenceMatrix.copy()
        return self.InvertMeasuredResponse(
            referenceMatrix,
            cast(np.ndarray, self.postImpulseResponses),
        )

    def BuildDacInput(
        self, predistortedPaInput: np.ndarray
    ) -> np.ndarray:
        """Pre-cancel the measured PA-input coupling network.

        Processing details:
            Algorithm: Validate the independently predistorted PA-drive
            matrix and, when enabled, solve the measured pre-network inverse
            so the physical coupling recreates those intended PA inputs.

        Args:
            predistortedPaInput: Desired actual input at every PA port.

        Returns:
            result: Raw DAC waveform matrix before physical pre-PA coupling.
        """

        paInputMatrix = np.asarray(
            predistortedPaInput, dtype=np.complex128
        )
        if (
            paInputMatrix.ndim != 2
            or paInputMatrix.shape[1] != self.chainCount
            or paInputMatrix.shape[0] == 0
            or not np.all(np.isfinite(paInputMatrix))
        ):
            raise ValueError(
                "predistortedPaInput must be a finite "
                "samples-by-chain matrix"
            )
        if not cast(
            bool, self.parameters["compensatePrePaCoupling"]
        ):
            return paInputMatrix.copy()
        return self.InvertMeasuredResponse(
            paInputMatrix,
            cast(np.ndarray, self.preImpulseResponses),
        )

    def ProcessFloating(self, inputSignal: np.ndarray) -> np.ndarray:
        """Apply post de-embedding, per-PA GMP, and pre-coupling cancellation.

        Processing details:
            Algorithm: Convert the desired final outputs to individual PA
            targets using the measured post inverse, evaluate every trained
            DpdGmp in its normalized floating domain, then transform intended
            PA drives to raw DAC samples using the measured pre inverse.

        Args:
            inputSignal: Normalized desired samples-by-chain outputs.

        Returns:
            result: Normalized raw DAC waveform matrix.
        """

        self.ValidateParameters()
        paOutputTargets = self.BuildPaOutputTargets(inputSignal)
        predistortedColumns = []
        for chainIndex, dpdModel in enumerate(self.dpdModels):
            predistortedColumns.append(
                dpdModel.ProcessFloating(
                    paOutputTargets[:, chainIndex]
                )
            )
        predistortedPaInput = np.column_stack(
            predistortedColumns
        )
        return self.BuildDacInput(predistortedPaInput)

    def Process(self, inputSignal: np.ndarray) -> np.ndarray:
        """Apply coupled MIMO DPD while preserving the public data convention.

        Processing details:
            Algorithm: Decode the complete matrix once, execute all channel
            inverses and GMP models in normalized floating point, and encode
            the raw DAC matrix once at the configured interface width.

        Args:
            inputSignal: Public desired waveform matrix.

        Returns:
            result: Public raw-DAC matrix in floating or integer-code form.
        """

        inputMatrix = self.PreparePublicMatrix(
            inputSignal, "inputSignal"
        )
        outputMatrix = self.ProcessFloating(inputMatrix)
        return FixedPoint(self.width).EncodeComplex(outputMatrix)

    def FitCoupledSegments(
        self,
        referenceSignals: Sequence[np.ndarray],
        paInputTargetSignals: Sequence[np.ndarray],
        segmentWeights: Optional[Sequence[float]] = None,
        sampleWeights: Optional[
            Sequence[Optional[np.ndarray]]
        ] = None,
    ) -> CouplingAwareDpdGmpTrainingResult:
        """Fit every PA inverse using post-deembedded references and labels.

        Processing details:
            Algorithm: Decode each desired final-output segment, remove the
            measured post-PA coupling to obtain individual PA-output targets,
            decode matching labels measured at the actual PA inputs after
            pre-coupling, and fit one DpdGmp per physical chain.  Pre-network
            cancellation is applied later during inference, not to labels.

        Args:
            referenceSignals: Desired final observed waveform segments.
            paInputTargetSignals: Matching learned actual PA-input labels.
            segmentWeights: Optional positive importance per segment.
            sampleWeights: Optional sample weights per segment.

        Returns:
            result: Per-chain coefficient and conditioning diagnostics.
        """

        if (
            isinstance(referenceSignals, (str, bytes))
            or isinstance(paInputTargetSignals, (str, bytes))
        ):
            raise TypeError("training signals must be matrix sequences")
        referenceSequence = tuple(referenceSignals)
        targetSequence = tuple(paInputTargetSignals)
        if (
            not referenceSequence
            or len(referenceSequence) != len(targetSequence)
        ):
            raise ValueError(
                "referenceSignals and paInputTargetSignals must have "
                "equal nonzero segment counts"
            )
        preparedPaOutputTargets = []
        preparedPaInputLabels = []
        for referenceSignal, targetSignal in zip(
            referenceSequence, targetSequence
        ):
            referenceMatrix = self.PreparePublicMatrix(
                referenceSignal, "referenceSignal"
            )
            targetMatrix = self.PreparePublicMatrix(
                targetSignal, "paInputTargetSignal"
            )
            if referenceMatrix.shape != targetMatrix.shape:
                raise ValueError(
                    "every reference and PA-input target matrix must "
                    "have identical shape"
                )
            preparedPaOutputTargets.append(
                self.BuildPaOutputTargets(referenceMatrix)
            )
            preparedPaInputLabels.append(targetMatrix)
        chainResults = []
        for chainIndex, dpdModel in enumerate(self.dpdModels):
            modelInterface = FixedPoint(dpdModel.width)
            chainReferences = tuple(
                modelInterface.EncodeComplex(
                    targetMatrix[:, chainIndex]
                )
                for targetMatrix in preparedPaOutputTargets
            )
            chainLabels = tuple(
                modelInterface.EncodeComplex(
                    labelMatrix[:, chainIndex]
                )
                for labelMatrix in preparedPaInputLabels
            )
            chainResults.append(
                dpdModel.FitSegments(
                    chainReferences,
                    chainLabels,
                    segmentWeights,
                    sampleWeights,
                )
            )
        trainingResult = CouplingAwareDpdGmpTrainingResult(
            chainResults=tuple(chainResults),
            segmentCount=len(referenceSequence),
            compensatePrePaCoupling=cast(
                bool,
                self.parameters["compensatePrePaCoupling"],
            ),
            compensatePostPaCoupling=cast(
                bool,
                self.parameters["compensatePostPaCoupling"],
            ),
        )
        self.lastTrainingResult = trainingResult
        return trainingResult

    def GetLastTrainingResult(
        self,
    ) -> Optional[CouplingAwareDpdGmpTrainingResult]:
        """Return the latest immutable coupled-training diagnostics.

        Processing details:
            Algorithm: Return the frozen result record directly because its
            chain result tuple contains only immutable scalar dataclasses.

        Returns:
            result: Last fit result or None before training.
        """

        return self.lastTrainingResult
