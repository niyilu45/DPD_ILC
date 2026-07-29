"""Object-oriented generalized-memory-polynomial DPD implementation."""

from collections import ChainMap
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, cast

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
        self, inputSignal: np.ndarray, signalName: str
    ) -> np.ndarray:
        """Decode and validate one public signal as a floating-point vector.

        Processing details:
            Algorithm: Use the configured FixedPoint boundary exactly once,
            flatten to one complex stream, and reject empty or nonfinite data.

        Args:
            inputSignal: Floating samples or public fixed-point complex codes.
            signalName: Name included in validation errors.

        Returns:
            result: Normalized finite complex training/inference vector.
        """

        decodedSignal = FixedPoint(self.width).DecodeComplex(
            inputSignal
        ).reshape(-1)
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
            basisChunk = BuildGmpBasisChunk(
                complexInput,
                self.featureSpecs,
                startIndex,
                stopIndex,
            )
            outputSignal[startIndex:stopIndex] = (
                basisChunk @ self.coefficients
            )
        return self.LimitMagnitude(outputSignal)

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
                basisChunk = BuildGmpBasisChunk(
                    referenceSignal,
                    self.featureSpecs,
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
                basisChunk = BuildGmpBasisChunk(
                    referenceSignal,
                    self.featureSpecs,
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
        regularizedMatrix = normalMatrix + (
            ridgeScale * np.eye(featureCount)
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
                basisChunk = BuildGmpBasisChunk(
                    referenceSignal,
                    self.featureSpecs,
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

        Returns:
            result: Indirect-learning coefficient-fit diagnostics.
        """

        floatingInput = self.PreparePublicSignal(
            paInputSignal, "paInputSignal"
        )
        floatingOutput = self.PreparePublicSignal(
            paOutputSignal, "paOutputSignal"
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
