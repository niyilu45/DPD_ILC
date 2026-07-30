"""Sample-by-sample LMS and NLMS adaptation for a GMP predistorter.

The adaptive model reuses the canonical main, lagging-envelope, and
leading-envelope feature order implemented by ``DpdGmp``. Training coefficients
are updated after every aligned sample. Active deployment coefficients can be
committed after every sample or once at the end of a frame.
"""

from collections import ChainMap
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Tuple, cast

import numpy as np

from .DpdGmp import DpdGmp

# Support both ``inc.lib`` and compatibility ``lib`` package imports.
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
class DpdLmsTrainingResult:
    """Store diagnostics from one sample-by-sample adaptation pass."""

    sampleCount: int
    updateCount: int
    featureCount: int
    adaptationMode: str
    beforeNmseDb: float
    onlineNmseDb: float
    afterNmseDb: float
    coefficientUpdateNorm: float
    maximumSampleUpdateNorm: float
    coefficientsCommitted: bool

    def ToDict(self) -> Dict[str, object]:
        """Return stable scalar diagnostics for JSON or CSV serialization.

        Processing details:
            Algorithm: Copy immutable counters, mode labels, NMSE values,
            coefficient-change diagnostics, and commit state without
            recalculating the adaptation.

        Returns:
            result: Ordinary dictionary containing one LMS training summary.
        """

        return {
            "sampleCount": self.sampleCount,
            "updateCount": self.updateCount,
            "featureCount": self.featureCount,
            "adaptationMode": self.adaptationMode,
            "beforeNmseDb": self.beforeNmseDb,
            "onlineNmseDb": self.onlineNmseDb,
            "afterNmseDb": self.afterNmseDb,
            "coefficientUpdateNorm": self.coefficientUpdateNorm,
            "maximumSampleUpdateNorm": self.maximumSampleUpdateNorm,
            "coefficientsCommitted": self.coefficientsCommitted,
        }


class DpdLms(DpdGmp):
    """Adapt one GMP DPD coefficient vector with complex LMS or NLMS.

    ``adaptiveCoefficients`` form the sample-updated shadow model.
    ``coefficients`` remain the active deployment model inherited from
    ``DpdGmp``. Frame commit mode prevents coefficient motion inside one
    transmitted OFDM frame, while sample commit mode exposes strict online
    adaptation for algorithm and hardware-reference experiments.
    """

    def __init__(
        self,
        parameters: Optional[Mapping[str, object]] = None,
        width: Optional[int] = None,
        **parameterOverrides: object,
    ) -> None:
        """Initialize GMP defaults, adaptive controls, and identity state.

        Processing details:
            Algorithm: Construct the proven DpdGmp base without caller data,
            extend its immutable defaults with LMS-specific controls, layer
            recognized direct and external overrides through ChainMap,
            validate both parameter groups, rebuild the selected GMP feature
            structure, and initialize active and shadow identity coefficients.

        Args:
            parameters: Optional caller-owned live parameter mapping.
            width: Optional public I/Q width override.
            parameterOverrides: Highest-priority recognized configuration.

        Returns:
            result: None. The object is ready for inference or adaptation.
        """

        super().__init__()
        combinedDefaults = dict(self.defaultParameters)
        combinedDefaults.update(
            {
                "adaptationMode": "nlms",
                "learningRate": 0.05,
                "normalizationEpsilon": 1.0e-6,
                "leakageFactor": 1.0e-7,
                "featureScaleMode": "frame",
                "featurePowerForgettingFactor": 0.999,
                "updateDecimation": 1,
                "coefficientCommitMode": "frame",
                "maximumSampleUpdateNorm": 0.05,
                "maximumSampleWeight": 8.0,
            }
        )
        self.defaultParameters = MappingProxyType(combinedDefaults)
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
                "DpdLms",
            )
        )
        recognizedOverrides = FilterRecognizedParameters(
            directOverrides,
            self.defaultParameters,
            "DpdLms",
        )
        self.parameters = ChainMap(
            recognizedOverrides,
            externalParameters,
            self.defaultParameters,
        )
        self.featureSpecs = []
        self.coefficients = np.zeros(0, dtype=np.complex128)
        self.activeStructure = (tuple(), 0, 0)
        self.lastTrainingResult = None
        self.lastLmsTrainingResult: Optional[
            DpdLmsTrainingResult
        ] = None
        self.ValidateParameters()
        self.ValidateLmsParameters()
        self.RebuildStructure(resetCoefficients=True)
        self.adaptiveCoefficients = self.coefficients.copy()
        self.identityCoefficients = self.coefficients.copy()
        self.featureScale = np.ones(
            len(self.featureSpecs), dtype=float
        )
        self.featurePower = np.ones(
            len(self.featureSpecs), dtype=float
        )
        self.sampleHistory = np.zeros(1, dtype=np.complex128)
        self.frameStartCoefficients = self.coefficients.copy()
        self.frameSampleCount = 0
        self.frameUpdateCount = 0
        self.frameErrorPower = 0.0
        self.frameTargetPower = 0.0
        self.frameMaximumUpdateNorm = 0.0
        self.InitializeAdaptiveState(copyActiveCoefficients=True)

    def ValidateLmsParameters(self) -> None:
        """Validate adaptive step, normalization, and commit controls.

        Processing details:
            Algorithm: Require a supported LMS family, finite positive step
            and denominator controls, bounded leakage and forgetting factors,
            positive update decimation and sample-weight limits, an optional
            positive update-norm projection, and sample or frame commit mode.

        Returns:
            result: None. Invalid adaptive settings raise a descriptive error.
        """

        adaptationMode = self.parameters["adaptationMode"]
        if (
            not isinstance(adaptationMode, str)
            or adaptationMode.strip().lower() not in ("lms", "nlms")
        ):
            raise ValueError(
                "adaptationMode must be either 'lms' or 'nlms'"
            )
        learningRate = self.parameters["learningRate"]
        if (
            not isinstance(learningRate, (int, float))
            or isinstance(learningRate, bool)
            or not np.isfinite(learningRate)
            or float(learningRate) <= 0.0
        ):
            raise ValueError("learningRate must be finite and positive")
        normalizationEpsilon = self.parameters[
            "normalizationEpsilon"
        ]
        if (
            not isinstance(normalizationEpsilon, (int, float))
            or isinstance(normalizationEpsilon, bool)
            or not np.isfinite(normalizationEpsilon)
            or float(normalizationEpsilon) <= 0.0
        ):
            raise ValueError(
                "normalizationEpsilon must be finite and positive"
            )
        leakageFactor = self.parameters["leakageFactor"]
        if (
            not isinstance(leakageFactor, (int, float))
            or isinstance(leakageFactor, bool)
            or not np.isfinite(leakageFactor)
            or not 0.0 <= float(leakageFactor) < 1.0
        ):
            raise ValueError(
                "leakageFactor must be finite in the interval [0, 1)"
            )
        featureScaleMode = self.parameters["featureScaleMode"]
        if (
            not isinstance(featureScaleMode, str)
            or featureScaleMode.strip().lower()
            not in ("frame", "running")
        ):
            raise ValueError(
                "featureScaleMode must be either 'frame' or 'running'"
            )
        forgettingFactor = self.parameters[
            "featurePowerForgettingFactor"
        ]
        if (
            not isinstance(forgettingFactor, (int, float))
            or isinstance(forgettingFactor, bool)
            or not np.isfinite(forgettingFactor)
            or not 0.0 <= float(forgettingFactor) < 1.0
        ):
            raise ValueError(
                "featurePowerForgettingFactor must be finite in [0, 1)"
            )
        updateDecimation = self.parameters["updateDecimation"]
        if (
            not isinstance(updateDecimation, int)
            or isinstance(updateDecimation, bool)
            or updateDecimation < 1
        ):
            raise ValueError("updateDecimation must be a positive integer")
        commitMode = self.parameters["coefficientCommitMode"]
        if (
            not isinstance(commitMode, str)
            or commitMode.strip().lower() not in ("sample", "frame")
        ):
            raise ValueError(
                "coefficientCommitMode must be 'sample' or 'frame'"
            )
        maximumUpdateNorm = self.parameters[
            "maximumSampleUpdateNorm"
        ]
        if maximumUpdateNorm is not None and (
            not isinstance(maximumUpdateNorm, (int, float))
            or isinstance(maximumUpdateNorm, bool)
            or not np.isfinite(maximumUpdateNorm)
            or float(maximumUpdateNorm) <= 0.0
        ):
            raise ValueError(
                "maximumSampleUpdateNorm must be positive finite or None"
            )
        maximumSampleWeight = self.parameters["maximumSampleWeight"]
        if (
            not isinstance(maximumSampleWeight, (int, float))
            or isinstance(maximumSampleWeight, bool)
            or not np.isfinite(maximumSampleWeight)
            or float(maximumSampleWeight) <= 0.0
        ):
            raise ValueError(
                "maximumSampleWeight must be finite and positive"
            )

    def UpdateParameters(self, **parameterOverrides: object) -> None:
        """Apply base and adaptive parameter changes transactionally.

        Processing details:
            Algorithm: Filter unknown keys with warnings, update the local
            ChainMap layer, validate both DpdGmp and LMS controls, roll back
            invalid changes, and rebuild identity active/shadow state only
            when order or memory dimensions change.

        Args:
            parameterOverrides: Recognized values placed in the local layer.

        Returns:
            result: None. Valid values affect later frames or samples.
        """

        recognizedOverrides = FilterRecognizedParameters(
            parameterOverrides,
            self.defaultParameters,
            "DpdLms.UpdateParameters",
        )
        previousOverrides = dict(self.parameters.maps[0])
        previousStructure = self.activeStructure
        self.parameters.maps[0].update(recognizedOverrides)
        try:
            self.ValidateParameters()
            self.ValidateLmsParameters()
            structureChanged = (
                self.ResolveStructure() != previousStructure
            )
        except (TypeError, ValueError):
            self.parameters.maps[0].clear()
            self.parameters.maps[0].update(previousOverrides)
            raise
        if structureChanged:
            self.RebuildStructure(resetCoefficients=True)
            self.InitializeAdaptiveState(copyActiveCoefficients=True)

    def SynchronizeStructure(self) -> None:
        """Synchronize live ChainMap structure and adaptive storage.

        Processing details:
            Algorithm: Validate current base and LMS parameters, detect an
            externally mutated order or memory configuration, rebuild the
            active identity GMP, and resize shadow coefficients, history,
            power estimates, and diagnostics as one atomic state transition.

        Returns:
            result: None. Nonstructural changes preserve learned coefficients.
        """

        self.ValidateParameters()
        self.ValidateLmsParameters()
        if self.ResolveStructure() != self.activeStructure:
            self.RebuildStructure(resetCoefficients=True)
            self.InitializeAdaptiveState(copyActiveCoefficients=True)

    def InitializeAdaptiveState(
        self, copyActiveCoefficients: bool
    ) -> None:
        """Initialize shadow coefficients and all streaming state arrays.

        Processing details:
            Algorithm: Optionally copy the active vector into the adaptive
            vector, recreate the canonical identity prior, size the causal
            history from the largest GMP delay, initialize feature scales and
            running powers, and clear frame-level counters and error sums.

        Args:
            copyActiveCoefficients: Copy active coefficients into the shadow
                vector before clearing streaming state.

        Returns:
            result: None. All adaptive arrays match the active feature count.
        """

        featureCount = len(self.featureSpecs)
        if (
            copyActiveCoefficients
            or not hasattr(self, "adaptiveCoefficients")
            or self.adaptiveCoefficients.size != featureCount
        ):
            self.adaptiveCoefficients = self.coefficients.copy()
        self.identityCoefficients = np.zeros(
            featureCount, dtype=np.complex128
        )
        identityIndex = self.featureSpecs.index(("main", 1, 0, 0))
        self.identityCoefficients[identityIndex] = 1.0 + 0.0j
        maximumDelay = max(
            memoryIndex + crossIndex
            for (
                _,
                _,
                memoryIndex,
                crossIndex,
            ) in self.featureSpecs
        )
        self.sampleHistory = np.zeros(
            maximumDelay + 1, dtype=np.complex128
        )
        normalizationEpsilon = float(
            self.parameters["normalizationEpsilon"]
        )
        self.featureScale = np.ones(featureCount, dtype=float)
        self.featurePower = np.full(
            featureCount,
            normalizationEpsilon,
            dtype=float,
        )
        self.frameStartCoefficients = (
            self.adaptiveCoefficients.copy()
        )
        self.frameSampleCount = 0
        self.frameUpdateCount = 0
        self.frameErrorPower = 0.0
        self.frameTargetPower = 0.0
        self.frameMaximumUpdateNorm = 0.0
        self.lastLmsTrainingResult = None

    def ResetAdaptiveState(
        self, copyActiveCoefficients: bool = True
    ) -> None:
        """Clear streaming history while optionally restoring the shadow.

        Processing details:
            Algorithm: Synchronize live structure and call the common state
            initializer so frame history, feature powers, and diagnostics
            cannot leak between independent captures. The active deployment
            vector is never modified by this operation.

        Args:
            copyActiveCoefficients: Reset shadow coefficients from the active
                vector when true; preserve them when false.

        Returns:
            result: None. A clean adaptive stream state is available.
        """

        self.SynchronizeStructure()
        self.InitializeAdaptiveState(copyActiveCoefficients)

    def ResetCoefficients(self) -> None:
        """Restore active and shadow coefficient vectors to identity.

        Processing details:
            Algorithm: Reuse DpdGmp's canonical identity rebuild, then reset
            all LMS histories, normalizers, counters, and the shadow vector so
            no coefficient or delayed sample survives the explicit reset.

        Returns:
            result: None. Both deployment and adaptation use identity.
        """

        super().ResetCoefficients()
        if hasattr(self, "adaptiveCoefficients"):
            self.InitializeAdaptiveState(copyActiveCoefficients=True)

    def SetCoefficients(self, coefficients: np.ndarray) -> None:
        """Replace active and shadow coefficients with one validated vector.

        Processing details:
            Algorithm: Delegate size and finiteness validation to DpdGmp,
            copy the accepted active vector into the adaptive model, and clear
            history and accumulated frame diagnostics.

        Args:
            coefficients: Complex vector in canonical GMP feature order.

        Returns:
            result: None. Both active and shadow models use the supplied data.
        """

        super().SetCoefficients(coefficients)
        self.InitializeAdaptiveState(copyActiveCoefficients=True)

    def CalculateFeatureScale(
        self, referenceSignal: np.ndarray
    ) -> np.ndarray:
        """Calculate frame-frozen RMS scale for every GMP feature.

        Processing details:
            Algorithm: Traverse the reference in existing bounded chunks,
            accumulate each basis-column energy without forming a full-frame
            matrix, divide by sample count, take square roots, and apply the
            configured numerical floor.

        Args:
            referenceSignal: Normalized floating reference vector.

        Returns:
            result: Positive feature RMS vector in canonical coefficient order.
        """

        featureEnergy = np.zeros(
            len(self.featureSpecs), dtype=float
        )
        chunkSize = cast(int, self.parameters["chunkSize"])
        for startIndex in range(0, referenceSignal.size, chunkSize):
            stopIndex = min(
                startIndex + chunkSize,
                referenceSignal.size,
            )
            basisChunk = self.BuildBasisChunk(
                referenceSignal,
                startIndex,
                stopIndex,
            )
            featureEnergy += np.sum(
                np.abs(basisChunk) ** 2,
                axis=0,
            )
        featureScale = np.sqrt(
            featureEnergy / float(referenceSignal.size)
        )
        return np.maximum(
            featureScale,
            float(self.parameters["normalizationEpsilon"]),
        )

    def PrepareFeatureScale(
        self, referenceSignal: np.ndarray
    ) -> np.ndarray:
        """Decode one public frame and install its frozen feature scales.

        Processing details:
            Algorithm: Synchronize the feature structure, decode the public
            signal once, calculate bounded-memory column RMS values, and copy
            them into owned state used by every later UpdateSample call.

        Args:
            referenceSignal: Public floating samples or fixed-point I/Q codes.

        Returns:
            result: Detached positive feature-scale vector.
        """

        self.SynchronizeStructure()
        preparedReference = self.PreparePublicSignal(
            referenceSignal,
            "referenceSignal",
        )
        self.featureScale = self.CalculateFeatureScale(
            preparedReference
        )
        return self.featureScale.copy()

    def BeginFrame(
        self,
        referenceSignal: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Start an independent adaptive frame without changing coefficients.

        Processing details:
            Algorithm: Synchronize parameters, clear causal sample history and
            frame statistics, reset running feature powers, and either
            calculate frozen frame scales from the supplied public reference
            or initialize running normalization for genuine streaming mode.

        Args:
            referenceSignal: Complete public reference required by frame-scale
                mode; optional and ignored by running-scale mode.

        Returns:
            result: Detached feature scales active at frame start.
        """

        self.SynchronizeStructure()
        self.sampleHistory.fill(0.0 + 0.0j)
        self.frameStartCoefficients = (
            self.adaptiveCoefficients.copy()
        )
        self.frameSampleCount = 0
        self.frameUpdateCount = 0
        self.frameErrorPower = 0.0
        self.frameTargetPower = 0.0
        self.frameMaximumUpdateNorm = 0.0
        normalizationEpsilon = float(
            self.parameters["normalizationEpsilon"]
        )
        self.featurePower.fill(normalizationEpsilon)
        featureScaleMode = str(
            self.parameters["featureScaleMode"]
        ).strip().lower()
        if featureScaleMode == "frame":
            if referenceSignal is None:
                raise ValueError(
                    "referenceSignal is required when "
                    "featureScaleMode is 'frame'"
                )
            self.PrepareFeatureScale(referenceSignal)
        else:
            self.featureScale.fill(1.0)
        return self.featureScale.copy()

    def BuildFeatureVector(
        self, referenceSample: complex
    ) -> np.ndarray:
        """Insert one sample and construct one causal GMP feature row.

        Processing details:
            Algorithm: Shift the fixed-size newest-first history by one,
            insert the current sample, then evaluate every canonical main,
            lagging-envelope, or leading-envelope term with exactly the same
            delay convention as the batch BuildGmpBasisChunk implementation.

        Args:
            referenceSample: One finite normalized complex reference sample.

        Returns:
            result: One complex feature vector in GetFeatureSpecs order.
        """

        complexSample = complex(referenceSample)
        if not (
            np.isfinite(complexSample.real)
            and np.isfinite(complexSample.imag)
        ):
            raise ValueError("referenceSample must be finite")
        if self.sampleHistory.size > 1:
            self.sampleHistory[1:] = self.sampleHistory[:-1]
        self.sampleHistory[0] = complexSample
        featureVector = np.empty(
            len(self.featureSpecs), dtype=np.complex128
        )
        for featureIndex, (
            branchName,
            nonlinearOrder,
            memoryIndex,
            crossIndex,
        ) in enumerate(self.featureSpecs):
            if branchName == "main":
                carrierSample = self.sampleHistory[memoryIndex]
                envelopeSample = carrierSample
            elif branchName == "lagging":
                carrierSample = self.sampleHistory[memoryIndex]
                envelopeSample = self.sampleHistory[
                    memoryIndex + crossIndex
                ]
            else:
                carrierSample = self.sampleHistory[
                    memoryIndex + crossIndex
                ]
                envelopeSample = self.sampleHistory[memoryIndex]
            featureVector[featureIndex] = (
                carrierSample
                * abs(envelopeSample) ** (nonlinearOrder - 1)
            )
        return featureVector

    def ResolveFeatureScale(
        self, featureVector: np.ndarray
    ) -> np.ndarray:
        """Return frozen scales or update running feature-power estimates.

        Processing details:
            Algorithm: In frame mode return the precomputed RMS vector. In
            running mode update one exponential power estimate per feature,
            floor every power, take square roots, and retain the resulting
            scales for diagnostics and coefficient-coordinate conversion.

        Args:
            featureVector: Current canonical complex GMP feature row.

        Returns:
            result: Positive scale used by the current sample update.
        """

        featureScaleMode = str(
            self.parameters["featureScaleMode"]
        ).strip().lower()
        if featureScaleMode == "frame":
            return self.featureScale
        forgettingFactor = float(
            self.parameters["featurePowerForgettingFactor"]
        )
        self.featurePower = (
            forgettingFactor * self.featurePower
            + (1.0 - forgettingFactor)
            * np.abs(featureVector) ** 2
        )
        self.featureScale = np.sqrt(
            np.maximum(
                self.featurePower,
                float(self.parameters["normalizationEpsilon"]),
            )
        )
        return self.featureScale

    def UpdateSampleFloating(
        self,
        referenceSample: complex,
        targetSample: complex,
        sampleWeight: float = 1.0,
    ) -> complex:
        """Predict and adapt shadow coefficients for one normalized sample.

        Processing details:
            Algorithm: Build one causal GMP feature row, predict the target
            with pre-update shadow coefficients, accumulate online error,
            optionally skip according to decimation, normalize feature and
            coefficient coordinates, apply complex LMS or NLMS plus leakage
            toward identity, project an excessive coefficient step, and
            optionally commit the new shadow vector immediately.

        Args:
            referenceSample: One normalized complex regression-input sample.
            targetSample: One normalized complex desired-output sample.
            sampleWeight: Nonnegative importance applied to error and metrics.

        Returns:
            result: Complex pre-update shadow prediction for this sample.
        """

        complexTarget = complex(targetSample)
        if not (
            np.isfinite(complexTarget.real)
            and np.isfinite(complexTarget.imag)
        ):
            raise ValueError("targetSample must be finite")
        if (
            not isinstance(sampleWeight, (int, float))
            or isinstance(sampleWeight, bool)
            or not np.isfinite(sampleWeight)
            or float(sampleWeight) < 0.0
        ):
            raise ValueError(
                "sampleWeight must be finite and nonnegative"
            )
        featureVector = self.BuildFeatureVector(referenceSample)
        predictedSample = complex(
            np.dot(featureVector, self.adaptiveCoefficients)
        )
        errorSample = complexTarget - predictedSample
        boundedWeight = min(
            float(sampleWeight),
            float(self.parameters["maximumSampleWeight"]),
        )
        self.frameSampleCount += 1
        self.frameErrorPower += (
            boundedWeight * abs(errorSample) ** 2
        )
        self.frameTargetPower += (
            boundedWeight * abs(complexTarget) ** 2
        )
        updateDecimation = cast(
            int, self.parameters["updateDecimation"]
        )
        shouldUpdate = (
            boundedWeight > 0.0
            and (self.frameSampleCount - 1) % updateDecimation == 0
            and float(np.vdot(featureVector, featureVector).real)
            > np.finfo(float).tiny
        )
        if not shouldUpdate:
            return predictedSample

        featureScale = self.ResolveFeatureScale(featureVector)
        normalizedFeature = featureVector / featureScale
        normalizedCoefficients = (
            self.adaptiveCoefficients * featureScale
        )
        normalizedIdentity = (
            self.identityCoefficients * featureScale
        )
        gradientVector = (
            boundedWeight
            * np.conj(normalizedFeature)
            * errorSample
        )
        adaptationMode = str(
            self.parameters["adaptationMode"]
        ).strip().lower()
        if adaptationMode == "nlms":
            denominator = (
                float(self.parameters["normalizationEpsilon"])
                + float(
                    np.vdot(
                        normalizedFeature,
                        normalizedFeature,
                    ).real
                )
            )
        else:
            denominator = 1.0
        learningRate = float(self.parameters["learningRate"])
        leakageFactor = float(self.parameters["leakageFactor"])
        normalizedUpdate = learningRate * (
            gradientVector / denominator
            - leakageFactor
            * (normalizedCoefficients - normalizedIdentity)
        )
        candidateCoefficients = (
            normalizedCoefficients + normalizedUpdate
        ) / featureScale
        coefficientUpdate = (
            candidateCoefficients - self.adaptiveCoefficients
        )
        coefficientUpdateNorm = float(
            np.linalg.norm(coefficientUpdate)
        )
        maximumUpdateNorm = self.parameters[
            "maximumSampleUpdateNorm"
        ]
        if (
            maximumUpdateNorm is not None
            and coefficientUpdateNorm > float(maximumUpdateNorm)
        ):
            coefficientUpdate *= (
                float(maximumUpdateNorm)
                / coefficientUpdateNorm
            )
            coefficientUpdateNorm = float(maximumUpdateNorm)
        candidateCoefficients = (
            self.adaptiveCoefficients + coefficientUpdate
        )
        if not np.all(np.isfinite(candidateCoefficients)):
            raise RuntimeError(
                "sample LMS coefficient update produced nonfinite values"
            )
        self.adaptiveCoefficients = candidateCoefficients
        self.frameUpdateCount += 1
        self.frameMaximumUpdateNorm = max(
            self.frameMaximumUpdateNorm,
            coefficientUpdateNorm,
        )
        commitMode = str(
            self.parameters["coefficientCommitMode"]
        ).strip().lower()
        if commitMode == "sample":
            self.coefficients = (
                self.adaptiveCoefficients.copy()
            )
        return predictedSample

    def UpdateSample(
        self,
        referenceSample: complex,
        targetSample: complex,
        sampleWeight: float = 1.0,
    ) -> complex:
        """Adapt one sample using the configured public data convention.

        Processing details:
            Algorithm: Decode one public reference and target code through the
            FixedPoint boundary, invoke the normalized scalar update exactly
            once, and encode the pre-update prediction back to the same public
            complex128 container convention.

        Args:
            referenceSample: Floating sample or complex I/Q integer code.
            targetSample: Floating target or complex I/Q integer code.
            sampleWeight: Nonnegative scalar training importance.

        Returns:
            result: Public-format pre-update prediction for this sample.
        """

        interfaceFormat = FixedPoint(self.width)
        floatingReference = interfaceFormat.DecodeComplex(
            np.asarray([referenceSample], dtype=np.complex128)
        )[0]
        floatingTarget = interfaceFormat.DecodeComplex(
            np.asarray([targetSample], dtype=np.complex128)
        )[0]
        floatingPrediction = self.UpdateSampleFloating(
            complex(floatingReference),
            complex(floatingTarget),
            sampleWeight,
        )
        return complex(
            interfaceFormat.EncodeComplex(
                np.asarray(
                    [floatingPrediction],
                    dtype=np.complex128,
                )
            )[0]
        )

    def CommitCoefficients(self) -> None:
        """Copy sample-updated shadow coefficients into active inference.

        Processing details:
            Algorithm: Validate feature-count and finiteness invariants, then
            copy the complete shadow vector in one assignment so a frame-mode
            deployment never observes a partially updated coefficient set.

        Returns:
            result: None. Subsequent Process calls use the adaptive vector.
        """

        if (
            self.adaptiveCoefficients.size
            != len(self.featureSpecs)
            or not np.all(np.isfinite(self.adaptiveCoefficients))
        ):
            raise RuntimeError(
                "adaptive coefficients are invalid for active structure"
            )
        self.coefficients = self.adaptiveCoefficients.copy()

    def EvaluateNmseWithCoefficients(
        self,
        referenceSignal: np.ndarray,
        targetSignal: np.ndarray,
        sampleWeights: np.ndarray,
        coefficients: np.ndarray,
    ) -> float:
        """Evaluate one fixed coefficient vector without adapting it.

        Processing details:
            Algorithm: Traverse the reference in bounded chunks, predict with
            the supplied fixed vector, apply the deployment magnitude limit,
            accumulate weighted residual and target powers, and return their
            ratio in decibels.

        Args:
            referenceSignal: Normalized complex regression input.
            targetSignal: Matching normalized desired output.
            sampleWeights: Nonnegative per-sample evaluation weights.
            coefficients: Fixed canonical coefficient vector to evaluate.

        Returns:
            result: Weighted fixed-model NMSE in decibels.
        """

        residualPower = 0.0
        targetPower = 0.0
        chunkSize = cast(int, self.parameters["chunkSize"])
        for startIndex in range(0, referenceSignal.size, chunkSize):
            stopIndex = min(
                startIndex + chunkSize,
                referenceSignal.size,
            )
            basisChunk = self.BuildBasisChunk(
                referenceSignal,
                startIndex,
                stopIndex,
            )
            predictedChunk = self.LimitMagnitude(
                basisChunk @ coefficients
            )
            targetChunk = targetSignal[startIndex:stopIndex]
            weightChunk = sampleWeights[startIndex:stopIndex]
            residualPower += float(
                np.sum(
                    weightChunk
                    * np.abs(targetChunk - predictedChunk) ** 2
                )
            )
            targetPower += float(
                np.sum(weightChunk * np.abs(targetChunk) ** 2)
            )
        numericFloor = np.finfo(float).tiny
        return float(
            10.0
            * np.log10(
                max(residualPower, numericFloor)
                / max(targetPower, numericFloor)
            )
        )

    def UpdateFromLabels(
        self,
        referenceSignal: np.ndarray,
        targetSignal: np.ndarray,
        sampleWeights: Optional[np.ndarray] = None,
    ) -> DpdLmsTrainingResult:
        """Run one ordered sample-by-sample pass over a waveform pair.

        Processing details:
            Algorithm: Decode and validate the public vectors, combine
            explicit and envelope-peak weights, evaluate the fixed pre-update
            shadow model, begin an independent frame, visit samples strictly
            in time order through UpdateSampleFloating, commit once in frame
            mode, evaluate the final fixed shadow model, and store online plus
            before/after diagnostics.

        Args:
            referenceSignal: Desired DPD input defining GMP feature samples.
            targetSignal: Matching PA-input, ILC-label, or inverse target.
            sampleWeights: Optional nonnegative per-sample importance.

        Returns:
            result: Immutable sample-update and fixed-model diagnostics.
        """

        self.SynchronizeStructure()
        preparedReference = self.PreparePublicSignal(
            referenceSignal,
            "referenceSignal",
        )
        preparedTarget = self.PreparePublicSignal(
            targetSignal,
            "targetSignal",
        )
        if preparedReference.size != preparedTarget.size:
            raise ValueError(
                "referenceSignal and targetSignal must have equal length"
            )
        resolvedWeights = self.BuildSampleWeights(
            preparedReference,
            sampleWeights,
            1.0,
        )
        initialCoefficients = self.adaptiveCoefficients.copy()
        beforeNmseDb = self.EvaluateNmseWithCoefficients(
            preparedReference,
            preparedTarget,
            resolvedWeights,
            initialCoefficients,
        )
        interfaceFormat = FixedPoint(self.width)
        publicReference = interfaceFormat.EncodeComplex(
            preparedReference
        )
        self.BeginFrame(publicReference)
        for sampleIndex in range(preparedReference.size):
            self.UpdateSampleFloating(
                complex(preparedReference[sampleIndex]),
                complex(preparedTarget[sampleIndex]),
                float(resolvedWeights[sampleIndex]),
            )
        commitMode = str(
            self.parameters["coefficientCommitMode"]
        ).strip().lower()
        if commitMode == "frame":
            self.CommitCoefficients()
        afterNmseDb = self.EvaluateNmseWithCoefficients(
            preparedReference,
            preparedTarget,
            resolvedWeights,
            self.adaptiveCoefficients,
        )
        numericFloor = np.finfo(float).tiny
        onlineNmseDb = float(
            10.0
            * np.log10(
                max(self.frameErrorPower, numericFloor)
                / max(self.frameTargetPower, numericFloor)
            )
        )
        trainingResult = DpdLmsTrainingResult(
            sampleCount=self.frameSampleCount,
            updateCount=self.frameUpdateCount,
            featureCount=len(self.featureSpecs),
            adaptationMode=str(
                self.parameters["adaptationMode"]
            ).strip().lower(),
            beforeNmseDb=beforeNmseDb,
            onlineNmseDb=onlineNmseDb,
            afterNmseDb=afterNmseDb,
            coefficientUpdateNorm=float(
                np.linalg.norm(
                    self.adaptiveCoefficients
                    - initialCoefficients
                )
            ),
            maximumSampleUpdateNorm=(
                self.frameMaximumUpdateNorm
            ),
            coefficientsCommitted=True,
        )
        self.lastLmsTrainingResult = trainingResult
        return trainingResult

    def UpdateIndirect(
        self,
        paInputSignal: np.ndarray,
        paOutputSignal: np.ndarray,
        sampleRateHz: float,
        signalProcessingParameters: Optional[
            Mapping[str, object]
        ] = None,
        sampleWeights: Optional[np.ndarray] = None,
    ) -> DpdLmsTrainingResult:
        """Adapt an indirect-learning postinverse sample by sample.

        Processing details:
            Algorithm: Decode the actual PA input and an arbitrary-length
            feedback capture, use SigProc to estimate frame-level delay, CFO,
            SFO, and common complex gain and return one aligned output sample
            per PA-input reference sample, encode both normalized vectors back
            through the public boundary, then train the output-to-input
            postinverse in strict chronological order.

        Args:
            paInputSignal: Public waveform actually presented to the PA.
            paOutputSignal: Public feedback capture; length may differ.
            sampleRateHz: Positive complex sample rate used by synchronization.
            signalProcessingParameters: Optional SigProc overrides.
            sampleWeights: Optional per-aligned-sample importance.

        Returns:
            result: Indirect sample-update and fixed-model diagnostics.
        """

        if (
            not isinstance(sampleRateHz, (int, float))
            or isinstance(sampleRateHz, bool)
            or not np.isfinite(sampleRateHz)
            or float(sampleRateHz) <= 0.0
        ):
            raise ValueError("sampleRateHz must be finite and positive")
        floatingInput = self.PreparePublicSignal(
            paInputSignal,
            "paInputSignal",
        )
        floatingOutput = self.PreparePublicSignal(
            paOutputSignal,
            "paOutputSignal",
        )
        signalProcessor = SigProc(
            floatingInput,
            float(sampleRateHz),
            parameters=signalProcessingParameters,
        )
        processedOutput = signalProcessor.Process(
            floatingOutput
        ).processedSignal
        interfaceFormat = FixedPoint(self.width)
        return self.UpdateFromLabels(
            interfaceFormat.EncodeComplex(processedOutput),
            interfaceFormat.EncodeComplex(floatingInput),
            sampleWeights,
        )

    def GetAdaptiveCoefficients(self) -> np.ndarray:
        """Return a detached copy of sample-updated shadow coefficients.

        Processing details:
            Algorithm: Synchronize live structure and copy owned adaptive
            storage so callers can inspect progress without modifying active
            or shadow inference state.

        Returns:
            result: Complex shadow vector in canonical feature order.
        """

        self.SynchronizeStructure()
        return self.adaptiveCoefficients.copy()

    def GetLastLmsTrainingResult(
        self,
    ) -> Optional[DpdLmsTrainingResult]:
        """Return the most recent sample-by-sample training summary.

        Processing details:
            Algorithm: Synchronize structure and return the immutable result
            object directly; structure rebuilds clear stale diagnostics.

        Returns:
            result: Last LMS result or None before a completed frame update.
        """

        self.SynchronizeStructure()
        return self.lastLmsTrainingResult
