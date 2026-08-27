"""Model transmitter, coupled PA, and parallel channel/feedback outputs.

The channel can apply flat or frequency-selective widely-linear transmitter and
feedback I/Q imbalance, causal complex coupling paths before a multi-chain PA
bank, and coupling after its nonlinear outputs. One PA evaluation always
produces a forward channel observation. Forward sampling copies it into the
feedback return, while feedback sampling additionally evaluates the embedded
receiver. Public fixed-point boundaries use raw integer I/Q codes while every
physical operation remains floating point.
"""

from collections import ChainMap
from difflib import SequenceMatcher
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np

# Cross-package imports support canonical ``inc.lib`` and compatibility
# ``lib`` package entry points without relying on a parent that may not exist.
if __package__ and "." in __package__:
    from ..utils.ConfigUtils import (
        FindUnknownParameterNames,
    )
    from ..utils.FixedPoint import FixedPoint
    from ..utils.SigProc import FeedbackIqCalibration, PowerCalibration
else:
    from utils.ConfigUtils import (
        FindUnknownParameterNames,
    )
    from utils.FixedPoint import FixedPoint
    from utils.SigProc import FeedbackIqCalibration, PowerCalibration


class Channel:
    """Apply one PA evaluation and expose channel plus feedback observations.

    ``noiseAmpMv`` is the RMS magnitude of the complete complex noise
    envelope, not the RMS of each individual I or Q component. Therefore the
    two real Gaussian components each use ``noiseAmpMv / sqrt(2)`` RMS.
    ``noiseSnrDb`` instead derives one noise RMS per chain from that chain's
    active-region signal RMS. Normalized PA output RMS equal to one represents
    ``maximumOutputPowerDbm`` at ``loadResistanceOhm``.
    """

    def __init__(
        self,
        paModel: Optional[Any] = None,
        parameters: Optional[Mapping[str, object]] = None,
        width: Optional[int] = None,
        **parameterOverrides: object,
    ) -> None:
        """Initialize a live ChainMap-backed PA-to-receiver channel.

        Processing details:
            Algorithm: Define every default inside the constructor, layer
            direct and caller-owned overrides ahead of the immutable default
            mapping, validate the physical settings, initialize one random
            generator, and optionally bind a PA processing object.

        Args:
            paModel: Optional object exposing ``Process`` and preferably
                ``ProcessFloating``. It is required by ``Process`` but not by
                ``ProcessPaOutput``.
            parameters: Optional caller-owned live mapping of channel values.
            width: Optional public I/Q component width. None selects the
                internal 16-bit default, zero selects floating point, and a
                positive value selects signed integer I/Q codes.
            parameterOverrides: Highest-priority local configuration values.
                Unsupported names raise ``TypeError`` because Channel uses a
                strict, case-sensitive public configuration vocabulary.

        Returns:
            result: None. The configured channel is ready for processing.
        """

        self.defaultParameters: Mapping[str, object] = MappingProxyType(
            {
                "sampleMode": "forward",
                "sampleRateHz": 1.0,
                "thermalRunMode": "steady_state",
                "thermalDutyCycle": 1.0,
                "thermalSteadyStateToleranceC": 1.0e-4,
                "maximumThermalSteadyStateIterations": 100,
                "phaseDegrees": 0,
                "noiseAmpMv": None,
                "noisePwrDbm": None,
                "noiseSnrDb": None,
                "txIqImbalanceEnabled": True,
                "txIqGainImbalanceDb": 0.0,
                "txIqPhaseImbalanceDegrees": 0.0,
                "txIqDirectFirTaps": None,
                "txIqImageFirTaps": None,
                "txDcOffset": 0.0 + 0.0j,
                "fbGainDb": 0.0,
                "fbPhaseDegrees": 0.0,
                "fbFirTaps": None,
                "fbIntegerDelaySamples": 0,
                "fbFractionalDelaySamples": 0.0,
                "fbCarrierFrequencyOffsetHz": 0.0,
                "fbSamplingFrequencyOffsetPpm": 0.0,
                "fbIqImbalanceEnabled": True,
                "fbIqGainImbalanceDb": 0.0,
                "fbIqPhaseImbalanceDegrees": 0.0,
                "fbIqDirectFirTaps": None,
                "fbIqImageFirTaps": None,
                "fbDcOffset": 0.0 + 0.0j,
                "fbIqCompensationMode": "none",
                "fbPhasePairResponses": (1.0 + 0.0j, 0.0 + 1.0j),
                "fbIqCompensationFilterLength": 1,
                "fbIqCompensationRegularization": 1.0e-6,
                "fbThirdOrderCoefficient": 0.0 + 0.0j,
                "fbClipAmplitude": None,
                "fbAdcWidth": None,
                "fbAdcFullScale": 1.0,
                "prePaCouplingPaths": None,
                "postPaCouplingPaths": None,
                "loadResistanceOhm": 50.0,
                "maximumOutputPowerDbm": 25.0,
                "calibrationToleranceDb": 0.25,
                "maximumCalibrationIterations": 60,
                "calibrationLearningRate": 0.8,
                "maximumDriveAdjustmentDb": 6.0,
                "calibrationDigitalHeadroomDb": 6.0,
                "jointPowerCalibration": None,
                "calibrationProbeStepDb": 0.05,
                "calibrationRegularization": 1.0e-6,
                "activePowerThresholdDb": -60.0,
                "activeGapToleranceSamples": 16,
                "randomSeed": 1701,
                "width": 16,
            }
        )
        directOverrides = dict(parameterOverrides)
        if width is not None:
            directOverrides["width"] = width
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping or None")
        externalParameters: Mapping[object, object] = (
            {} if parameters is None else parameters
        )
        unknownExternalNames = FindUnknownParameterNames(
            externalParameters,
            self.defaultParameters,
        )
        if unknownExternalNames:
            raise TypeError(
                self.FormatUnknownParameterError(
                    "Channel",
                    unknownExternalNames,
                    tuple(self.defaultParameters),
                )
            )
        unknownOverrideNames = FindUnknownParameterNames(
            directOverrides,
            self.defaultParameters,
        )
        if unknownOverrideNames:
            raise TypeError(
                self.FormatUnknownParameterError(
                    "Channel",
                    unknownOverrideNames,
                    tuple(self.defaultParameters),
                )
            )
        recognizedOverrides = {
            str(parameterName): parameterValue
            for parameterName, parameterValue in directOverrides.items()
        }
        self.parameters: ChainMap[str, object] = ChainMap(
            recognizedOverrides,
            externalParameters,
            self.defaultParameters,
        )
        self.paModel: Optional[Any] = None
        self._paProcessMethod: Optional[Any] = None
        self._powerCalibration: Optional[PowerCalibration] = None
        self._calibrationDriveDbPerChain: Tuple[float, ...] = tuple()
        self._lastTransmitterOutput: Optional[np.ndarray] = None
        self._lastActualPaInput: Optional[np.ndarray] = None
        self._lastCalibrationOutputPowerDbm: Optional[
            Union[float, Tuple[float, ...]]
        ] = None
        self._feedbackIqCalibration: Optional[
            FeedbackIqCalibration
        ] = None
        self._feedbackIqCalibrationSignature: Optional[
            Tuple[object, ...]
        ] = None
        self._lastFeedbackPhasePair: Optional[
            Tuple[np.ndarray, np.ndarray]
        ] = None
        self._activeRandomSeed: Optional[int] = None
        self._randomGenerator = np.random.default_rng()
        self.ValidateParameters()
        self.SynchronizeRandomGenerator(forceReset=True)
        if paModel is not None:
            self.SetPaModel(paModel)

    @property
    def Width(self) -> int:
        """Return the public I/Q component width.

        Processing details:
            Algorithm: Read the validated value from the current ChainMap so
            live caller-owned parameter changes affect the next operation.

        Returns:
            result: Zero for floating mode or a positive fixed-point width.
        """

        return cast(int, self.parameters["width"])

    width = Width

    @property
    def SampleMode(self) -> str:
        """Return the normalized feedback-observation sampling mode.

        Processing details:
            Algorithm: Strip surrounding whitespace, convert the configured
            name to lowercase, and return the validated canonical value so
            live caller-owned mapping changes affect the next call. Forward
            mode makes public ``fbOut`` an exact copy of ``chOut``; feedback
            mode evaluates the embedded receiver for ``fbOut``. Compatibility
            methods still return only the selected observation.

        Returns:
            result: ``"forward"`` for instrument sampling or ``"fb"`` for
                the nonideal embedded feedback receiver.
        """

        return str(self.parameters["sampleMode"]).strip().lower()

    sampleMode = SampleMode

    @staticmethod
    def FormatUnknownParameterError(
        ownerName: str,
        unknownParameterNames: Tuple[str, ...],
        supportedParameterNames: Tuple[str, ...],
    ) -> str:
        """Build a closest-first error for unknown configuration names.

        Processing details:
            Algorithm: Compare names case-insensitively with Ratcliff-Obershelp
            sequence similarity, use length difference and lexical order as
            deterministic tie breakers, and list every supported name for
            each unknown name from highest to lowest relationship.

        Args:
            ownerName: Configuration owner and optional nested-path context.
            unknownParameterNames: Unsupported caller-provided names.
            supportedParameterNames: Complete legal vocabulary for the owner.

        Returns:
            result: Multi-line TypeError message with closest candidates first.
        """

        candidateSections = []
        for unknownParameterName in unknownParameterNames:
            foldedUnknownName = unknownParameterName.casefold()
            rankedParameterNames = sorted(
                supportedParameterNames,
                key=lambda supportedName: (
                    -SequenceMatcher(
                        None,
                        foldedUnknownName,
                        supportedName.casefold(),
                    ).ratio(),
                    abs(len(supportedName) - len(unknownParameterName)),
                    supportedName.casefold(),
                    supportedName,
                ),
            )
            candidateSections.append(
                f"{unknownParameterName}: "
                + ", ".join(rankedParameterNames)
            )
        return (
            f"{ownerName} received unknown configuration parameter(s): "
            + ", ".join(unknownParameterNames)
            + ". Parameter names are case-sensitive. All supported "
            "parameter names ordered from highest to lowest similarity:\n"
            + "\n".join(candidateSections)
        )

    def GetParameters(self) -> Dict[str, object]:
        """Return one flattened snapshot of all effective channel settings.

        Processing details:
            Algorithm: Resolve the ChainMap precedence into a new dictionary
            without exposing or mutating any internal configuration layer.

        Returns:
            result: Ordinary dictionary containing every supported setting.
        """

        self.ValidateParameters()
        return dict(self.parameters)

    def UpdateParameters(self, **parameterOverrides: object) -> None:
        """Apply validated high-priority channel overrides transactionally.

        Processing details:
            Algorithm: Reject unsupported names before changing state, update
            the local ChainMap layer, validate the complete resolved state,
            and restore the prior state when any recognized value is invalid.

        Args:
            parameterOverrides: Supported values to replace locally.

        Returns:
            result: None. Valid updates affect subsequent channel calls.
        """

        unknownParameterNames = FindUnknownParameterNames(
            parameterOverrides,
            self.defaultParameters,
        )
        if unknownParameterNames:
            raise TypeError(
                self.FormatUnknownParameterError(
                    "Channel.UpdateParameters",
                    unknownParameterNames,
                    tuple(self.defaultParameters),
                )
            )
        recognizedOverrides = {
            str(parameterName): parameterValue
            for parameterName, parameterValue in parameterOverrides.items()
        }
        previousOverrides = dict(self.parameters.maps[0])
        previousSeed = self._activeRandomSeed
        self.parameters.maps[0].update(recognizedOverrides)
        try:
            self.ValidateParameters()
            self.SynchronizeRandomGenerator()
        except (TypeError, ValueError):
            self.parameters.maps[0].clear()
            self.parameters.maps[0].update(previousOverrides)
            self._activeRandomSeed = previousSeed
            self.SynchronizeRandomGenerator(forceReset=True)
            raise
        calibrationSensitiveNames = {
            "sampleRateHz",
            "phaseDegrees",
            "fbGainDb",
            "fbPhaseDegrees",
            "fbFirTaps",
            "fbIntegerDelaySamples",
            "fbFractionalDelaySamples",
            "fbCarrierFrequencyOffsetHz",
            "fbSamplingFrequencyOffsetPpm",
            "fbIqImbalanceEnabled",
            "fbIqGainImbalanceDb",
            "fbIqPhaseImbalanceDegrees",
            "fbIqDirectFirTaps",
            "fbIqImageFirTaps",
            "fbDcOffset",
            "fbPhasePairResponses",
            "fbIqCompensationFilterLength",
            "fbIqCompensationRegularization",
            "fbThirdOrderCoefficient",
            "fbClipAmplitude",
            "fbAdcWidth",
            "fbAdcFullScale",
            "width",
        }
        if any(
            parameterName in calibrationSensitiveNames
            for parameterName in recognizedOverrides
        ):
            self.ResetFeedbackIqCalibration()

    def ValidateParameters(self) -> None:
        """Validate phase, noise, physical scaling, seed, and interface width.

        Processing details:
            Algorithm: Reject unknown names in both live caller layers,
            restrict phase to minus 90, zero, or plus 90 degrees, enforce
            mutual exclusion of the three noise controls, check all physical
            scalars for finite values and valid domains, then use
            ``FixedPoint`` as the authoritative width validator.

        Returns:
            result: None. Invalid recognized settings raise an exception.
        """

        for layerIndex, parameterLayer in enumerate(
            self.parameters.maps[:2]
        ):
            unknownParameterNames = FindUnknownParameterNames(
                parameterLayer,
                self.defaultParameters,
            )
            if unknownParameterNames:
                layerName = (
                    "local override"
                    if layerIndex == 0
                    else "external parameter"
                )
                raise TypeError(
                    self.FormatUnknownParameterError(
                        f"Channel {layerName} layer",
                        unknownParameterNames,
                        tuple(self.defaultParameters),
                    )
                )

        sampleMode = self.parameters["sampleMode"]
        if (
            not isinstance(sampleMode, str)
            or sampleMode.strip().lower() not in ("forward", "fb")
        ):
            raise ValueError(
                "sampleMode has an invalid value. Allowed values: "
                "'forward' or 'fb'."
            )
        fbIqCompensationMode = self.parameters[
            "fbIqCompensationMode"
        ]
        if (
            not isinstance(fbIqCompensationMode, str)
            or fbIqCompensationMode.strip().lower()
            not in ("none", "phase_pair", "filter")
        ):
            raise ValueError(
                "fbIqCompensationMode has an invalid value. Allowed values: "
                "'none', 'phase_pair', or 'filter'."
            )
        normalizedCompensationMode = fbIqCompensationMode.strip().lower()
        if (
            normalizedCompensationMode == "phase_pair"
            and sampleMode.strip().lower() != "fb"
        ):
            raise ValueError(
                "fbIqCompensationMode='phase_pair' requires "
                "sampleMode='fb' so the two embedded feedback captures can "
                "be acquired before DPD training"
            )
        fbPhasePairResponses = self.parameters["fbPhasePairResponses"]
        if isinstance(fbPhasePairResponses, (str, bytes)) or not (
            isinstance(fbPhasePairResponses, Sequence)
            or isinstance(fbPhasePairResponses, np.ndarray)
        ):
            raise TypeError(
                "fbPhasePairResponses has an invalid type. Allowed values: "
                "a sequence of exactly two finite nonzero complex responses."
            )
        if len(fbPhasePairResponses) != 2:
            raise ValueError(
                "fbPhasePairResponses has an invalid length. Allowed value: "
                "exactly two responses for the nominal 0- and 90-degree "
                "states."
            )
        try:
            phaseResponseArray = np.asarray(
                fbPhasePairResponses, dtype=np.complex128
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise TypeError(
                "fbPhasePairResponses must contain finite complex scalars"
            ) from error
        if not np.all(np.isfinite(phaseResponseArray)):
            raise ValueError(
                "fbPhasePairResponses has an invalid value. Every response "
                "must be finite."
            )
        if np.any(np.abs(phaseResponseArray) <= np.finfo(float).tiny):
            raise ValueError(
                "fbPhasePairResponses has an invalid value. Responses must "
                "be nonzero."
            )
        phaseSeparationMatrix = np.asarray(
            (
                (
                    phaseResponseArray[0],
                    np.conj(phaseResponseArray[0]),
                ),
                (
                    phaseResponseArray[1],
                    np.conj(phaseResponseArray[1]),
                ),
            ),
            dtype=np.complex128,
        )
        phaseResponseScale = float(np.max(np.abs(phaseResponseArray)))
        if (
            abs(np.linalg.det(phaseSeparationMatrix))
            <= np.finfo(float).eps * phaseResponseScale**2
        ):
            raise ValueError(
                "fbPhasePairResponses cannot separate direct and image "
                "components. Their relative phase must not be 0 or 180 "
                "degrees."
            )
        fbIqCompensationFilterLength = self.parameters[
            "fbIqCompensationFilterLength"
        ]
        if (
            not isinstance(fbIqCompensationFilterLength, int)
            or isinstance(fbIqCompensationFilterLength, bool)
            or fbIqCompensationFilterLength < 1
        ):
            raise ValueError(
                "fbIqCompensationFilterLength has an invalid value. Allowed "
                "range: integer in [1, +inf) taps."
            )
        fbIqCompensationRegularization = self.parameters[
            "fbIqCompensationRegularization"
        ]
        if (
            not isinstance(
                fbIqCompensationRegularization, (int, float)
            )
            or isinstance(fbIqCompensationRegularization, bool)
            or not np.isfinite(fbIqCompensationRegularization)
            or float(fbIqCompensationRegularization) <= 0.0
        ):
            raise ValueError(
                "fbIqCompensationRegularization has an invalid value. "
                "Allowed range: finite real number in (0, +inf)."
            )
        sampleRateHz = self.parameters["sampleRateHz"]
        if (
            not isinstance(sampleRateHz, (int, float))
            or isinstance(sampleRateHz, bool)
            or not np.isfinite(sampleRateHz)
            or float(sampleRateHz) <= 0.0
        ):
            raise ValueError(
                "sampleRateHz has an invalid value. Allowed range: "
                "finite real number in (0, +inf) Hz."
            )

        thermalRunMode = self.parameters["thermalRunMode"]
        if (
            not isinstance(thermalRunMode, str)
            or thermalRunMode.strip().lower()
            not in ("steady_state", "transient")
        ):
            raise ValueError(
                "thermalRunMode has an invalid value. Allowed values: "
                "'steady_state' or 'transient'."
            )
        thermalDutyCycle = self.parameters["thermalDutyCycle"]
        if (
            not isinstance(thermalDutyCycle, (int, float))
            or isinstance(thermalDutyCycle, bool)
            or not np.isfinite(thermalDutyCycle)
            or not 0.0 < float(thermalDutyCycle) <= 1.0
        ):
            raise ValueError(
                "thermalDutyCycle has an invalid value. Allowed range: "
                "finite real number in (0, 1]."
            )
        thermalSteadyStateToleranceC = self.parameters[
            "thermalSteadyStateToleranceC"
        ]
        if (
            not isinstance(
                thermalSteadyStateToleranceC, (int, float)
            )
            or isinstance(thermalSteadyStateToleranceC, bool)
            or not np.isfinite(thermalSteadyStateToleranceC)
            or float(thermalSteadyStateToleranceC) <= 0.0
        ):
            raise ValueError(
                "thermalSteadyStateToleranceC has an invalid value. "
                "Allowed range: finite real number in (0, +inf) C."
            )
        maximumThermalSteadyStateIterations = self.parameters[
            "maximumThermalSteadyStateIterations"
        ]
        if (
            not isinstance(maximumThermalSteadyStateIterations, int)
            or isinstance(maximumThermalSteadyStateIterations, bool)
            or maximumThermalSteadyStateIterations < 1
        ):
            raise ValueError(
                "maximumThermalSteadyStateIterations has an invalid value. "
                "Allowed range: integer in [1, +inf)."
            )

        phaseDegrees = self.parameters["phaseDegrees"]
        if (
            not isinstance(phaseDegrees, (int, float))
            or isinstance(phaseDegrees, bool)
            or not np.isfinite(phaseDegrees)
            or float(phaseDegrees) not in (-90.0, 0.0, 90.0)
        ):
            raise ValueError(
                "phaseDegrees has an invalid value. Allowed values: "
                "-90, 0, or 90 degrees."
            )

        noiseAmpMv = self.parameters["noiseAmpMv"]
        noisePwrDbm = self.parameters["noisePwrDbm"]
        noiseSnrDb = self.parameters["noiseSnrDb"]
        configuredNoiseControls = sum(
            noiseControl is not None
            for noiseControl in (
                noiseAmpMv,
                noisePwrDbm,
                noiseSnrDb,
            )
        )
        if configuredNoiseControls > 1:
            raise ValueError(
                "noise controls have an invalid combination. Allowed "
                "combination: at most one of noiseAmpMv, noisePwrDbm, and "
                "noiseSnrDb may be non-None."
            )
        if noiseAmpMv is not None and (
            not isinstance(noiseAmpMv, (int, float))
            or isinstance(noiseAmpMv, bool)
            or not np.isfinite(noiseAmpMv)
            or float(noiseAmpMv) < 0.0
        ):
            raise ValueError(
                "noiseAmpMv has an invalid value. Allowed range: None or "
                "a finite real number in [0, +inf) mV RMS."
            )
        if noisePwrDbm is not None and (
            not isinstance(noisePwrDbm, (int, float))
            or isinstance(noisePwrDbm, bool)
            or not np.isfinite(noisePwrDbm)
        ):
            raise ValueError(
                "noisePwrDbm has an invalid value. Allowed range: None or "
                "any finite real number in (-inf, +inf) dBm."
            )
        if noiseSnrDb is not None and (
            not isinstance(noiseSnrDb, (int, float))
            or isinstance(noiseSnrDb, bool)
            or not np.isfinite(noiseSnrDb)
        ):
            raise ValueError(
                "noiseSnrDb has an invalid value. Allowed range: None or "
                "any finite real number in (-inf, +inf) dB."
            )

        for iqEnableParameterName in (
            "txIqImbalanceEnabled",
            "fbIqImbalanceEnabled",
        ):
            iqStageEnabled = self.parameters[iqEnableParameterName]
            if not isinstance(iqStageEnabled, bool):
                raise TypeError(
                    f"{iqEnableParameterName} has an invalid type. Allowed "
                    "values: True or False."
                )

        for realImpairmentParameterName in (
            "txIqGainImbalanceDb",
            "txIqPhaseImbalanceDegrees",
            "fbGainDb",
            "fbPhaseDegrees",
            "fbCarrierFrequencyOffsetHz",
            "fbIqGainImbalanceDb",
            "fbIqPhaseImbalanceDegrees",
        ):
            impairmentParameterValue = self.parameters[
                realImpairmentParameterName
            ]
            if (
                not isinstance(impairmentParameterValue, (int, float))
                or isinstance(impairmentParameterValue, bool)
                or not np.isfinite(impairmentParameterValue)
            ):
                raise ValueError(
                    f"{realImpairmentParameterName} has an invalid value. "
                    "Allowed range: any finite real number in "
                    "(-inf, +inf)."
                )
        fbSamplingFrequencyOffsetPpm = self.parameters[
            "fbSamplingFrequencyOffsetPpm"
        ]
        if (
            not isinstance(
                fbSamplingFrequencyOffsetPpm, (int, float)
            )
            or isinstance(fbSamplingFrequencyOffsetPpm, bool)
            or not np.isfinite(fbSamplingFrequencyOffsetPpm)
            or not -1.0e6
            < float(fbSamplingFrequencyOffsetPpm)
            < 1.0e6
        ):
            raise ValueError(
                "fbSamplingFrequencyOffsetPpm has an invalid value. Allowed "
                "range: finite real number in (-1000000, 1000000) ppm."
            )
        fbIntegerDelaySamples = self.parameters[
            "fbIntegerDelaySamples"
        ]
        if (
            not isinstance(fbIntegerDelaySamples, int)
            or isinstance(fbIntegerDelaySamples, bool)
            or fbIntegerDelaySamples < 0
        ):
            raise ValueError(
                "fbIntegerDelaySamples has an invalid value. Allowed range: "
                "integer in [0, +inf) samples."
            )
        fbFractionalDelaySamples = self.parameters[
            "fbFractionalDelaySamples"
        ]
        if (
            not isinstance(fbFractionalDelaySamples, (int, float))
            or isinstance(fbFractionalDelaySamples, bool)
            or not np.isfinite(fbFractionalDelaySamples)
            or not -0.5
            <= float(fbFractionalDelaySamples)
            < 0.5
        ):
            raise ValueError(
                "fbFractionalDelaySamples has an invalid value. Allowed "
                "range: finite real number in [-0.5, 0.5) samples."
            )
        fbFirTaps = self.parameters["fbFirTaps"]
        if fbFirTaps is not None:
            if isinstance(fbFirTaps, (str, bytes)):
                raise TypeError(
                    "fbFirTaps has an invalid type. Allowed values: None or "
                    "a nonempty one-dimensional sequence of finite complex "
                    "numbers."
                )
            firTapArray = np.asarray(
                fbFirTaps, dtype=np.complex128
            )
            if (
                firTapArray.ndim != 1
                or firTapArray.size == 0
                or not np.all(np.isfinite(firTapArray))
            ):
                raise ValueError(
                    "fbFirTaps has an invalid value. Allowed values: None or "
                    "a nonempty one-dimensional sequence of finite complex "
                    "numbers."
                )
        for iqFirParameterName in (
            "txIqDirectFirTaps",
            "txIqImageFirTaps",
            "fbIqDirectFirTaps",
            "fbIqImageFirTaps",
        ):
            iqFirTaps = self.parameters[iqFirParameterName]
            if iqFirTaps is None:
                continue
            if isinstance(iqFirTaps, (str, bytes)):
                raise TypeError(
                    f"{iqFirParameterName} has an invalid type. Allowed "
                    "values: None or a nonempty one-dimensional sequence of "
                    "finite complex numbers."
                )
            try:
                iqFirTapArray = np.asarray(
                    iqFirTaps, dtype=np.complex128
                )
            except (TypeError, ValueError, OverflowError) as error:
                raise TypeError(
                    f"{iqFirParameterName} has an invalid type. Allowed "
                    "values: None or a nonempty one-dimensional sequence of "
                    "finite complex numbers."
                ) from error
            if (
                iqFirTapArray.ndim != 1
                or iqFirTapArray.size == 0
                or not np.all(np.isfinite(iqFirTapArray))
            ):
                raise ValueError(
                    f"{iqFirParameterName} has an invalid value. Allowed "
                    "values: None or a nonempty one-dimensional sequence of "
                    "finite complex numbers."
                )
        for complexParameterName in (
            "txDcOffset",
            "fbDcOffset",
            "fbThirdOrderCoefficient",
        ):
            complexParameterValue = self.parameters[
                complexParameterName
            ]
            if (
                not isinstance(
                    complexParameterValue,
                    (int, float, complex),
                )
                or isinstance(complexParameterValue, bool)
            ):
                raise TypeError(
                    f"{complexParameterName} has an invalid type. Allowed "
                    "values: finite int, float, or complex scalar."
                )
            resolvedComplexValue = complex(complexParameterValue)
            if not (
                np.isfinite(resolvedComplexValue.real)
                and np.isfinite(resolvedComplexValue.imag)
            ):
                raise ValueError(
                    f"{complexParameterName} has an invalid value. Allowed "
                    "range: finite real and imaginary components."
                )
        fbClipAmplitude = self.parameters["fbClipAmplitude"]
        if fbClipAmplitude is not None and (
            not isinstance(fbClipAmplitude, (int, float))
            or isinstance(fbClipAmplitude, bool)
            or not np.isfinite(fbClipAmplitude)
            or float(fbClipAmplitude) <= 0.0
        ):
            raise ValueError(
                "fbClipAmplitude has an invalid value. Allowed range: None "
                "or a finite real number in (0, +inf)."
            )
        fbAdcWidth = self.parameters["fbAdcWidth"]
        if fbAdcWidth is not None and (
            not isinstance(fbAdcWidth, int)
            or isinstance(fbAdcWidth, bool)
            or not 2 <= fbAdcWidth <= 32
        ):
            raise ValueError(
                "fbAdcWidth has an invalid value. Allowed range: None or an "
                "integer in [2, 32] bits."
            )
        fbAdcFullScale = self.parameters["fbAdcFullScale"]
        if (
            not isinstance(fbAdcFullScale, (int, float))
            or isinstance(fbAdcFullScale, bool)
            or not np.isfinite(fbAdcFullScale)
            or float(fbAdcFullScale) <= 0.0
        ):
            raise ValueError(
                "fbAdcFullScale has an invalid value. Allowed range: finite "
                "real number in (0, +inf)."
            )
        self.ResolveCouplingPaths("prePaCouplingPaths")
        self.ResolveCouplingPaths("postPaCouplingPaths")

        loadResistanceOhm = self.parameters["loadResistanceOhm"]
        if (
            not isinstance(loadResistanceOhm, (int, float))
            or isinstance(loadResistanceOhm, bool)
            or not np.isfinite(loadResistanceOhm)
            or float(loadResistanceOhm) <= 0.0
        ):
            raise ValueError(
                "loadResistanceOhm has an invalid value. Allowed range: "
                "finite real number in (0, +inf) ohms."
            )
        maximumOutputPowerDbm = self.parameters[
            "maximumOutputPowerDbm"
        ]
        if (
            not isinstance(maximumOutputPowerDbm, (int, float))
            or isinstance(maximumOutputPowerDbm, bool)
            or not np.isfinite(maximumOutputPowerDbm)
        ):
            raise ValueError(
                "maximumOutputPowerDbm has an invalid value. Allowed range: "
                "any finite real number in (-inf, +inf) dBm."
            )

        calibrationToleranceDb = self.parameters[
            "calibrationToleranceDb"
        ]
        if (
            not isinstance(calibrationToleranceDb, (int, float))
            or isinstance(calibrationToleranceDb, bool)
            or not np.isfinite(calibrationToleranceDb)
            or float(calibrationToleranceDb) <= 0.0
        ):
            raise ValueError(
                "calibrationToleranceDb has an invalid value. Allowed range: "
                "finite real number in (0, +inf) dB."
            )
        maximumCalibrationIterations = self.parameters[
            "maximumCalibrationIterations"
        ]
        if (
            not isinstance(maximumCalibrationIterations, int)
            or isinstance(maximumCalibrationIterations, bool)
            or maximumCalibrationIterations < 1
        ):
            raise ValueError(
                "maximumCalibrationIterations has an invalid value. Allowed "
                "range: integer in [1, +inf)."
            )
        calibrationLearningRate = self.parameters[
            "calibrationLearningRate"
        ]
        if (
            not isinstance(calibrationLearningRate, (int, float))
            or isinstance(calibrationLearningRate, bool)
            or not np.isfinite(calibrationLearningRate)
            or not 0.0 < float(calibrationLearningRate) <= 1.0
        ):
            raise ValueError(
                "calibrationLearningRate has an invalid value. Allowed "
                "range: finite real number in (0, 1]."
            )
        maximumDriveAdjustmentDb = self.parameters[
            "maximumDriveAdjustmentDb"
        ]
        if (
            not isinstance(maximumDriveAdjustmentDb, (int, float))
            or isinstance(maximumDriveAdjustmentDb, bool)
            or not np.isfinite(maximumDriveAdjustmentDb)
            or float(maximumDriveAdjustmentDb) <= 0.0
        ):
            raise ValueError(
                "maximumDriveAdjustmentDb has an invalid value. Allowed "
                "range: finite real number in (0, +inf) dB."
            )
        calibrationDigitalHeadroomDb = self.parameters[
            "calibrationDigitalHeadroomDb"
        ]
        if (
            not isinstance(calibrationDigitalHeadroomDb, (int, float))
            or isinstance(calibrationDigitalHeadroomDb, bool)
            or not np.isfinite(calibrationDigitalHeadroomDb)
            or not 0.0 <= float(calibrationDigitalHeadroomDb) <= 60.0
        ):
            raise ValueError(
                "calibrationDigitalHeadroomDb has an invalid value. Allowed "
                "range: finite real number in [0, 60] dB."
            )
        jointPowerCalibration = self.parameters[
            "jointPowerCalibration"
        ]
        if (
            jointPowerCalibration is not None
            and not isinstance(jointPowerCalibration, bool)
        ):
            raise TypeError(
                "jointPowerCalibration has an invalid value. Allowed values: "
                "True, False, or None."
            )
        calibrationProbeStepDb = self.parameters[
            "calibrationProbeStepDb"
        ]
        if (
            not isinstance(calibrationProbeStepDb, (int, float))
            or isinstance(calibrationProbeStepDb, bool)
            or not np.isfinite(calibrationProbeStepDb)
            or float(calibrationProbeStepDb) <= 0.0
        ):
            raise ValueError(
                "calibrationProbeStepDb has an invalid value. Allowed range: "
                "finite real number in (0, +inf) dB."
            )
        calibrationRegularization = self.parameters[
            "calibrationRegularization"
        ]
        if (
            not isinstance(calibrationRegularization, (int, float))
            or isinstance(calibrationRegularization, bool)
            or not np.isfinite(calibrationRegularization)
            or float(calibrationRegularization) <= 0.0
        ):
            raise ValueError(
                "calibrationRegularization has an invalid value. Allowed "
                "range: finite real number in (0, +inf)."
            )
        activePowerThresholdDb = self.parameters[
            "activePowerThresholdDb"
        ]
        if (
            not isinstance(activePowerThresholdDb, (int, float))
            or isinstance(activePowerThresholdDb, bool)
            or not np.isfinite(activePowerThresholdDb)
            or float(activePowerThresholdDb) >= 0.0
        ):
            raise ValueError(
                "activePowerThresholdDb has an invalid value. Allowed range: "
                "finite real number in (-inf, 0) dB."
            )
        activeGapToleranceSamples = self.parameters[
            "activeGapToleranceSamples"
        ]
        if (
            not isinstance(activeGapToleranceSamples, int)
            or isinstance(activeGapToleranceSamples, bool)
            or activeGapToleranceSamples < 0
        ):
            raise ValueError(
                "activeGapToleranceSamples has an invalid value. Allowed "
                "range: integer in [0, +inf) samples."
            )

        randomSeed = self.parameters["randomSeed"]
        if randomSeed is not None and (
            not isinstance(randomSeed, int)
            or isinstance(randomSeed, bool)
            or int(randomSeed) < 0
        ):
            raise ValueError(
                "randomSeed has an invalid value. Allowed range: None or an "
                "integer in [0, +inf)."
            )
        width = self.parameters["width"]
        if (
            not isinstance(width, int)
            or isinstance(width, bool)
            or not 0 <= width <= 53
        ):
            raise ValueError(
                "width has an invalid value. Allowed range: integer in "
                "[0, 53] bits, where 0 selects floating-point mode."
            )
        FixedPoint(width)

    def SetPaModel(self, paModel: Any) -> None:
        """Bind the PA evaluated before the channel impairments.

        Processing details:
            Algorithm: Require a callable public ``Process`` entry, retain
            the PA object and bound method, and allow ``ProcessFloating`` to
            be selected dynamically when the PA provides that efficient
            normalized-domain path.

        Args:
            paModel: PA-like object exposing a callable ``Process`` method.

        Returns:
            result: None. Later ``Process`` calls use the new PA.
        """

        paProcessMethod = getattr(paModel, "Process", None)
        if not callable(paProcessMethod):
            raise TypeError("paModel must expose a callable Process method")
        self.paModel = paModel
        self._paProcessMethod = paProcessMethod
        self._powerCalibration = None
        self._calibrationDriveDbPerChain = tuple()
        self._lastTransmitterOutput = None
        self._lastActualPaInput = None
        self._lastCalibrationOutputPowerDbm = None
        self.ResetFeedbackIqCalibration()

    @staticmethod
    def ResolveCalibrationTargets(
        outputPowerDbm: Union[float, Sequence[float], np.ndarray],
    ) -> Tuple[float, Optional[Tuple[float, ...]]]:
        """Normalize one SISO target or one target per MIMO PA chain.

        Processing details:
            Algorithm: Treat a finite real scalar as one shared output-power
            target. Treat a nonempty one-dimensional sequence as independent
            per-chain targets, convert every element to a plain float, and
            reject booleans, strings, nested arrays, or nonfinite values.

        Args:
            outputPowerDbm: Shared target dBm or a sequence ordered by chain.

        Returns:
            result: Scalar fallback target and optional per-chain target tuple.
        """

        if (
            isinstance(outputPowerDbm, (int, float, np.integer, np.floating))
            and not isinstance(outputPowerDbm, (bool, np.bool_))
        ):
            scalarTarget = float(outputPowerDbm)
            if not np.isfinite(scalarTarget):
                raise ValueError(
                    "outputPowerDbm has an invalid value. Allowed range: "
                    "any finite real scalar in (-inf, +inf) dBm, or a "
                    "nonempty one-dimensional sequence of such values."
                )
            return scalarTarget, None
        if isinstance(outputPowerDbm, (str, bytes)):
            raise TypeError(
                "outputPowerDbm has an invalid type. Allowed values: any "
                "finite real scalar, or a nonempty one-dimensional sequence "
                "of finite real values in dBm."
            )
        targetArray = np.asarray(outputPowerDbm, dtype=object)
        if targetArray.ndim != 1 or targetArray.size == 0:
            raise ValueError(
                "outputPowerDbm has an invalid sequence shape. Allowed "
                "values: a finite real scalar, or a nonempty "
                "one-dimensional sequence of finite real values in dBm."
            )
        targetValues = []
        for targetValue in targetArray:
            if (
                not isinstance(
                    targetValue,
                    (int, float, np.integer, np.floating),
                )
                or isinstance(targetValue, (bool, np.bool_))
            ):
                raise TypeError(
                    "outputPowerDbm contains an invalid element type. Every "
                    "element must be a finite real number in (-inf, +inf) "
                    "dBm."
                )
            floatingTarget = float(targetValue)
            if not np.isfinite(floatingTarget):
                raise ValueError(
                    "outputPowerDbm contains an invalid element value. Every "
                    "element must be a finite real number in (-inf, +inf) "
                    "dBm."
                )
            targetValues.append(floatingTarget)
        targetTuple = tuple(targetValues)
        return targetTuple[0], targetTuple

    def ConfigurePowerCalibration(
        self,
        outputPowerDbm: Union[float, Sequence[float], np.ndarray],
    ) -> PowerCalibration:
        """Prepare the hidden PA-input calibration loop for one request.

        Processing details:
            Algorithm: Resolve scalar or per-chain targets, copy the current
            Channel power detector, digital-headroom and convergence settings
            into a private ``PowerCalibration`` instance, bind the Channel's
            clean calibration-drive protocol, and reuse the helper so its
            converged total drive can accelerate later requests. Rebuild it
            only after a PA replacement or public-width change.

        Args:
            outputPowerDbm: Shared target dBm or one target for every PA chain.

        Returns:
            result: Configured private calibration helper bound to this PA.
        """

        if self.paModel is None or self._paProcessMethod is None:
            raise RuntimeError(
                "power calibration requires a PA bound through paModel "
                "or SetPaModel"
            )
        self.ValidateParameters()
        self.ValidateThermalReferencePlanes()
        scalarTarget, perChainTargets = self.ResolveCalibrationTargets(
            outputPowerDbm
        )
        calibrationParameters = {
            "loadResistanceOhm": self.parameters["loadResistanceOhm"],
            "maximumOutputPowerDbm": self.parameters[
                "maximumOutputPowerDbm"
            ],
            "outputPowerDbm": scalarTarget,
            "outputPowerDbmPerChain": perChainTargets,
            "calibrationToleranceDb": self.parameters[
                "calibrationToleranceDb"
            ],
            "maximumCalibrationIterations": self.parameters[
                "maximumCalibrationIterations"
            ],
            "calibrationLearningRate": self.parameters[
                "calibrationLearningRate"
            ],
            "maximumDriveAdjustmentDb": self.parameters[
                "maximumDriveAdjustmentDb"
            ],
            "calibrationDigitalHeadroomDb": self.parameters[
                "calibrationDigitalHeadroomDb"
            ],
            "enableJointCalibration": (
                self.HasPrePaCoupling()
                if self.parameters["jointPowerCalibration"] is None
                else bool(
                    self.parameters["jointPowerCalibration"]
                )
            ),
            "calibrationProbeStepDb": self.parameters[
                "calibrationProbeStepDb"
            ],
            "calibrationRegularization": self.parameters[
                "calibrationRegularization"
            ],
            "activePowerThresholdDb": self.parameters[
                "activePowerThresholdDb"
            ],
            "activeGapToleranceSamples": self.parameters[
                "activeGapToleranceSamples"
            ],
            "width": self.width,
        }
        if (
            self._powerCalibration is None
            or self._powerCalibration.width != self.width
        ):
            self._powerCalibration = PowerCalibration(
                paModel=self,
                parameters=calibrationParameters,
            )
        else:
            self._powerCalibration.UpdateParameters(
                **calibrationParameters
            )
        return self._powerCalibration

    def CalibratePaInput(
        self,
        inputSignal: np.ndarray,
        outputPowerDbm: Union[float, Sequence[float], np.ndarray],
    ) -> np.ndarray:
        """Generate a public Tx waveform and hidden drive for requested dBm.

        Processing details:
            Algorithm: Configure the private closed loop and normalize only the
            active part of the caller's arbitrary waveform. In fixed mode,
            retain legal public DAC codes with digital headroom and repeatedly
            vary a post-decode analog drive before Tx I/Q impairment, pre-PA
            coupling, and the bound PA. Measure actual active PA output power
            until every chain enters tolerance, committing the hidden analog
            drive only on success. Suspend and snapshot any active PA thermal
            network before the trials and restore exact temperature and elapsed
            time in a finally block. Padding and long idle intervals remain
            outside the active-region detector.

        Args:
            inputSignal: Arbitrarily scaled public SISO or MIMO waveform.
            outputPowerDbm: Shared target dBm or one target per PA chain.

        Returns:
            result: Public waveform before the committed analog drive stage.
        """

        self.ValidateParameters()
        # Direct CalibratePaInput and PrepareThermalTest callers must receive
        # the same pre-transaction reference-plane checks as Process callers.
        self.ValidateThermalReferencePlanes()
        powerCalibration = self.ConfigurePowerCalibration(outputPowerDbm)
        # PowerCalibration owns the common thermal transaction so the same
        # reference-temperature behavior also applies to direct calibrator use.
        calibratedInput = powerCalibration.Calibrate(inputSignal)
        scalarTarget, perChainTargets = self.ResolveCalibrationTargets(
            outputPowerDbm
        )
        self._lastCalibrationOutputPowerDbm = (
            scalarTarget if perChainTargets is None else perChainTargets
        )
        return calibratedInput

    def PrepareThermalTest(
        self,
        inputSignal: np.ndarray,
        calibrationOutputPowerDbm: Union[
            float, Sequence[float], np.ndarray
        ],
        initialJunctionTemperatureC: Optional[
            Union[float, Sequence[float]]
        ] = None,
        ambientTemperatureC: Optional[float] = None,
    ) -> np.ndarray:
        """Calibrate once without heating and freeze one Channel drive state.

        Processing details:
            Algorithm: Suspend the bound SISO or MIMO PA thermal network,
            execute the existing dBm closed loop using only reference
            electrical parameters, retain the converged analog drive inside
            this Channel and copy its paired public codes before Tx I/Q,
            restore thermal modeling without accepting calibration heat,
            optionally reset the requested starting temperatures, and return
            the codes. A later Process call on this instance reapplies both the
            committed analog drive and Tx I/Q stage.

        Args:
            inputSignal: Arbitrarily scaled public SISO or MIMO waveform.
            calibrationOutputPowerDbm: Reference-temperature target dBm value.
            initialJunctionTemperatureC: Optional scalar or per-chain test start.
            ambientTemperatureC: Optional shared ambient test temperature.

        Returns:
            result: Frozen public codes for this Channel's open-loop tests.
        """

        if self.paModel is None:
            raise RuntimeError("PrepareThermalTest requires a bound PA model")
        self.CalibratePaInput(
            inputSignal,
            calibrationOutputPowerDbm,
        )
        calibratedInput = np.array(
            self.GetLastPaInput(), dtype=np.complex128, copy=True
        )
        if (
            initialJunctionTemperatureC is not None
            or ambientTemperatureC is not None
        ):
            resetMethod = getattr(self.paModel, "ResetThermalState", None)
            if not callable(resetMethod):
                raise RuntimeError(
                    "bound PA does not support thermal-state reset"
                )
            resetMethod(
                initialJunctionTemperatureC,
                ambientTemperatureC,
            )
        return calibratedInput

    def AdvanceThermalIdle(
        self,
        idleTimeSec: float,
    ) -> object:
        """Advance the bound PA through a physical idle interval.

        Processing details:
            Algorithm: Delegate the nonnegative gap to the PA thermal model so
            idle bias power and cooling evolve without emitting an RF waveform.

        Args:
            idleTimeSec: Physical frame-to-frame idle interval in seconds.

        Returns:
            result: Junction temperature or per-chain temperature tuple.
        """

        if self.paModel is None:
            raise RuntimeError("AdvanceThermalIdle requires a bound PA model")
        advanceMethod = getattr(self.paModel, "AdvanceIdle", None)
        if not callable(advanceMethod):
            raise RuntimeError("bound PA does not support thermal idle advance")
        return advanceMethod(idleTimeSec)

    def SuspendThermalModel(self) -> object:
        """Suspend the bound PA thermal state during power calibration.

        Processing details:
            Algorithm: Validate the wrapped PA's paired transaction protocol
            before changing state, then delegate snapshot creation. A PA with
            no thermal protocol returns None and remains electrically usable.

        Returns:
            result: Opaque bound-PA thermal snapshot or None.
        """

        if self.paModel is None:
            return None
        suspendMethod = getattr(
            self.paModel, "SuspendThermalModel", None
        )
        restoreMethod = getattr(
            self.paModel, "RestoreThermalModel", None
        )
        if callable(suspendMethod) != callable(restoreMethod):
            raise TypeError(
                "a bound PA thermal transaction must expose both "
                "SuspendThermalModel and RestoreThermalModel, or neither"
            )
        return suspendMethod() if callable(suspendMethod) else None

    def RestoreThermalModel(self, thermalSnapshot: object) -> None:
        """Restore the bound PA state after electrical power calibration.

        Processing details:
            Algorithm: Treat a None snapshot as a no-op and otherwise forward
            the opaque snapshot to the paired bound-PA restore method. The PA
            remains responsible for honoring a live ``enabled=False`` setting
            instead of reviving an obsolete enabled snapshot.

        Args:
            thermalSnapshot: Value returned by ``SuspendThermalModel``.

        Returns:
            result: None. Bound thermal state is restored when still enabled.
        """

        if thermalSnapshot is None:
            return
        if self.paModel is None:
            raise RuntimeError(
                "cannot restore thermal state without a bound PA model"
            )
        restoreMethod = getattr(
            self.paModel, "RestoreThermalModel", None
        )
        if not callable(restoreMethod):
            raise TypeError(
                "bound PA does not expose RestoreThermalModel"
            )
        restoreMethod(thermalSnapshot)

    def GetThermalMetrics(self) -> Dict[str, object]:
        """Return thermal diagnostics from the bound PA without reprocessing.

        Processing details:
            Algorithm: Delegate to the PA thermal observer and copy the result
            so callers can log temperature, heat, duty cycle, and elapsed time.

        Returns:
            result: Ordinary diagnostic dictionary for SISO or MIMO operation.
        """

        if self.paModel is None:
            raise RuntimeError("GetThermalMetrics requires a bound PA model")
        metricsMethod = getattr(self.paModel, "GetThermalMetrics", None)
        if not callable(metricsMethod):
            return {"enabled": False}
        return dict(metricsMethod())

    def IsThermalModelEnabled(self) -> bool:
        """Report whether at least one bound PA has active thermal state.

        Processing details:
            Algorithm: Inspect the bound PA's read-only thermal metrics. A
            SISO dictionary uses its top-level ``enabled`` flag, while a MIMO
            dictionary is active when any chain reports enabled operation.

        Returns:
            result: True when periodic thermal scheduling is required.
        """

        if self.paModel is None:
            return False
        metricsMethod = getattr(self.paModel, "GetThermalMetrics", None)
        if not callable(metricsMethod):
            return False
        thermalMetrics = metricsMethod()
        if not isinstance(thermalMetrics, Mapping):
            raise TypeError(
                "bound PA GetThermalMetrics must return a mapping"
            )
        if "enabled" in thermalMetrics:
            return bool(thermalMetrics["enabled"])
        chainMetrics = thermalMetrics.get("chains", tuple())
        if isinstance(chainMetrics, Sequence) and not isinstance(
            chainMetrics, (str, bytes)
        ):
            return any(
                isinstance(chainMetric, Mapping)
                and bool(chainMetric.get("enabled", False))
                for chainMetric in chainMetrics
            )
        return False

    def ValidateThermalReferencePlanes(
        self,
    ) -> Tuple[Mapping[str, object], ...]:
        """Validate shared thermal time, power, and activity reference planes.

        Processing details:
            Algorithm: Collect every enabled SISO or MIMO PA thermal metrics
            mapping and require its sample rate, normalized-output dBm scale,
            and RF-active threshold to equal the corresponding Channel value.
            One data window cannot represent different physical durations or
            different normalized-power meanings across connected modules.

        Returns:
            result: Immutable tuple of enabled per-chain thermal mappings.
        """

        if self.paModel is None:
            return tuple()
        thermalMetrics = self.GetThermalMetrics()
        if "enabled" in thermalMetrics:
            enabledChainMetrics = (
                (thermalMetrics,) if thermalMetrics["enabled"] else tuple()
            )
        else:
            rawChainMetrics = thermalMetrics.get("chains", tuple())
            if not isinstance(rawChainMetrics, Sequence) or isinstance(
                rawChainMetrics, (str, bytes)
            ):
                raise TypeError(
                    "MIMO thermal metrics must expose a chain sequence"
                )
            enabledChainMetrics = tuple(
                chainMetric
                for chainMetric in rawChainMetrics
                if isinstance(chainMetric, Mapping)
                and bool(chainMetric.get("enabled", False))
            )
        expectedValues = (
            (
                "sampleRateHz",
                float(self.parameters["sampleRateHz"]),
                "Hz",
            ),
            (
                "referenceOutputPowerDbm",
                float(self.parameters["maximumOutputPowerDbm"]),
                "dBm",
            ),
            (
                "activePowerThresholdDb",
                float(self.parameters["activePowerThresholdDb"]),
                "dB",
            ),
        )
        for chainIndex, chainMetric in enumerate(enabledChainMetrics):
            for metricName, channelValue, unitName in expectedValues:
                if metricName not in chainMetric:
                    raise TypeError(
                        "enabled thermal PA metrics must expose "
                        f"{metricName}"
                    )
                paValue = float(chainMetric[metricName])
                if not np.isclose(
                    paValue,
                    channelValue,
                    rtol=1.0e-12,
                    atol=1.0e-12,
                ):
                    raise ValueError(
                        f"Channel {metricName} must equal enabled PA chain "
                        f"{chainIndex} ThermalConfig.{metricName}; received "
                        f"{channelValue:.12g} {unitName} and "
                        f"{paValue:.12g} {unitName}"
                    )
        return tuple(enabledChainMetrics)

    def GetActualDutyCycle(
        self,
        inputSignal: Optional[np.ndarray] = None,
    ) -> Union[float, Tuple[float, ...]]:
        """Return measured RF-active duty for a full scheduled period.

        Processing details:
            Algorithm: Without an input, read the accepted value from the most
            recent thermal period. With an input, cross the public fixed-point
            boundary, apply committed analog drive, Tx I/Q impairment, and
            pre-PA coupling, then ask the built-in PA bank to classify activity
            at the actual PA-input reference plane. SISO returns a scalar and
            MIMO returns one value per column.

        Args:
            inputSignal: Optional raw public data-window vector or matrix.

        Returns:
            result: Actual RF-active fraction of the complete period.
        """

        if self.paModel is None:
            raise RuntimeError("GetActualDutyCycle requires a bound PA model")
        self.ValidateParameters()
        self.ValidateThermalReferencePlanes()
        if inputSignal is None:
            thermalMetrics = self.GetThermalMetrics()
            if "actualDutyCycle" in thermalMetrics:
                return float(thermalMetrics["actualDutyCycle"])
            chainMetrics = thermalMetrics.get("chains", tuple())
            if isinstance(chainMetrics, Sequence) and not isinstance(
                chainMetrics, (str, bytes)
            ):
                dutyValues = tuple(
                    float(chainMetric.get("actualDutyCycle", 0.0))
                    for chainMetric in chainMetrics
                    if isinstance(chainMetric, Mapping)
                )
                if dutyValues:
                    return dutyValues
            raise RuntimeError(
                "Process must complete one thermal period before the "
                "no-argument duty-cycle query"
            )
        interfaceFormat = FixedPoint(self.width)
        normalizedInput = interfaceFormat.DecodeComplex(
            self.ValidateSignal(inputSignal, "inputSignal")
        )
        drivenInput = self.ApplyCalibrationDrive(normalizedInput)
        transmitterOutput = self.ApplyTransmitterIqImbalance(drivenInput)
        actualPaInput = self.ApplyPrePaCoupling(transmitterOutput)
        if not self.IsThermalModelEnabled():
            # Duty-cycle observation is useful before thermal modeling is
            # enabled. Classify the actual PA-input columns with the Channel's
            # global activity threshold instead of reporting a false zero.
            inputMatrix = (
                actualPaInput.reshape(-1, 1)
                if actualPaInput.ndim == 1
                else actualPaInput
            )
            thresholdScale = 10.0 ** (
                float(self.parameters["activePowerThresholdDb"]) / 10.0
            )
            dutyValues = []
            for chainIndex in range(inputMatrix.shape[1]):
                chainPower = np.abs(inputMatrix[:, chainIndex]) ** 2
                peakPower = float(np.max(chainPower))
                waveformDutyCycle = (
                    0.0
                    if peakPower <= np.finfo(float).tiny
                    else float(
                        np.mean(chainPower >= peakPower * thresholdScale)
                    )
                )
                dutyValues.append(
                    float(self.parameters["thermalDutyCycle"])
                    * waveformDutyCycle
                )
            if actualPaInput.ndim == 1:
                return dutyValues[0]
            return tuple(dutyValues)
        dutyMethod = getattr(
            self.paModel, "CalculateActualDutyCycle", None
        )
        if not callable(dutyMethod):
            raise TypeError(
                "bound PA must expose CalculateActualDutyCycle for a "
                "pre-processing duty query"
            )
        dutyResult = dutyMethod(
            actualPaInput,
            float(self.parameters["thermalDutyCycle"]),
        )
        if isinstance(dutyResult, (int, float, np.integer, np.floating)):
            return float(dutyResult)
        dutyValues = tuple(float(value) for value in dutyResult)
        if actualPaInput.ndim == 1 and len(dutyValues) == 1:
            return dutyValues[0]
        return dutyValues

    def GetLastPaInput(self) -> np.ndarray:
        """Return the latest calibrated digital input before Tx I/Q error.

        Processing details:
            Algorithm: Delegate to the private calibration helper and return
            its defensive copy. For compatibility the historical method name
            is retained, but the returned waveform is now explicitly the
            digital Channel input before the Tx I/Q and coupling modules.

        Returns:
            result: Last converged digital transmitter-input waveform.
        """

        if self._powerCalibration is None:
            raise RuntimeError(
                "calibrated Process must run before GetLastPaInput"
            )
        return self._powerCalibration.GetLastPaInput()

    def GetLastTransmitterOutput(self) -> np.ndarray:
        """Return the latest waveform after Tx I/Q but before PA coupling.

        Processing details:
            Algorithm: Return a defensive copy of the waveform cached whenever
            the normal or calibration path evaluates the transmitter I/Q stage.

        Returns:
            result: Tx modulator output before ``prePaCouplingPaths``.
        """

        if self._lastTransmitterOutput is None:
            raise RuntimeError(
                "Process must run before GetLastTransmitterOutput"
            )
        return self._lastTransmitterOutput.copy()

    def GetLastActualPaInput(self) -> np.ndarray:
        """Return the latest physical PA input after Tx I/Q and coupling.

        Processing details:
            Algorithm: Return a defensive copy cached after transmitter I/Q
            impairment and all configured pre-PA inter-chain coupling paths.

        Returns:
            result: Actual waveform matrix or vector presented to the PA bank.
        """

        if self._lastActualPaInput is None:
            raise RuntimeError(
                "Process must run before GetLastActualPaInput"
            )
        return self._lastActualPaInput.copy()

    def GetLastPaOutput(self) -> np.ndarray:
        """Return the reference-temperature PA output from calibration.

        Processing details:
            Algorithm: Return the cached converged PA observation measured
            while thermal effects were suspended, before phase rotation and
            receiver noise. A temperature-aware ``Process`` call subsequently
            evaluates the accepted drive once more after thermal restoration,
            so this diagnostic is intentionally not that live PA waveform.

        Returns:
            result: Last clean reference-calibration PA output waveform.
        """

        if self._powerCalibration is None:
            raise RuntimeError(
                "calibrated Process must run before GetLastPaOutput"
            )
        return self._powerCalibration.GetLastPaOutput()

    def GetLastCalibrationMetrics(self) -> Dict[str, object]:
        """Return reference-calibration power, error, and iteration data.

        Processing details:
            Algorithm: Delegate to the private calibration helper while
            preserving its ordinary dictionary result and hidden drive state.
            Thermal live-output power remains available through
            ``GetThermalMetrics`` instead of overwriting this reference result.

        Returns:
            result: Dictionary describing the converged PA power loop.
        """

        if self._powerCalibration is None:
            raise RuntimeError(
                "calibrated Process must run before "
                "GetLastCalibrationMetrics"
            )
        return self._powerCalibration.GetLastCalibrationMetrics()

    def FeedbackIqCalibrationSignature(self) -> Tuple[object, ...]:
        """Return the configuration identity of one valid FB I/Q inverse.

        Processing details:
            Algorithm: Normalize every feedback-path value that can change the
            phase-pair separation or single-capture inverse into immutable
            scalar tuples. The compensation-mode selector is intentionally
            excluded so a filter learned in ``phase_pair`` mode remains usable
            after an explicit switch to ``filter`` mode.

        Returns:
            result: Immutable signature used to reject stale cached filters.
        """

        self.ValidateParameters()
        phaseResponses = tuple(
            complex(responseValue)
            for responseValue in cast(
                Sequence[complex],
                self.parameters["fbPhasePairResponses"],
            )
        )
        feedbackFirTaps = tuple(
            complex(tapValue) for tapValue in self.ResolveFeedbackFirTaps()
        )
        feedbackIqDirectFirTaps, feedbackIqImageFirTaps = (
            self.FeedbackIqFilterTaps()
        )
        return (
            id(self.paModel),
            float(self.parameters["sampleRateHz"]),
            float(self.parameters["phaseDegrees"]),
            float(self.parameters["fbGainDb"]),
            float(self.parameters["fbPhaseDegrees"]),
            feedbackFirTaps,
            int(self.parameters["fbIntegerDelaySamples"]),
            float(self.parameters["fbFractionalDelaySamples"]),
            float(self.parameters["fbCarrierFrequencyOffsetHz"]),
            float(self.parameters["fbSamplingFrequencyOffsetPpm"]),
            bool(self.parameters["fbIqImbalanceEnabled"]),
            float(self.parameters["fbIqGainImbalanceDb"]),
            float(self.parameters["fbIqPhaseImbalanceDegrees"]),
            tuple(complex(tapValue) for tapValue in feedbackIqDirectFirTaps),
            tuple(complex(tapValue) for tapValue in feedbackIqImageFirTaps),
            complex(self.parameters["fbDcOffset"]),
            phaseResponses,
            int(self.parameters["fbIqCompensationFilterLength"]),
            float(self.parameters["fbIqCompensationRegularization"]),
            complex(self.parameters["fbThirdOrderCoefficient"]),
            (
                None
                if self.parameters["fbClipAmplitude"] is None
                else float(self.parameters["fbClipAmplitude"])
            ),
            self.parameters["fbAdcWidth"],
            float(self.parameters["fbAdcFullScale"]),
            int(self.parameters["width"]),
        )

    def ResetFeedbackIqCalibration(self) -> None:
        """Invalidate every cached FB phase-pair calibration artifact.

        Processing details:
            Algorithm: Discard the calibration object, its configuration
            signature, and the latest raw pair together so a PA replacement or
            feedback-path update cannot silently reuse stale coefficients.

        Returns:
            result: None. ``filter`` mode requires a new successful phase-pair
                acquisition before it can process another signal.
        """

        self._feedbackIqCalibration = None
        self._feedbackIqCalibrationSignature = None
        self._lastFeedbackPhasePair = None

    def ConfigureFeedbackIqCalibration(self) -> FeedbackIqCalibration:
        """Create or synchronize the normalized-domain FB I/Q calibrator.

        Processing details:
            Algorithm: Translate Channel parameter names into the reusable
            signal-processing class vocabulary, always select width zero because
            the Channel has already decoded its public boundary, and rebuild the
            helper only when its own resolved configuration differs. Rebuilding
            deliberately invalidates any old filter.

        Returns:
            result: Configured floating-domain calibration object.
        """

        self.ValidateParameters()
        desiredParameters: Dict[str, object] = {
            "phaseResponses": tuple(
                complex(responseValue)
                for responseValue in cast(
                    Sequence[complex],
                    self.parameters["fbPhasePairResponses"],
                )
            ),
            "commonDcOffset": (
                complex(self.parameters["fbDcOffset"])
                if bool(self.parameters["fbIqImbalanceEnabled"])
                else 0.0 + 0.0j
            ),
            "filterLength": int(
                self.parameters["fbIqCompensationFilterLength"]
            ),
            "regularization": float(
                self.parameters["fbIqCompensationRegularization"]
            ),
            "width": 0,
        }
        if self._feedbackIqCalibration is None:
            self._feedbackIqCalibration = FeedbackIqCalibration(
                parameters=desiredParameters
            )
        elif self._feedbackIqCalibration.GetParameters() != desiredParameters:
            self._feedbackIqCalibration = FeedbackIqCalibration(
                parameters=desiredParameters
            )
            self._feedbackIqCalibrationSignature = None
        return self._feedbackIqCalibration

    def RequireCurrentFeedbackIqCalibration(
        self,
    ) -> FeedbackIqCalibration:
        """Return a valid cached inverse or reject stale filter operation.

        Processing details:
            Algorithm: Compare the live PA/feedback configuration signature with
            the signature stored after the latest successful phase-pair fit.
            Any mismatch atomically clears the cache and raises an actionable
            error instead of applying coefficients at the wrong reference plane.

        Returns:
            result: Current calibrated floating-domain compensation object.
        """

        currentSignature = self.FeedbackIqCalibrationSignature()
        if (
            self._feedbackIqCalibration is None
            or self._feedbackIqCalibrationSignature is None
        ):
            raise RuntimeError(
                "fbIqCompensationMode='filter' requires a valid feedback I/Q "
                "calibration. Run Channel.Process once with sampleMode='fb' "
                "and fbIqCompensationMode='phase_pair', then switch only the "
                "compensation mode to 'filter'."
            )
        if self._feedbackIqCalibrationSignature != currentSignature:
            self.ResetFeedbackIqCalibration()
            raise RuntimeError(
                "the cached feedback I/Q compensation filter is stale because "
                "the PA, feedback path, phase responses, filter controls, or "
                "public width changed. Re-run phase_pair calibration before "
                "using fbIqCompensationMode='filter'."
            )
        return self._feedbackIqCalibration

    def GetLastFeedbackPhasePair(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return defensive public copies of the latest raw phase captures.

        Processing details:
            Algorithm: Require a completed phase-pair Channel pass, encode both
            internally normalized captures through the current public
            FixedPoint boundary, and copy each result independently.

        Returns:
            result: Raw nominal-zero and nominal-ninety feedback arrays in the
                Channel's configured floating or fixed integer-code convention.
        """

        if self._lastFeedbackPhasePair is None:
            raise RuntimeError(
                "phase_pair processing must complete before "
                "GetLastFeedbackPhasePair"
            )
        self.RequireCurrentFeedbackIqCalibration()
        interfaceFormat = FixedPoint(self.width)
        return tuple(
            interfaceFormat.EncodeComplex(phaseSignal).copy()
            for phaseSignal in self._lastFeedbackPhasePair
        )

    def GetFeedbackIqCalibrationMetrics(self) -> Dict[str, object]:
        """Return defensive metrics for the latest current phase-pair fit.

        Processing details:
            Algorithm: Require a non-stale configuration signature, delegate to
            the reusable calibrator's defensive metric getter, and copy the
            result once more at the Channel boundary.

        Returns:
            result: Independent dictionary containing phase separation, image
                ratio, FIR fit NMSE, and conditioning diagnostics.
        """

        calibration = self.RequireCurrentFeedbackIqCalibration()
        return dict(calibration.GetCalibrationMetrics())

    def SynchronizeRandomGenerator(
        self, forceReset: bool = False
    ) -> None:
        """Synchronize random state with a live ``randomSeed`` setting.

        Processing details:
            Algorithm: Rebuild the NumPy generator only when explicitly
            requested or when a caller-owned parameter mapping changes its
            seed. A None seed requests NumPy entropy; an integer gives a
            repeatable noise sequence.

        Args:
            forceReset: True to restart the configured random sequence now.

        Returns:
            result: None. Subsequent noise samples use synchronized state.
        """

        randomSeedValue = self.parameters["randomSeed"]
        resolvedSeed = (
            None if randomSeedValue is None else int(randomSeedValue)
        )
        if forceReset or resolvedSeed != self._activeRandomSeed:
            self._randomGenerator = np.random.default_rng(resolvedSeed)
            self._activeRandomSeed = resolvedSeed

    def ResetRandomGenerator(self) -> None:
        """Restart the configured white-noise sequence.

        Processing details:
            Algorithm: Validate the current live mapping and force a new
            NumPy generator initialized from the currently resolved seed.

        Returns:
            result: None. A fixed seed reproduces prior channel noise exactly.
        """

        self.ValidateParameters()
        self.SynchronizeRandomGenerator(forceReset=True)

    @staticmethod
    def ValidateSignal(
        inputSignal: np.ndarray, signalName: str
    ) -> np.ndarray:
        """Convert and validate one complex vector or matrix.

        Processing details:
            Algorithm: Convert without flattening so SISO vectors and MIMO
            sample-by-chain matrices retain shape, then reject empty,
            unsupported-dimensional, NaN, or infinite inputs.

        Args:
            inputSignal: Public or normalized complex samples.
            signalName: Human-readable name used in validation messages.

        Returns:
            result: Finite complex128 vector or matrix of unchanged shape.
        """

        complexSignal = np.asarray(inputSignal, dtype=np.complex128)
        if complexSignal.ndim not in (1, 2):
            raise ValueError(f"{signalName} must be a vector or matrix")
        if complexSignal.size == 0:
            raise ValueError(f"{signalName} cannot be empty")
        if not np.all(np.isfinite(complexSignal)):
            raise ValueError(
                f"{signalName} contains NaN or infinite values"
            )
        return complexSignal

    def ResolveCouplingPaths(
        self,
        parameterName: str,
        chainCount: Optional[int] = None,
    ) -> Tuple[Dict[str, object], ...]:
        """Validate and canonicalize PA-side cross-coupling path mappings.

        Processing details:
            Algorithm: Accept only the pre- or post-PA path parameter, replace
            None with no coupling, reject unknown nested keys, validate
            source/destination indices, gain, phase, causal integer delay,
            bounded fractional delay, and an optional finite complex FIR,
            then return defensive ordinary dictionaries. The direct path
            remains an implicit identity and every configured path is added
            to its destination.

        Args:
            parameterName: ``prePaCouplingPaths`` or ``postPaCouplingPaths``.
            chainCount: Optional matrix column count for index-range checks.

        Returns:
            result: Immutable-order tuple of canonical coupling path mappings.
        """

        if parameterName not in (
            "prePaCouplingPaths",
            "postPaCouplingPaths",
        ):
            raise ValueError(
                "parameterName has an invalid value. Allowed values: "
                "'prePaCouplingPaths' or 'postPaCouplingPaths'."
            )
        rawPaths = self.parameters[parameterName]
        if rawPaths is None:
            return tuple()
        if isinstance(rawPaths, (str, bytes)) or not isinstance(
            rawPaths, Sequence
        ):
            raise TypeError(
                f"{parameterName} has an invalid type. Allowed values: None "
                "or a sequence of coupling-path mappings."
            )
        pathDefaults: Mapping[str, object] = MappingProxyType(
            {
                "sourceChain": 0,
                "destinationChain": 1,
                "gainDb": -30.0,
                "phaseDegrees": 0.0,
                "integerDelaySamples": 0,
                "fractionalDelaySamples": 0.0,
                "firTaps": None,
            }
        )
        resolvedPaths = []
        for pathIndex, rawPath in enumerate(rawPaths):
            if not isinstance(rawPath, Mapping):
                raise TypeError(
                    f"{parameterName}[{pathIndex}] has an invalid type. "
                    "Allowed value: a mapping containing supported path "
                    "fields."
                )
            unknownPathNames = FindUnknownParameterNames(
                rawPath,
                pathDefaults,
            )
            if unknownPathNames:
                raise TypeError(
                    self.FormatUnknownParameterError(
                        f"Channel.{parameterName}[{pathIndex}]",
                        unknownPathNames,
                        tuple(pathDefaults),
                    )
                )
            recognizedPath = {
                str(pathName): pathValue
                for pathName, pathValue in rawPath.items()
            }
            resolvedPath = {
                **pathDefaults,
                **recognizedPath,
            }
            sourceChain = resolvedPath["sourceChain"]
            destinationChain = resolvedPath["destinationChain"]
            for indexName, indexValue in (
                ("sourceChain", sourceChain),
                ("destinationChain", destinationChain),
            ):
                if (
                    not isinstance(indexValue, (int, np.integer))
                    or isinstance(indexValue, (bool, np.bool_))
                    or int(indexValue) < 0
                ):
                    raise ValueError(
                        f"{parameterName}[{pathIndex}].{indexName} "
                        "has an invalid value. Allowed range: integer in "
                        "[0, +inf)."
                    )
                if (
                    chainCount is not None
                    and int(indexValue) >= chainCount
                ):
                    raise IndexError(
                        f"{parameterName}[{pathIndex}].{indexName} "
                        "is outside the waveform chain range. Allowed range: "
                        f"integer in [0, {chainCount - 1}]."
                    )
            if int(sourceChain) == int(destinationChain):
                raise ValueError(
                    f"{parameterName}[{pathIndex}] has identical source and "
                    "destination chains. Allowed values: two different "
                    "nonnegative chain indices; direct paths are implicit "
                    "identities."
                )
            for scalarName in ("gainDb", "phaseDegrees"):
                scalarValue = resolvedPath[scalarName]
                if (
                    not isinstance(scalarValue, (int, float))
                    or isinstance(scalarValue, bool)
                    or not np.isfinite(scalarValue)
                ):
                    raise ValueError(
                        f"{parameterName}[{pathIndex}].{scalarName} "
                        "has an invalid value. Allowed range: any finite real "
                        "number in (-inf, +inf)."
                    )
            integerDelay = resolvedPath["integerDelaySamples"]
            if (
                not isinstance(integerDelay, (int, np.integer))
                or isinstance(integerDelay, (bool, np.bool_))
                or int(integerDelay) < 0
            ):
                raise ValueError(
                    f"{parameterName}[{pathIndex}]."
                    "integerDelaySamples has an invalid value. Allowed range: "
                    "integer in [0, +inf) samples."
                )
            fractionalDelay = resolvedPath[
                "fractionalDelaySamples"
            ]
            if (
                not isinstance(fractionalDelay, (int, float))
                or isinstance(fractionalDelay, bool)
                or not np.isfinite(fractionalDelay)
                or not -0.5 <= float(fractionalDelay) < 0.5
            ):
                raise ValueError(
                    f"{parameterName}[{pathIndex}]."
                    "fractionalDelaySamples has an invalid value. Allowed "
                    "range: finite real number in [-0.5, 0.5) samples."
                )
            firTaps = resolvedPath["firTaps"]
            if firTaps is not None:
                if isinstance(firTaps, (str, bytes)):
                    raise TypeError(
                        f"{parameterName}[{pathIndex}].firTaps has an invalid "
                        "type. Allowed values: None or a nonempty "
                        "one-dimensional sequence of finite complex numbers."
                    )
                firArray = np.asarray(
                    firTaps, dtype=np.complex128
                )
                if (
                    firArray.ndim != 1
                    or firArray.size == 0
                    or not np.all(np.isfinite(firArray))
                ):
                    raise ValueError(
                        f"{parameterName}[{pathIndex}].firTaps has an invalid "
                        "value. Allowed values: None or a nonempty "
                        "one-dimensional sequence of finite complex numbers."
                    )
                resolvedPath["firTaps"] = tuple(
                    complex(value) for value in firArray
                )
            resolvedPath["sourceChain"] = int(sourceChain)
            resolvedPath["destinationChain"] = int(
                destinationChain
            )
            resolvedPath["gainDb"] = float(resolvedPath["gainDb"])
            resolvedPath["phaseDegrees"] = float(
                resolvedPath["phaseDegrees"]
            )
            resolvedPath["integerDelaySamples"] = int(
                integerDelay
            )
            resolvedPath["fractionalDelaySamples"] = float(
                fractionalDelay
            )
            resolvedPaths.append(resolvedPath)
        return tuple(resolvedPaths)

    def HasPrePaCoupling(self) -> bool:
        """Return whether at least one PA-input cross-coupling path exists.

        Processing details:
            Algorithm: Resolve the live pre-PA path sequence without requiring
            a waveform chain count and test whether the canonical tuple is
            nonempty. This result selects joint power calibration by default.

        Returns:
            result: True when PA inputs are electrically cross-coupled.
        """

        return bool(self.ResolveCouplingPaths("prePaCouplingPaths"))

    @staticmethod
    def ApplyCouplingPath(
        sourceSignal: np.ndarray,
        couplingPath: Mapping[str, object],
    ) -> np.ndarray:
        """Apply one causal complex FIR, fractional delay, and path gain.

        Processing details:
            Algorithm: Convolve the source with the configured FIR or an
            identity tap, retain the source record length, interpolate real
            and imaginary components at ``n-fractionalDelay`` with zero
            extrapolation, prefix the nonnegative integer delay, and multiply
            by the voltage gain and phase coefficient.

        Args:
            sourceSignal: One normalized complex source-chain vector.
            couplingPath: Canonical mapping returned by ResolveCouplingPaths.

        Returns:
            result: Same-length complex contribution at the destination.
        """

        sourceVector = np.asarray(
            sourceSignal, dtype=np.complex128
        ).reshape(-1)
        firTapsValue = couplingPath["firTaps"]
        firTaps = (
            np.asarray((1.0 + 0.0j,), dtype=np.complex128)
            if firTapsValue is None
            else np.asarray(firTapsValue, dtype=np.complex128)
        )
        filteredSignal = np.convolve(
            sourceVector, firTaps, mode="full"
        )[: sourceVector.size]
        fractionalDelay = float(
            couplingPath["fractionalDelaySamples"]
        )
        if fractionalDelay != 0.0:
            nominalPositions = np.arange(
                sourceVector.size, dtype=float
            )
            sourcePositions = nominalPositions - fractionalDelay
            filteredSignal = (
                np.interp(
                    sourcePositions,
                    nominalPositions,
                    filteredSignal.real,
                    left=0.0,
                    right=0.0,
                )
                + 1j
                * np.interp(
                    sourcePositions,
                    nominalPositions,
                    filteredSignal.imag,
                    left=0.0,
                    right=0.0,
                )
            )
        integerDelay = int(
            couplingPath["integerDelaySamples"]
        )
        if integerDelay > 0:
            delayedSignal = np.zeros_like(filteredSignal)
            if integerDelay < sourceVector.size:
                delayedSignal[integerDelay:] = filteredSignal[
                    : sourceVector.size - integerDelay
                ]
        else:
            delayedSignal = filteredSignal
        pathCoefficient = (
            np.power(
                10.0, float(couplingPath["gainDb"]) / 20.0
            )
            * np.exp(
                1j
                * np.deg2rad(
                    float(couplingPath["phaseDegrees"])
                )
            )
        )
        pathOutput = pathCoefficient * delayedSignal
        if not np.all(np.isfinite(pathOutput)):
            raise ValueError(
                "coupling path exceeded the numeric range"
            )
        return np.asarray(pathOutput, dtype=np.complex128)

    def ApplyMimoCoupling(
        self,
        inputSignal: np.ndarray,
        parameterName: str,
    ) -> np.ndarray:
        """Add every configured cross-coupling contribution to a matrix.

        Processing details:
            Algorithm: Preserve an identity direct path for every column,
            resolve and range-check all configured paths against the current
            matrix, evaluate each source path independently, and accumulate
            its contribution into the selected destination. Paths may be
            asymmetric and multiple paths may share a source/destination.

        Args:
            inputSignal: Normalized vector or samples-by-chains matrix.
            parameterName: Pre- or post-PA path configuration name.

        Returns:
            result: Same-shape waveform after additive complex coupling.
        """

        complexInput = self.ValidateSignal(inputSignal, "inputSignal")
        inputWasVector = complexInput.ndim == 1
        inputMatrix = (
            complexInput.reshape(-1, 1)
            if inputWasVector
            else complexInput
        )
        couplingPaths = self.ResolveCouplingPaths(
            parameterName, inputMatrix.shape[1]
        )
        outputMatrix = inputMatrix.copy()
        for couplingPath in couplingPaths:
            sourceChain = int(couplingPath["sourceChain"])
            destinationChain = int(
                couplingPath["destinationChain"]
            )
            outputMatrix[:, destinationChain] += (
                self.ApplyCouplingPath(
                    inputMatrix[:, sourceChain],
                    couplingPath,
                )
            )
        return outputMatrix[:, 0] if inputWasVector else outputMatrix

    def ResolveIqImbalanceCoefficients(
        self,
        gainImbalanceDb: float,
        phaseImbalanceDegrees: float,
    ) -> Tuple[complex, complex]:
        """Convert I/Q gain and quadrature errors to a widely-linear pair.

        Processing details:
            Algorithm: Apply reciprocal half-gain changes to I and Q, rotate
            the imperfect Q axis by the configured quadrature error, and
            algebraically rewrite the real I/Q mapping as ``alpha*x +
            beta*conj(x)``. The direct coefficient is alpha and the image
            coefficient is beta.

        Args:
            gainImbalanceDb: I-to-Q gain ratio error in decibels.
            phaseImbalanceDegrees: Departure from ideal quadrature in degrees.

        Returns:
            result: Direct and conjugate-image complex coefficients.
        """

        iGain = np.power(10.0, float(gainImbalanceDb) / 40.0)
        qGain = np.power(10.0, -float(gainImbalanceDb) / 40.0)
        phaseErrorRadians = np.deg2rad(float(phaseImbalanceDegrees))
        directCoefficient = 0.5 * (
            iGain
            + qGain * np.cos(phaseErrorRadians)
            + 1j * qGain * np.sin(phaseErrorRadians)
        )
        imageCoefficient = 0.5 * (
            iGain
            - qGain * np.cos(phaseErrorRadians)
            + 1j * qGain * np.sin(phaseErrorRadians)
        )
        return complex(directCoefficient), complex(imageCoefficient)

    def ApplyIqImbalanceStage(
        self,
        inputSignal: np.ndarray,
        gainImbalanceDb: float,
        phaseImbalanceDegrees: float,
        dcOffset: complex,
        stageName: str,
        directFirTaps: Optional[Sequence[complex]] = None,
        imageFirTaps: Optional[Sequence[complex]] = None,
    ) -> np.ndarray:
        """Apply one frequency-selective widely-linear I/Q impairment stage.

        Processing details:
            Algorithm: Causally convolve every chain with the direct FIR,
            convolve its complex conjugate with the image FIR, add both paths
            and one normalized complex DC offset, retain the input record
            length, and reject numeric overflow. One-tap FIRs reduce exactly to
            the historical flat ``alpha*x + beta*conj(x)`` model.

        Args:
            inputSignal: Normalized complex waveform entering the I/Q stage.
            gainImbalanceDb: Legacy I-to-Q gain ratio error in decibels.
            phaseImbalanceDegrees: Legacy quadrature error in degrees.
            dcOffset: Normalized complex LO-leakage or receiver DC term.
            stageName: Human-readable stage name used in error diagnostics.
            directFirTaps: Optional complete causal FIR for the desired ``x``
                branch. None selects its legacy one-tap coefficient.
            imageFirTaps: Optional complete causal FIR for the conjugate ``x*``
                branch. None selects its legacy one-tap coefficient.

        Returns:
            result: Same-shape waveform containing filtered direct, image, and
                DC terms.
        """

        complexInput = self.ValidateSignal(inputSignal, "inputSignal")
        directCoefficient, imageCoefficient = (
            self.ResolveIqImbalanceCoefficients(
                gainImbalanceDb,
                phaseImbalanceDegrees,
            )
        )
        directTapArray = np.asarray(
            (directCoefficient,)
            if directFirTaps is None
            else directFirTaps,
            dtype=np.complex128,
        )
        imageTapArray = np.asarray(
            (imageCoefficient,)
            if imageFirTaps is None
            else imageFirTaps,
            dtype=np.complex128,
        )
        if (
            directTapArray.ndim != 1
            or directTapArray.size == 0
            or not np.all(np.isfinite(directTapArray))
        ):
            raise ValueError(
                "directFirTaps must be a nonempty one-dimensional sequence "
                "of finite complex numbers"
            )
        if (
            imageTapArray.ndim != 1
            or imageTapArray.size == 0
            or not np.all(np.isfinite(imageTapArray))
        ):
            raise ValueError(
                "imageFirTaps must be a nonempty one-dimensional sequence "
                "of finite complex numbers"
            )
        directTaps = directTapArray.reshape(-1)
        imageTaps = imageTapArray.reshape(-1)
        if (
            directTaps.size == 1
            and directTaps[0] == 1.0 + 0.0j
            and imageTaps.size == 1
            and imageTaps[0] == 0.0 + 0.0j
            and complex(dcOffset) == 0.0 + 0.0j
        ):
            # An enabled but ideal I/Q stage is physically an identity. Keep
            # the public method's independent-array behavior while avoiding a
            # conjugate array and three full-waveform arithmetic passes.
            return complexInput.copy()
        inputWasVector = complexInput.ndim == 1
        inputMatrix = (
            complexInput.reshape(-1, 1)
            if inputWasVector
            else complexInput
        )
        sampleCount, chainCount = inputMatrix.shape
        directOutput = np.empty_like(inputMatrix)
        imageOutput = np.empty_like(inputMatrix)
        conjugateInput = np.conj(inputMatrix)
        for chainIndex in range(chainCount):
            if directTaps.size == 1:
                directOutput[:, chainIndex] = (
                    directTaps[0] * inputMatrix[:, chainIndex]
                )
            else:
                directOutput[:, chainIndex] = np.convolve(
                    inputMatrix[:, chainIndex], directTaps, mode="full"
                )[:sampleCount]
            if imageTaps.size == 1:
                imageOutput[:, chainIndex] = (
                    imageTaps[0] * conjugateInput[:, chainIndex]
                )
            else:
                imageOutput[:, chainIndex] = np.convolve(
                    conjugateInput[:, chainIndex], imageTaps, mode="full"
                )[:sampleCount]
        iqOutputMatrix = (
            directOutput + imageOutput + complex(dcOffset)
        )
        if not np.all(np.isfinite(iqOutputMatrix)):
            raise ValueError(
                f"{stageName} I/Q imbalance exceeded the numeric range"
            )
        if inputWasVector:
            return np.asarray(iqOutputMatrix[:, 0], dtype=np.complex128)
        return np.asarray(iqOutputMatrix, dtype=np.complex128)

    def TransmitterIqCoefficients(self) -> Tuple[complex, complex]:
        """Return the legacy flat Tx direct and image fallback coefficients.

        Processing details:
            Algorithm: Validate the live Channel configuration. Return the
            ideal direct/image pair when the Tx stage is disabled; otherwise
            convert transmitter gain and phase mismatch parameters to the
            one-tap fallback pair. Explicit Tx FIR parameters replace their
            respective branch; use ``TransmitterIqFilterTaps`` to query the
            actual effective response.

        Returns:
            result: Tx direct coefficient followed by Tx image coefficient.
        """

        self.ValidateParameters()
        if not bool(self.parameters["txIqImbalanceEnabled"]):
            return 1.0 + 0.0j, 0.0 + 0.0j
        return self.ResolveIqImbalanceCoefficients(
            float(self.parameters["txIqGainImbalanceDb"]),
            float(self.parameters["txIqPhaseImbalanceDegrees"]),
        )

    def FeedbackIqCoefficients(self) -> Tuple[complex, complex]:
        """Return the legacy flat FB direct and image fallback coefficients.

        Processing details:
            Algorithm: Validate the live Channel configuration. Return the
            ideal direct/image pair when the feedback stage is disabled;
            otherwise convert receiver gain and phase mismatch parameters to
            the one-tap fallback pair. Explicit FB FIR parameters replace their
            respective branch; use ``FeedbackIqFilterTaps`` for the effective
            response used before feedback DC addition.

        Returns:
            result: Feedback direct coefficient followed by image coefficient.
        """

        self.ValidateParameters()
        if not bool(self.parameters["fbIqImbalanceEnabled"]):
            return 1.0 + 0.0j, 0.0 + 0.0j
        return self.ResolveIqImbalanceCoefficients(
            float(self.parameters["fbIqGainImbalanceDb"]),
            float(self.parameters["fbIqPhaseImbalanceDegrees"]),
        )

    def TransmitterIqFilterTaps(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the effective Tx direct and conjugate-image FIR taps.

        Processing details:
            Algorithm: When the Tx I/Q stage is disabled, return the exact
            identity/zero pair. Otherwise each explicitly configured FIR is the
            complete response of that widely-linear branch; a None branch falls
            back to the corresponding legacy gain/phase-derived one-tap
            coefficient. Return defensive complex vectors.

        Returns:
            result: Effective direct FIR followed by effective image FIR.
        """

        self.ValidateParameters()
        if not bool(self.parameters["txIqImbalanceEnabled"]):
            return (
                np.asarray((1.0 + 0.0j,), dtype=np.complex128),
                np.asarray((0.0 + 0.0j,), dtype=np.complex128),
            )
        directCoefficient, imageCoefficient = (
            self.TransmitterIqCoefficients()
        )
        directFirTaps = self.parameters["txIqDirectFirTaps"]
        imageFirTaps = self.parameters["txIqImageFirTaps"]
        return (
            np.asarray(
                (directCoefficient,)
                if directFirTaps is None
                else directFirTaps,
                dtype=np.complex128,
            ).reshape(-1).copy(),
            np.asarray(
                (imageCoefficient,)
                if imageFirTaps is None
                else imageFirTaps,
                dtype=np.complex128,
            ).reshape(-1).copy(),
        )

    def FeedbackIqFilterTaps(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the effective FB direct and conjugate-image FIR taps.

        Processing details:
            Algorithm: When the feedback I/Q stage is disabled, return the
            identity/zero pair. Otherwise use each configured FIR as that
            branch's complete frequency-selective response and let a None
            branch fall back to the legacy gain/phase-derived one-tap value.

        Returns:
            result: Effective feedback direct FIR followed by image FIR.
        """

        self.ValidateParameters()
        if not bool(self.parameters["fbIqImbalanceEnabled"]):
            return (
                np.asarray((1.0 + 0.0j,), dtype=np.complex128),
                np.asarray((0.0 + 0.0j,), dtype=np.complex128),
            )
        directCoefficient, imageCoefficient = self.FeedbackIqCoefficients()
        directFirTaps = self.parameters["fbIqDirectFirTaps"]
        imageFirTaps = self.parameters["fbIqImageFirTaps"]
        return (
            np.asarray(
                (directCoefficient,)
                if directFirTaps is None
                else directFirTaps,
                dtype=np.complex128,
            ).reshape(-1).copy(),
            np.asarray(
                (imageCoefficient,)
                if imageFirTaps is None
                else imageFirTaps,
                dtype=np.complex128,
            ).reshape(-1).copy(),
        )

    def ApplyTransmitterIqImbalance(
        self, inputSignal: np.ndarray
    ) -> np.ndarray:
        """Apply flat or frequency-selective Tx I/Q error before every PA.

        Processing details:
            Algorithm: Validate and return an unchanged complex copy when
            ``txIqImbalanceEnabled`` is False, bypassing Tx gain mismatch,
            phase mismatch, both frequency-selective FIR branches, and DC
            leakage together. When enabled, use the
            common widely-linear FIR stage with the resolved ``tx`` direct and
            image responses before PA-input coupling. This impairment changes
            physical PA drive, PA nonlinear products, both observations, and
            power calibration.

        Args:
            inputSignal: Digital transmitter waveform before RF modulation.

        Returns:
            result: Same-shape waveform emitted by the nonideal Tx I/Q stage.
        """

        self.ValidateParameters()
        if not bool(self.parameters["txIqImbalanceEnabled"]):
            return np.asarray(
                self.ValidateSignal(inputSignal, "inputSignal"),
                dtype=np.complex128,
            ).copy()
        return self.ApplyIqImbalanceStage(
            inputSignal,
            float(self.parameters["txIqGainImbalanceDb"]),
            float(self.parameters["txIqPhaseImbalanceDegrees"]),
            complex(self.parameters["txDcOffset"]),
            "transmitter",
            cast(
                Optional[Sequence[complex]],
                self.parameters["txIqDirectFirTaps"],
            ),
            cast(
                Optional[Sequence[complex]],
                self.parameters["txIqImageFirTaps"],
            ),
        )

    def ApplyPrePaCoupling(
        self, inputSignal: np.ndarray
    ) -> np.ndarray:
        """Apply all configured coupling paths before nonlinear PA branches.

        Processing details:
            Algorithm: Delegate to ApplyMimoCoupling with the pre-PA path
            sequence so every PA receives its own drive plus delayed complex
            leakage from the other physical input chains.

        Args:
            inputSignal: Tx-modulator output vector or samples-by-chains matrix.

        Returns:
            result: Same-shape actual drive presented to the PA bank.
        """

        return self.ApplyMimoCoupling(
            inputSignal, "prePaCouplingPaths"
        )

    def ApplyPostPaCoupling(
        self, paOutputSignal: np.ndarray
    ) -> np.ndarray:
        """Apply all configured coupling paths after nonlinear PA branches.

        Processing details:
            Algorithm: Delegate to ApplyMimoCoupling with the post-PA path
            sequence so independently generated nonlinear outputs leak into
            other conducted output chains before forward or feedback sampling.

        Args:
            paOutputSignal: Normalized output vector or matrix from the PA bank.

        Returns:
            result: Same-shape coupled PA output presented to the sampler.
        """

        return self.ApplyMimoCoupling(
            paOutputSignal, "postPaCouplingPaths"
        )

    def ResolveNoiseRmsVolts(self) -> float:
        """Resolve the requested complex-envelope noise RMS in volts.

        Processing details:
            Algorithm: Convert ``noiseAmpMv`` directly from millivolts, or
            convert ``noisePwrDbm`` to watts and then RMS voltage through the
            configured resistive port. All three controls set to None resolve
            to zero noise. An SNR-controlled value is signal-dependent and
            must instead use ``ResolveSnrNoiseRmsPerChain``.

        Returns:
            result: Nonnegative complex-envelope RMS noise voltage.
        """

        self.ValidateParameters()
        if self.parameters["noiseSnrDb"] is not None:
            raise RuntimeError(
                "noiseSnrDb has no signal-independent RMS voltage; use "
                "ResolveSnrNoiseRmsPerChain(inputSignal)"
            )
        noiseAmpMv = self.parameters["noiseAmpMv"]
        if noiseAmpMv is not None:
            return float(noiseAmpMv) * 1.0e-3
        noisePwrDbm = self.parameters["noisePwrDbm"]
        if noisePwrDbm is None:
            return 0.0
        loadResistanceOhm = float(
            cast(float, self.parameters["loadResistanceOhm"])
        )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            rmsVoltage = (
                np.sqrt(1.0e-3 * loadResistanceOhm)
                * np.power(10.0, float(noisePwrDbm) / 20.0)
            )
        if not np.isfinite(rmsVoltage):
            raise ValueError("noisePwrDbm is outside the numeric range")
        return float(rmsVoltage)

    def ResolveNoiseRmsNormalized(self) -> float:
        """Convert physical noise RMS to normalized PA output units.

        Processing details:
            Algorithm: Calculate the RMS voltage represented by normalized
            PA output magnitude one at ``maximumOutputPowerDbm`` and divide
            the requested physical noise RMS by that full-scale voltage.

        Returns:
            result: Nonnegative complex-noise RMS in normalized units.
        """

        noiseRmsVolts = self.ResolveNoiseRmsVolts()
        if noiseRmsVolts == 0.0:
            return 0.0
        loadResistanceOhm = float(
            cast(float, self.parameters["loadResistanceOhm"])
        )
        maximumOutputPowerDbm = float(
            cast(float, self.parameters["maximumOutputPowerDbm"])
        )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            fullScaleRmsVolts = (
                np.sqrt(1.0e-3 * loadResistanceOhm)
                * np.power(10.0, maximumOutputPowerDbm / 20.0)
            )
        if (
            not np.isfinite(fullScaleRmsVolts)
            or fullScaleRmsVolts <= 0.0
        ):
            raise ValueError(
                "maximumOutputPowerDbm is outside the numeric range"
            )
        return float(noiseRmsVolts / fullScaleRmsVolts)

    def ResolveSnrNoiseRmsPerChain(
        self, inputSignal: np.ndarray
    ) -> Tuple[float, ...]:
        """Derive per-chain normalized noise RMS from active-signal SNR.

        Processing details:
            Algorithm: Detect each chain's active burst with the same relative
            power threshold and short-gap rule used by PA power calibration,
            measure signal RMS only over those samples, and multiply by
            ``10**(-noiseSnrDb/20)``. This keeps leading/trailing padding and
            long duty-cycle off intervals from artificially reducing noise.

        Args:
            inputSignal: Normalized phase-rotated SISO or MIMO signal.

        Returns:
            result: One total complex-envelope noise RMS per signal chain.
        """

        self.ValidateParameters()
        noiseSnrDb = self.parameters["noiseSnrDb"]
        if noiseSnrDb is None:
            raise RuntimeError(
                "ResolveSnrNoiseRmsPerChain requires noiseSnrDb"
            )
        complexInput = self.ValidateSignal(inputSignal, "inputSignal")
        activePowerDetector = PowerCalibration(
            parameters={
                "activePowerThresholdDb": self.parameters[
                    "activePowerThresholdDb"
                ],
                "activeGapToleranceSamples": self.parameters[
                    "activeGapToleranceSamples"
                ],
                "width": 0,
            },
        )
        signalRmsPerChain = activePowerDetector.CalculateActiveRmsPerChain(
            complexInput
        )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            noiseToSignalScale = np.power(
                10.0, -float(noiseSnrDb) / 20.0
            )
        if not np.isfinite(noiseToSignalScale):
            raise ValueError("noiseSnrDb is outside the numeric range")
        return tuple(
            float(signalRms * noiseToSignalScale)
            for signalRms in signalRmsPerChain
        )

    def ApplyPhaseRotation(
        self, inputSignal: np.ndarray
    ) -> np.ndarray:
        """Rotate normalized complex samples by the configured phase.

        Processing details:
            Algorithm: Convert the allowed degree value to radians, multiply
            every vector or matrix element by one unit-magnitude complex
            exponential, and preserve signal shape and average power.

        Args:
            inputSignal: Normalized floating complex PA output.

        Returns:
            result: Phase-rotated normalized complex samples.
        """

        self.ValidateParameters()
        complexInput = self.ValidateSignal(inputSignal, "inputSignal")
        phaseRadians = np.deg2rad(
            float(cast(float, self.parameters["phaseDegrees"]))
        )
        if phaseRadians == 0.0:
            return complexInput.copy()
        phaseFactor = np.exp(1j * phaseRadians)
        return np.asarray(
            complexInput * phaseFactor, dtype=np.complex128
        )

    def AddNoise(self, inputSignal: np.ndarray) -> np.ndarray:
        """Add circular complex white Gaussian noise in normalized units.

        Processing details:
            Algorithm: For an SNR control, derive one total noise RMS from
            each chain's active signal RMS. Otherwise resolve the common
            physical amplitude or power setting. Divide total RMS by square
            root two for independent I and Q Gaussian components, draw one
            sample per array element, and add it after phase rotation. No
            configured noise returns an exact copy.

        Args:
            inputSignal: Normalized floating complex samples after rotation.

        Returns:
            result: Normalized samples including the configured white noise.
        """

        complexInput = self.ValidateSignal(inputSignal, "inputSignal")
        self.SynchronizeRandomGenerator()
        noiseSnrDb = self.parameters["noiseSnrDb"]
        if noiseSnrDb is None:
            noiseRmsNormalized = self.ResolveNoiseRmsNormalized()
            if noiseRmsNormalized == 0.0:
                return complexInput.copy()
            componentScale: Union[float, np.ndarray] = (
                noiseRmsNormalized / np.sqrt(2.0)
            )
        else:
            noiseRmsPerChain = self.ResolveSnrNoiseRmsPerChain(
                complexInput
            )
            if complexInput.ndim == 1:
                componentScale = noiseRmsPerChain[0] / np.sqrt(2.0)
            else:
                componentScale = (
                    np.asarray(noiseRmsPerChain, dtype=float).reshape(1, -1)
                    / np.sqrt(2.0)
                )
        complexNoise = componentScale * (
            self._randomGenerator.standard_normal(complexInput.shape)
            + 1j
            * self._randomGenerator.standard_normal(complexInput.shape)
        )
        return np.asarray(
            complexInput + complexNoise, dtype=np.complex128
        )

    def ResolveFeedbackFirTaps(self) -> np.ndarray:
        """Return the configured causal feedback-channel impulse response.

        Processing details:
            Algorithm: Replace a disabled setting with one identity tap and
            otherwise convert the already validated caller sequence into a
            defensive complex128 vector.

        Returns:
            result: Nonempty one-dimensional complex FIR tap vector.
        """

        self.ValidateParameters()
        fbFirTaps = self.parameters["fbFirTaps"]
        if fbFirTaps is None:
            return np.asarray((1.0 + 0.0j,), dtype=np.complex128)
        return np.asarray(
            fbFirTaps, dtype=np.complex128
        ).reshape(-1).copy()

    def ApplyFeedbackLinearResponse(
        self, inputSignal: np.ndarray
    ) -> np.ndarray:
        """Apply feedback coupling gain, phase, and causal FIR response.

        Processing details:
            Algorithm: Convolve each SISO or MIMO chain independently with
            ``fbFirTaps``, retain the original record length, and multiply by
            the configured logarithmic voltage gain and complex phase. The
            forward instrument mode never calls this operation.

        Args:
            inputSignal: Normalized PA output after common phase rotation.

        Returns:
            result: Feedback analog signal after linear path distortion.
        """

        complexInput = self.ValidateSignal(inputSignal, "inputSignal")
        inputWasVector = complexInput.ndim == 1
        inputMatrix = (
            complexInput.reshape(-1, 1)
            if inputWasVector
            else complexInput
        )
        firTaps = self.ResolveFeedbackFirTaps()
        feedbackGainDb = float(self.parameters["fbGainDb"])
        feedbackPhaseDegrees = float(
            self.parameters["fbPhaseDegrees"]
        )
        if (
            firTaps.size == 1
            and firTaps[0] == 1.0 + 0.0j
            and feedbackGainDb == 0.0
            and feedbackPhaseDegrees == 0.0
        ):
            return complexInput.copy()
        filteredMatrix = np.empty_like(inputMatrix)
        for chainIndex in range(inputMatrix.shape[1]):
            filteredMatrix[:, chainIndex] = np.convolve(
                inputMatrix[:, chainIndex],
                firTaps,
                mode="full",
            )[: inputMatrix.shape[0]]
        with np.errstate(over="ignore", invalid="ignore"):
            feedbackGain = np.power(
                10.0,
                feedbackGainDb / 20.0,
            )
        feedbackPhase = np.exp(
            1j
            * np.deg2rad(
                feedbackPhaseDegrees
            )
        )
        linearOutput = filteredMatrix * feedbackGain * feedbackPhase
        if not np.all(np.isfinite(linearOutput)):
            raise ValueError(
                "feedback linear response exceeded the numeric range"
            )
        return (
            linearOutput[:, 0]
            if inputWasVector
            else np.asarray(linearOutput, dtype=np.complex128)
        )

    def ApplyFeedbackNonlinearity(
        self, inputSignal: np.ndarray
    ) -> np.ndarray:
        """Apply receiver third-order distortion and envelope clipping.

        Processing details:
            Algorithm: Evaluate the memoryless complex polynomial
            ``x + c3*abs(x)**2*x`` and then, when configured, radially limit
            samples whose complex-envelope magnitude exceeds
            ``fbClipAmplitude``. These effects model feedback front-end
            compression rather than PA nonlinearity.

        Args:
            inputSignal: Feedback signal after linear coupling response.

        Returns:
            result: Nonlinearly distorted normalized feedback waveform.
        """

        self.ValidateParameters()
        complexInput = self.ValidateSignal(inputSignal, "inputSignal")
        thirdOrderCoefficient = complex(
            self.parameters["fbThirdOrderCoefficient"]
        )
        clipAmplitudeValue = self.parameters["fbClipAmplitude"]
        if (
            thirdOrderCoefficient == 0.0 + 0.0j
            and clipAmplitudeValue is None
        ):
            return complexInput.copy()
        nonlinearOutput = complexInput + (
            thirdOrderCoefficient
            * np.abs(complexInput) ** 2
            * complexInput
        )
        if clipAmplitudeValue is not None:
            clipAmplitude = float(clipAmplitudeValue)
            outputMagnitude = np.abs(nonlinearOutput)
            overLimit = outputMagnitude > clipAmplitude
            if np.any(overLimit):
                nonlinearOutput = nonlinearOutput.copy()
                nonlinearOutput[overLimit] *= (
                    clipAmplitude / outputMagnitude[overLimit]
                )
        if not np.all(np.isfinite(nonlinearOutput)):
            raise ValueError(
                "feedback nonlinearity exceeded the numeric range"
            )
        return np.asarray(nonlinearOutput, dtype=np.complex128)

    def ApplyFeedbackTimingAndFrequency(
        self, inputSignal: np.ndarray
    ) -> np.ndarray:
        """Apply fractional/integer delay, SFO, and carrier offset.

        Processing details:
            Algorithm: Sample every chain at positions
            ``n*(1+sfo)-fractionalDelay`` with linear interpolation and zero
            extrapolation, prefix the configured integer delay while
            preserving record length, then multiply by the carrier-frequency
            phase ramp derived from ``sampleRateHz``.

        Args:
            inputSignal: Feedback analog waveform before oscillator effects.

        Returns:
            result: Same-shape waveform with timing and frequency offsets.
        """

        self.ValidateParameters()
        complexInput = self.ValidateSignal(inputSignal, "inputSignal")
        inputWasVector = complexInput.ndim == 1
        inputMatrix = (
            complexInput.reshape(-1, 1)
            if inputWasVector
            else complexInput
        )
        sampleCount = inputMatrix.shape[0]
        nominalPositions = np.arange(sampleCount, dtype=float)
        samplingOffsetRatio = (
            float(self.parameters["fbSamplingFrequencyOffsetPpm"])
            * 1.0e-6
        )
        fractionalDelay = float(
            self.parameters["fbFractionalDelaySamples"]
        )
        sourcePositions = (
            nominalPositions * (1.0 + samplingOffsetRatio)
            - fractionalDelay
        )
        if samplingOffsetRatio == 0.0 and fractionalDelay == 0.0:
            resampledMatrix = inputMatrix.copy()
        else:
            resampledMatrix = np.empty_like(inputMatrix)
            for chainIndex in range(inputMatrix.shape[1]):
                inputColumn = inputMatrix[:, chainIndex]
                resampledMatrix[:, chainIndex] = (
                    np.interp(
                        sourcePositions,
                        nominalPositions,
                        inputColumn.real,
                        left=0.0,
                        right=0.0,
                    )
                    + 1j
                    * np.interp(
                        sourcePositions,
                        nominalPositions,
                        inputColumn.imag,
                        left=0.0,
                        right=0.0,
                    )
                )
        integerDelay = int(
            self.parameters["fbIntegerDelaySamples"]
        )
        if integerDelay > 0:
            delayedMatrix = np.zeros_like(resampledMatrix)
            if integerDelay < sampleCount:
                delayedMatrix[integerDelay:, :] = resampledMatrix[
                    : sampleCount - integerDelay, :
                ]
        else:
            delayedMatrix = resampledMatrix
        carrierFrequencyOffsetHz = float(
            self.parameters["fbCarrierFrequencyOffsetHz"]
        )
        if carrierFrequencyOffsetHz != 0.0:
            carrierPhasor = np.exp(
                1j
                * 2.0
                * np.pi
                * carrierFrequencyOffsetHz
                * nominalPositions
                / float(self.parameters["sampleRateHz"])
            ).reshape(-1, 1)
            delayedMatrix = delayedMatrix * carrierPhasor
        return (
            delayedMatrix[:, 0]
            if inputWasVector
            else np.asarray(delayedMatrix, dtype=np.complex128)
        )

    def ApplyFeedbackIqImbalance(
        self, inputSignal: np.ndarray
    ) -> np.ndarray:
        """Apply flat or frequency-selective FB I/Q error and receiver DC.

        Processing details:
            Algorithm: Validate and return an unchanged complex copy when
            ``fbIqImbalanceEnabled`` is False, bypassing feedback gain
            mismatch, quadrature phase error, both FIR branches, and DC offset
            together. When
            enabled, use the shared widely-linear FIR stage. This function
            appears in the processing chain only for ``sampleMode="fb"``.

        Args:
            inputSignal: Feedback waveform after timing and frequency errors.

        Returns:
            result: Feedback baseband waveform with image and DC impairment.
        """

        self.ValidateParameters()
        if not bool(self.parameters["fbIqImbalanceEnabled"]):
            return np.asarray(
                self.ValidateSignal(inputSignal, "inputSignal"),
                dtype=np.complex128,
            ).copy()
        return self.ApplyIqImbalanceStage(
            inputSignal,
            float(self.parameters["fbIqGainImbalanceDb"]),
            float(self.parameters["fbIqPhaseImbalanceDegrees"]),
            complex(self.parameters["fbDcOffset"]),
            "feedback",
            cast(
                Optional[Sequence[complex]],
                self.parameters["fbIqDirectFirTaps"],
            ),
            cast(
                Optional[Sequence[complex]],
                self.parameters["fbIqImageFirTaps"],
            ),
        )

    def ApplyFeedbackAdc(
        self, inputSignal: np.ndarray
    ) -> np.ndarray:
        """Apply optional feedback ADC component clipping and quantization.

        Processing details:
            Algorithm: When ``fbAdcWidth`` is enabled, normalize I and Q by
            ``fbAdcFullScale``, round each component to a signed W-bit code,
            saturate to ``[-2**(W-1), 2**(W-1)-1]``, and decode the code back
            to a floating receiver sample. None leaves the signal unchanged.

        Args:
            inputSignal: Noisy feedback receiver waveform before ADC.

        Returns:
            result: Same-shape floating waveform after ADC quantization.
        """

        self.ValidateParameters()
        complexInput = self.ValidateSignal(inputSignal, "inputSignal")
        adcWidthValue = self.parameters["fbAdcWidth"]
        if adcWidthValue is None:
            return complexInput.copy()
        adcWidth = int(adcWidthValue)
        fullScale = float(self.parameters["fbAdcFullScale"])
        codeScale = float(2 ** (adcWidth - 1))
        minimumCode = -codeScale
        maximumCode = codeScale - 1.0
        realCodes = np.clip(
            np.rint(complexInput.real / fullScale * codeScale),
            minimumCode,
            maximumCode,
        )
        imagCodes = np.clip(
            np.rint(complexInput.imag / fullScale * codeScale),
            minimumCode,
            maximumCode,
        )
        return np.asarray(
            fullScale * (realCodes + 1j * imagCodes) / codeScale,
            dtype=np.complex128,
        )

    def ApplyFeedbackAnalogImpairments(
        self, inputSignal: np.ndarray
    ) -> np.ndarray:
        """Apply every configured embedded-feedback analog impairment.

        Processing details:
            Algorithm: Execute coupling/FIR distortion, receiver third-order
            compression and clipping, timing/CFO/SFO errors, then I/Q mismatch
            and DC offset. AWGN and ADC quantization are intentionally applied
            later so noise is quantized by the modeled converter.

        Args:
            inputSignal: Normalized PA output after common phase rotation.

        Returns:
            result: Feedback analog baseband waveform immediately before noise.
        """

        timingOutput = self.ApplyFeedbackPreIqImpairments(inputSignal)
        return self.ApplyFeedbackIqImbalance(timingOutput)

    def ApplyFeedbackPreIqImpairments(
        self, inputSignal: np.ndarray
    ) -> np.ndarray:
        """Evaluate all embedded feedback effects before I/Q conversion.

        Processing details:
            Algorithm: Apply feedback coupling/FIR response, receiver
            third-order compression and clipping, then timing, sampling-rate,
            and carrier-frequency offsets. Stop immediately before the I/Q
            imbalance/DC stage so a measured calibration phase-switch response
            can be inserted at one unambiguous physical reference plane.

        Args:
            inputSignal: Normalized PA observation after common phase rotation.

        Returns:
            result: Feedback waveform at the input of the imperfect I/Q stage.
        """

        linearOutput = self.ApplyFeedbackLinearResponse(inputSignal)
        nonlinearOutput = self.ApplyFeedbackNonlinearity(linearOutput)
        return self.ApplyFeedbackTimingAndFrequency(nonlinearOutput)

    def FeedbackDirectSmallSignalGain(self) -> complex:
        """Return the feedback path's analytic direct small-signal coefficient.

        Processing details:
            Algorithm: Multiply coupling gain/phase, ordinary FB FIR DC
            response, and the sum of the direct widely-linear I/Q FIR taps.
            Third-order distortion vanishes at zero amplitude; delay, CFO,
            SFO, DC, noise, clipping, and ADC are not representable by one
            stationary scalar and therefore do not modify this diagnostic.

        Returns:
            result: Complex direct small-signal coefficient before common PA
                and ``phaseDegrees`` gain.
        """

        self.ValidateParameters()
        feedbackGain = np.power(
            10.0, float(self.parameters["fbGainDb"]) / 20.0
        )
        feedbackPhase = np.exp(
            1j
            * np.deg2rad(
                float(self.parameters["fbPhaseDegrees"])
            )
        )
        firDcResponse = np.sum(self.ResolveFeedbackFirTaps())
        directIqFirTaps, _ = self.FeedbackIqFilterTaps()
        directIqDcResponse = np.sum(directIqFirTaps)
        return complex(
            feedbackGain
            * feedbackPhase
            * firDcResponse
            * directIqDcResponse
        )

    def ApplyChannelEffects(
        self, paOutputSignal: np.ndarray
    ) -> np.ndarray:
        """Apply the legacy sampleMode-selected receiver path.

        Processing details:
            Algorithm: Delegate to the explicit forward or feedback helper
            selected by ``sampleMode``. This compatibility entry remains
            single-output. Public ``Process`` always returns two arrays, but
            forward mode makes its feedback array equal to its channel array.

        Args:
            paOutputSignal: Normalized floating PA output samples.

        Returns:
            result: Normalized floating receiver-input samples.
        """

        if self.sampleMode == "forward":
            return self.ApplyForwardChannelEffects(paOutputSignal)
        return self.ApplyCompensatedFeedbackChannelEffects(paOutputSignal)

    def ApplyCompensatedFeedbackChannelEffects(
        self, paOutputSignal: np.ndarray
    ) -> np.ndarray:
        """Apply the selected raw, phase-pair, or cached-filter FB path.

        Processing details:
            Algorithm: Preserve the historical single raw capture in ``none``
            mode. In ``phase_pair`` mode evaluate two measured switch responses
            from the same supplied PA output, separate direct and image terms,
            fit and cache a widely-linear FIR, and return the direct term. In
            ``filter`` mode evaluate only the first switch state and apply the
            most recent non-stale cached inverse.

        Args:
            paOutputSignal: One already evaluated normalized PA-output waveform
                after post-PA coupling.

        Returns:
            result: Normalized feedback observation selected for DPD training.
        """

        self.ValidateParameters()
        compensationMode = str(
            self.parameters["fbIqCompensationMode"]
        ).strip().lower()
        if compensationMode == "none":
            return self.ApplyFeedbackChannelEffects(paOutputSignal)
        phaseResponses = tuple(
            complex(responseValue)
            for responseValue in cast(
                Sequence[complex],
                self.parameters["fbPhasePairResponses"],
            )
        )
        if compensationMode == "filter":
            calibration = self.RequireCurrentFeedbackIqCalibration()
            rawFeedbackOutput = self.ApplyFeedbackChannelEffectsAtResponse(
                paOutputSignal, phaseResponses[0]
            )
            return calibration.Apply(rawFeedbackOutput)

        # The caller supplies one already evaluated PA output, so both receiver
        # states differ only in the extra phase-switch response and independent
        # receiver noise/quantization realizations.
        phaseZeroFeedback = self.ApplyFeedbackChannelEffectsAtResponse(
            paOutputSignal, phaseResponses[0]
        )
        phaseNinetyFeedback = self.ApplyFeedbackChannelEffectsAtResponse(
            paOutputSignal, phaseResponses[1]
        )
        calibration = self.ConfigureFeedbackIqCalibration()
        directFeedback, _ = calibration.SeparatePhasePair(
            phaseZeroFeedback, phaseNinetyFeedback
        )
        calibration.Calibrate(phaseZeroFeedback, phaseNinetyFeedback)
        self._feedbackIqCalibrationSignature = (
            self.FeedbackIqCalibrationSignature()
        )
        self._lastFeedbackPhasePair = (
            np.array(phaseZeroFeedback, dtype=np.complex128, copy=True),
            np.array(phaseNinetyFeedback, dtype=np.complex128, copy=True),
        )
        return directFeedback

    def ApplyForwardChannelEffects(
        self, paOutputSignal: np.ndarray
    ) -> np.ndarray:
        """Create the forward channel observation used for final RF metrics.

        Processing details:
            Algorithm: Apply the common PA-to-receiver phase rotation and the
            configured measurement noise without applying any embedded-
            feedback FIR, nonlinearity, oscillator, I/Q, or ADC impairment.

        Args:
            paOutputSignal: Normalized PA output after post-PA coupling.

        Returns:
            result: Forward channel waveform used by Analysis for EVM, SNR,
                ACLR, power, IRR, and two-tone measurements.
        """

        self.ValidateParameters()
        phaseRotatedSignal = self.ApplyPhaseRotation(paOutputSignal)
        return self.AddNoise(phaseRotatedSignal)

    def ApplyFeedbackChannelEffects(
        self, paOutputSignal: np.ndarray
    ) -> np.ndarray:
        """Create the embedded-feedback observation used by DPD learning.

        Processing details:
            Algorithm: Apply the common phase rotation, the complete feedback
            analog impairment chain, independently generated measurement
            noise, and the optional feedback ADC in their physical order.

        Args:
            paOutputSignal: Normalized PA output after post-PA coupling.

        Returns:
            result: Feedback waveform presented to ILC synchronization,
                error calculation, and coefficient updates.
        """

        return self.ApplyFeedbackChannelEffectsAtResponse(
            paOutputSignal, 1.0 + 0.0j
        )

    def ApplyFeedbackChannelEffectsAtResponse(
        self,
        paOutputSignal: np.ndarray,
        phaseResponse: complex,
    ) -> np.ndarray:
        """Evaluate one measured phase-switch state before FB I/Q error.

        Processing details:
            Algorithm: Apply the existing common ``phaseDegrees`` rotation and
            every feedback impairment up to the I/Q-converter input, multiply
            by the supplied measured complex response of the additional phase
            switch, then execute feedback I/Q/DC, noise, and ADC stages. This
            explicit reference plane makes unequal measured switch magnitudes
            separable even when an earlier feedback amplifier is nonlinear.

        Args:
            paOutputSignal: Normalized PA output after post-PA coupling.
            phaseResponse: Finite nonzero complex voltage response measured for
                the selected phase-switch state.

        Returns:
            result: Raw normalized feedback waveform for that switch state.
        """

        self.ValidateParameters()
        try:
            complexPhaseResponse = complex(phaseResponse)
        except (TypeError, ValueError, OverflowError) as error:
            raise TypeError(
                "phaseResponse must be one finite nonzero complex scalar"
            ) from error
        if (
            not np.isfinite(complexPhaseResponse)
            or abs(complexPhaseResponse) <= np.finfo(float).tiny
        ):
            raise ValueError(
                "phaseResponse must be one finite nonzero complex scalar"
            )
        phaseRotatedSignal = self.ApplyPhaseRotation(paOutputSignal)
        preIqFeedbackSignal = self.ApplyFeedbackPreIqImpairments(
            phaseRotatedSignal
        )
        switchedFeedbackSignal = np.asarray(
            preIqFeedbackSignal * complexPhaseResponse,
            dtype=np.complex128,
        )
        feedbackAnalogSignal = self.ApplyFeedbackIqImbalance(
            switchedFeedbackSignal
        )
        noisyFeedbackSignal = self.AddNoise(feedbackAnalogSignal)
        return self.ApplyFeedbackAdc(noisyFeedbackSignal)

    def ProcessPaOutput(
        self, paOutputSignal: np.ndarray
    ) -> np.ndarray:
        """Apply channel effects to an already evaluated public PA output.

        Processing details:
            Algorithm: Decode public integer I/Q codes once, apply configured
            post-PA inter-chain coupling, execute the selected forward or
            feedback sampling path in normalized floating units, and encode
            the receiver signal back to the same public interface convention.

        Args:
            paOutputSignal: Public PA output vector or matrix.

        Returns:
            result: Public receiver waveform with matching shape and width.
        """

        interfaceFormat = FixedPoint(self.width)
        normalizedPaOutput = interfaceFormat.DecodeComplex(
            self.ValidateSignal(paOutputSignal, "paOutputSignal")
        )
        coupledPaOutput = self.ApplyPostPaCoupling(
            normalizedPaOutput
        )
        normalizedReceiverSignal = self.ApplyChannelEffects(
            coupledPaOutput
        )
        return interfaceFormat.EncodeComplex(normalizedReceiverSignal)

    def ProcessBoundPaFloating(
        self, inputSignal: np.ndarray
    ) -> np.ndarray:
        """Evaluate only the bound PA bank in normalized floating units.

        Processing details:
            Algorithm: Prefer the PA's direct ``ProcessFloating`` entry. For
            a fixed-only third-party PA, encode and decode around its public
            width exactly once. No pre/post coupling or sampling impairment is
            applied here, which keeps one unambiguous PA-bank calculation
            reusable by normal processing and closed-loop power calibration.

        Args:
            inputSignal: Normalized floating PA input vector or matrix.

        Returns:
            result: Normalized floating output of the bound PA bank.
        """

        if self.paModel is None or self._paProcessMethod is None:
            raise RuntimeError(
                "Process requires a PA bound through paModel or SetPaModel"
            )
        normalizedInput = self.ValidateSignal(
            inputSignal, "inputSignal"
        )
        floatingProcessor = getattr(
            self.paModel, "ProcessFloating", None
        )
        if callable(floatingProcessor):
            normalizedPaOutput = floatingProcessor(normalizedInput)
        else:
            paWidthValue = getattr(self.paModel, "width", 0)
            paInterfaceFormat = FixedPoint(int(paWidthValue))
            publicPaInput = paInterfaceFormat.EncodeComplex(
                normalizedInput
            )
            publicPaOutput = self._paProcessMethod(publicPaInput)
            normalizedPaOutput = paInterfaceFormat.DecodeComplex(
                publicPaOutput
            )
        normalizedOutput = self.ValidateSignal(
            normalizedPaOutput, "paOutputSignal"
        )
        if normalizedOutput.shape != normalizedInput.shape:
            raise ValueError(
                "bound PA must preserve the input vector or matrix shape"
            )
        return normalizedOutput

    def ProcessBoundPaThermalPeriodFloating(
        self, inputSignal: np.ndarray
    ) -> np.ndarray:
        """Evaluate the bound PA over one Channel-configured thermal period.

        Processing details:
            Algorithm: Prefer the built-in periodic thermal protocol and pass
            the Channel run mode, scheduling duty, closure tolerance, and
            iteration limit. A nonthermal third-party PA retains the ordinary
            floating path. A third-party PA that advertises enabled thermal
            state but lacks the protocol is rejected instead of silently
            ignoring steady-state or scheduled-idle semantics.

        Args:
            inputSignal: Actual normalized PA-input vector or matrix.

        Returns:
            result: PA output data window with matching shape.
        """

        if self.paModel is None:
            raise RuntimeError(
                "Process requires a PA bound through paModel or SetPaModel"
            )
        self.ValidateParameters()
        normalizedInput = self.ValidateSignal(inputSignal, "inputSignal")
        periodicProcessor = getattr(
            self.paModel, "ProcessThermalPeriodFloating", None
        )
        if callable(periodicProcessor):
            self.ValidateThermalReferencePlanes()
            normalizedPaOutput = periodicProcessor(
                normalizedInput,
                thermalRunMode=str(
                    self.parameters["thermalRunMode"]
                ).strip().lower(),
                thermalDutyCycle=float(
                    self.parameters["thermalDutyCycle"]
                ),
                steadyStateToleranceC=float(
                    self.parameters["thermalSteadyStateToleranceC"]
                ),
                maximumSteadyStateIterations=int(
                    self.parameters[
                        "maximumThermalSteadyStateIterations"
                    ]
                ),
            )
        else:
            if self.IsThermalModelEnabled():
                raise TypeError(
                    "an enabled thermal PA must expose "
                    "ProcessThermalPeriodFloating for Channel scheduling"
                )
            return self.ProcessBoundPaFloating(normalizedInput)
        normalizedOutput = self.ValidateSignal(
            normalizedPaOutput, "paOutputSignal"
        )
        if normalizedOutput.shape != normalizedInput.shape:
            raise ValueError(
                "bound PA must preserve the input vector or matrix shape"
            )
        return normalizedOutput

    def ResolveCalibrationDriveDbPerChain(
        self,
        driveDbPerChain: Sequence[float],
        chainCount: int,
    ) -> Tuple[float, ...]:
        """Validate one post-DAC calibration drive per waveform chain.

        Processing details:
            Algorithm: Require a nonempty one-dimensional sequence whose length
            matches the current waveform, reject boolean and nonfinite values,
            and return immutable plain floats before exponential conversion.

        Args:
            driveDbPerChain: Chain-ordered analog drive values in decibels.
            chainCount: Number of SISO or MIMO columns being processed.

        Returns:
            result: Validated drive tuple in physical transmit-chain order.
        """

        if (
            not isinstance(chainCount, int)
            or isinstance(chainCount, bool)
            or chainCount < 1
        ):
            raise ValueError("chainCount must be a positive integer")
        if isinstance(driveDbPerChain, (str, bytes)):
            raise TypeError("driveDbPerChain must be a numeric sequence")
        driveArray = np.asarray(driveDbPerChain, dtype=object)
        if driveArray.ndim != 1 or driveArray.size != chainCount:
            raise ValueError(
                "driveDbPerChain must contain one value per waveform chain"
            )
        resolvedValues = []
        for driveValue in driveArray:
            if (
                not isinstance(
                    driveValue,
                    (int, float, np.integer, np.floating),
                )
                or isinstance(driveValue, (bool, np.bool_))
                or not np.isfinite(driveValue)
            ):
                raise ValueError(
                    "every driveDbPerChain entry must be a finite real dB "
                    "value"
                )
            resolvedValues.append(float(driveValue))
        return tuple(resolvedValues)

    def ApplyCalibrationDrive(
        self,
        inputSignal: np.ndarray,
        driveDbPerChain: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Apply committed or explicit analog drive after public decoding.

        Processing details:
            Algorithm: Preserve vector or matrix orientation, resolve one dB
            gain per column, multiply in floating point, and reject numerical
            overflow. An empty committed tuple means unity drive before the
            first successful closed-loop calibration.

        Args:
            inputSignal: Decoded normalized SISO or MIMO transmit waveform.
            driveDbPerChain: Optional explicit trial drives. None selects the
                most recently committed calibration values.

        Returns:
            result: Floating waveform at the output of the simulated analog
                drive stage and before transmitter I/Q impairment.
        """

        normalizedInput = self.ValidateSignal(inputSignal, "inputSignal")
        inputWasVector = normalizedInput.ndim == 1
        inputMatrix = (
            normalizedInput.reshape(-1, 1)
            if inputWasVector
            else normalizedInput
        )
        chainCount = inputMatrix.shape[1]
        selectedDriveValues: Sequence[float]
        if driveDbPerChain is not None:
            selectedDriveValues = driveDbPerChain
        elif self._calibrationDriveDbPerChain:
            selectedDriveValues = self._calibrationDriveDbPerChain
        else:
            selectedDriveValues = tuple(0.0 for _ in range(chainCount))
        resolvedDriveDb = self.ResolveCalibrationDriveDbPerChain(
            selectedDriveValues,
            chainCount,
        )
        if all(driveValue == 0.0 for driveValue in resolvedDriveDb):
            return normalizedInput.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            driveScale = np.power(
                10.0, np.asarray(resolvedDriveDb, dtype=float) / 20.0
            )
            drivenMatrix = inputMatrix * driveScale.reshape(1, -1)
        if not np.all(np.isfinite(drivenMatrix)):
            raise ValueError(
                "driveDbPerChain is outside the numeric amplitude range"
            )
        if inputWasVector:
            return np.asarray(drivenMatrix[:, 0], dtype=np.complex128)
        return np.asarray(drivenMatrix, dtype=np.complex128)

    def SetCalibrationDriveDb(
        self, driveDbPerChain: Sequence[float]
    ) -> None:
        """Commit hidden post-DAC drives after closed-loop convergence.

        Processing details:
            Algorithm: Validate the complete candidate tuple before replacing
            private state. The operation is atomic and is never called by
            ``PowerCalibration`` for an unsuccessful or probe iteration.

        Args:
            driveDbPerChain: One converged analog drive in dB per chain.

        Returns:
            result: None. Later normal processing reuses the accepted drives.
        """

        if isinstance(driveDbPerChain, (str, bytes)):
            raise TypeError("driveDbPerChain must be a numeric sequence")
        driveArray = np.asarray(driveDbPerChain, dtype=object)
        if driveArray.ndim != 1 or driveArray.size == 0:
            raise ValueError(
                "driveDbPerChain must be a nonempty one-dimensional sequence"
            )
        self._calibrationDriveDbPerChain = (
            self.ResolveCalibrationDriveDbPerChain(
                driveDbPerChain,
                int(driveArray.size),
            )
        )

    def ProcessCalibrationDrive(
        self,
        inputSignal: np.ndarray,
        driveDbPerChain: Sequence[float],
    ) -> np.ndarray:
        """Measure a clean PA trial at explicit post-decode analog drives.

        Processing details:
            Algorithm: Decode the legal public waveform once, apply the trial
            drive without changing committed state, then execute Tx I/Q
            impairment, pre-PA coupling, and the nonlinear PA bank. Encode the
            clean per-PA outputs while excluding post-coupling and receiver
            impairments from the power-control reference plane.

        Args:
            inputSignal: Public floating waveform or fixed-point I/Q codes.
            driveDbPerChain: Trial analog drive in dB for every waveform chain.

        Returns:
            result: Public clean PA output before post-PA coupling and sampling.
        """

        interfaceFormat = FixedPoint(self.width)
        normalizedInput = interfaceFormat.DecodeComplex(
            self.ValidateSignal(inputSignal, "inputSignal")
        )
        drivenInput = self.ApplyCalibrationDrive(
            normalizedInput, driveDbPerChain
        )
        transmitterOutput = self.ApplyTransmitterIqImbalance(drivenInput)
        actualPaInput = self.ApplyPrePaCoupling(transmitterOutput)
        self._lastTransmitterOutput = np.array(
            transmitterOutput, dtype=np.complex128, copy=True
        )
        self._lastActualPaInput = np.array(
            actualPaInput, dtype=np.complex128, copy=True
        )
        normalizedPaOutput = self.ProcessBoundPaFloating(actualPaInput)
        return interfaceFormat.EncodeComplex(normalizedPaOutput)

    def ProcessPaBankForCalibration(
        self, inputSignal: np.ndarray
    ) -> np.ndarray:
        """Evaluate Tx I/Q, pre-PA coupling, and PA at the public boundary.

        Processing details:
            Algorithm: Decode the trial waveform once, apply the committed
            post-DAC drive, transmitter I/Q mismatch and PA-input cross-coupling,
            evaluate every nonlinear PA, and encode the clean per-PA outputs.
            Post-PA coupling and receiver effects are excluded so
            ``outputPowerDbm`` continues to mean each physical PA's own output.

        Args:
            inputSignal: Public SISO or samples-by-chains calibration trial.

        Returns:
            result: Public clean PA outputs before post-PA coupling.
        """

        interfaceFormat = FixedPoint(self.width)
        normalizedInput = interfaceFormat.DecodeComplex(
            self.ValidateSignal(inputSignal, "inputSignal")
        )
        drivenInput = self.ApplyCalibrationDrive(normalizedInput)
        transmitterOutput = self.ApplyTransmitterIqImbalance(
            drivenInput
        )
        actualPaInput = self.ApplyPrePaCoupling(transmitterOutput)
        self._lastTransmitterOutput = np.array(
            transmitterOutput, dtype=np.complex128, copy=True
        )
        self._lastActualPaInput = np.array(
            actualPaInput, dtype=np.complex128, copy=True
        )
        normalizedPaOutput = self.ProcessBoundPaFloating(actualPaInput)
        return interfaceFormat.EncodeComplex(normalizedPaOutput)

    def ProcessCoupledPaFloating(
        self, inputSignal: np.ndarray
    ) -> np.ndarray:
        """Evaluate the transmitter, PA bank, and post-PA coupling once.

        Processing details:
            Algorithm: Apply the committed post-DAC calibration drive, Tx I/Q
            mismatch, additive complex FIR coupling before the PA bank, all
            nonlinear branches over one thermal period, and output coupling.
            Stop at the common branch point so forward and feedback outputs
            can be generated without evaluating or heating the PA twice.

        Args:
            inputSignal: Normalized digital Tx input vector or matrix.

        Returns:
            result: Normalized floating waveform at the common receiver
                branch point before common phase and receiver impairments.
        """

        normalizedInput = self.ValidateSignal(
            inputSignal, "inputSignal"
        )
        drivenInput = self.ApplyCalibrationDrive(normalizedInput)
        transmitterOutput = self.ApplyTransmitterIqImbalance(
            drivenInput
        )
        actualPaInput = self.ApplyPrePaCoupling(transmitterOutput)
        self._lastTransmitterOutput = np.array(
            transmitterOutput, dtype=np.complex128, copy=True
        )
        self._lastActualPaInput = np.array(
            actualPaInput, dtype=np.complex128, copy=True
        )
        normalizedPaOutput = self.ProcessBoundPaThermalPeriodFloating(
            actualPaInput
        )
        return self.ApplyPostPaCoupling(normalizedPaOutput)

    def ProcessFloating(
        self, inputSignal: np.ndarray
    ) -> np.ndarray:
        """Evaluate the legacy sampleMode-selected floating receiver path.

        Processing details:
            Algorithm: Evaluate the common transmitter, PA, thermal period,
            and output coupling once, then apply only the path selected by
            ``sampleMode``. New DPD code should use
            ``ProcessOutputPathsFloating`` so it receives both observations.

        Args:
            inputSignal: Normalized digital Tx input vector or matrix.

        Returns:
            result: Legacy single floating output selected by ``sampleMode``.
        """

        coupledPaOutput = self.ProcessCoupledPaFloating(inputSignal)
        return self.ApplyChannelEffects(coupledPaOutput)

    def ProcessOutputPathsFloating(
        self, inputSignal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return forward and feedback observations from one PA evaluation.

        Processing details:
            Algorithm: Run the transmitter, coupled PA bank, and thermal
            period exactly once, then evaluate the forward measurement path.
            In forward mode return an independent copy of that same waveform
            as ``fbOut``. In feedback mode additionally evaluate the complete
            embedded receiver, with an independent noise realization, while
            retaining the common PA memory and temperature state.

        Args:
            inputSignal: Normalized digital Tx input vector or matrix.

        Returns:
            result: ``(chOut, fbOut)`` in normalized floating units. ``fbOut``
                equals ``chOut`` in forward mode and contains the embedded-
                feedback observation in feedback mode.
        """

        coupledPaOutput = self.ProcessCoupledPaFloating(inputSignal)
        channelOutput = self.ApplyForwardChannelEffects(coupledPaOutput)
        if self.sampleMode == "forward":
            return channelOutput, channelOutput.copy()
        feedbackOutput = self.ApplyCompensatedFeedbackChannelEffects(
            coupledPaOutput
        )
        return channelOutput, feedbackOutput

    def ProcessNormalizedOutputPaths(
        self, inputSignal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply public Channel scheduling semantics to normalized ILC samples.

        Processing details:
            Algorithm: Validate the current configuration and keep the ordinary
            normalized dual-output fast path for non-steady thermal operation.
            When periodic steady-state thermal scheduling is active, cross the
            configured public fixed-point boundary, call ``Process`` so its
            cached output-power target is recalibrated for this candidate, then
            decode both outputs. This prevents an ILC amplitude update from
            silently changing the conducted power used for EVM comparison.

        Args:
            inputSignal: Normalized floating SISO or samples-by-chains input.

        Returns:
            result: Normalized ``(chOut, fbOut)`` from one committed live period.
        """

        self.ValidateParameters()
        normalizedInput = self.ValidateSignal(inputSignal, "inputSignal")
        usesSteadyStateThermalMode = (
            self.IsThermalModelEnabled()
            and str(self.parameters["thermalRunMode"]).strip().lower()
            == "steady_state"
        )
        if not usesSteadyStateThermalMode:
            return self.ProcessOutputPathsFloating(normalizedInput)
        interfaceFormat = FixedPoint(self.width)
        publicInput = interfaceFormat.EncodeComplex(normalizedInput)
        publicChannelOutput, publicFeedbackOutput = self.Process(publicInput)
        return (
            interfaceFormat.DecodeComplex(publicChannelOutput),
            interfaceFormat.DecodeComplex(publicFeedbackOutput),
        )

    def Process(
        self,
        inputSignal: np.ndarray,
        outputPowerDbm: Optional[
            Union[float, Sequence[float], np.ndarray]
        ] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return channel and feedback outputs at the public boundary.

        Processing details:
            Algorithm: When ``outputPowerDbm`` is provided, first run the
            private closed loop that selects a legal public waveform and a
            post-decode analog drive before I/Q impairment while observing clean
            PA output at the reference temperature. Restore the suspended
            thermal state, commit only the converged drive, then pass the public
            waveform through the complete live PA and sampling path exactly once
            so temperature drift remains visible. When the target is None,
            preserve the direct one-pass behavior with the most recently
            committed drive, as required by iterative algorithms. When an
            enabled PA uses the default steady-state thermal mode, a missing
            target reuses the most recent successful target and repeats the
            reference-temperature calibration; the first such call therefore
            requires an explicit target. The accepted waveform is then
            evaluated on the periodic steady-state temperature curve exactly
            once. Forward mode copies ``chOut`` into ``fbOut``; feedback mode
            evaluates the embedded receiver for ``fbOut``. DPD/ILC uses the
            selected ``fbOut`` observation and final RF metrics use ``chOut``.

        Args:
            inputSignal: Public digital Tx vector or samples-by-chains matrix.
            outputPowerDbm: Optional shared target dBm or per-chain sequence.

        Returns:
            result: ``(chOut, fbOut)`` with matching input shape and public
                floating or integer-code convention for both arrays.
        """

        self.ValidateParameters()
        if (
            self.sampleMode == "fb"
            and str(self.parameters["fbIqCompensationMode"])
            .strip()
            .lower()
            == "filter"
        ):
            # Fail before power calibration or a live PA/thermal evaluation so
            # an absent or stale inverse cannot consume a transmission period.
            self.RequireCurrentFeedbackIqCalibration()
        # Reject cross-module reference-plane mismatches before a closed-loop
        # calibration can commit a new target or analog drive.
        self.ValidateThermalReferencePlanes()
        processingInput = inputSignal
        resolvedOutputPowerDbm = outputPowerDbm
        usesSteadyStateThermalMode = (
            self.IsThermalModelEnabled()
            and str(self.parameters["thermalRunMode"]).strip().lower()
            == "steady_state"
        )
        if resolvedOutputPowerDbm is None and usesSteadyStateThermalMode:
            if self._lastCalibrationOutputPowerDbm is None:
                raise ValueError(
                    "the first steady-state thermal Channel.Process call "
                    "requires outputPowerDbm so every steady-state period "
                    "can repeat reference-temperature power calibration"
                )
            resolvedOutputPowerDbm = self._lastCalibrationOutputPowerDbm
        if resolvedOutputPowerDbm is not None:
            processingInput = self.CalibratePaInput(
                inputSignal,
                resolvedOutputPowerDbm,
            )
            # CalibratePaInput has already restored the PA's original thermal
            # state. Re-evaluate the accepted drive once so this public call
            # represents a real temperature-aware transmission rather than the
            # cold calibration observation cached by PowerCalibration.
        interfaceFormat = FixedPoint(self.width)
        normalizedInput = interfaceFormat.DecodeComplex(
            self.ValidateSignal(processingInput, "inputSignal")
        )
        normalizedChannelOutput, normalizedFeedbackOutput = (
            self.ProcessOutputPathsFloating(normalizedInput)
        )
        return (
            interfaceFormat.EncodeComplex(normalizedChannelOutput),
            interfaceFormat.EncodeComplex(normalizedFeedbackOutput),
        )

    def SmallSignalGain(self) -> complex:
        """Return the deterministic direct small-signal sampling-path gain.

        Processing details:
            Algorithm: Multiply the committed SISO analog drive, the Tx direct
            I/Q FIR response at DC, bound PA small-signal gain, and common
            phase. In feedback mode also multiply by the feedback direct I/Q
            FIR response at DC and ordinary linear path. Image and DC terms are
            excluded. This scalar cannot describe off-DC frequency selectivity.

        Returns:
            result: Complex direct small-signal gain for the selected mode.
        """

        if self.paModel is None:
            raise RuntimeError(
                "SmallSignalGain requires a bound PA model"
            )
        smallSignalGainMethod = getattr(
            self.paModel, "SmallSignalGain", None
        )
        if not callable(smallSignalGainMethod):
            raise TypeError(
                "bound paModel must expose SmallSignalGain"
            )
        self.ValidateParameters()
        phaseRadians = np.deg2rad(
            float(cast(float, self.parameters["phaseDegrees"]))
        )
        transmitterDirectFirTaps, _ = self.TransmitterIqFilterTaps()
        transmitterDirectDcResponse = np.sum(transmitterDirectFirTaps)
        if len(self._calibrationDriveDbPerChain) > 1:
            raise ValueError(
                "SmallSignalGain is scalar and cannot represent multiple "
                "calibrated channel drives"
            )
        calibrationDriveDb = (
            0.0
            if not self._calibrationDriveDbPerChain
            else self._calibrationDriveDbPerChain[0]
        )
        calibrationDriveScale = np.power(
            10.0, calibrationDriveDb / 20.0
        )
        selectedPathGain = complex(
            calibrationDriveScale
            * transmitterDirectDcResponse
            * smallSignalGainMethod()
            * np.exp(1j * phaseRadians)
        )
        if self.sampleMode == "fb":
            selectedPathGain *= self.FeedbackDirectSmallSignalGain()
        return complex(selectedPathGain)
