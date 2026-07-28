"""Model the receive link between a nonlinear PA and a measurement input.

The current channel is intentionally compact: it evaluates a bound PA,
rotates the PA output by one supported constant phase, and optionally adds
circular complex white Gaussian noise. Public fixed-point boundaries use raw
integer I/Q codes while every physical operation remains floating point.
"""

from collections import ChainMap
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
        FilterRecognizedParameters,
        RecognizedParameterView,
    )
    from ..utils.FixedPoint import FixedPoint
    from ..utils.SigProc import PowerCalibration
else:
    from utils.ConfigUtils import (
        FilterRecognizedParameters,
        RecognizedParameterView,
    )
    from utils.FixedPoint import FixedPoint
    from utils.SigProc import PowerCalibration


class Channel:
    """Apply PA processing, constant phase rotation, and optional AWGN.

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
                Unsupported names produce a warning and are ignored.

        Returns:
            result: None. The configured channel is ready for processing.
        """

        self.defaultParameters: Mapping[str, object] = MappingProxyType(
            {
                "phaseDegrees": 0,
                "noiseAmpMv": None,
                "noisePwrDbm": None,
                "noiseSnrDb": None,
                "loadResistanceOhm": 50.0,
                "maximumOutputPowerDbm": 25.0,
                "calibrationToleranceDb": 0.25,
                "maximumCalibrationIterations": 60,
                "calibrationLearningRate": 0.8,
                "maximumDriveAdjustmentDb": 6.0,
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
        externalParameters: Mapping[str, object] = (
            {}
            if parameters is None
            else RecognizedParameterView(
                parameters,
                self.defaultParameters,
                "Channel",
            )
        )
        recognizedOverrides = FilterRecognizedParameters(
            directOverrides,
            self.defaultParameters,
            "Channel",
        )
        self.parameters: ChainMap[str, object] = ChainMap(
            recognizedOverrides,
            externalParameters,
            self.defaultParameters,
        )
        self.paModel: Optional[Any] = None
        self._paProcessMethod: Optional[Any] = None
        self._powerCalibration: Optional[PowerCalibration] = None
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

    def GetParameters(self) -> Dict[str, object]:
        """Return one flattened snapshot of all effective channel settings.

        Processing details:
            Algorithm: Resolve the ChainMap precedence into a new dictionary
            without exposing or mutating any internal configuration layer.

        Returns:
            result: Ordinary dictionary containing every supported setting.
        """

        return dict(self.parameters)

    def UpdateParameters(self, **parameterOverrides: object) -> None:
        """Apply validated high-priority channel overrides transactionally.

        Processing details:
            Algorithm: Filter unsupported keys with a warning, update the
            local ChainMap layer, validate the complete resolved state, and
            restore the prior state when any recognized value is invalid.

        Args:
            parameterOverrides: Supported values to replace locally.

        Returns:
            result: None. Valid updates affect subsequent channel calls.
        """

        recognizedOverrides = FilterRecognizedParameters(
            parameterOverrides,
            self.defaultParameters,
            "Channel.UpdateParameters",
        )
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

    def ValidateParameters(self) -> None:
        """Validate phase, noise, physical scaling, seed, and interface width.

        Processing details:
            Algorithm: Restrict phase to minus 90, zero, or plus 90 degrees;
            enforce mutual exclusion of the three noise controls; check all
            physical scalars for finite values and valid domains; then use
            ``FixedPoint`` as the authoritative width validator.

        Returns:
            result: None. Invalid recognized settings raise an exception.
        """

        phaseDegrees = self.parameters["phaseDegrees"]
        if (
            not isinstance(phaseDegrees, (int, float))
            or isinstance(phaseDegrees, bool)
            or not np.isfinite(phaseDegrees)
            or float(phaseDegrees) not in (-90.0, 0.0, 90.0)
        ):
            raise ValueError(
                "phaseDegrees must be exactly -90, 0, or 90"
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
                "noiseAmpMv, noisePwrDbm, and noiseSnrDb are mutually "
                "exclusive"
            )
        if noiseAmpMv is not None and (
            not isinstance(noiseAmpMv, (int, float))
            or isinstance(noiseAmpMv, bool)
            or not np.isfinite(noiseAmpMv)
            or float(noiseAmpMv) < 0.0
        ):
            raise ValueError(
                "noiseAmpMv must be finite, nonnegative, or None"
            )
        if noisePwrDbm is not None and (
            not isinstance(noisePwrDbm, (int, float))
            or isinstance(noisePwrDbm, bool)
            or not np.isfinite(noisePwrDbm)
        ):
            raise ValueError("noisePwrDbm must be finite or None")
        if noiseSnrDb is not None and (
            not isinstance(noiseSnrDb, (int, float))
            or isinstance(noiseSnrDb, bool)
            or not np.isfinite(noiseSnrDb)
        ):
            raise ValueError("noiseSnrDb must be finite or None")

        loadResistanceOhm = self.parameters["loadResistanceOhm"]
        if (
            not isinstance(loadResistanceOhm, (int, float))
            or isinstance(loadResistanceOhm, bool)
            or not np.isfinite(loadResistanceOhm)
            or float(loadResistanceOhm) <= 0.0
        ):
            raise ValueError(
                "loadResistanceOhm must be finite and positive"
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
                "maximumOutputPowerDbm must be finite"
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
                "calibrationToleranceDb must be finite and positive"
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
                "maximumCalibrationIterations must be a positive integer"
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
                "calibrationLearningRate must be in the interval (0, 1]"
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
                "maximumDriveAdjustmentDb must be finite and positive"
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
                "activePowerThresholdDb must be finite and negative"
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
                "activeGapToleranceSamples must be a nonnegative integer"
            )

        randomSeed = self.parameters["randomSeed"]
        if randomSeed is not None and (
            not isinstance(randomSeed, int)
            or isinstance(randomSeed, bool)
            or int(randomSeed) < 0
        ):
            raise ValueError(
                "randomSeed must be a nonnegative integer or None"
            )
        FixedPoint(self.width)

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
                raise ValueError("outputPowerDbm must be finite")
            return scalarTarget, None
        if isinstance(outputPowerDbm, (str, bytes)):
            raise TypeError(
                "outputPowerDbm must be a real scalar or numeric sequence"
            )
        targetArray = np.asarray(outputPowerDbm, dtype=object)
        if targetArray.ndim != 1 or targetArray.size == 0:
            raise ValueError(
                "outputPowerDbm sequence must be nonempty and "
                "one-dimensional"
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
                    "every outputPowerDbm target must be a real number"
                )
            floatingTarget = float(targetValue)
            if not np.isfinite(floatingTarget):
                raise ValueError(
                    "every outputPowerDbm target must be finite"
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
            Channel power detector and convergence settings into a private
            ``PowerCalibration`` instance, and reuse that instance so its
            converged drive preset can accelerate later requests. Rebuild the
            helper only after a PA replacement or public-width change.

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
            or self._powerCalibration.paModel is not self.paModel
            or self._powerCalibration.width != self.width
        ):
            self._powerCalibration = PowerCalibration(
                paModel=self.paModel,
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
        """Generate a hidden-drive PA input that meets requested output dBm.

        Processing details:
            Algorithm: Configure the private closed loop, normalize only the
            active part of the caller's arbitrary waveform, repeatedly send a
            newly scaled input through the bound PA, measure actual active PA
            output power, and update the input preset until every chain falls
            inside the configured tolerance. Padding and long idle intervals
            are excluded by the active-region detector.

        Args:
            inputSignal: Arbitrarily scaled public SISO or MIMO waveform.
            outputPowerDbm: Shared target dBm or one target per PA chain.

        Returns:
            result: Public PA-input waveform accepted by the closed loop.
        """

        powerCalibration = self.ConfigurePowerCalibration(outputPowerDbm)
        return powerCalibration.Calibrate(inputSignal)

    def GetLastPaInput(self) -> np.ndarray:
        """Return the most recent internally calibrated public PA input.

        Processing details:
            Algorithm: Delegate to the private calibration helper and return
            its defensive copy without exposing the hidden dB drive preset.

        Returns:
            result: Last converged waveform sent to the bound PA.
        """

        if self._powerCalibration is None:
            raise RuntimeError(
                "calibrated Process must run before GetLastPaInput"
            )
        return self._powerCalibration.GetLastPaInput()

    def GetLastPaOutput(self) -> np.ndarray:
        """Return the clean PA output measured by the last calibration.

        Processing details:
            Algorithm: Return the cached converged PA observation before
            phase rotation and receiver noise, avoiding another PA evaluation.

        Returns:
            result: Last clean public PA output waveform.
        """

        if self._powerCalibration is None:
            raise RuntimeError(
                "calibrated Process must run before GetLastPaOutput"
            )
        return self._powerCalibration.GetLastPaOutput()

    def GetLastCalibrationMetrics(self) -> Dict[str, object]:
        """Return the latest target, measured power, error, and iteration data.

        Processing details:
            Algorithm: Delegate to the private calibration helper while
            preserving its ordinary dictionary result and hidden drive state.

        Returns:
            result: Dictionary describing the converged PA power loop.
        """

        if self._powerCalibration is None:
            raise RuntimeError(
                "calibrated Process must run before "
                "GetLastCalibrationMetrics"
            )
        return self._powerCalibration.GetLastCalibrationMetrics()

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

    def ApplyChannelEffects(
        self, paOutputSignal: np.ndarray
    ) -> np.ndarray:
        """Apply phase rotation followed by receiver white noise.

        Processing details:
            Algorithm: Validate the live configuration, rotate the normalized
            PA output first, then add AWGN so the implemented order exactly
            matches ``PA -> phase -> AddNoise``.

        Args:
            paOutputSignal: Normalized floating PA output samples.

        Returns:
            result: Normalized floating receiver-input samples.
        """

        self.ValidateParameters()
        phaseRotatedSignal = self.ApplyPhaseRotation(paOutputSignal)
        return self.AddNoise(phaseRotatedSignal)

    def ProcessPaOutput(
        self, paOutputSignal: np.ndarray
    ) -> np.ndarray:
        """Apply channel effects to an already evaluated public PA output.

        Processing details:
            Algorithm: Decode public integer I/Q codes once, execute phase
            rotation and noise in normalized floating units, and encode the
            receiver signal back to the same public interface convention.

        Args:
            paOutputSignal: Public PA output vector or matrix.

        Returns:
            result: Public receiver waveform with matching shape and width.
        """

        interfaceFormat = FixedPoint(self.width)
        normalizedPaOutput = interfaceFormat.DecodeComplex(
            self.ValidateSignal(paOutputSignal, "paOutputSignal")
        )
        normalizedReceiverSignal = self.ApplyChannelEffects(
            normalizedPaOutput
        )
        return interfaceFormat.EncodeComplex(normalizedReceiverSignal)

    def ProcessFloating(
        self, inputSignal: np.ndarray
    ) -> np.ndarray:
        """Evaluate the bound PA and channel in normalized floating units.

        Processing details:
            Algorithm: Prefer the PA's direct ``ProcessFloating`` entry. For
            a fixed-only third-party PA, encode and decode around its public
            width exactly once. Apply phase rotation and AWGN only after the
            PA output has returned to normalized floating units.

        Args:
            inputSignal: Normalized floating PA input vector or matrix.

        Returns:
            result: Normalized floating receiver-input waveform.
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
        return self.ApplyChannelEffects(normalizedPaOutput)

    def Process(
        self,
        inputSignal: np.ndarray,
        outputPowerDbm: Optional[
            Union[float, Sequence[float], np.ndarray]
        ] = None,
    ) -> np.ndarray:
        """Evaluate the complete PA-to-receiver path at the public boundary.

        Processing details:
            Algorithm: When ``outputPowerDbm`` is provided, first run the
            private closed loop that repeatedly adjusts only the PA input and
            observes clean PA output until its active-region power converges.
            Apply phase rotation and receiver noise exactly once to the cached
            accepted PA output. When the target is None, preserve the direct
            one-pass PA-to-receiver behavior required by iterative algorithms.

        Args:
            inputSignal: Public PA input vector or samples-by-chains matrix.
            outputPowerDbm: Optional shared target dBm or per-chain sequence.

        Returns:
            result: Public receiver waveform with matching shape and type.
        """

        if outputPowerDbm is not None:
            self.CalibratePaInput(inputSignal, outputPowerDbm)
            return self.ProcessPaOutput(self.GetLastPaOutput())
        interfaceFormat = FixedPoint(self.width)
        normalizedInput = interfaceFormat.DecodeComplex(
            self.ValidateSignal(inputSignal, "inputSignal")
        )
        normalizedOutput = self.ProcessFloating(normalizedInput)
        return interfaceFormat.EncodeComplex(normalizedOutput)

    def SmallSignalGain(self) -> complex:
        """Return the deterministic small-signal gain through phase rotation.

        Processing details:
            Algorithm: Query the bound PA's small-signal complex gain and
            multiply it by the configured unit-magnitude phase factor. AWGN
            has zero mean and therefore does not contribute deterministic
            small-signal gain.

        Returns:
            result: Complex PA-plus-phase small-signal gain.
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
        return complex(
            smallSignalGainMethod() * np.exp(1j * phaseRadians)
        )
