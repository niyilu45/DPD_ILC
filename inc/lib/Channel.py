"""Model the receive link between a nonlinear PA and a measurement input.

The current channel is intentionally compact: it evaluates a bound PA,
rotates the PA output by one supported constant phase, and optionally adds
circular complex white Gaussian noise. Public fixed-point boundaries use raw
integer I/Q codes while every physical operation remains floating point.
"""

from collections import ChainMap
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple, cast

import numpy as np

# Cross-package imports support canonical ``inc.lib`` and compatibility
# ``lib`` package entry points without relying on a parent that may not exist.
if __package__ and "." in __package__:
    from ..utils.ConfigUtils import (
        FilterRecognizedParameters,
        RecognizedParameterView,
    )
    from ..utils.FixedPoint import FixedPoint
else:
    from utils.ConfigUtils import (
        FilterRecognizedParameters,
        RecognizedParameterView,
    )
    from utils.FixedPoint import FixedPoint


class Channel:
    """Apply PA processing, constant phase rotation, and optional AWGN.

    ``noiseAmpMv`` is the RMS magnitude of the complete complex noise
    envelope, not the RMS of each individual I or Q component. Therefore the
    two real Gaussian components each use ``noiseAmpMv / sqrt(2)`` RMS.
    Normalized PA output RMS equal to one represents
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
                "loadResistanceOhm": 50.0,
                "maximumOutputPowerDbm": 25.0,
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
            enforce mutual exclusion of the two noise controls; check all
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
        if noiseAmpMv is not None and noisePwrDbm is not None:
            raise ValueError(
                "noiseAmpMv and noisePwrDbm cannot both be non-None"
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
            configured resistive port. Two None values resolve to zero noise.

        Returns:
            result: Nonnegative complex-envelope RMS noise voltage.
        """

        self.ValidateParameters()
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
            Algorithm: Resolve the total complex-envelope RMS, divide it by
            square root two for independent I and Q Gaussian components,
            draw an independent sample for every array element, and add it
            after phase rotation. Zero configured RMS returns an exact copy.

        Args:
            inputSignal: Normalized floating complex samples after rotation.

        Returns:
            result: Normalized samples including the configured white noise.
        """

        complexInput = self.ValidateSignal(inputSignal, "inputSignal")
        self.SynchronizeRandomGenerator()
        noiseRmsNormalized = self.ResolveNoiseRmsNormalized()
        if noiseRmsNormalized == 0.0:
            return complexInput.copy()
        componentScale = noiseRmsNormalized / np.sqrt(2.0)
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

    def Process(self, inputSignal: np.ndarray) -> np.ndarray:
        """Evaluate the complete PA-to-receiver path at the public boundary.

        Processing details:
            Algorithm: Decode public PA-input codes to normalized values,
            evaluate the bound PA, phase rotation, and white noise entirely
            in floating point, then encode the receiver waveform back to raw
            integer I/Q codes when ``width`` is positive.

        Args:
            inputSignal: Public PA input vector or samples-by-chains matrix.

        Returns:
            result: Public receiver waveform with matching shape and type.
        """

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
