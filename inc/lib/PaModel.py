"""Power-amplifier behavioral models used by the DPD-ILC simulation.

Callers construct ``PaModel`` with ``modelName="rapp"``, ``"wiener"``,
``"gmp"``, or ``"doherty"`` and then call ``Process``. Four nonlinear model
families are
provided internally:

* ``RappPA`` applies the classic memoryless solid-state PA AM-AM curve and
  preserves input phase, providing a deliberate zero-memory reference model.
* ``WienerPA`` applies a linear memory filter followed by a smooth Rapp
  AM-AM characteristic and a saturating AM-PM characteristic.
* ``GMPPA`` implements the generalized memory polynomial main, lagging,
  and leading cross terms described in the project theory document.
* ``DohertyPA`` combines independently configurable carrier and peaking
  behavioral branches with envelope-dependent peaking turn-on, branch delay,
  complex combining, and simplified load modulation.

``PaModel`` accepts one complex stream. ``MimoPaModel`` owns one independent
``PaModel`` per transmit chain, accepts a samples-by-chains matrix, and applies
independent input drive and output-power calibration on every chain. Optional
``ThermalConfig`` and ``ThermalNetwork`` objects add power- and duty-cycle-
driven junction-temperature state without changing the electrical families.
"""

from collections import ChainMap
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union, cast

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


@dataclass(frozen=True)
class ThermalConfig:
    """Configure PA self-heating and temperature-dependent electrical drift.

    The thermal network is driven by estimated dissipated power. The public
    RF waveform remains normalized, while ``referenceOutputPowerDbm`` maps a
    normalized output power of one to physical watts for the heat calculation.
    """

    enabled: bool = False
    modelName: str = "foster"
    sampleRateHz: float = 80.0e6
    ambientTemperatureC: float = 25.0
    initialJunctionTemperatureC: float = 25.0
    referenceTemperatureC: float = 25.0
    thermalResistancesCPerW: Tuple[float, ...] = (2.0, 8.0, 20.0)
    thermalTimeConstantsSec: Tuple[float, ...] = (50.0e-6, 5.0e-3, 0.5)
    thermalUpdateIntervalSamples: int = 256
    idleDissipatedPowerW: float = 0.15
    efficiencyModelName: str = "power_dependent"
    peakDrainEfficiency: float = 0.45
    minimumDrainEfficiency: float = 0.10
    efficiencyKneeOutputPowerDbm: float = 15.0
    referenceOutputPowerDbm: float = 25.0
    activePowerThresholdDb: float = -60.0
    gainTemperatureCoefficientDbPerC: float = -0.012
    phaseTemperatureCoefficientDegreesPerC: float = 0.03
    saturationTemperatureCoefficientPerC: float = -0.0015
    nonlinearityTemperatureCoefficientPerC: float = 0.0020
    maximumJunctionTemperatureC: float = 150.0

    @classmethod
    def Recommended(
        cls,
        modelName: str,
        sampleRateHz: float = 80.0e6,
        **parameterOverrides: object,
    ) -> "ThermalConfig":
        """Build one complete, valid thermal-model starting profile.

        Processing details:
            Algorithm: Select model-specific resistance and time-constant
            vectors for a static temperature point, one-RC trend model, or
            three-branch Foster model; combine them with a common 25 dBm-class
            demonstration heat-source and electrical-drift profile; apply
            explicit caller overrides; construct the immutable configuration;
            and validate every field before returning it. These values are
            simulation starting points rather than device specifications.

        Args:
            modelName: ``"static"``, ``"single_rc"``, or ``"foster"``.
            sampleRateHz: Actual public waveform sample rate in hertz.
            parameterOverrides: Optional explicit replacements for any
                ThermalConfig dataclass field other than the selected model
                name and sample rate.

        Returns:
            result: Validated enabled ThermalConfig with model-specific values.
        """

        if not isinstance(modelName, str):
            raise TypeError("modelName must be a string")
        normalizedModelName = modelName.strip().lower()
        modelProfiles: Mapping[str, Mapping[str, object]] = {
            "static": {
                "initialJunctionTemperatureC": 55.0,
                "thermalResistancesCPerW": (1.0,),
                "thermalTimeConstantsSec": (1.0,),
            },
            "single_rc": {
                "initialJunctionTemperatureC": 25.0,
                "thermalResistancesCPerW": (20.0,),
                "thermalTimeConstantsSec": (20.0e-3,),
            },
            "foster": {
                "initialJunctionTemperatureC": 25.0,
                "thermalResistancesCPerW": (2.0, 8.0, 20.0),
                "thermalTimeConstantsSec": (50.0e-6, 5.0e-3, 0.5),
            },
        }
        if normalizedModelName not in modelProfiles:
            raise ValueError(
                "modelName has an invalid value. Allowed values: "
                "'static', 'single_rc', or 'foster'."
            )
        supportedNames = tuple(cls.__dataclass_fields__)
        unknownNames = tuple(
            parameterName
            for parameterName in parameterOverrides
            if parameterName not in supportedNames
        )
        if unknownNames:
            raise TypeError(
                "unknown ThermalConfig parameter(s): "
                + ", ".join(sorted(unknownNames))
                + ". Supported parameters: "
                + ", ".join(supportedNames)
            )
        commonParameters: Dict[str, object] = {
            "enabled": True,
            "modelName": normalizedModelName,
            "sampleRateHz": sampleRateHz,
            "ambientTemperatureC": 25.0,
            "referenceTemperatureC": 25.0,
            "thermalUpdateIntervalSamples": 256,
            "idleDissipatedPowerW": 0.15,
            "efficiencyModelName": "power_dependent",
            "peakDrainEfficiency": 0.45,
            "minimumDrainEfficiency": 0.10,
            "efficiencyKneeOutputPowerDbm": 15.0,
            "referenceOutputPowerDbm": 25.0,
            "activePowerThresholdDb": -60.0,
            "gainTemperatureCoefficientDbPerC": -0.012,
            "phaseTemperatureCoefficientDegreesPerC": 0.03,
            "saturationTemperatureCoefficientPerC": -0.0015,
            "nonlinearityTemperatureCoefficientPerC": 0.0020,
            "maximumJunctionTemperatureC": 150.0,
        }
        resolvedParameters = {
            **commonParameters,
            **modelProfiles[normalizedModelName],
            **parameterOverrides,
        }
        recommendedConfig = cls(**resolvedParameters)
        recommendedConfig.Validate()
        return recommendedConfig

    def Validate(self) -> None:
        """Validate thermal topology, physical units, and drift coefficients.

        Processing details:
            Algorithm: Normalize the selected thermal and efficiency model
            names conceptually, verify matching positive Foster vectors,
            require finite temperatures, powers, coefficients, and limits,
            and reject efficiencies outside the physical open interval.

        Returns:
            result: None. Invalid thermal settings raise a descriptive error.
        """

        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean")
        if not isinstance(self.modelName, str):
            raise TypeError("modelName must be a string")
        normalizedModelName = self.modelName.strip().lower()
        if normalizedModelName not in ("static", "single_rc", "foster"):
            raise ValueError(
                "thermal modelName must be 'static', 'single_rc', or 'foster'"
            )
        if not isinstance(self.efficiencyModelName, str):
            raise TypeError("efficiencyModelName must be a string")
        normalizedEfficiencyName = self.efficiencyModelName.strip().lower()
        if normalizedEfficiencyName not in ("constant", "power_dependent"):
            raise ValueError(
                "efficiencyModelName must be 'constant' or 'power_dependent'"
            )
        resistanceValues = tuple(self.thermalResistancesCPerW)
        timeConstantValues = tuple(self.thermalTimeConstantsSec)
        if len(resistanceValues) == 0 or len(resistanceValues) != len(
            timeConstantValues
        ):
            raise ValueError(
                "thermal resistance and time-constant vectors must have "
                "matching nonzero lengths"
            )
        for parameterName, values in (
            ("thermalResistancesCPerW", resistanceValues),
            ("thermalTimeConstantsSec", timeConstantValues),
        ):
            if any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isfinite(value)
                or float(value) <= 0.0
                for value in values
            ):
                raise ValueError(
                    f"{parameterName} entries must be finite and positive"
                )
        if normalizedModelName == "single_rc" and len(resistanceValues) != 1:
            raise ValueError(
                "single_rc requires exactly one thermal resistance and time constant"
            )
        finiteParameters = {
            "sampleRateHz": self.sampleRateHz,
            "ambientTemperatureC": self.ambientTemperatureC,
            "initialJunctionTemperatureC": self.initialJunctionTemperatureC,
            "referenceTemperatureC": self.referenceTemperatureC,
            "idleDissipatedPowerW": self.idleDissipatedPowerW,
            "efficiencyKneeOutputPowerDbm": self.efficiencyKneeOutputPowerDbm,
            "referenceOutputPowerDbm": self.referenceOutputPowerDbm,
            "activePowerThresholdDb": self.activePowerThresholdDb,
            "gainTemperatureCoefficientDbPerC": (
                self.gainTemperatureCoefficientDbPerC
            ),
            "phaseTemperatureCoefficientDegreesPerC": (
                self.phaseTemperatureCoefficientDegreesPerC
            ),
            "saturationTemperatureCoefficientPerC": (
                self.saturationTemperatureCoefficientPerC
            ),
            "nonlinearityTemperatureCoefficientPerC": (
                self.nonlinearityTemperatureCoefficientPerC
            ),
            "maximumJunctionTemperatureC": self.maximumJunctionTemperatureC,
        }
        for parameterName, parameterValue in finiteParameters.items():
            if (
                not isinstance(parameterValue, (int, float))
                or isinstance(parameterValue, bool)
                or not np.isfinite(parameterValue)
            ):
                raise ValueError(f"{parameterName} must be finite")
        if float(self.sampleRateHz) <= 0.0:
            raise ValueError("sampleRateHz must be positive")
        if (
            not isinstance(self.thermalUpdateIntervalSamples, int)
            or isinstance(self.thermalUpdateIntervalSamples, bool)
            or self.thermalUpdateIntervalSamples < 1
        ):
            raise ValueError(
                "thermalUpdateIntervalSamples must be a positive integer"
            )
        if float(self.idleDissipatedPowerW) < 0.0:
            raise ValueError("idleDissipatedPowerW cannot be negative")
        if float(self.activePowerThresholdDb) > 0.0:
            raise ValueError("activePowerThresholdDb cannot exceed zero dB")
        for parameterName, efficiencyValue in (
            ("peakDrainEfficiency", self.peakDrainEfficiency),
            ("minimumDrainEfficiency", self.minimumDrainEfficiency),
        ):
            if (
                not isinstance(efficiencyValue, (int, float))
                or isinstance(efficiencyValue, bool)
                or not np.isfinite(efficiencyValue)
                or not 0.0 < float(efficiencyValue) < 1.0
            ):
                raise ValueError(f"{parameterName} must be between zero and one")
        if float(self.minimumDrainEfficiency) > float(
            self.peakDrainEfficiency
        ):
            raise ValueError(
                "minimumDrainEfficiency cannot exceed peakDrainEfficiency"
            )
        if float(self.maximumJunctionTemperatureC) <= float(
            self.ambientTemperatureC
        ):
            raise ValueError(
                "maximumJunctionTemperatureC must exceed ambient temperature"
            )
        if float(self.initialJunctionTemperatureC) > float(
            self.maximumJunctionTemperatureC
        ):
            raise ValueError(
                "initialJunctionTemperatureC cannot exceed the maximum limit"
            )


class ThermalNetwork:
    """Maintain the causal thermal state of a static, single-RC, or Foster model."""

    def __init__(self, config: ThermalConfig) -> None:
        """Create the requested thermal topology and initialize its state.

        Processing details:
            Algorithm: Validate the immutable configuration, select all Foster
            branches or the one requested RC branch, and distribute the initial
            junction-to-ambient temperature rise in proportion to resistance.

        Args:
            config: Validated thermal-network and temperature-drift settings.

        Returns:
            result: None. Thermal nodes are ready for causal time advancement.
        """

        config.Validate()
        self.config = config
        self.ambientTemperatureC = float(config.ambientTemperatureC)
        self.elapsedTimeSec = 0.0
        self.temperatureRisePerBranchC = np.zeros(
            len(self.ResolveBranches()[0]), dtype=float
        )
        self.Reset()

    def ResolveBranches(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the active resistance and time-constant vectors.

        Processing details:
            Algorithm: Use one branch for ``single_rc``, retain every supplied
            branch for ``foster``, and keep the vectors available for static
            initialization even though static mode does not advance them.

        Returns:
            result: Two float arrays containing C/W and second values.
        """

        resistanceValues = np.asarray(
            self.config.thermalResistancesCPerW, dtype=float
        )
        timeConstantValues = np.asarray(
            self.config.thermalTimeConstantsSec, dtype=float
        )
        if self.config.modelName.strip().lower() == "single_rc":
            return resistanceValues[:1], timeConstantValues[:1]
        return resistanceValues, timeConstantValues

    def Reset(
        self,
        junctionTemperatureC: Optional[float] = None,
        ambientTemperatureC: Optional[float] = None,
    ) -> None:
        """Reset elapsed time and establish a new physical starting condition.

        Processing details:
            Algorithm: Resolve optional finite ambient and junction values,
            distribute the initial temperature rise over active branches in
            proportion to their thermal resistances, and clear elapsed time.

        Args:
            junctionTemperatureC: Optional initial junction temperature.
            ambientTemperatureC: Optional ambient or cold-plate temperature.

        Returns:
            result: None. The next waveform begins from the requested state.
        """

        resolvedAmbient = (
            float(self.config.ambientTemperatureC)
            if ambientTemperatureC is None
            else float(ambientTemperatureC)
        )
        resolvedJunction = (
            float(self.config.initialJunctionTemperatureC)
            if junctionTemperatureC is None
            else float(junctionTemperatureC)
        )
        if not np.isfinite(resolvedAmbient) or not np.isfinite(
            resolvedJunction
        ):
            raise ValueError("thermal reset temperatures must be finite")
        if resolvedJunction > float(
            self.config.maximumJunctionTemperatureC
        ):
            raise ValueError(
                "thermal reset junction temperature exceeds the maximum limit"
            )
        resistanceValues, _ = self.ResolveBranches()
        resistanceSum = float(np.sum(resistanceValues))
        initialRise = resolvedJunction - resolvedAmbient
        self.ambientTemperatureC = resolvedAmbient
        self.temperatureRisePerBranchC = (
            initialRise * resistanceValues / resistanceSum
        )
        self.elapsedTimeSec = 0.0

    def CurrentTemperatureC(self) -> float:
        """Return the present junction temperature in degrees Celsius.

        Processing details:
            Algorithm: Add every causal branch temperature rise to the current
            ambient reference without advancing time or changing any state.

        Returns:
            result: Scalar junction temperature in degrees Celsius.
        """

        return float(
            self.ambientTemperatureC
            + np.sum(self.temperatureRisePerBranchC)
        )

    def Advance(self, dissipatedPowerW: float, durationSec: float) -> float:
        """Advance every thermal branch for a constant mean dissipated power.

        Processing details:
            Algorithm: Apply the exact zero-order-hold solution of each Foster
            RC differential equation. Static mode advances time but preserves
            the explicitly selected junction temperature.

        Args:
            dissipatedPowerW: Nonnegative mean heat input during the interval.
            durationSec: Nonnegative physical interval duration in seconds.

        Returns:
            result: Junction temperature after the interval in Celsius.
        """

        resolvedPower = float(dissipatedPowerW)
        resolvedDuration = float(durationSec)
        if (
            not np.isfinite(resolvedPower)
            or resolvedPower < 0.0
            or not np.isfinite(resolvedDuration)
            or resolvedDuration < 0.0
        ):
            raise ValueError("thermal power and duration must be finite and nonnegative")
        if self.config.modelName.strip().lower() != "static":
            resistanceValues, timeConstantValues = self.ResolveBranches()
            decayValues = np.exp(-resolvedDuration / timeConstantValues)
            self.temperatureRisePerBranchC = (
                decayValues * self.temperatureRisePerBranchC
                + resistanceValues
                * resolvedPower
                * (1.0 - decayValues)
            )
        self.elapsedTimeSec += resolvedDuration
        return self.CurrentTemperatureC()

    def GetMetrics(self) -> Dict[str, object]:
        """Return a defensive dictionary describing the current thermal state.

        Processing details:
            Algorithm: Copy branch rises and expose topology, ambient,
            junction temperature, and elapsed physical time without mutation.

        Returns:
            result: Ordinary dictionary suitable for logs and result files.
        """

        return {
            "modelName": self.config.modelName.strip().lower(),
            "ambientTemperatureC": float(self.ambientTemperatureC),
            "junctionTemperatureC": self.CurrentTemperatureC(),
            "temperatureRisePerBranchC": tuple(
                float(value) for value in self.temperatureRisePerBranchC
            ),
            "elapsedTimeSec": float(self.elapsedTimeSec),
        }


@dataclass(frozen=True)
class RappConfig:
    """Configure the classic memoryless Rapp solid-state PA model."""

    linearGain: float = 1.0
    saturationAmplitude: float = 1.0
    rappSmoothness: float = 3.0

    def Validate(self) -> None:
        """Validate the memoryless Rapp AM-AM parameters.

        Processing details:
            Algorithm: Require every scalar to be a finite real number and
            require positive gain, saturation amplitude, and smoothness so
            the AM-AM curve remains monotonic, bounded, and well defined.

        Returns:
            result: None. Invalid settings raise a descriptive exception.
        """

        for parameterName, parameterValue in (
            ("linearGain", self.linearGain),
            ("saturationAmplitude", self.saturationAmplitude),
            ("rappSmoothness", self.rappSmoothness),
        ):
            if (
                not isinstance(parameterValue, (int, float))
                or isinstance(parameterValue, bool)
                or not np.isfinite(parameterValue)
                or float(parameterValue) <= 0.0
            ):
                raise ValueError(
                    f"{parameterName} must be finite and positive"
                )


class RappPA:
    """Implement a phase-preserving memoryless solid-state PA model."""

    def __init__(self, config: RappConfig = RappConfig()) -> None:
        """Initialize the classic Rapp model from validated settings.

        Processing details:
            Algorithm: Validate and retain the immutable configuration. No
            delay line, filter state, or envelope state is created because
            every output sample depends only on the same-index input sample.

        Args:
            config: Memoryless AM-AM gain, saturation, and smoothness values.

        Returns:
            result: None. The model is ready for independent sample mapping.
        """

        config.Validate()
        self.config = config

    def Process(self, inputSignal: np.ndarray) -> np.ndarray:
        """Apply the memoryless Rapp AM-AM law without AM-PM rotation.

        Processing details:
            Algorithm: Convert each complex sample to magnitude and phase,
            evaluate the Rapp compression denominator independently at that
            same sample, and restore the unchanged input phase. A logarithmic
            denominator calculation prevents avoidable overflow for large
            finite floating-point inputs while preserving zero exactly.

        Args:
            inputSignal: One-dimensional normalized complex input waveform.

        Returns:
            result: Same-length complex output with no delay or stored state.
        """

        complexInput = AsComplexVector(inputSignal)
        inputMagnitude = np.abs(complexInput)
        outputMagnitude = np.zeros(inputMagnitude.shape, dtype=float)
        positiveMask = inputMagnitude > 0.0
        if np.any(positiveMask):
            positiveMagnitude = inputMagnitude[positiveMask]
            logarithmicRatio = np.log(
                positiveMagnitude / float(self.config.saturationAmplitude)
            )
            twiceSmoothness = 2.0 * float(
                self.config.rappSmoothness
            )
            logarithmicDenominator = np.logaddexp(
                0.0,
                twiceSmoothness * logarithmicRatio,
            ) / twiceSmoothness
            logarithmicOutput = (
                np.log(float(self.config.linearGain))
                + np.log(positiveMagnitude)
                - logarithmicDenominator
            )
            outputMagnitude[positiveMask] = np.exp(logarithmicOutput)

        # The original Rapp SSPA model has zero AM-PM conversion, so the
        # output phase is exactly the input phase for every nonzero sample.
        outputSignal = outputMagnitude * np.exp(1j * np.angle(complexInput))
        if not np.all(np.isfinite(outputSignal)):
            raise ValueError("Rapp PA output exceeded the numeric range")
        return np.asarray(outputSignal, dtype=np.complex128)

    def SmallSignalGain(self) -> complex:
        """Return the real small-signal gain of the memoryless Rapp curve.

        Processing details:
            Algorithm: Use the zero-amplitude limit of the Rapp denominator,
            which equals one and therefore leaves ``linearGain`` unchanged.

        Returns:
            result: Positive real gain represented as a complex scalar.
        """

        return complex(float(self.config.linearGain), 0.0)


@dataclass(frozen=True)
class WienerConfig:
    """Configure the linear-memory and memoryless-nonlinearity cascade."""

    linearTaps: Tuple[complex, ...] = (
        1.0 + 0.0j,
        0.055 - 0.025j,
        -0.018 + 0.012j,
    )
    linearGain: float = 1.0
    saturationAmplitude: float = 1.0
    rappSmoothness: float = 3.0
    ampmCoefficient: float = 0.18

    def Validate(self) -> None:
        """Reject nonphysical settings before processing a waveform.

        Processing details:
            Algorithm: Evaluate every documented constraint in deterministic order and stop at the first invalid condition without changing valid state.

        Returns:
            result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
        """

        if len(self.linearTaps) == 0:
            raise ValueError("linearTaps must contain at least one coefficient")
        if self.linearGain <= 0.0:
            raise ValueError("linearGain must be positive")
        if self.saturationAmplitude <= 0.0:
            raise ValueError("saturationAmplitude must be positive")
        if self.rappSmoothness <= 0.0:
            raise ValueError("rappSmoothness must be positive")


class WienerPA:
    """Model a PA as an FIR memory filter followed by AM-AM and AM-PM curves."""

    def __init__(self, config: WienerConfig = WienerConfig()) -> None:
        """Initialize the Wiener PA from validated memory and nonlinearity settings.

        Processing details:
            Algorithm: Carry out the described operation using validated inputs, explicit array-shape handling, and deterministic project conventions.

        Args:
            config: Validated configuration object controlling this operation.

        Returns:
            result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
        """
        config.Validate()
        self.config = config
        self.linearTaps = np.asarray(config.linearTaps, dtype=np.complex128)

    def Process(self, inputSignal: np.ndarray) -> np.ndarray:
        """Pass a complex waveform through the nonlinear Wiener model.

        The Rapp equation used for the output magnitude is

        ``Aout = G*Ain / (1 + (Ain/Asat)^(2p))^(1/(2p))``.

        A bounded quadratic phase term represents AM-PM conversion. The
        causal FIR stage makes the model frequency selective and gives the
        ILC algorithm a genuine memory effect to compensate.
        """

        complexInput = AsComplexVector(inputSignal)
        filteredSignal = np.convolve(
            complexInput, self.linearTaps, mode="full"
        )[: complexInput.size]

        inputMagnitude = np.abs(filteredSignal)
        normalizedMagnitude = inputMagnitude / self.config.saturationAmplitude
        smoothness = self.config.rappSmoothness
        compressionDenominator = (
            1.0 + normalizedMagnitude ** (2.0 * smoothness)
        ) ** (1.0 / (2.0 * smoothness))
        outputMagnitude = (
            self.config.linearGain
            * inputMagnitude
            / compressionDenominator
        )

        # The rational form is small around the origin and approaches a
        # bounded phase rotation under heavy compression.
        phaseRotation = (
            self.config.ampmCoefficient
            * normalizedMagnitude**2
            / (1.0 + normalizedMagnitude**2)
        )
        inputPhase = np.angle(filteredSignal)
        return outputMagnitude * np.exp(1j * (inputPhase + phaseRotation))

    def SmallSignalGain(self) -> complex:
        """Return the DC small-signal gain of the linear Wiener cascade.

        Processing details:
            Algorithm: Carry out the described operation using validated inputs, explicit array-shape handling, and deterministic project conventions.

        Returns:
            result: complex. The computed value described by the summary, with documented units, shape, and normalization.
        """

        return self.config.linearGain * np.sum(self.linearTaps)


@dataclass(frozen=True)
class GMPConfig:
    """Configure a generalized memory-polynomial PA model.

    Coefficient dictionaries use ``(order, memoryIndex)`` for main terms and
    ``(order, memoryIndex, crossIndex)`` for lagging/leading terms. Missing
    entries are treated as zero. When no dictionaries are supplied, a stable
    compressive model with memory is generated automatically.
    """

    nonlinearOrders: Tuple[int, ...] = (1, 3, 5, 7)
    memoryDepth: int = 3
    crossMemoryDepth: int = 2
    mainCoefficients: Optional[Mapping[Tuple[int, int], complex]] = None
    laggingCoefficients: Optional[Mapping[Tuple[int, int, int], complex]] = None
    leadingCoefficients: Optional[Mapping[Tuple[int, int, int], complex]] = None

    def Validate(self) -> None:
        """Validate order and memory dimensions used by the GMP expansion.

        Processing details:
            Algorithm: Evaluate every documented constraint in deterministic order and stop at the first invalid condition without changing valid state.

        Returns:
            result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
        """

        if len(self.nonlinearOrders) == 0:
            raise ValueError("nonlinearOrders cannot be empty")
        if any(order < 1 or order % 2 == 0 for order in self.nonlinearOrders):
            raise ValueError("nonlinearOrders must contain positive odd integers")
        if self.memoryDepth < 1:
            raise ValueError("memoryDepth must be positive")
        if self.crossMemoryDepth < 0:
            raise ValueError("crossMemoryDepth cannot be negative")


class GMPPA:
    """Implement a complex-baseband generalized memory polynomial PA."""

    def __init__(self, config: GMPConfig = GMPConfig()) -> None:
        """Initialize GMP coefficients from validated order and memory settings.

        Processing details:
            Algorithm: Carry out the described operation using validated inputs, explicit array-shape handling, and deterministic project conventions.

        Args:
            config: Validated configuration object controlling this operation.

        Returns:
            result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
        """
        config.Validate()
        self.config = config
        defaultMain, defaultLagging, defaultLeading = DefaultGmpCoefficients(
            config.nonlinearOrders,
            config.memoryDepth,
            config.crossMemoryDepth,
        )
        self.mainCoefficients = dict(
            defaultMain
            if config.mainCoefficients is None
            else config.mainCoefficients
        )
        self.laggingCoefficients = dict(
            defaultLagging
            if config.laggingCoefficients is None
            else config.laggingCoefficients
        )
        self.leadingCoefficients = dict(
            defaultLeading
            if config.leadingCoefficients is None
            else config.leadingCoefficients
        )

    def Process(self, inputSignal: np.ndarray) -> np.ndarray:
        """Evaluate the main, lagging, and leading GMP basis expansions.

        Processing details:
            Algorithm: Execute the configured signal-processing path, preserve sample alignment, and return the complete downstream result.

        Args:
            inputSignal: One-dimensional complex baseband samples supplied to the operation.

        Returns:
            result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
        """

        complexInput = AsComplexVector(inputSignal)
        outputSignal = np.zeros_like(complexInput)

        # Main branch: x[n-m] * |x[n-m]|^(p-1).
        for (nonlinearOrder, memoryIndex), coefficient in self.mainCoefficients.items():
            delayedSignal = DelaySignal(complexInput, memoryIndex)
            outputSignal += (
                coefficient
                * delayedSignal
                * np.abs(delayedSignal) ** (nonlinearOrder - 1)
            )

        # Lagging envelope branch:
        # x[n-m] * |x[n-m-l]|^(p-1).
        for (
            nonlinearOrder,
            memoryIndex,
            crossIndex,
        ), coefficient in self.laggingCoefficients.items():
            carrierSignal = DelaySignal(complexInput, memoryIndex)
            envelopeSignal = DelaySignal(
                complexInput, memoryIndex + crossIndex
            )
            outputSignal += (
                coefficient
                * carrierSignal
                * np.abs(envelopeSignal) ** (nonlinearOrder - 1)
            )

        # Leading envelope branch:
        # x[n-m-l] * |x[n-m]|^(p-1).
        for (
            nonlinearOrder,
            memoryIndex,
            crossIndex,
        ), coefficient in self.leadingCoefficients.items():
            carrierSignal = DelaySignal(
                complexInput, memoryIndex + crossIndex
            )
            envelopeSignal = DelaySignal(complexInput, memoryIndex)
            outputSignal += (
                coefficient
                * carrierSignal
                * np.abs(envelopeSignal) ** (nonlinearOrder - 1)
            )
        return outputSignal

    def SmallSignalGain(self) -> complex:
        """Return the DC gain contributed by all first-order main terms.

        Processing details:
            Algorithm: Carry out the described operation using validated inputs, explicit array-shape handling, and deterministic project conventions.

        Returns:
            result: complex. The computed value described by the summary, with documented units, shape, and normalization.
        """

        return sum(
            coefficient
            for (nonlinearOrder, _), coefficient in self.mainCoefficients.items()
            if nonlinearOrder == 1
        )


@dataclass(frozen=True)
class DohertyConfig:
    """Configure a behavioral carrier-plus-peaking Doherty architecture."""

    carrierModelName: str = "wiener"
    peakingModelName: str = "wiener"
    carrierWienerConfig: Optional[WienerConfig] = None
    carrierGmpConfig: Optional[GMPConfig] = None
    peakingWienerConfig: Optional[WienerConfig] = None
    peakingGmpConfig: Optional[GMPConfig] = None
    carrierInputGain: float = 1.0
    peakingInputGain: float = 1.0
    peakingTurnOnAmplitude: float = 0.45
    peakingTransitionWidth: float = 0.15
    carrierCombineCoefficient: complex = 1.0 + 0.0j
    peakingCombineCoefficient: complex = 0.50 + 0.0j
    peakingDelaySamples: int = 0
    loadModulationStrength: float = 0.10

    def Validate(self) -> None:
        """Validate both branch families and all Doherty physical controls.

        Processing details:
            Algorithm: Normalize each branch name, require Wiener or GMP,
            type-check optional branch configurations, and reject nonfinite,
            nonpositive, or unsupported envelope, delay, gain, combining, and
            load-modulation settings before either PA branch is constructed.

        Returns:
            result: None. Invalid architecture settings raise an exception.
        """

        for branchName, modelName in (
            ("carrierModelName", self.carrierModelName),
            ("peakingModelName", self.peakingModelName),
        ):
            if not isinstance(modelName, str):
                raise TypeError(f"{branchName} must be a string")
            if modelName.strip().lower() not in ("wiener", "gmp"):
                raise ValueError(
                    f"{branchName} must be either 'wiener' or 'gmp'"
                )
        for configName, branchConfig, expectedType in (
            (
                "carrierWienerConfig",
                self.carrierWienerConfig,
                WienerConfig,
            ),
            ("carrierGmpConfig", self.carrierGmpConfig, GMPConfig),
            (
                "peakingWienerConfig",
                self.peakingWienerConfig,
                WienerConfig,
            ),
            ("peakingGmpConfig", self.peakingGmpConfig, GMPConfig),
        ):
            if branchConfig is not None and not isinstance(
                branchConfig, expectedType
            ):
                raise TypeError(
                    f"{configName} must be a {expectedType.__name__} "
                    "or None"
                )
        for gainName, gainValue in (
            ("carrierInputGain", self.carrierInputGain),
            ("peakingInputGain", self.peakingInputGain),
        ):
            if (
                not isinstance(gainValue, (int, float))
                or isinstance(gainValue, bool)
                or not np.isfinite(gainValue)
                or float(gainValue) <= 0.0
            ):
                raise ValueError(f"{gainName} must be finite and positive")
        if (
            not isinstance(self.peakingTurnOnAmplitude, (int, float))
            or isinstance(self.peakingTurnOnAmplitude, bool)
            or not np.isfinite(self.peakingTurnOnAmplitude)
            or float(self.peakingTurnOnAmplitude) <= 0.0
        ):
            raise ValueError(
                "peakingTurnOnAmplitude must be finite and positive"
            )
        if (
            not isinstance(self.peakingTransitionWidth, (int, float))
            or isinstance(self.peakingTransitionWidth, bool)
            or not np.isfinite(self.peakingTransitionWidth)
            or float(self.peakingTransitionWidth) <= 0.0
        ):
            raise ValueError(
                "peakingTransitionWidth must be finite and positive"
            )
        for coefficientName, coefficientValue in (
            (
                "carrierCombineCoefficient",
                self.carrierCombineCoefficient,
            ),
            (
                "peakingCombineCoefficient",
                self.peakingCombineCoefficient,
            ),
        ):
            if not isinstance(
                coefficientValue, (int, float, complex)
            ) or isinstance(coefficientValue, bool):
                raise TypeError(
                    f"{coefficientName} must be a finite complex scalar"
                )
            complexCoefficient = complex(coefficientValue)
            if not (
                np.isfinite(complexCoefficient.real)
                and np.isfinite(complexCoefficient.imag)
            ):
                raise ValueError(f"{coefficientName} must be finite")
        if (
            abs(complex(self.carrierCombineCoefficient))
            <= np.finfo(float).tiny
        ):
            raise ValueError(
                "carrierCombineCoefficient cannot be zero"
            )
        if (
            not isinstance(self.peakingDelaySamples, int)
            or isinstance(self.peakingDelaySamples, bool)
            or self.peakingDelaySamples < 0
        ):
            raise ValueError(
                "peakingDelaySamples must be a nonnegative integer"
            )
        if (
            not isinstance(self.loadModulationStrength, (int, float))
            or isinstance(self.loadModulationStrength, bool)
            or not np.isfinite(self.loadModulationStrength)
            or float(self.loadModulationStrength) < 0.0
        ):
            raise ValueError(
                "loadModulationStrength must be finite and nonnegative"
            )


class DohertyPA:
    """Model a behavioral Doherty carrier and envelope-gated peaking PA."""

    def __init__(
        self, config: DohertyConfig = DohertyConfig()
    ) -> None:
        """Construct independently configurable carrier and peaking branches.

        Processing details:
            Algorithm: Validate the architecture, construct a Wiener or GMP
            behavioral model for each branch, and retain the immutable
            envelope-gating, combining, branch-delay, and load-modulation
            controls used by every waveform evaluation.

        Args:
            config: Doherty architecture and branch-model configuration.

        Returns:
            result: None. Both nonlinear branches are ready for processing.
        """

        config.Validate()
        self.config = config
        self.carrierModel = self.BuildBranchModel(
            config.carrierModelName,
            config.carrierWienerConfig,
            config.carrierGmpConfig,
        )
        self.peakingModel = self.BuildBranchModel(
            config.peakingModelName,
            config.peakingWienerConfig,
            config.peakingGmpConfig,
        )

    @staticmethod
    def BuildBranchModel(
        modelName: str,
        wienerConfig: Optional[WienerConfig],
        gmpConfig: Optional[GMPConfig],
    ) -> Any:
        """Construct one validated Wiener or GMP Doherty branch.

        Processing details:
            Algorithm: Normalize the already validated family name and select
            the matching branch configuration, falling back to that family's
            immutable built-in defaults when the optional object is None.

        Args:
            modelName: Branch family name, either Wiener or GMP.
            wienerConfig: Optional Wiener settings for this branch.
            gmpConfig: Optional GMP settings for this branch.

        Returns:
            result: WienerPA or GMPPA object exposing Process and gain methods.
        """

        if modelName.strip().lower() == "wiener":
            return WienerPA(
                WienerConfig() if wienerConfig is None else wienerConfig
            )
        return GMPPA(GMPConfig() if gmpConfig is None else gmpConfig)

    def PeakingActivation(
        self, inputMagnitude: np.ndarray
    ) -> np.ndarray:
        """Return the smooth envelope-dependent peaking turn-on factor.

        Processing details:
            Algorithm: Map magnitudes below ``peakingTurnOnAmplitude`` to
            zero, transition over ``peakingTransitionWidth``, clamp to one,
            and apply the cubic smooth-step polynomial so branch derivatives
            remain continuous at both transition endpoints.

        Args:
            inputMagnitude: Nonnegative input-envelope magnitude array.

        Returns:
            result: Real activation array in the closed interval zero to one.
        """

        transitionPosition = (
            np.asarray(inputMagnitude, dtype=float)
            - float(self.config.peakingTurnOnAmplitude)
        ) / float(self.config.peakingTransitionWidth)
        clippedPosition = np.clip(transitionPosition, 0.0, 1.0)
        return (
            clippedPosition**2
            * (3.0 - 2.0 * clippedPosition)
        )

    def Process(self, inputSignal: np.ndarray) -> np.ndarray:
        """Evaluate carrier, peaking turn-on, load modulation, and combining.

        Processing details:
            Algorithm: Drive the carrier continuously, gate the peaking input
            with the smooth envelope activation, evaluate both nonlinear
            behavioral branches, delay the peaking output, multiply the
            carrier by an activation-dependent load-modulation factor, and
            coherently combine both complex outputs without changing length.

        Args:
            inputSignal: One-dimensional normalized complex input waveform.

        Returns:
            result: Combined normalized Doherty PA output waveform.
        """

        complexInput = AsComplexVector(inputSignal)
        activation = self.PeakingActivation(np.abs(complexInput))
        carrierInput = (
            float(self.config.carrierInputGain) * complexInput
        )
        peakingInput = (
            float(self.config.peakingInputGain)
            * activation
            * complexInput
        )
        carrierOutput = self.carrierModel.Process(carrierInput)
        peakingOutput = DelaySignal(
            self.peakingModel.Process(peakingInput),
            self.config.peakingDelaySamples,
        )
        carrierLoadFactor = (
            1.0
            + float(self.config.loadModulationStrength) * activation
        )
        combinedOutput = (
            complex(self.config.carrierCombineCoefficient)
            * carrierLoadFactor
            * carrierOutput
            + complex(self.config.peakingCombineCoefficient)
            * peakingOutput
        )
        if not np.all(np.isfinite(combinedOutput)):
            raise ValueError(
                "Doherty branch combination exceeded the numeric range"
            )
        return np.asarray(combinedOutput, dtype=np.complex128)

    def SmallSignalGain(self) -> complex:
        """Return the low-power gain while the peaking branch is disabled.

        Processing details:
            Algorithm: At the origin the peaking activation and simplified
            load modulation are zero, so multiply only the carrier branch's
            small-signal gain by its input and complex combining coefficients.

        Returns:
            result: Complex carrier-path small-signal gain.
        """

        return complex(
            complex(self.config.carrierCombineCoefficient)
            * float(self.config.carrierInputGain)
            * self.carrierModel.SmallSignalGain()
        )


class PaModel:
    """Configure and operate one Rapp, Wiener, GMP, or Doherty PA model.

    The facade gives every caller the same object-oriented construction and
    processing interface while retaining the dedicated model implementations.

    Example:
        ``paModel = PaModel(modelName="wiener")``
        ``outputSignal = paModel.Process(inputSignal)``
    """

    def __init__(
        self,
        modelName: Optional[str] = None,
        rappConfig: Optional[RappConfig] = None,
        wienerConfig: Optional[WienerConfig] = None,
        gmpConfig: Optional[GMPConfig] = None,
        dohertyConfig: Optional[DohertyConfig] = None,
        thermalConfig: Optional[ThermalConfig] = None,
        parameters: Optional[Mapping[str, object]] = None,
        width: Optional[int] = None,
        **parameterOverrides: object,
    ) -> None:
        """Initialize the PA facade and select its active model family.

        Processing details:
            Algorithm: Define immutable PA defaults inside this constructor,
            then layer direct arguments and the caller-owned mapping ahead of
            them so callers provide only values they intend to override.

        Args:
            modelName: Selected PA model family name.
            rappConfig: Optional memoryless Rapp configuration.
            wienerConfig: Optional Wiener configuration; None selects built-in values.
            gmpConfig: Optional GMP configuration; None selects built-in values.
            dohertyConfig: Optional carrier/peaking Doherty configuration.
            thermalConfig: Optional self-heating and temperature-drift model.
            parameters: Optional external mapping layered ahead of the built-in defaults.
            width: Optional external I/Q width. None selects the internal
                16-bit default, zero selects floating point, and a positive
                value selects signed integer I/Q codes in complex128.
            parameterOverrides: Additional keyword settings. Unsupported names
                produce a warning and are ignored.

        Returns:
            result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
        """
        self.defaultParameters: Mapping[str, object] = MappingProxyType(
            {
                "modelName": "wiener",
                "rappConfig": None,
                "wienerConfig": None,
                "gmpConfig": None,
                "dohertyConfig": None,
                "thermalConfig": None,
                "width": 16,
            }
        )
        directOverrides = dict(parameterOverrides)
        if modelName is not None:
            directOverrides["modelName"] = modelName
        if rappConfig is not None:
            directOverrides["rappConfig"] = rappConfig
        if wienerConfig is not None:
            directOverrides["wienerConfig"] = wienerConfig
        if gmpConfig is not None:
            directOverrides["gmpConfig"] = gmpConfig
        if dohertyConfig is not None:
            directOverrides["dohertyConfig"] = dohertyConfig
        if thermalConfig is not None:
            directOverrides["thermalConfig"] = thermalConfig
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
                "PaModel",
            )
        )
        recognizedOverrides = FilterRecognizedParameters(
            directOverrides,
            self.defaultParameters,
            "PaModel",
        )
        self.parameters: ChainMap[str, object] = ChainMap(
            recognizedOverrides,
            externalParameters,
            self.defaultParameters,
        )
        self.model = None
        self.thermalNetwork: Optional[ThermalNetwork] = None
        self._activeThermalConfig: Optional[ThermalConfig] = None
        self._externalTemperatureOffsetC = 0.0
        self._lastThermalMetrics: Dict[str, object] = {}
        self._activeConfiguration: Optional[
            Tuple[
                str,
                Optional[RappConfig],
                Optional[WienerConfig],
                Optional[GMPConfig],
                Optional[DohertyConfig],
            ]
        ] = None
        self.SynchronizeModel()
        self.SynchronizeThermalModel()

    @property
    def ModelName(self) -> str:
        """Return the normalized model name resolved by the ChainMap.

        Processing details:
            Algorithm: Carry out the described operation using validated inputs, explicit array-shape handling, and deterministic project conventions.

        Returns:
            result: str. The computed value described by the summary, with documented units, shape, and normalization.
        """

        normalizedName, _, _, _, _ = self.ResolveConfiguration()
        return normalizedName

    modelName = ModelName

    @property
    def Width(self) -> int:
        """Return the external I/Q component width.

        Processing details:
            Algorithm: Resolve the current ChainMap value so updates affect
            subsequent input and output boundary quantization.

        Returns:
            result: Zero for floating mode or a positive fixed-point width.
        """

        return cast(int, self.parameters["width"])

    width = Width

    def GetParameters(self) -> Dict[str, object]:
        """Return a flattened snapshot of all resolved PA parameters.

        Processing details:
            Algorithm: Resolve values according to state and ChainMap precedence, keeping caller-owned configuration behavior explicit.

        Returns:
            result: Dict[str, object]. The computed value described by the summary, with documented units, shape, and normalization.
        """

        return dict(self.parameters)

    def UpdateParameters(self, **parameterOverrides: object) -> None:
        """Apply validated high-priority PA configuration overrides.

        Processing details:
            Algorithm: Resolve values according to state and ChainMap precedence, keeping caller-owned configuration behavior explicit.

        Args:
            parameterOverrides: Highest-priority keyword values applied to the local ChainMap layer.

        Returns:
            result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
        """

        recognizedOverrides = FilterRecognizedParameters(
            parameterOverrides,
            self.defaultParameters,
            "PaModel.UpdateParameters",
        )
        previousOverrides = dict(self.parameters.maps[0])
        self.parameters.maps[0].update(recognizedOverrides)
        try:
            self.SynchronizeModel()
            self.SynchronizeThermalModel()
        except (TypeError, ValueError):
            self.parameters.maps[0].clear()
            self.parameters.maps[0].update(previousOverrides)
            self.SynchronizeModel()
            self.SynchronizeThermalModel()
            raise

    def ResolveThermalConfig(self) -> Optional[ThermalConfig]:
        """Validate and return the optional resolved thermal configuration.

        Processing details:
            Algorithm: Read the active ChainMap value, require ``ThermalConfig``
            or None, and validate every recognized physical field before use.

        Returns:
            result: Validated immutable thermal configuration or None.
        """

        rawThermalConfig = self.parameters["thermalConfig"]
        if rawThermalConfig is None:
            return None
        if not isinstance(rawThermalConfig, ThermalConfig):
            raise TypeError("thermalConfig must be a ThermalConfig or None")
        rawThermalConfig.Validate()
        return rawThermalConfig

    def SynchronizeThermalModel(self) -> None:
        """Create or update thermal state after a configuration change.

        Processing details:
            Algorithm: Disable the network for None or disabled configuration,
            preserve state while the immutable configuration is unchanged, and
            initialize a new network only when the caller changes its settings.

        Returns:
            result: None. Thermal state matches the active parameter mapping.
        """

        thermalConfig = self.ResolveThermalConfig()
        if thermalConfig is None or not thermalConfig.enabled:
            self.thermalNetwork = None
            self._activeThermalConfig = thermalConfig
            self._lastThermalMetrics = {}
            return
        if thermalConfig == self._activeThermalConfig:
            return
        self.thermalNetwork = ThermalNetwork(thermalConfig)
        self._activeThermalConfig = thermalConfig
        self._lastThermalMetrics = self.thermalNetwork.GetMetrics()

    def SuspendThermalModel(self) -> Optional[Dict[str, object]]:
        """Suspend self-heating while preserving an exact restorable snapshot.

        Processing details:
            Algorithm: Copy the active configuration, branch temperatures,
            ambient value, elapsed time, and latest diagnostics, then disable
            only the network reference so calibration evaluates the electrical
            model at its fixed reference temperature without producing heat.

        Returns:
            result: Snapshot dictionary, or None when thermal modeling is off.
        """

        self.SynchronizeThermalModel()
        if self.thermalNetwork is None:
            return None
        snapshot = {
            "thermalConfig": self._activeThermalConfig,
            "temperatureRisePerBranchC": np.array(
                self.thermalNetwork.temperatureRisePerBranchC,
                dtype=float,
                copy=True,
            ),
            "ambientTemperatureC": float(
                self.thermalNetwork.ambientTemperatureC
            ),
            "elapsedTimeSec": float(self.thermalNetwork.elapsedTimeSec),
            "lastThermalMetrics": dict(self._lastThermalMetrics),
            "externalTemperatureOffsetC": float(
                self._externalTemperatureOffsetC
            ),
        }
        self.thermalNetwork = None
        return snapshot

    def RestoreThermalModel(
        self,
        thermalSnapshot: Optional[Mapping[str, object]],
    ) -> None:
        """Restore thermal state saved before temperature-independent work.

        Processing details:
            Algorithm: Treat None as a no-op, reconstruct the validated active
            network, restore branch temperatures, ambient and elapsed time,
            and recover the latest public diagnostics without advancing heat.

        Args:
            thermalSnapshot: Snapshot returned by ``SuspendThermalModel``.

        Returns:
            result: None. Thermal testing resumes from the original state.
        """

        if thermalSnapshot is None:
            return
        thermalConfig = thermalSnapshot.get("thermalConfig")
        if not isinstance(thermalConfig, ThermalConfig):
            raise TypeError("thermal snapshot contains an invalid configuration")
        restoredNetwork = ThermalNetwork(thermalConfig)
        restoredRises = np.asarray(
            thermalSnapshot["temperatureRisePerBranchC"], dtype=float
        )
        if restoredRises.shape != restoredNetwork.temperatureRisePerBranchC.shape:
            raise ValueError("thermal snapshot branch count is incompatible")
        restoredNetwork.temperatureRisePerBranchC = restoredRises.copy()
        restoredNetwork.ambientTemperatureC = float(
            thermalSnapshot["ambientTemperatureC"]
        )
        restoredNetwork.elapsedTimeSec = float(
            thermalSnapshot["elapsedTimeSec"]
        )
        self.thermalNetwork = restoredNetwork
        self._activeThermalConfig = thermalConfig
        self._lastThermalMetrics = dict(
            cast(Mapping[str, object], thermalSnapshot["lastThermalMetrics"])
        )
        self._externalTemperatureOffsetC = float(
            thermalSnapshot.get("externalTemperatureOffsetC", 0.0)
        )

    def ResolveConfiguration(
        self,
    ) -> Tuple[
        str,
        Optional[RappConfig],
        Optional[WienerConfig],
        Optional[GMPConfig],
        Optional[DohertyConfig],
    ]:
        """Validate and return the currently resolved PA configuration.

        Processing details:
            Algorithm: Resolve values according to state and ChainMap precedence, keeping caller-owned configuration behavior explicit.

        Returns:
            result: Model name followed by optional Rapp, Wiener, GMP, and
                Doherty configurations in deterministic order.
        """

        rawModelName = self.parameters["modelName"]
        if not isinstance(rawModelName, str):
            raise TypeError("modelName must be a string")
        normalizedName = rawModelName.strip().lower()
        if normalizedName not in ("rapp", "wiener", "gmp", "doherty"):
            raise ValueError(
                "modelName must be 'rapp', 'wiener', 'gmp', or 'doherty'"
            )

        rawRappConfig = self.parameters["rappConfig"]
        if rawRappConfig is not None and not isinstance(
            rawRappConfig, RappConfig
        ):
            raise TypeError("rappConfig must be a RappConfig or None")
        rawWienerConfig = self.parameters["wienerConfig"]
        if rawWienerConfig is not None and not isinstance(
            rawWienerConfig, WienerConfig
        ):
            raise TypeError("wienerConfig must be a WienerConfig or None")
        rawGmpConfig = self.parameters["gmpConfig"]
        if rawGmpConfig is not None and not isinstance(
            rawGmpConfig, GMPConfig
        ):
            raise TypeError("gmpConfig must be a GMPConfig or None")
        rawDohertyConfig = self.parameters["dohertyConfig"]
        if rawDohertyConfig is not None and not isinstance(
            rawDohertyConfig, DohertyConfig
        ):
            raise TypeError(
                "dohertyConfig must be a DohertyConfig or None"
            )
        FixedPoint(self.width)
        self.ResolveThermalConfig()
        return (
            normalizedName,
            cast(Optional[RappConfig], rawRappConfig),
            cast(Optional[WienerConfig], rawWienerConfig),
            cast(Optional[GMPConfig], rawGmpConfig),
            cast(Optional[DohertyConfig], rawDohertyConfig),
        )

    def SynchronizeModel(self) -> None:
        """Rebuild the PA when a live external parameter mapping changes.

        Processing details:
            Algorithm: Resolve values according to state and ChainMap precedence, keeping caller-owned configuration behavior explicit.

        Returns:
            result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
        """

        selectedConfiguration = self.ResolveConfiguration()
        if selectedConfiguration == self._activeConfiguration:
            return
        (
            normalizedName,
            rappConfig,
            wienerConfig,
            gmpConfig,
            dohertyConfig,
        ) = selectedConfiguration
        if normalizedName == "rapp":
            selectedModel = RappPA(
                RappConfig() if rappConfig is None else rappConfig
            )
        elif normalizedName == "wiener":
            selectedModel = WienerPA(
                WienerConfig() if wienerConfig is None else wienerConfig
            )
        elif normalizedName == "gmp":
            selectedModel = GMPPA(
                GMPConfig() if gmpConfig is None else gmpConfig
            )
        else:
            selectedModel = DohertyPA(
                DohertyConfig()
                if dohertyConfig is None
                else dohertyConfig
            )
        self.model = selectedModel
        self._activeConfiguration = selectedConfiguration

    def Process(self, inputSignal: np.ndarray) -> np.ndarray:
        """Pass a complex waveform through the configured PA model.

        Processing details:
            Algorithm: Round and saturate the public fixed-point I/Q codes,
            decode them to a normalized floating envelope, evaluate the PA,
            and encode the floating result back to integer-valued public codes.

        Args:
            inputSignal: One-dimensional complex baseband samples supplied to the operation.

        Returns:
            result: Complex128 samples containing raw I/Q codes in fixed mode
                or physical floating samples when ``width`` equals zero.
        """

        self.SynchronizeModel()
        self.SynchronizeThermalModel()
        interfaceFormat = FixedPoint(self.width)
        floatingInput = interfaceFormat.DecodeComplex(inputSignal)
        floatingOutput = self.ProcessFloating(floatingInput)
        return interfaceFormat.EncodeComplex(floatingOutput)

    def ProcessFloating(self, inputSignal: np.ndarray) -> np.ndarray:
        """Evaluate the PA directly in its normalized floating-point domain.

        Processing details:
            Algorithm: Validate finite normalized complex samples and pass
            them to the active Rapp, Wiener, GMP, or Doherty calculation without
            applying the public fixed-point encoding. This method is used by
            internal algorithms such as ILC after they have crossed the public
            boundary.

        Args:
            inputSignal: Normalized physical complex samples of any shape.

        Returns:
            result: Floating complex PA output with the same shape.
        """

        self.SynchronizeModel()
        self.SynchronizeThermalModel()
        complexInput = np.asarray(inputSignal, dtype=np.complex128)
        if complexInput.size == 0 or not np.all(np.isfinite(complexInput)):
            raise ValueError("inputSignal must contain finite samples")
        if self.thermalNetwork is None:
            return np.asarray(
                self.model.Process(complexInput), dtype=np.complex128
            )
        return self.ProcessThermalFloating(complexInput)

    def ProcessAtTemperatureFloating(
        self,
        inputSignal: np.ndarray,
        junctionTemperatureC: float,
    ) -> np.ndarray:
        """Evaluate the electrical model at one fixed junction temperature.

        Processing details:
            Algorithm: Run the unchanged Rapp, Wiener, GMP, or Doherty base model,
            then apply temperature-dependent complex gain, saturation-like
            envelope scaling, and nonlinear phase without advancing heat.

        Args:
            inputSignal: Normalized finite complex waveform segment.
            junctionTemperatureC: Fixed junction temperature in Celsius.

        Returns:
            result: Temperature-modified complex output segment.
        """

        baseOutput = np.asarray(
            self.model.Process(inputSignal), dtype=np.complex128
        )
        return self.ApplyTemperatureDrift(
            baseOutput,
            junctionTemperatureC,
        )

    def ApplyTemperatureDrift(
        self,
        baseOutput: np.ndarray,
        junctionTemperatureC: float,
    ) -> np.ndarray:
        """Apply temperature drift to an already evaluated electrical output.

        Processing details:
            Algorithm: Preserve full-frame electrical memory by accepting the
            previously evaluated base output, then apply configured complex
            gain and compression drift for one thermal update interval.

        Args:
            baseOutput: Electrical-model samples before temperature drift.
            junctionTemperatureC: Fixed junction temperature in Celsius.

        Returns:
            result: Same-shape temperature-modified complex output samples.
        """

        thermalConfig = self.ResolveThermalConfig()
        complexOutput = np.asarray(baseOutput, dtype=np.complex128)
        if thermalConfig is None or not thermalConfig.enabled:
            return complexOutput
        temperatureDeltaC = (
            float(junctionTemperatureC)
            - float(thermalConfig.referenceTemperatureC)
        )
        gainScale = 10.0 ** (
            float(thermalConfig.gainTemperatureCoefficientDbPerC)
            * temperatureDeltaC
            / 20.0
        )
        phaseRotation = np.deg2rad(
            float(thermalConfig.phaseTemperatureCoefficientDegreesPerC)
            * temperatureDeltaC
        )
        saturationScale = max(
            0.05,
            1.0
            + float(thermalConfig.saturationTemperatureCoefficientPerC)
            * temperatureDeltaC,
        )
        nonlinearityScale = max(
            0.0,
            float(thermalConfig.nonlinearityTemperatureCoefficientPerC)
            * temperatureDeltaC,
        )
        outputMagnitude = np.abs(complexOutput)
        nonlinearEnvelopeScale = 1.0 / (
            1.0
            + nonlinearityScale
            * (outputMagnitude / saturationScale) ** 2
        )
        return np.asarray(
            gainScale
            * nonlinearEnvelopeScale
            * complexOutput
            * np.exp(1j * phaseRotation),
            dtype=np.complex128,
        )

    def EstimateDissipatedPowerW(
        self,
        outputSignal: np.ndarray,
    ) -> float:
        """Estimate mean heat power from normalized RF output and efficiency.

        Processing details:
            Algorithm: Map normalized instantaneous envelope power to watts,
            classify samples below the relative active threshold as idle,
            calculate constant or smooth output-power-dependent efficiency for
            active samples, and average RF loss plus idle bias dissipation.

        Args:
            outputSignal: Temperature-modified normalized PA output segment.

        Returns:
            result: Mean dissipated heat power in watts for the segment.
        """

        thermalConfig = self.ResolveThermalConfig()
        if thermalConfig is None:
            return 0.0
        outputPowerNormalized = np.abs(outputSignal) ** 2
        referencePowerW = 10.0 ** (
            (float(thermalConfig.referenceOutputPowerDbm) - 30.0) / 10.0
        )
        outputPowerW = referencePowerW * outputPowerNormalized
        peakPower = float(np.max(outputPowerNormalized))
        if peakPower <= np.finfo(float).tiny:
            return float(thermalConfig.idleDissipatedPowerW)
        activeThresholdLinear = 10.0 ** (
            float(thermalConfig.activePowerThresholdDb) / 10.0
        )
        activeMask = outputPowerNormalized >= activeThresholdLinear * peakPower
        efficiencyValues = np.full(
            outputPowerW.shape,
            float(thermalConfig.minimumDrainEfficiency),
            dtype=float,
        )
        if thermalConfig.efficiencyModelName.strip().lower() == "constant":
            efficiencyValues.fill(float(thermalConfig.peakDrainEfficiency))
        else:
            kneePowerW = 10.0 ** (
                (
                    float(thermalConfig.efficiencyKneeOutputPowerDbm)
                    - 30.0
                )
                / 10.0
            )
            normalizedKneePower = outputPowerW / max(
                kneePowerW, np.finfo(float).tiny
            )
            efficiencyValues = (
                float(thermalConfig.minimumDrainEfficiency)
                + (
                    float(thermalConfig.peakDrainEfficiency)
                    - float(thermalConfig.minimumDrainEfficiency)
                )
                * normalizedKneePower
                / (1.0 + normalizedKneePower)
            )
        activeDissipation = (
            float(thermalConfig.idleDissipatedPowerW)
            + outputPowerW
            * (1.0 / efficiencyValues - 1.0)
        )
        dissipatedPowerPerSample = np.where(
            activeMask,
            activeDissipation,
            float(thermalConfig.idleDissipatedPowerW),
        )
        return float(np.mean(dissipatedPowerPerSample))

    def ProcessThermalFloating(self, inputSignal: np.ndarray) -> np.ndarray:
        """Process a waveform causally while RF power advances thermal state.

        Processing details:
            Algorithm: Split the waveform into configured thermal update
            intervals, evaluate each segment at its starting junction
            temperature, estimate mean heat from all active and idle samples,
            and advance the Foster state for the exact segment duration.

        Args:
            inputSignal: Normalized finite complex waveform to transmit.

        Returns:
            result: Same-length complex waveform including thermal drift.
        """

        thermalConfig = self.ResolveThermalConfig()
        if thermalConfig is None or self.thermalNetwork is None:
            return np.asarray(
                self.model.Process(inputSignal), dtype=np.complex128
            )
        baseOutput = np.asarray(
            self.model.Process(inputSignal), dtype=np.complex128
        )
        outputSignal = np.empty_like(inputSignal, dtype=np.complex128)
        intervalLength = int(thermalConfig.thermalUpdateIntervalSamples)
        dissipatedEnergyJ = 0.0
        startingTemperatureC = (
            self.thermalNetwork.CurrentTemperatureC()
            + self._externalTemperatureOffsetC
        )
        for startIndex in range(0, inputSignal.size, intervalLength):
            stopIndex = min(startIndex + intervalLength, inputSignal.size)
            inputSegment = inputSignal[startIndex:stopIndex]
            baseOutputSegment = baseOutput[startIndex:stopIndex]
            junctionTemperatureC = (
                self.thermalNetwork.CurrentTemperatureC()
                + self._externalTemperatureOffsetC
            )
            outputSegment = self.ApplyTemperatureDrift(
                baseOutputSegment,
                junctionTemperatureC,
            )
            outputSignal[startIndex:stopIndex] = outputSegment
            dissipatedPowerW = self.EstimateDissipatedPowerW(outputSegment)
            durationSec = inputSegment.size / float(thermalConfig.sampleRateHz)
            dissipatedEnergyJ += dissipatedPowerW * durationSec
            selfHeatingTemperatureC = self.thermalNetwork.Advance(
                dissipatedPowerW,
                durationSec,
            )
            junctionTemperatureC = (
                selfHeatingTemperatureC
                + self._externalTemperatureOffsetC
            )
            if junctionTemperatureC > float(
                thermalConfig.maximumJunctionTemperatureC
            ):
                raise RuntimeError(
                    "PA junction temperature exceeded maximumJunctionTemperatureC"
                )
        totalDurationSec = inputSignal.size / float(thermalConfig.sampleRateHz)
        inputPower = np.abs(inputSignal) ** 2
        inputPeakPower = float(np.max(inputPower))
        activeThreshold = inputPeakPower * 10.0 ** (
            float(thermalConfig.activePowerThresholdDb) / 10.0
        )
        activeMask = inputPower >= activeThreshold
        if inputPeakPower <= np.finfo(float).tiny:
            activeMask = np.zeros(inputSignal.size, dtype=bool)
        activeOutputPower = (
            float(np.mean(np.abs(outputSignal[activeMask]) ** 2))
            if np.any(activeMask)
            else np.finfo(float).tiny
        )
        self._lastThermalMetrics = {
            **self.thermalNetwork.GetMetrics(),
            "junctionTemperatureC": (
                self.thermalNetwork.CurrentTemperatureC()
                + self._externalTemperatureOffsetC
            ),
            "selfHeatingJunctionTemperatureC": (
                self.thermalNetwork.CurrentTemperatureC()
            ),
            "mutualHeatingTemperatureRiseC": (
                self._externalTemperatureOffsetC
            ),
            "startingJunctionTemperatureC": startingTemperatureC,
            "endingJunctionTemperatureC": (
                self.thermalNetwork.CurrentTemperatureC()
                + self._externalTemperatureOffsetC
            ),
            "averageDissipatedPowerW": (
                dissipatedEnergyJ / totalDurationSec
            ),
            "activeSampleDutyCycle": self.CalculateActiveDutyCycle(inputSignal),
            "outputPowerDbm": (
                float(thermalConfig.referenceOutputPowerDbm)
                + 10.0
                * np.log10(
                    max(
                        activeOutputPower,
                        np.finfo(float).tiny,
                    )
                )
            ),
        }
        return outputSignal

    def CalculateActiveDutyCycle(self, inputSignal: np.ndarray) -> float:
        """Measure active RF duty cycle while retaining idle thermal samples.

        Processing details:
            Algorithm: Compare instantaneous input power with a peak-relative
            threshold and return the active fraction strictly as a diagnostic;
            every inactive sample still advances idle heat and physical time.

        Args:
            inputSignal: Normalized finite complex waveform under test.

        Returns:
            result: Active sample fraction in the closed interval zero to one.
        """

        thermalConfig = self.ResolveThermalConfig()
        if thermalConfig is None:
            return 0.0
        inputPower = np.abs(inputSignal) ** 2
        peakPower = float(np.max(inputPower))
        if peakPower <= np.finfo(float).tiny:
            return 0.0
        threshold = peakPower * 10.0 ** (
            float(thermalConfig.activePowerThresholdDb) / 10.0
        )
        return float(np.mean(inputPower >= threshold))

    def ResetThermalState(
        self,
        junctionTemperatureC: Optional[float] = None,
        ambientTemperatureC: Optional[float] = None,
    ) -> None:
        """Reset the optional thermal state without changing PA coefficients.

        Processing details:
            Algorithm: Synchronize the configured network, reject use while
            disabled, delegate the physical reset, and refresh diagnostics.

        Args:
            junctionTemperatureC: Optional new starting junction temperature.
            ambientTemperatureC: Optional new ambient or cold-plate value.

        Returns:
            result: None. Subsequent test frames start from the reset state.
        """

        self.SynchronizeThermalModel()
        if self.thermalNetwork is None:
            raise RuntimeError("thermal model is not enabled")
        self.thermalNetwork.Reset(
            junctionTemperatureC,
            ambientTemperatureC,
        )
        self._externalTemperatureOffsetC = 0.0
        self._lastThermalMetrics = self.thermalNetwork.GetMetrics()

    def AdvanceIdle(self, idleTimeSec: float) -> float:
        """Advance cooling or quiescent heating between transmitted frames.

        Processing details:
            Algorithm: Apply idle dissipated power for the requested physical
            gap through the same thermal network and refresh diagnostics.

        Args:
            idleTimeSec: Nonnegative frame-to-frame idle interval in seconds.

        Returns:
            result: Junction temperature after the idle interval in Celsius.
        """

        self.SynchronizeThermalModel()
        thermalConfig = self.ResolveThermalConfig()
        if self.thermalNetwork is None or thermalConfig is None:
            raise RuntimeError("thermal model is not enabled")
        selfHeatingTemperatureC = self.thermalNetwork.Advance(
            float(thermalConfig.idleDissipatedPowerW),
            idleTimeSec,
        )
        junctionTemperatureC = (
            selfHeatingTemperatureC + self._externalTemperatureOffsetC
        )
        self._lastThermalMetrics = {
            **self.thermalNetwork.GetMetrics(),
            "junctionTemperatureC": junctionTemperatureC,
            "selfHeatingJunctionTemperatureC": selfHeatingTemperatureC,
            "mutualHeatingTemperatureRiseC": (
                self._externalTemperatureOffsetC
            ),
            "averageDissipatedPowerW": float(
                thermalConfig.idleDissipatedPowerW
            ),
            "activeSampleDutyCycle": 0.0,
        }
        return junctionTemperatureC

    def SetExternalTemperatureOffsetC(
        self,
        externalTemperatureOffsetC: float,
    ) -> None:
        """Set the temperature rise contributed by neighboring PA chains.

        Processing details:
            Algorithm: Validate a finite nonnegative mutual-heating offset and
            retain it separately from the PA's own Foster nodes so a MIMO bank
            can update coupling once per common frame without double counting.

        Args:
            externalTemperatureOffsetC: Neighbor-induced junction rise in C.

        Returns:
            result: None. Subsequent electrical segments use the total junction temperature.
        """

        resolvedOffset = float(externalTemperatureOffsetC)
        if not np.isfinite(resolvedOffset) or resolvedOffset < 0.0:
            raise ValueError(
                "externalTemperatureOffsetC must be finite and nonnegative"
            )
        self._externalTemperatureOffsetC = resolvedOffset
        if self.thermalNetwork is not None:
            self._lastThermalMetrics = {
                **self._lastThermalMetrics,
                "junctionTemperatureC": (
                    self.thermalNetwork.CurrentTemperatureC()
                    + resolvedOffset
                ),
                "selfHeatingJunctionTemperatureC": (
                    self.thermalNetwork.CurrentTemperatureC()
                ),
                "mutualHeatingTemperatureRiseC": resolvedOffset,
            }

    def GetThermalMetrics(self) -> Dict[str, object]:
        """Return the latest temperature, heat, duty-cycle, and timing values.

        Processing details:
            Algorithm: Synchronize live settings and return a defensive copy;
            disabled thermal operation returns an explicit status dictionary.

        Returns:
            result: Ordinary dictionary containing current thermal diagnostics.
        """

        self.SynchronizeThermalModel()
        if self.thermalNetwork is None:
            return {"enabled": False}
        return {"enabled": True, **dict(self._lastThermalMetrics)}

    def SmallSignalGain(self) -> complex:
        """Return the configured model's DC small-signal complex gain.

        Processing details:
            Algorithm: Carry out the described operation using validated inputs, explicit array-shape handling, and deterministic project conventions.

        Returns:
            result: complex. The computed value described by the summary, with documented units, shape, and normalization.
        """

        self.SynchronizeModel()
        self.SynchronizeThermalModel()
        baseGain = complex(self.model.SmallSignalGain())
        thermalConfig = self.ResolveThermalConfig()
        if thermalConfig is None or self.thermalNetwork is None:
            return baseGain
        junctionTemperatureC = (
            self.thermalNetwork.CurrentTemperatureC()
            + self._externalTemperatureOffsetC
        )
        temperatureDeltaC = (
            junctionTemperatureC
            - float(thermalConfig.referenceTemperatureC)
        )
        gainScale = 10.0 ** (
            float(thermalConfig.gainTemperatureCoefficientDbPerC)
            * temperatureDeltaC
            / 20.0
        )
        phaseRotation = np.deg2rad(
            float(thermalConfig.phaseTemperatureCoefficientDegreesPerC)
            * temperatureDeltaC
        )
        return complex(
            baseGain * gainScale * np.exp(1j * phaseRotation)
        )


class MimoPaModel:
    """Operate independent nonlinear PA models on all transmit chains.

    Each chain can select its own Rapp, Wiener, GMP, or Doherty configuration, input
    drive, relative output power, and optional absolute output-power target in
    dBm. A legacy RMS-voltage target remains available for compatibility.
    This class intentionally owns only the independent nonlinear PA bank;
    ``Channel`` applies configurable coupling before and after this bank while
    preserving the same samples-by-chains interface.
    """

    def __init__(
        self,
        parameters: Optional[Mapping[str, object]] = None,
        width: Optional[int] = None,
        **parameterOverrides: object,
    ) -> None:
        """Initialize all chain models with internal default parameters.

        Processing details:
            Algorithm: Define immutable defaults locally, layer caller
            overrides with ``ChainMap``, validate every per-chain sequence,
            and construct one independent ``PaModel`` for each transmit chain.

        Args:
            parameters: Optional caller-owned mapping containing overrides.
            width: Optional matrix-interface I/Q width. None selects the
                internal 16-bit default and zero selects floating point.
            parameterOverrides: Highest-priority MIMO PA settings.

        Returns:
            result: None. Validated chain models and power settings are stored.
        """

        self.defaultParameters: Mapping[str, object] = MappingProxyType(
            {
                "numTransmitChains": 1,
                "paParametersPerChain": None,
                "inputPowerDbPerChain": None,
                "outputPowerDbPerChain": None,
                "targetOutputRmsPerChain": None,
                "targetOutputPowerDbmPerChain": None,
                "loadResistanceOhm": 50.0,
                "maximumOutputPowerDbm": 25.0,
                "thermalCouplingCPerW": None,
                "width": 16,
            }
        )
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping or None")
        externalParameters: Mapping[str, object] = (
            {}
            if parameters is None
            else RecognizedParameterView(
                parameters,
                self.defaultParameters,
                "MimoPaModel",
            )
        )
        recognizedOverrides = FilterRecognizedParameters(
            parameterOverrides,
            self.defaultParameters,
            "MimoPaModel",
        )
        if width is not None:
            recognizedOverrides["width"] = width
        self.parameters: ChainMap[str, object] = ChainMap(
            recognizedOverrides,
            externalParameters,
            self.defaultParameters,
        )
        self.paModels = []
        self._activePaParameterSnapshot = None
        self.lastOutputRmsPerChain: Tuple[float, ...] = tuple()
        self.lastDissipatedPowerWPerChain: Tuple[float, ...] = tuple()
        self.SynchronizeModels()

    @property
    def NumTransmitChains(self) -> int:
        """Return the configured number of physical PA chains.

        Processing details:
            Algorithm: Resolve the validated integer from the parameter layers.

        Returns:
            result: Positive number of independent transmit chains.
        """

        return cast(int, self.parameters["numTransmitChains"])

    numTransmitChains = NumTransmitChains

    @property
    def Width(self) -> int:
        """Return the MIMO external I/Q component width.

        Processing details:
            Algorithm: Resolve the live ChainMap setting used by both the
            matrix and individual-chain public processing paths.

        Returns:
            result: Zero for floating mode or a positive fixed-point width.
        """

        return cast(int, self.parameters["width"])

    width = Width

    def GetParameters(self) -> Dict[str, object]:
        """Return a flattened snapshot of effective MIMO PA parameters.

        Processing details:
            Algorithm: Resolve and copy every ``ChainMap`` entry so callers
            cannot mutate the local highest-priority layer through the result.

        Returns:
            result: Dictionary containing chain count, model, and power values.
        """

        return dict(self.parameters)

    def UpdateParameters(self, **parameterOverrides: object) -> None:
        """Apply validated MIMO PA parameter overrides transactionally.

        Processing details:
            Algorithm: Update the local layer, rebuild chain models when their
            configurations change, and restore the previous layer on failure.

        Args:
            parameterOverrides: Supported chain or power settings to update.

        Returns:
            result: None. Valid settings affect subsequent ``Process`` calls.
        """

        recognizedOverrides = FilterRecognizedParameters(
            parameterOverrides,
            self.defaultParameters,
            "MimoPaModel.UpdateParameters",
        )
        previousOverrides = dict(self.parameters.maps[0])
        self.parameters.maps[0].update(recognizedOverrides)
        try:
            self.SynchronizeModels()
        except (TypeError, ValueError):
            self.parameters.maps[0].clear()
            self.parameters.maps[0].update(previousOverrides)
            self.SynchronizeModels()
            raise

    def ResolveNumericSequence(
        self,
        parameterName: str,
        defaultValue: float,
        allowNoneEntries: bool = False,
    ) -> Tuple[Optional[float], ...]:
        """Resolve one scalar-per-chain numeric configuration sequence.

        Processing details:
            Algorithm: Expand a missing sequence to one default per chain,
            require exact length, and convert every finite entry to float.

        Args:
            parameterName: Name of the sequence in the active parameter map.
            defaultValue: Value used for every chain when the sequence is None.
            allowNoneEntries: Whether individual entries may disable a target.

        Returns:
            result: Tuple containing one numeric or optional value per chain.
        """

        rawSequence = self.parameters[parameterName]
        if rawSequence is None:
            if allowNoneEntries:
                return tuple(
                    None for _ in range(self.numTransmitChains)
                )
            return tuple(
                float(defaultValue) for _ in range(self.numTransmitChains)
            )
        if isinstance(rawSequence, (str, bytes)) or not isinstance(
            rawSequence, Sequence
        ):
            raise TypeError(f"{parameterName} must be a sequence or None")
        if len(rawSequence) != self.numTransmitChains:
            raise ValueError(
                f"{parameterName} must contain one value per transmit chain"
            )
        resolvedValues = []
        for rawValue in rawSequence:
            if rawValue is None and allowNoneEntries:
                resolvedValues.append(None)
                continue
            if (
                not isinstance(rawValue, (int, float))
                or isinstance(rawValue, bool)
                or not np.isfinite(rawValue)
            ):
                raise ValueError(
                    f"{parameterName} entries must be finite numeric values"
                )
            resolvedValues.append(float(rawValue))
        return tuple(resolvedValues)

    def ResolvePaParametersPerChain(self) -> Tuple[Mapping[str, object], ...]:
        """Resolve one ordinary ``PaModel`` override mapping per chain.

        Processing details:
            Algorithm: Expand ``None`` to empty mappings, validate exact chain
            count and mapping types, then copy entries for stable comparison.

        Returns:
            result: Tuple of independent per-chain PA parameter dictionaries.
        """

        rawParameters = self.parameters["paParametersPerChain"]
        if rawParameters is None:
            return tuple({} for _ in range(self.numTransmitChains))
        if isinstance(rawParameters, (str, bytes)) or not isinstance(
            rawParameters, Sequence
        ):
            raise TypeError(
                "paParametersPerChain must be a sequence of mappings or None"
            )
        if len(rawParameters) != self.numTransmitChains:
            raise ValueError(
                "paParametersPerChain must contain one mapping per chain"
            )
        resolvedParameters = []
        for chainParameters in rawParameters:
            if not isinstance(chainParameters, Mapping):
                raise TypeError(
                    "each paParametersPerChain entry must be a mapping"
                )
            resolvedParameters.append(dict(chainParameters))
        return tuple(resolvedParameters)

    def ValidateParameters(self) -> None:
        """Validate chain count, model mappings, and power controls.

        Processing details:
            Algorithm: Validate the positive chain count, resolve all per-chain
            sequences, validate the resistive port, and reject conflicting
            legacy RMS and absolute dBm targets. Unknown keys have already
            been warned about and filtered at the configuration boundary.

        Returns:
            result: None. Invalid settings raise descriptive exceptions.
        """

        numTransmitChains = self.parameters["numTransmitChains"]
        if (
            not isinstance(numTransmitChains, int)
            or isinstance(numTransmitChains, bool)
            or numTransmitChains < 1
            or numTransmitChains > 16
        ):
            raise ValueError("numTransmitChains must be an integer from 1 to 16")
        self.ResolvePaParametersPerChain()
        self.ResolveNumericSequence("inputPowerDbPerChain", 0.0)
        self.ResolveNumericSequence("outputPowerDbPerChain", 0.0)
        targetOutputRmsValues = self.ResolveNumericSequence(
            "targetOutputRmsPerChain", 0.0, allowNoneEntries=True
        )
        if any(
            targetValue is not None and targetValue <= 0.0
            for targetValue in targetOutputRmsValues
        ):
            raise ValueError(
                "targetOutputRmsPerChain entries must be positive or None"
            )
        targetOutputPowerDbmValues = self.ResolveNumericSequence(
            "targetOutputPowerDbmPerChain",
            0.0,
            allowNoneEntries=True,
        )
        if any(
            rmsTarget is not None and dbmTarget is not None
            for rmsTarget, dbmTarget in zip(
                targetOutputRmsValues,
                targetOutputPowerDbmValues,
            )
        ):
            raise ValueError(
                "one chain cannot set both target output RMS and dBm"
            )
        powerCalibration = PowerCalibration(
            loadResistanceOhm=self.parameters["loadResistanceOhm"],
            maximumOutputPowerDbm=self.parameters[
                "maximumOutputPowerDbm"
            ],
        )
        for targetPowerDbm in targetOutputPowerDbmValues:
            if targetPowerDbm is not None:
                powerCalibration.OutputPowerToDriveScale(
                    targetPowerDbm
                )
        FixedPoint(self.width)
        self.ResolveThermalCouplingMatrix()

    def ResolveThermalCouplingMatrix(self) -> np.ndarray:
        """Return the optional mutual steady-state thermal-resistance matrix.

        Processing details:
            Algorithm: Expand None to a zero matrix, require an exact finite
            nonnegative chains-by-chains array, and force the diagonal to zero
            because each PaModel already owns its self-heating Foster network.

        Returns:
            result: Matrix in degrees Celsius per watt from source to victim.
        """

        rawMatrix = self.parameters["thermalCouplingCPerW"]
        if rawMatrix is None:
            return np.zeros(
                (self.numTransmitChains, self.numTransmitChains), dtype=float
            )
        couplingMatrix = np.asarray(rawMatrix, dtype=float)
        expectedShape = (
            self.numTransmitChains,
            self.numTransmitChains,
        )
        if couplingMatrix.shape != expectedShape:
            raise ValueError(
                "thermalCouplingCPerW must be a square matrix with one row "
                "and column per transmit chain"
            )
        if not np.all(np.isfinite(couplingMatrix)) or np.any(
            couplingMatrix < 0.0
        ):
            raise ValueError(
                "thermalCouplingCPerW entries must be finite and nonnegative"
            )
        couplingMatrix = couplingMatrix.copy()
        np.fill_diagonal(couplingMatrix, 0.0)
        return couplingMatrix

    def UpdateMutualHeating(self) -> None:
        """Map recent per-chain heat power to neighbor temperature offsets.

        Processing details:
            Algorithm: Multiply the source-chain heat vector by the configured
            off-diagonal C/W matrix and apply each victim offset for the next
            common frame. This low-rate approximation preserves independent
            self-heating dynamics while exposing package or board heat sharing.

        Returns:
            result: None. Per-chain external temperature offsets are updated.
        """

        couplingMatrix = self.ResolveThermalCouplingMatrix()
        if len(self.lastDissipatedPowerWPerChain) != self.numTransmitChains:
            return
        sourcePowerVector = np.asarray(
            self.lastDissipatedPowerWPerChain, dtype=float
        )
        victimTemperatureRise = couplingMatrix @ sourcePowerVector
        for paModel, temperatureRiseC in zip(
            self.paModels, victimTemperatureRise
        ):
            paModel.SetExternalTemperatureOffsetC(float(temperatureRiseC))

    def SynchronizeModels(self) -> None:
        """Rebuild per-chain PA objects after live configuration changes.

        Processing details:
            Algorithm: Validate all settings, compare copied chain mappings
            with the last snapshot, and reconstruct only when they differ.

        Returns:
            result: None. ``paModels`` always matches current configuration.
        """

        self.ValidateParameters()
        paParameterSnapshot = self.ResolvePaParametersPerChain()
        if paParameterSnapshot == self._activePaParameterSnapshot:
            return
        self.paModels = [
            # Each facade is an internal floating-point calculation block.
            # MimoPaModel applies the selected fixed-point format only at its
            # public Process and ProcessChain boundaries.
            PaModel(
                parameters={
                    **chainParameters,
                    "width": 0,
                }
            )
            for chainParameters in paParameterSnapshot
        ]
        self._activePaParameterSnapshot = paParameterSnapshot

    def SetOutputPowerDb(self, chainIndex: int, outputPowerDb: float) -> None:
        """Set one chain's relative output-power calibration in decibels.

        Processing details:
            Algorithm: Copy the resolved per-chain dB tuple, replace one
            indexed value, and commit it through transactional validation.

        Args:
            chainIndex: Zero-based transmit-chain index.
            outputPowerDb: Desired relative output power in decibels.

        Returns:
            result: None. The selected chain changes on the next processing call.
        """

        if not isinstance(chainIndex, int) or isinstance(chainIndex, bool):
            raise TypeError("chainIndex must be an integer")
        if chainIndex < 0 or chainIndex >= self.numTransmitChains:
            raise IndexError("chainIndex is outside the configured chain range")
        resolvedValues = list(
            self.ResolveNumericSequence("outputPowerDbPerChain", 0.0)
        )
        resolvedValues[chainIndex] = float(outputPowerDb)
        self.UpdateParameters(outputPowerDbPerChain=tuple(resolvedValues))

    def SetTargetOutputRms(
        self, chainIndex: int, targetOutputRms: Optional[float]
    ) -> None:
        """Set or disable one chain's absolute output RMS target.

        Processing details:
            Algorithm: Copy the optional target sequence, update one entry,
            and validate positive enabled targets transactionally.

        Args:
            chainIndex: Zero-based transmit-chain index.
            targetOutputRms: Positive complex-envelope RMS, or None to disable.

        Returns:
            result: None. The new target applies to subsequent outputs.
        """

        if not isinstance(chainIndex, int) or isinstance(chainIndex, bool):
            raise TypeError("chainIndex must be an integer")
        if chainIndex < 0 or chainIndex >= self.numTransmitChains:
            raise IndexError("chainIndex is outside the configured chain range")
        rawTargets = self.parameters["targetOutputRmsPerChain"]
        targetValues = (
            [None] * self.numTransmitChains
            if rawTargets is None
            else list(rawTargets)
        )
        targetValues[chainIndex] = targetOutputRms
        rawPowerTargets = self.parameters["targetOutputPowerDbmPerChain"]
        powerTargetValues = (
            [None] * self.numTransmitChains
            if rawPowerTargets is None
            else list(rawPowerTargets)
        )
        powerTargetValues[chainIndex] = None
        self.UpdateParameters(
            targetOutputRmsPerChain=tuple(targetValues),
            targetOutputPowerDbmPerChain=tuple(powerTargetValues),
        )

    def SetTargetOutputPowerDbm(
        self,
        chainIndex: int,
        targetOutputPowerDbm: Optional[float],
    ) -> None:
        """Set or disable one chain's absolute output-power target in dBm.

        Processing details:
            Algorithm: Replace one dBm target, clear the legacy RMS target for
            the same chain, and validate the requested power through the
            configured resistive-port calibration.

        Args:
            chainIndex: Zero-based transmit-chain index.
            targetOutputPowerDbm: Finite absolute output power in dBm, or None.

        Returns:
            result: None. The selected chain is recalibrated on later calls.
        """

        if not isinstance(chainIndex, int) or isinstance(chainIndex, bool):
            raise TypeError("chainIndex must be an integer")
        if chainIndex < 0 or chainIndex >= self.numTransmitChains:
            raise IndexError("chainIndex is outside the configured chain range")
        rawPowerTargets = self.parameters["targetOutputPowerDbmPerChain"]
        powerTargetValues = (
            [None] * self.numTransmitChains
            if rawPowerTargets is None
            else list(rawPowerTargets)
        )
        powerTargetValues[chainIndex] = targetOutputPowerDbm
        rawRmsTargets = self.parameters["targetOutputRmsPerChain"]
        rmsTargetValues = (
            [None] * self.numTransmitChains
            if rawRmsTargets is None
            else list(rawRmsTargets)
        )
        rmsTargetValues[chainIndex] = None
        self.UpdateParameters(
            targetOutputPowerDbmPerChain=tuple(powerTargetValues),
            targetOutputRmsPerChain=tuple(rmsTargetValues),
        )

    def Process(self, inputSignal: np.ndarray) -> np.ndarray:
        """Process every transmit column through its independent PA chain.

        Processing details:
            Algorithm: Decode the public matrix of fixed I/Q codes once,
            process every normalized floating column through its PA and power
            controls, encode every result column back to public integer codes,
            and preserve the original vector or matrix orientation.

        Args:
            inputSignal: Complex vector for one chain or matrix shaped samples
                by the configured number of transmit chains.

        Returns:
            result: Processed complex array containing raw I/Q codes in fixed
                mode and physical floating samples in floating mode.
        """

        self.SynchronizeModels()
        interfaceFormat = FixedPoint(self.width)
        complexInput = interfaceFormat.DecodeComplex(inputSignal)
        inputWasVector = complexInput.ndim == 1
        if inputWasVector:
            complexInput = complexInput.reshape(-1, 1)
        if (
            complexInput.ndim != 2
            or complexInput.shape[1] != self.numTransmitChains
        ):
            raise ValueError(
                "inputSignal must have one column per transmit chain"
            )
        if complexInput.shape[0] == 0 or not np.all(np.isfinite(complexInput)):
            raise ValueError("inputSignal must contain finite samples")
        outputColumns = []
        outputRmsValues = []
        for chainIndex in range(self.numTransmitChains):
            floatingChainOutput = self.ProcessChainFloating(
                complexInput[:, chainIndex], chainIndex
            )
            outputColumns.append(
                interfaceFormat.EncodeComplex(floatingChainOutput)
            )
            outputRmsValues.append(
                float(
                    np.sqrt(
                        np.mean(np.abs(floatingChainOutput) ** 2)
                    )
                )
            )
        outputMatrix = np.column_stack(outputColumns)
        self.lastOutputRmsPerChain = tuple(outputRmsValues)
        self.lastDissipatedPowerWPerChain = tuple(
            float(
                paModel.GetThermalMetrics().get(
                    "averageDissipatedPowerW", 0.0
                )
            )
            for paModel in self.paModels
        )
        self.UpdateMutualHeating()
        if inputWasVector and self.numTransmitChains == 1:
            return outputMatrix[:, 0]
        return outputMatrix

    def ProcessFloating(
        self, inputSignal: np.ndarray
    ) -> np.ndarray:
        """Evaluate every PA chain without public fixed-point conversion.

        Processing details:
            Algorithm: Validate a normalized samples-by-chains matrix, call
            each independently configured PA's floating processing path,
            update the most recent per-chain RMS diagnostics, and preserve a
            SISO vector only when the configured PA bank has one chain.

        Args:
            inputSignal: Normalized vector or samples-by-chains matrix.

        Returns:
            result: Same-orientation floating PA-bank output.
        """

        self.SynchronizeModels()
        complexInput = np.asarray(
            inputSignal, dtype=np.complex128
        )
        inputWasVector = complexInput.ndim == 1
        inputMatrix = (
            complexInput.reshape(-1, 1)
            if inputWasVector
            else complexInput
        )
        if (
            inputMatrix.ndim != 2
            or inputMatrix.shape[1] != self.numTransmitChains
            or inputMatrix.shape[0] == 0
            or not np.all(np.isfinite(inputMatrix))
        ):
            raise ValueError(
                "inputSignal must have one finite column per transmit chain"
            )
        outputColumns = [
            self.ProcessChainFloating(
                inputMatrix[:, chainIndex], chainIndex
            )
            for chainIndex in range(self.numTransmitChains)
        ]
        outputMatrix = np.column_stack(outputColumns)
        self.lastOutputRmsPerChain = tuple(
            float(np.sqrt(np.mean(np.abs(outputColumn) ** 2)))
            for outputColumn in outputColumns
        )
        self.lastDissipatedPowerWPerChain = tuple(
            float(
                paModel.GetThermalMetrics().get(
                    "averageDissipatedPowerW", 0.0
                )
            )
            for paModel in self.paModels
        )
        self.UpdateMutualHeating()
        if inputWasVector and self.numTransmitChains == 1:
            return outputMatrix[:, 0]
        return outputMatrix

    def ProcessChain(
        self, inputSignal: np.ndarray, chainIndex: int
    ) -> np.ndarray:
        """Process a vector through one selected PA and power calibration.

        Processing details:
            Algorithm: Decode raw external codes, apply the selected chain's
            floating PA and power controls, then encode raw external codes.

        Args:
            inputSignal: One-dimensional complex samples for one RF chain.
            chainIndex: Zero-based physical PA index.

        Returns:
            result: Complex128 vector containing raw fixed I/Q codes or
                floating samples according to the configured width.
        """

        self.SynchronizeModels()
        if not isinstance(chainIndex, int) or isinstance(chainIndex, bool):
            raise TypeError("chainIndex must be an integer")
        if chainIndex < 0 or chainIndex >= self.numTransmitChains:
            raise IndexError("chainIndex is outside the configured chain range")
        interfaceFormat = FixedPoint(self.width)
        complexInput = interfaceFormat.DecodeComplex(inputSignal)
        if complexInput.ndim != 1 or complexInput.size == 0:
            raise ValueError("inputSignal must be a nonempty vector")
        if not np.all(np.isfinite(complexInput)):
            raise ValueError("inputSignal must contain finite samples")
        chainOutput = self.ProcessChainFloating(complexInput, chainIndex)
        return interfaceFormat.EncodeComplex(chainOutput)

    def ProcessChainFloating(
        self, inputSignal: np.ndarray, chainIndex: int
    ) -> np.ndarray:
        """Evaluate one MIMO PA chain in normalized floating-point units.

        Processing details:
            Algorithm: Validate one normalized complex vector, apply its input
            drive, internal floating PA model, relative output calibration,
            and optional absolute RMS target without external code conversion.

        Args:
            inputSignal: Normalized physical complex samples for one chain.
            chainIndex: Zero-based physical PA index.

        Returns:
            result: Floating complex output before public fixed-point encoding.
        """

        self.SynchronizeModels()
        if not isinstance(chainIndex, int) or isinstance(chainIndex, bool):
            raise TypeError("chainIndex must be an integer")
        if chainIndex < 0 or chainIndex >= self.numTransmitChains:
            raise IndexError("chainIndex is outside the configured chain range")
        complexInput = np.asarray(inputSignal, dtype=np.complex128)
        if complexInput.ndim != 1 or complexInput.size == 0:
            raise ValueError("inputSignal must be a nonempty vector")
        if not np.all(np.isfinite(complexInput)):
            raise ValueError("inputSignal must contain finite samples")
        inputPowerDbValues = self.ResolveNumericSequence(
            "inputPowerDbPerChain", 0.0
        )
        outputPowerDbValues = self.ResolveNumericSequence(
            "outputPowerDbPerChain", 0.0
        )
        targetOutputRmsValues = self.ResolveNumericSequence(
            "targetOutputRmsPerChain", 0.0, allowNoneEntries=True
        )
        targetOutputPowerDbmValues = self.ResolveNumericSequence(
            "targetOutputPowerDbmPerChain",
            0.0,
            allowNoneEntries=True,
        )
        inputScale = 10.0 ** (
            float(inputPowerDbValues[chainIndex]) / 20.0
        )
        chainOutput = self.paModels[chainIndex].ProcessFloating(
            inputScale * complexInput
        )
        outputScale = 10.0 ** (
            float(outputPowerDbValues[chainIndex]) / 20.0
        )
        chainOutput = outputScale * chainOutput
        targetOutputRms = targetOutputRmsValues[chainIndex]
        targetOutputPowerDbm = targetOutputPowerDbmValues[chainIndex]
        if targetOutputPowerDbm is not None:
            targetOutputRms = PowerCalibration(
                loadResistanceOhm=self.parameters["loadResistanceOhm"],
                maximumOutputPowerDbm=self.parameters[
                    "maximumOutputPowerDbm"
                ],
            ).DbmToRms(targetOutputPowerDbm)
        if targetOutputRms is not None:
            currentRms = np.sqrt(np.mean(np.abs(chainOutput) ** 2))
            if currentRms <= np.finfo(float).tiny:
                raise ValueError(
                    "cannot set target RMS on a zero-power PA output"
                )
            chainOutput = float(targetOutputRms) * chainOutput / currentRms
        return np.asarray(chainOutput, dtype=np.complex128)

    def GetOutputRmsPerChain(self) -> Tuple[float, ...]:
        """Return legacy RMS output voltages measured by the most recent call.

        Processing details:
            Algorithm: Return an immutable tuple already calculated from each
            output column, without reprocessing the waveform.

        Returns:
            result: One complex-envelope RMS voltage per transmit chain.
        """

        return tuple(self.lastOutputRmsPerChain)

    def GetOutputPowerDbmPerChain(self) -> Tuple[float, ...]:
        """Return the most recently measured output powers in dBm.

        Processing details:
            Algorithm: Convert every retained chain RMS through the same
            resistive-port calibration used by absolute dBm targets.

        Returns:
            result: Chain-ordered absolute output powers in dBm, or an empty
            tuple before the first complete matrix ``Process`` call.
        """

        powerCalibration = PowerCalibration(
            loadResistanceOhm=self.parameters["loadResistanceOhm"],
            maximumOutputPowerDbm=self.parameters[
                "maximumOutputPowerDbm"
            ],
        )
        return tuple(
            powerCalibration.RmsToDbm(outputRms)
            for outputRms in self.lastOutputRmsPerChain
        )

    def SuspendThermalModel(self) -> Tuple[Optional[Dict[str, object]], ...]:
        """Suspend self-heating on every physical PA during calibration.

        Processing details:
            Algorithm: Synchronize the per-chain model bank and collect one
            exact restorable thermal snapshot from every PaModel in chain order.

        Returns:
            result: Immutable tuple of optional per-chain thermal snapshots.
        """

        self.SynchronizeModels()
        return tuple(
            paModel.SuspendThermalModel() for paModel in self.paModels
        )

    def RestoreThermalModel(
        self,
        thermalSnapshots: Optional[
            Sequence[Optional[Mapping[str, object]]]
        ],
    ) -> None:
        """Restore all per-chain thermal states after calibration completes.

        Processing details:
            Algorithm: Accept None as a no-op, require one snapshot per chain,
            and delegate exact state restoration to each physical PaModel.

        Args:
            thermalSnapshots: Chain-ordered snapshots from suspension.

        Returns:
            result: None. Every enabled PA resumes from its prior temperature.
        """

        if thermalSnapshots is None:
            return
        self.SynchronizeModels()
        if len(thermalSnapshots) != self.numTransmitChains:
            raise ValueError("thermalSnapshots must contain one entry per chain")
        for paModel, thermalSnapshot in zip(
            self.paModels, thermalSnapshots
        ):
            paModel.RestoreThermalModel(thermalSnapshot)

    def ResetThermalState(
        self,
        junctionTemperatureC: Optional[
            Union[float, Sequence[float]]
        ] = None,
        ambientTemperatureC: Optional[float] = None,
    ) -> None:
        """Reset every chain to shared or independently specified temperature.

        Processing details:
            Algorithm: Expand a scalar junction temperature across chains or
            validate an exact sequence, then reset each enabled PA with the
            shared ambient reference without processing an RF waveform.

        Args:
            junctionTemperatureC: Optional scalar or one value per chain.
            ambientTemperatureC: Optional shared ambient temperature.

        Returns:
            result: None. All configured thermal states restart coherently.
        """

        self.SynchronizeModels()
        if junctionTemperatureC is None or isinstance(
            junctionTemperatureC, (int, float)
        ):
            junctionValues = tuple(
                junctionTemperatureC
                for _ in range(self.numTransmitChains)
            )
        else:
            if len(junctionTemperatureC) != self.numTransmitChains:
                raise ValueError(
                    "junctionTemperatureC must contain one value per chain"
                )
            junctionValues = tuple(junctionTemperatureC)
        for paModel, junctionValue in zip(self.paModels, junctionValues):
            thermalConfig = paModel.ResolveThermalConfig()
            if thermalConfig is not None and thermalConfig.enabled:
                paModel.ResetThermalState(
                    None if junctionValue is None else float(junctionValue),
                    ambientTemperatureC,
                )

    def AdvanceIdle(self, idleTimeSec: float) -> Tuple[Optional[float], ...]:
        """Advance all enabled PA thermal states through one common idle gap.

        Processing details:
            Algorithm: Apply each chain's own configured idle dissipation for
            the same elapsed duration and retain None for thermally disabled PAs.

        Args:
            idleTimeSec: Nonnegative physical frame-to-frame gap in seconds.

        Returns:
            result: Chain-ordered junction temperatures or None entries.
        """

        self.SynchronizeModels()
        temperatures = []
        for paModel in self.paModels:
            thermalConfig = paModel.ResolveThermalConfig()
            if thermalConfig is None or not thermalConfig.enabled:
                temperatures.append(None)
            else:
                temperatures.append(paModel.AdvanceIdle(idleTimeSec))
        return tuple(temperatures)

    def GetThermalMetrics(self) -> Dict[str, object]:
        """Return one thermal diagnostic dictionary per physical PA chain.

        Processing details:
            Algorithm: Synchronize models and copy each PaModel's latest
            temperature, duty-cycle, heat-power, and elapsed-time diagnostics.

        Returns:
            result: Dictionary containing a chain-ordered immutable tuple.
        """

        self.SynchronizeModels()
        return {
            "chains": tuple(
                paModel.GetThermalMetrics() for paModel in self.paModels
            )
        }


class IQImbalancePA:
    """Wrap any PA with a widely-linear output IQ-imbalance model."""

    def __init__(
        self,
        paModel: Any,
        directCoefficient: complex = 1.0 + 0.0j,
        imageCoefficient: complex = 0.045 * np.exp(1j * 0.35),
    ) -> None:
        """Initialize an IQ-imbalance wrapper around an existing PA model.

        Processing details:
            Algorithm: Carry out the described operation using validated inputs, explicit array-shape handling, and deterministic project conventions.

        Args:
            paModel: PA object exposing Process and SmallSignalGain operations.
            directCoefficient: Complex gain of the desired direct IQ path.
            imageCoefficient: Complex gain multiplying the conjugate image path.

        Returns:
            result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
        """
        self.paModel = paModel
        self.directCoefficient = complex(directCoefficient)
        self.imageCoefficient = complex(imageCoefficient)

    @property
    def Width(self) -> int:
        """Return the wrapped PA public I/Q component width.

        Processing details:
            Algorithm: Forward the wrapped PA width and use zero for a
            third-party floating PA that does not expose this setting.

        Returns:
            result: Nonnegative public fixed-point component width.
        """

        return int(getattr(self.paModel, "width", 0))

    width = Width

    def Process(self, inputSignal: np.ndarray) -> np.ndarray:
        """Apply the base PA and then add its conjugate image component.

        Processing details:
            Algorithm: Execute the configured signal-processing path, preserve sample alignment, and return the complete downstream result.

        Args:
            inputSignal: One-dimensional complex baseband samples supplied to the operation.

        Returns:
            result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
        """

        interfaceFormat = FixedPoint(self.width)
        floatingInput = interfaceFormat.DecodeComplex(inputSignal)
        floatingOutput = self.ProcessFloating(floatingInput)
        return interfaceFormat.EncodeComplex(floatingOutput)

    def ProcessFloating(self, inputSignal: np.ndarray) -> np.ndarray:
        """Apply PA nonlinearity and IQ imbalance in floating-point units.

        Processing details:
            Algorithm: Evaluate the wrapped PA without its public code
            conversion when supported, then combine direct and conjugated
            image paths before the outer interface performs one encoding.

        Args:
            inputSignal: Normalized floating complex samples.

        Returns:
            result: Normalized floating samples including IQ imbalance.
        """

        complexInput = np.asarray(inputSignal, dtype=np.complex128)
        floatingProcessor = getattr(self.paModel, "ProcessFloating", None)
        if callable(floatingProcessor):
            paOutput = floatingProcessor(complexInput)
        else:
            paOutput = self.paModel.Process(complexInput)
        return np.asarray(
            self.directCoefficient * paOutput
            + self.imageCoefficient * np.conj(paOutput),
            dtype=np.complex128,
        )

    def SmallSignalGain(self) -> complex:
        """Return the direct-path small-signal gain of the wrapped PA.

        Processing details:
            Algorithm: Carry out the described operation using validated inputs, explicit array-shape handling, and deterministic project conventions.

        Returns:
            result: complex. The computed value described by the summary, with documented units, shape, and normalization.
        """

        return self.directCoefficient * self.paModel.SmallSignalGain()


def AsComplexVector(inputSignal: np.ndarray) -> np.ndarray:
    """Convert input to a finite one-dimensional complex array.

    Processing details:
        Algorithm: Carry out the described operation using validated inputs, explicit array-shape handling, and deterministic project conventions.

    Args:
        inputSignal: One-dimensional complex baseband samples supplied to the operation.

    Returns:
        result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
    """

    complexInput = np.asarray(inputSignal, dtype=np.complex128)
    if complexInput.ndim != 1:
        raise ValueError("inputSignal must be one-dimensional")
    if not np.all(np.isfinite(complexInput)):
        raise ValueError("inputSignal contains NaN or infinite values")
    return complexInput


def DelaySignal(inputSignal: np.ndarray, sampleDelay: int) -> np.ndarray:
    """Apply a causal integer delay without changing the array length.

    Processing details:
        Algorithm: Apply the bounded sample-domain transformation without changing array length or causal indexing conventions.

    Args:
        inputSignal: One-dimensional complex baseband samples supplied to the operation.
        sampleDelay: Nonnegative causal delay measured in complex samples.

    Returns:
        result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
    """

    if sampleDelay < 0:
        raise ValueError("sampleDelay cannot be negative")
    if sampleDelay == 0:
        return inputSignal
    delayedSignal = np.zeros_like(inputSignal)
    if sampleDelay < inputSignal.size:
        delayedSignal[sampleDelay:] = inputSignal[:-sampleDelay]
    return delayedSignal


def DefaultGmpCoefficients(
    nonlinearOrders: Sequence[int],
    memoryDepth: int,
    crossMemoryDepth: int,
) -> Tuple[
    Dict[Tuple[int, int], complex],
    Dict[Tuple[int, int, int], complex],
    Dict[Tuple[int, int, int], complex],
]:
    """Create stable default coefficients with compression and memory effects.

    Processing details:
        Algorithm: Construct the requested model structure in deterministic order so coefficient indices and delayed samples remain reproducible.

    Args:
        nonlinearOrders: Positive odd polynomial orders included in the model.
        memoryDepth: Number of causal sample delays included in the model.
        crossMemoryDepth: Number of envelope cross-delays included in the GMP model.

    Returns:
        result: Tuple[Dict[Tuple[int, int], complex], Dict[Tuple[int, int, int], complex], Dict[Tuple[int, int, int], complex]]. The computed value described by the summary, with documented units, shape, and normalization.
    """

    # Zero-memory coefficients define the dominant AM-AM/AM-PM behavior.
    orderCoefficient = {
        1: 1.0 + 0.0j,
        3: -0.62 + 0.16j,
        5: 0.18 - 0.08j,
        7: -0.024 + 0.014j,
    }
    mainCoefficients: Dict[Tuple[int, int], complex] = {}
    laggingCoefficients: Dict[Tuple[int, int, int], complex] = {}
    leadingCoefficients: Dict[Tuple[int, int, int], complex] = {}

    for nonlinearOrder in nonlinearOrders:
        baseCoefficient = orderCoefficient.get(
            nonlinearOrder,
            (-0.12 + 0.03j) / max(nonlinearOrder - 1, 1),
        )
        for memoryIndex in range(memoryDepth):
            if nonlinearOrder == 1:
                # The first-order tail creates a mild frequency response.
                linearTail = (
                    1.0 + 0.0j
                    if memoryIndex == 0
                    else (0.045 - 0.020j) * ((-0.45) ** (memoryIndex - 1))
                )
                mainCoefficients[(nonlinearOrder, memoryIndex)] = linearTail
            else:
                memoryDecay = (0.34**memoryIndex) * np.exp(
                    -1j * 0.18 * memoryIndex
                )
                mainCoefficients[(nonlinearOrder, memoryIndex)] = (
                    baseCoefficient * memoryDecay
                )

        if nonlinearOrder == 1:
            continue
        for memoryIndex in range(memoryDepth):
            for crossIndex in range(1, crossMemoryDepth + 1):
                crossDecay = (0.22**memoryIndex) * (0.42**crossIndex)
                laggingCoefficients[
                    (nonlinearOrder, memoryIndex, crossIndex)
                ] = (0.040 - 0.018j) * crossDecay / (nonlinearOrder - 1)
                leadingCoefficients[
                    (nonlinearOrder, memoryIndex, crossIndex)
                ] = (-0.026 + 0.012j) * crossDecay / (nonlinearOrder - 1)

    return mainCoefficients, laggingCoefficients, leadingCoefficients


def AddAwgn(
    inputSignal: np.ndarray,
    snrDb: Optional[float],
    randomGenerator: np.random.Generator,
) -> np.ndarray:
    """Add complex white Gaussian feedback noise at the requested SNR.

    Processing details:
        Algorithm: Carry out the described operation using validated inputs, explicit array-shape handling, and deterministic project conventions.

    Args:
        inputSignal: One-dimensional complex baseband samples supplied to the operation.
        snrDb: Requested signal-to-noise ratio in decibels, or None for no noise.
        randomGenerator: NumPy random generator that makes results reproducible.

    Returns:
        result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
    """

    complexInput = AsComplexVector(inputSignal)
    if snrDb is None:
        return complexInput.copy()
    signalPower = np.mean(np.abs(complexInput) ** 2)
    noisePower = signalPower / (10.0 ** (snrDb / 10.0))
    noiseScale = np.sqrt(noisePower / 2.0)
    complexNoise = noiseScale * (
        randomGenerator.standard_normal(complexInput.size)
        + 1j * randomGenerator.standard_normal(complexInput.size)
    )
    return complexInput + complexNoise
