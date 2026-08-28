"""Power-amplifier behavioral models used by the DPD-ILC simulation.

Callers construct ``PaModel`` with ``modelName="rapp"``, ``"wiener"``,
``"gmp"``, ``"piecewise_gmp"``, or ``"doherty"`` and then call ``Process``.
Five nonlinear model families are
provided internally:

* ``RappPA`` applies the classic memoryless solid-state PA AM-AM curve and
  preserves input phase, providing a deliberate zero-memory reference model.
* ``WienerPA`` applies a linear memory filter followed by a smooth Rapp
  AM-AM characteristic and a saturating AM-PM characteristic.
* ``GMPPA`` implements the generalized memory polynomial main, lagging,
  and leading cross terms described in the project theory document.
* ``PiecewiseGMPPA`` blends sparse GMP coefficient sets over adjacent
  envelope regions with compact, twice-continuously-differentiable gates.
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


@dataclass(frozen=True, eq=False)
class _ThermalRuntime:
    """Cache validated thermal constants for one waveform-period operation.

    ``ThermalConfig`` is immutable at the public boundary, so a period and all
    of its steady-state solver probes may safely share these normalized names,
    derived scalars, and read-only branch arrays.  The cache is deliberately
    local to one top-level operation; live ChainMap updates are therefore still
    observed and validated by the next call.
    """

    config: ThermalConfig
    modelName: str
    efficiencyModelName: str
    resistanceValues: np.ndarray
    timeConstantValues: np.ndarray
    sampleRateHz: float
    thermalUpdateIntervalSamples: int
    activeThresholdLinear: float
    referencePowerW: float
    efficiencyKneePowerW: float
    minimumDrainEfficiency: float
    peakDrainEfficiency: float
    idleDissipatedPowerW: float
    maximumJunctionTemperatureC: float

    @classmethod
    def FromValidatedConfig(
        cls, config: ThermalConfig
    ) -> "_ThermalRuntime":
        """Build one operation-local cache from a validated configuration.

        Processing details:
            Algorithm: Normalize model names once, copy the selected thermal
            branch vectors into read-only arrays, and precompute scalar unit
            conversions reused by every interval in one waveform period.

        Args:
            config: Thermal configuration already validated at the public
                operation boundary.

        Returns:
            result: Immutable operation-local thermal runtime constants.
        """

        modelName = config.modelName.strip().lower()
        resistanceValues = np.array(
            config.thermalResistancesCPerW, dtype=float, copy=True
        )
        timeConstantValues = np.array(
            config.thermalTimeConstantsSec, dtype=float, copy=True
        )
        if modelName == "single_rc":
            resistanceValues = resistanceValues[:1]
            timeConstantValues = timeConstantValues[:1]
        resistanceValues.setflags(write=False)
        timeConstantValues.setflags(write=False)
        return cls(
            config=config,
            modelName=modelName,
            efficiencyModelName=(
                config.efficiencyModelName.strip().lower()
            ),
            resistanceValues=resistanceValues,
            timeConstantValues=timeConstantValues,
            sampleRateHz=float(config.sampleRateHz),
            thermalUpdateIntervalSamples=int(
                config.thermalUpdateIntervalSamples
            ),
            activeThresholdLinear=(
                10.0 ** (float(config.activePowerThresholdDb) / 10.0)
            ),
            referencePowerW=(
                10.0
                ** (
                    (float(config.referenceOutputPowerDbm) - 30.0)
                    / 10.0
                )
            ),
            efficiencyKneePowerW=(
                10.0
                ** (
                    (
                        float(config.efficiencyKneeOutputPowerDbm)
                        - 30.0
                    )
                    / 10.0
                )
            ),
            minimumDrainEfficiency=float(config.minimumDrainEfficiency),
            peakDrainEfficiency=float(config.peakDrainEfficiency),
            idleDissipatedPowerW=float(config.idleDissipatedPowerW),
            maximumJunctionTemperatureC=float(
                config.maximumJunctionTemperatureC
            ),
        )


class ThermalNetwork:
    """Maintain the causal thermal state of a static, single-RC, or Foster model."""

    def __init__(self, config: ThermalConfig) -> None:
        """Create the requested thermal topology and initialize its state.

        Processing details:
            Algorithm: Validate the immutable configuration, select all Foster
            branches or the one requested RC branch, and distribute the initial
            junction-to-ambient temperature rise in proportion to resistance.
            Reject disabled configurations because PaModel represents disabled
            operation by omitting the ThermalNetwork entirely.

        Args:
            config: Validated thermal-network and temperature-drift settings.

        Returns:
            result: None. Thermal nodes are ready for causal time advancement.
        """

        config.Validate()
        if not config.enabled:
            raise ValueError(
                "ThermalNetwork requires ThermalConfig.enabled=True; "
                "bind a disabled ThermalConfig to PaModel to bypass all "
                "temperature effects"
            )
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
        self.temperatureRisePerBranchC = self.CalculateAdvancedState(
            self.temperatureRisePerBranchC,
            resolvedPower,
            resolvedDuration,
        )
        self.elapsedTimeSec += resolvedDuration
        return self.CurrentTemperatureC()

    def CalculateAdvancedState(
        self,
        startingTemperatureRisePerBranchC: np.ndarray,
        dissipatedPowerW: float,
        durationSec: float,
    ) -> np.ndarray:
        """Calculate one exact thermal step without changing live state.

        Processing details:
            Algorithm: Validate a complete branch-state vector and apply the
            zero-order-hold RC solution independently to every Foster branch.
            Static mode returns an unchanged copy because its junction
            temperature is selected explicitly rather than integrated.

        Args:
            startingTemperatureRisePerBranchC: Branch rises at interval start.
            dissipatedPowerW: Nonnegative mean heat input during the interval.
            durationSec: Nonnegative interval duration in seconds.

        Returns:
            result: New branch-temperature-rise vector without side effects.
        """

        thermalRuntime = _ThermalRuntime.FromValidatedConfig(self.config)
        return self.CalculateAdvancedStateResolved(
            startingTemperatureRisePerBranchC,
            dissipatedPowerW,
            durationSec,
            thermalRuntime,
        )

    def CalculateAdvancedStateResolved(
        self,
        startingTemperatureRisePerBranchC: np.ndarray,
        dissipatedPowerW: float,
        durationSec: float,
        thermalRuntime: _ThermalRuntime,
    ) -> np.ndarray:
        """Advance one state using operation-local thermal constants.

        Processing details:
            Algorithm: Retain the public state, power, and duration checks,
            then apply the exact RC step with the already selected read-only
            resistance and time-constant arrays.

        Args:
            startingTemperatureRisePerBranchC: Branch rises before the step.
            dissipatedPowerW: Nonnegative mean heat power for the step.
            durationSec: Nonnegative physical step duration in seconds.
            thermalRuntime: Validated constants shared by the current period.

        Returns:
            result: Branch temperature rises after the requested step.
        """

        startingState = np.asarray(
            startingTemperatureRisePerBranchC, dtype=float
        )
        if startingState.shape != self.temperatureRisePerBranchC.shape:
            raise ValueError(
                "startingTemperatureRisePerBranchC has an incompatible shape"
            )
        if not np.all(np.isfinite(startingState)):
            raise ValueError(
                "startingTemperatureRisePerBranchC must contain finite values"
            )
        resolvedPower = float(dissipatedPowerW)
        resolvedDuration = float(durationSec)
        if (
            not np.isfinite(resolvedPower)
            or resolvedPower < 0.0
            or not np.isfinite(resolvedDuration)
            or resolvedDuration < 0.0
        ):
            raise ValueError(
                "thermal power and duration must be finite and nonnegative"
            )
        if thermalRuntime.modelName == "static":
            return startingState.copy()
        decayValues = np.exp(
            -resolvedDuration / thermalRuntime.timeConstantValues
        )
        return np.asarray(
            decayValues * startingState
            + thermalRuntime.resistanceValues
            * resolvedPower
            * (1.0 - decayValues),
            dtype=float,
        )

    def CalculatePeriodicSteadyState(
        self,
        dissipatedPowersW: Sequence[float],
        durationsSec: Sequence[float],
    ) -> np.ndarray:
        """Solve the branch state that repeats after one power schedule.

        Processing details:
            Algorithm: Starting from zero branch rise, compose every exact
            constant-power interval to obtain the additive term of the full
            cycle. Divide it by one minus the full-cycle decay on each branch,
            using ``expm1`` for short-period numerical accuracy. This is an
            analytic periodic solution for a frozen dissipated-power trace.

        Args:
            dissipatedPowersW: Mean heat power for each consecutive interval.
            durationsSec: Physical duration paired with every power value.

        Returns:
            result: Branch rises at both the beginning and end of the cycle.
        """

        thermalRuntime = _ThermalRuntime.FromValidatedConfig(self.config)
        return self.CalculatePeriodicSteadyStateResolved(
            dissipatedPowersW,
            durationsSec,
            thermalRuntime,
        )

    def CalculatePeriodicSteadyStateResolved(
        self,
        dissipatedPowersW: Sequence[float],
        durationsSec: Sequence[float],
        thermalRuntime: _ThermalRuntime,
    ) -> np.ndarray:
        """Solve one frozen heat schedule using cached branch constants.

        Processing details:
            Algorithm: Validate the complete power-duration schedule once,
            compose every exact RC step with the same branch arrays, and solve
            the repeating initial state using the full-period decay.

        Args:
            dissipatedPowersW: Consecutive nonnegative heat-power values.
            durationsSec: Physical durations paired with the heat values.
            thermalRuntime: Validated constants shared by the current solver.

        Returns:
            result: Branch rises that repeat at the period boundaries.
        """

        powerValues = np.asarray(dissipatedPowersW, dtype=float)
        durationValues = np.asarray(durationsSec, dtype=float)
        if (
            powerValues.ndim != 1
            or durationValues.ndim != 1
            or powerValues.size == 0
            or powerValues.size != durationValues.size
        ):
            raise ValueError(
                "dissipatedPowersW and durationsSec must be matching "
                "nonempty one-dimensional sequences"
            )
        if (
            not np.all(np.isfinite(powerValues))
            or np.any(powerValues < 0.0)
            or not np.all(np.isfinite(durationValues))
            or np.any(durationValues < 0.0)
            or float(np.sum(durationValues)) <= 0.0
        ):
            raise ValueError(
                "periodic powers must be nonnegative and durations must "
                "define one positive finite period"
            )
        if thermalRuntime.modelName == "static":
            return self.temperatureRisePerBranchC.copy()
        additiveState = np.zeros_like(
            self.temperatureRisePerBranchC, dtype=float
        )
        for dissipatedPowerW, durationSec in zip(
            powerValues, durationValues
        ):
            additiveState = self.CalculateAdvancedStateResolved(
                additiveState,
                float(dissipatedPowerW),
                float(durationSec),
                thermalRuntime,
            )
        totalDurationSec = float(np.sum(durationValues))
        oneMinusCycleDecay = -np.expm1(
            -totalDurationSec / thermalRuntime.timeConstantValues
        )
        return np.asarray(
            additiveState / oneMinusCycleDecay, dtype=float
        )

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
    entries are treated as zero. When no dictionaries are supplied, the
    generator creates a mildly compressive Rapp-like static curve that is
    monotonic for normalized constant-envelope amplitudes from zero through
    two, then adds small zero-sum dynamic memory residuals around that curve.
    Custom dictionaries remain unconstrained behavioral fits and may therefore
    contain stronger droop, hysteresis, or polynomial foldback.
    """

    nonlinearOrders: Tuple[int, ...] = (1, 3, 5, 7)
    memoryDepth: int = 3
    crossMemoryDepth: int = 2
    nonlinearScale: float = 0.135
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
        if (
            not isinstance(self.nonlinearScale, (int, float))
            or isinstance(self.nonlinearScale, bool)
            or not np.isfinite(self.nonlinearScale)
            or not 0.0 <= float(self.nonlinearScale) <= 1.0
        ):
            raise ValueError(
                "nonlinearScale must be finite and between zero and one"
            )


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
            config.nonlinearScale,
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

        # A default GMP expansion references only a handful of unique causal
        # delays, although the same delayed vector appears in many main,
        # lagging, and leading terms. Build each delayed waveform and each
        # delay/order envelope power once per call. The coefficient iteration
        # order below remains unchanged, so accumulation and causal boundary
        # behavior match the direct polynomial expansion.
        requiredDelays = set()
        requiredEnvelopePowers = set()
        for nonlinearOrder, memoryIndex in self.mainCoefficients:
            requiredDelays.add(memoryIndex)
            requiredEnvelopePowers.add((memoryIndex, nonlinearOrder))
        for (
            nonlinearOrder,
            memoryIndex,
            crossIndex,
        ) in self.laggingCoefficients:
            requiredDelays.add(memoryIndex)
            requiredDelays.add(memoryIndex + crossIndex)
            requiredEnvelopePowers.add(
                (memoryIndex + crossIndex, nonlinearOrder)
            )
        for (
            nonlinearOrder,
            memoryIndex,
            crossIndex,
        ) in self.leadingCoefficients:
            requiredDelays.add(memoryIndex)
            requiredDelays.add(memoryIndex + crossIndex)
            requiredEnvelopePowers.add(
                (memoryIndex, nonlinearOrder)
            )
        delayedSignals = {
            sampleDelay: DelaySignal(complexInput, sampleDelay)
            for sampleDelay in requiredDelays
        }
        envelopePowers = {
            (sampleDelay, nonlinearOrder): (
                np.abs(delayedSignals[sampleDelay])
                ** (nonlinearOrder - 1)
            )
            for sampleDelay, nonlinearOrder in requiredEnvelopePowers
        }

        # Main branch: x[n-m] * |x[n-m]|^(p-1).
        for (nonlinearOrder, memoryIndex), coefficient in self.mainCoefficients.items():
            delayedSignal = delayedSignals[memoryIndex]
            outputSignal += (
                coefficient
                * delayedSignal
                * envelopePowers[(memoryIndex, nonlinearOrder)]
            )

        # Lagging envelope branch:
        # x[n-m] * |x[n-m-l]|^(p-1).
        for (
            nonlinearOrder,
            memoryIndex,
            crossIndex,
        ), coefficient in self.laggingCoefficients.items():
            carrierSignal = delayedSignals[memoryIndex]
            outputSignal += (
                coefficient
                * carrierSignal
                * envelopePowers[
                    (memoryIndex + crossIndex, nonlinearOrder)
                ]
            )

        # Leading envelope branch:
        # x[n-m-l] * |x[n-m]|^(p-1).
        for (
            nonlinearOrder,
            memoryIndex,
            crossIndex,
        ), coefficient in self.leadingCoefficients.items():
            carrierSignal = delayedSignals[memoryIndex + crossIndex]
            outputSignal += (
                coefficient
                * carrierSignal
                * envelopePowers[(memoryIndex, nonlinearOrder)]
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
class PiecewiseGMPConfig:
    """Configure smooth envelope-region blending of multiple GMP models.

    ``regionBoundaries`` contains the normalized input-envelope boundaries
    between adjacent GMP regions. ``transitionWidths`` gives the complete
    compact blending interval around each matching boundary. An explicit
    ``regionConfigs`` tuple supplies one ``GMPConfig`` per region. When it is
    omitted, three sparse built-in GMP regions provide modest changes in
    static compression, AM-PM, and electrical memory without adding noise.
    """

    regionBoundaries: Tuple[float, ...] = (0.25, 0.60)
    transitionWidths: Tuple[float, ...] = (0.12, 0.18)
    regionConfigs: Optional[Tuple[GMPConfig, ...]] = None

    def Validate(self) -> None:
        """Validate region ordering, compact transitions, and GMP configs.

        Processing details:
            Algorithm: Require at least two amplitude regions, pair every
            positive finite boundary with one positive finite transition
            width, keep the first transition above zero, and validate one
            optional GMP configuration per resulting region. Ordered gates may
            overlap because hierarchical region weights remain nonnegative.

        Returns:
            result: None. Invalid region geometry or coefficients raise an
                exception before the PA is constructed.
        """

        if not isinstance(self.regionBoundaries, tuple):
            raise TypeError("regionBoundaries must be a tuple")
        if not isinstance(self.transitionWidths, tuple):
            raise TypeError("transitionWidths must be a tuple")
        if len(self.regionBoundaries) == 0:
            raise ValueError("regionBoundaries cannot be empty")
        if len(self.transitionWidths) != len(self.regionBoundaries):
            raise ValueError(
                "transitionWidths must match regionBoundaries in length"
            )
        boundaries = []
        widths = []
        for boundaryIndex, (boundaryValue, widthValue) in enumerate(
            zip(self.regionBoundaries, self.transitionWidths)
        ):
            if (
                not isinstance(boundaryValue, (int, float))
                or isinstance(boundaryValue, bool)
                or not np.isfinite(boundaryValue)
                or float(boundaryValue) <= 0.0
            ):
                raise ValueError(
                    f"regionBoundaries[{boundaryIndex}] must be finite "
                    "and positive"
                )
            if (
                not isinstance(widthValue, (int, float))
                or isinstance(widthValue, bool)
                or not np.isfinite(widthValue)
                or float(widthValue) <= 0.0
            ):
                raise ValueError(
                    f"transitionWidths[{boundaryIndex}] must be finite "
                    "and positive"
                )
            boundaries.append(float(boundaryValue))
            widths.append(float(widthValue))
        if any(
            laterBoundary <= earlierBoundary
            for earlierBoundary, laterBoundary in zip(
                boundaries[:-1], boundaries[1:]
            )
        ):
            raise ValueError("regionBoundaries must be strictly increasing")
        if boundaries[0] - 0.5 * widths[0] <= 0.0:
            raise ValueError(
                "the first transition interval must start above zero"
            )
        if self.regionConfigs is None:
            if len(boundaries) != 2:
                raise ValueError(
                    "built-in piecewise GMP requires exactly two region "
                    "boundaries; supply regionConfigs for another count"
                )
            return
        if not isinstance(self.regionConfigs, tuple):
            raise TypeError("regionConfigs must be a tuple or None")
        expectedRegionCount = len(boundaries) + 1
        if len(self.regionConfigs) != expectedRegionCount:
            raise ValueError(
                "regionConfigs must contain exactly one GMPConfig per region"
            )
        for regionIndex, regionConfig in enumerate(self.regionConfigs):
            if not isinstance(regionConfig, GMPConfig):
                raise TypeError(
                    f"regionConfigs[{regionIndex}] must be a GMPConfig"
                )
            regionConfig.Validate()


class PiecewiseGMPPA:
    """Blend adjacent sparse GMP models over smooth envelope regions."""

    def __init__(
        self,
        config: PiecewiseGMPConfig = PiecewiseGMPConfig(),
    ) -> None:
        """Initialize explicit or deterministic built-in regional GMPs.

        Processing details:
            Algorithm: Validate compact transition geometry, use caller-owned
            regional GMP configurations when supplied, or derive three sparse
            defaults from one common GMP expansion. The built-in profiles
            preserve each region's settled target while varying gain, AM-PM,
            nonlinearity, and zero-sum memory residual strength so ordinary
            GMP predistortion sees controlled structural model mismatch.

        Args:
            config: Piecewise transition and regional GMP configuration.

        Returns:
            result: None. The immutable configuration and regional models are
                ready for deterministic processing.
        """

        config.Validate()
        self.config = config
        if config.regionConfigs is None:
            nonlinearOrders = (1, 3, 5)
            memoryDepth = 2
            crossMemoryDepth = 1
            baseMain, baseLagging, baseLeading = DefaultGmpCoefficients(
                nonlinearOrders,
                memoryDepth,
                crossMemoryDepth,
                nonlinearScale=1.0,
            )
            regionProfiles = (
                (1.025, -0.004, 0.78, 0.72),
                (1.000, 0.018, 1.00, 1.00),
                (0.955, 0.055, 1.04, 1.35),
            )
            regionConfigurations = []
            for (
                gainScale,
                phaseRotationRadians,
                nonlinearScale,
                memoryScale,
            ) in regionProfiles:
                complexScale = gainScale * np.exp(
                    1j * phaseRotationRadians
                )
                mainCoefficients: Dict[Tuple[int, int], complex] = {}
                laggingCoefficients: Dict[
                    Tuple[int, int, int], complex
                ] = {}
                leadingCoefficients: Dict[
                    Tuple[int, int, int], complex
                ] = {}
                for nonlinearOrder in nonlinearOrders:
                    orderScale = complexScale * (
                        1.0 if nonlinearOrder == 1 else nonlinearScale
                    )
                    targetCoefficient = orderScale * (
                        sum(
                            coefficient
                            for (
                                order,
                                _memoryIndex,
                            ), coefficient in baseMain.items()
                            if order == nonlinearOrder
                        )
                        + sum(
                            coefficient
                            for (
                                order,
                                _memoryIndex,
                                _crossIndex,
                            ), coefficient in baseLagging.items()
                            if order == nonlinearOrder
                        )
                        + sum(
                            coefficient
                            for (
                                order,
                                _memoryIndex,
                                _crossIndex,
                            ), coefficient in baseLeading.items()
                            if order == nonlinearOrder
                        )
                    )
                    for (
                        order,
                        memoryIndex,
                    ), coefficient in baseMain.items():
                        if order != nonlinearOrder or memoryIndex == 0:
                            continue
                        mainCoefficients[(order, memoryIndex)] = (
                            orderScale * memoryScale * coefficient
                        )
                    for coefficientKey, coefficient in baseLagging.items():
                        if coefficientKey[0] != nonlinearOrder:
                            continue
                        laggingCoefficients[coefficientKey] = (
                            orderScale * memoryScale * coefficient
                        )
                    for coefficientKey, coefficient in baseLeading.items():
                        if coefficientKey[0] != nonlinearOrder:
                            continue
                        leadingCoefficients[coefficientKey] = (
                            orderScale * memoryScale * coefficient
                        )
                    dynamicCoefficientSum = sum(
                        coefficient
                        for (order, memoryIndex), coefficient in (
                            mainCoefficients.items()
                        )
                        if order == nonlinearOrder and memoryIndex > 0
                    ) + sum(
                        coefficient
                        for (order, _, _), coefficient in (
                            laggingCoefficients.items()
                        )
                        if order == nonlinearOrder
                    ) + sum(
                        coefficient
                        for (order, _, _), coefficient in (
                            leadingCoefficients.items()
                        )
                        if order == nonlinearOrder
                    )
                    mainCoefficients[(nonlinearOrder, 0)] = (
                        targetCoefficient - dynamicCoefficientSum
                    )
                regionConfigurations.append(
                    GMPConfig(
                        nonlinearOrders=nonlinearOrders,
                        memoryDepth=memoryDepth,
                        crossMemoryDepth=crossMemoryDepth,
                        mainCoefficients=mainCoefficients,
                        laggingCoefficients=laggingCoefficients,
                        leadingCoefficients=leadingCoefficients,
                    )
                )
            resolvedRegionConfigs = tuple(regionConfigurations)
        else:
            resolvedRegionConfigs = config.regionConfigs
        self.regionModels = tuple(
            GMPPA(regionConfig) for regionConfig in resolvedRegionConfigs
        )
        mainCoefficientSets = tuple(
            regionModel.mainCoefficients
            for regionModel in self.regionModels
        )
        laggingCoefficientSets = tuple(
            regionModel.laggingCoefficients
            for regionModel in self.regionModels
        )
        leadingCoefficientSets = tuple(
            regionModel.leadingCoefficients
            for regionModel in self.regionModels
        )
        mainKeys = set().union(
            *(coefficientSet.keys() for coefficientSet in mainCoefficientSets)
        )
        laggingKeys = set().union(
            *(
                coefficientSet.keys()
                for coefficientSet in laggingCoefficientSets
            )
        )
        leadingKeys = set().union(
            *(
                coefficientSet.keys()
                for coefficientSet in leadingCoefficientSets
            )
        )
        self._mainTerms = []
        for coefficientKey in sorted(mainKeys):
            coefficientValues = tuple(
                coefficientSet.get(coefficientKey, 0.0 + 0.0j)
                for coefficientSet in mainCoefficientSets
            )
            coefficientDifferences = tuple(
                laterCoefficient - earlierCoefficient
                for earlierCoefficient, laterCoefficient in zip(
                    coefficientValues[:-1], coefficientValues[1:]
                )
            )
            if coefficientValues[0] != 0.0 + 0.0j or any(
                value != 0.0 + 0.0j
                for value in coefficientDifferences
            ):
                self._mainTerms.append(
                    (
                        coefficientKey,
                        coefficientValues[0],
                        coefficientDifferences,
                    )
                )
        self._laggingTerms = []
        for coefficientKey in sorted(laggingKeys):
            coefficientValues = tuple(
                coefficientSet.get(coefficientKey, 0.0 + 0.0j)
                for coefficientSet in laggingCoefficientSets
            )
            coefficientDifferences = tuple(
                laterCoefficient - earlierCoefficient
                for earlierCoefficient, laterCoefficient in zip(
                    coefficientValues[:-1], coefficientValues[1:]
                )
            )
            if coefficientValues[0] != 0.0 + 0.0j or any(
                value != 0.0 + 0.0j
                for value in coefficientDifferences
            ):
                self._laggingTerms.append(
                    (
                        coefficientKey,
                        coefficientValues[0],
                        coefficientDifferences,
                    )
                )
        self._leadingTerms = []
        for coefficientKey in sorted(leadingKeys):
            coefficientValues = tuple(
                coefficientSet.get(coefficientKey, 0.0 + 0.0j)
                for coefficientSet in leadingCoefficientSets
            )
            coefficientDifferences = tuple(
                laterCoefficient - earlierCoefficient
                for earlierCoefficient, laterCoefficient in zip(
                    coefficientValues[:-1], coefficientValues[1:]
                )
            )
            if coefficientValues[0] != 0.0 + 0.0j or any(
                value != 0.0 + 0.0j
                for value in coefficientDifferences
            ):
                self._leadingTerms.append(
                    (
                        coefficientKey,
                        coefficientValues[0],
                        coefficientDifferences,
                    )
                )
        requiredDelays = set()
        requiredEnvelopePowers = set()
        for (nonlinearOrder, memoryIndex), _, _ in self._mainTerms:
            requiredDelays.add(memoryIndex)
            requiredEnvelopePowers.add((memoryIndex, nonlinearOrder))
        for (
            nonlinearOrder,
            memoryIndex,
            crossIndex,
        ), _, _ in self._laggingTerms:
            requiredDelays.add(memoryIndex)
            requiredDelays.add(memoryIndex + crossIndex)
            requiredEnvelopePowers.add(
                (memoryIndex + crossIndex, nonlinearOrder)
            )
        for (
            nonlinearOrder,
            memoryIndex,
            crossIndex,
        ), _, _ in self._leadingTerms:
            requiredDelays.add(memoryIndex)
            requiredDelays.add(memoryIndex + crossIndex)
            requiredEnvelopePowers.add((memoryIndex, nonlinearOrder))
        self._requiredDelays = tuple(sorted(requiredDelays))
        self._requiredEnvelopePowers = tuple(
            sorted(requiredEnvelopePowers)
        )

    def Process(self, inputSignal: np.ndarray) -> np.ndarray:
        """Evaluate shared GMP bases with smoothly varying coefficients.

        Processing details:
            Algorithm: Convert each compact transition interval into a C2
            smootherstep gate, cache every unique delayed signal and envelope
            power needed by any region once, and evaluate each unioned GMP
            basis with a coefficient equal to the first region plus gated
            adjacent-region coefficient differences. Each successive gate is
            multiplied by all earlier gates, yielding low/middle/high weights
            ``1-S1``, ``S1*(1-S2)``, and ``S1*S2`` for the default model; the
            weights remain nonnegative and sum to one even if gates overlap.

        Args:
            inputSignal: One-dimensional normalized complex baseband samples.

        Returns:
            result: Same-length deterministic complex piecewise-GMP output.
        """

        complexInput = AsComplexVector(inputSignal)
        inputEnvelope = np.abs(complexInput)
        transitionFunctions = []
        for boundaryValue, widthValue in zip(
            self.config.regionBoundaries,
            self.config.transitionWidths,
        ):
            transitionStart = float(boundaryValue) - 0.5 * float(widthValue)
            normalizedPosition = np.clip(
                (inputEnvelope - transitionStart) / float(widthValue),
                0.0,
                1.0,
            )
            transitionFunctions.append(
                normalizedPosition**3
                * (
                    10.0
                    + normalizedPosition
                    * (-15.0 + 6.0 * normalizedPosition)
                )
            )
        regionActivationFunctions = []
        cumulativeActivation = np.ones_like(inputEnvelope)
        for transitionFunction in transitionFunctions:
            cumulativeActivation = (
                cumulativeActivation * transitionFunction
            )
            regionActivationFunctions.append(cumulativeActivation)

        delayedSignals = {
            sampleDelay: DelaySignal(complexInput, sampleDelay)
            for sampleDelay in self._requiredDelays
        }
        envelopePowers = {
            (sampleDelay, nonlinearOrder): (
                np.abs(delayedSignals[sampleDelay])
                ** (nonlinearOrder - 1)
            )
            for sampleDelay, nonlinearOrder in (
                self._requiredEnvelopePowers
            )
        }
        outputSignal = np.zeros_like(complexInput)

        for (
            (nonlinearOrder, memoryIndex),
            initialCoefficient,
            coefficientDifferences,
        ) in self._mainTerms:
            basisSignal = (
                delayedSignals[memoryIndex]
                * envelopePowers[(memoryIndex, nonlinearOrder)]
            )
            outputSignal += initialCoefficient * basisSignal
            for coefficientDifference, regionActivation in zip(
                coefficientDifferences,
                regionActivationFunctions,
            ):
                if coefficientDifference != 0.0 + 0.0j:
                    outputSignal += (
                        coefficientDifference
                        * regionActivation
                        * basisSignal
                    )

        for (
            (nonlinearOrder, memoryIndex, crossIndex),
            initialCoefficient,
            coefficientDifferences,
        ) in self._laggingTerms:
            basisSignal = (
                delayedSignals[memoryIndex]
                * envelopePowers[
                    (memoryIndex + crossIndex, nonlinearOrder)
                ]
            )
            outputSignal += initialCoefficient * basisSignal
            for coefficientDifference, regionActivation in zip(
                coefficientDifferences,
                regionActivationFunctions,
            ):
                if coefficientDifference != 0.0 + 0.0j:
                    outputSignal += (
                        coefficientDifference
                        * regionActivation
                        * basisSignal
                    )

        for (
            (nonlinearOrder, memoryIndex, crossIndex),
            initialCoefficient,
            coefficientDifferences,
        ) in self._leadingTerms:
            basisSignal = (
                delayedSignals[memoryIndex + crossIndex]
                * envelopePowers[(memoryIndex, nonlinearOrder)]
            )
            outputSignal += initialCoefficient * basisSignal
            for coefficientDifference, regionActivation in zip(
                coefficientDifferences,
                regionActivationFunctions,
            ):
                if coefficientDifference != 0.0 + 0.0j:
                    outputSignal += (
                        coefficientDifference
                        * regionActivation
                        * basisSignal
                    )
        return outputSignal

    def SmallSignalGain(self) -> complex:
        """Return the first region's exact small-envelope gain.

        Processing details:
            Algorithm: The first compact transition begins above zero, so the
            origin uses only region zero; return that GMP model's first-order
            DC gain without evaluating any transition arrays.

        Returns:
            result: Complex small-signal gain of the lowest-amplitude region.
        """

        return self.regionModels[0].SmallSignalGain()


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
    peakingInputGain: float = 0.85
    peakingTurnOnAmplitude: float = 0.45
    peakingTransitionWidth: float = 0.50
    carrierCombineCoefficient: complex = 1.0 + 0.0j
    peakingCombineCoefficient: complex = 0.15 + 0.0j
    peakingDelaySamples: int = 0
    loadModulationStrength: float = 0.02

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
    """Operate one Rapp, Wiener, GMP, piecewise-GMP, or Doherty PA model.

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
        piecewiseGmpConfig: Optional[PiecewiseGMPConfig] = None,
        dohertyConfig: Optional[DohertyConfig] = None,
        thermalConfig: Optional[ThermalConfig] = None,
        parameters: Optional[Mapping[str, object]] = None,
        width: Optional[int] = None,
        outputFullScaleAmplitude: Optional[float] = None,
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
            piecewiseGmpConfig: Optional smooth regional GMP configuration.
            dohertyConfig: Optional carrier/peaking Doherty configuration.
            thermalConfig: Optional self-heating and temperature-drift model.
            parameters: Optional external mapping layered ahead of the built-in defaults.
            width: Optional external I/Q width. None selects the internal
                16-bit default, zero selects floating point, and a positive
                value selects signed integer I/Q codes in complex128.
            outputFullScaleAmplitude: Optional physical magnitude represented
                by the PA output code rail. The 2.0 default gives the
                observation path 6.02 dB of component headroom while the
                input DAC remains normalized to one.
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
                "piecewiseGmpConfig": None,
                "dohertyConfig": None,
                "thermalConfig": None,
                "width": 16,
                "outputFullScaleAmplitude": 2.0,
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
        if piecewiseGmpConfig is not None:
            directOverrides["piecewiseGmpConfig"] = piecewiseGmpConfig
        if dohertyConfig is not None:
            directOverrides["dohertyConfig"] = dohertyConfig
        if thermalConfig is not None:
            directOverrides["thermalConfig"] = thermalConfig
        if width is not None:
            directOverrides["width"] = width
        if outputFullScaleAmplitude is not None:
            directOverrides["outputFullScaleAmplitude"] = (
                outputFullScaleAmplitude
            )
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
        self._thermalEffectsSuspended = False
        self._externalTemperatureOffsetC = 0.0
        self._lastThermalMetrics: Dict[str, object] = {}
        self._calibrationDriveDb = 0.0
        self._activeConfiguration: Optional[
            Tuple[
                str,
                Optional[RappConfig],
                Optional[WienerConfig],
                Optional[GMPConfig],
                Optional[PiecewiseGMPConfig],
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

        normalizedName, _, _, _, _, _ = self.ResolveConfiguration()
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

    @property
    def OutputFullScaleAmplitude(self) -> float:
        """Return the physical magnitude represented by a PA output code rail.

        Processing details:
            Algorithm: Resolve the live observation scale independently of
            the normalized input DAC convention.

        Returns:
            result: Positive physical I/Q component full-scale amplitude.
        """

        return float(
            cast(float, self.parameters["outputFullScaleAmplitude"])
        )

    outputFullScaleAmplitude = OutputFullScaleAmplitude

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
            preserve the explicit electrical-only suspension state during a
            calibration transaction, retain state while an enabled immutable
            configuration is unchanged, and initialize a new network only when
            the caller changes enabled settings outside that transaction.

        Returns:
            result: None. Thermal state matches the active parameter mapping.
        """

        thermalConfig = self.ResolveThermalConfig()
        if thermalConfig is None or not thermalConfig.enabled:
            self.thermalNetwork = None
            self._activeThermalConfig = thermalConfig
            # Disabled means no self-heating and no retained neighbor-induced
            # temperature shift. Clearing both prevents a previous MIMO run
            # from resurfacing stale thermal drift after later re-enablement.
            self._externalTemperatureOffsetC = 0.0
            self._lastThermalMetrics = {}
            return
        if self._thermalEffectsSuspended:
            # Suspension is an explicit transaction state. Do not recreate a
            # thermal network merely because processing synchronizes live
            # parameters while electrical-only calibration is in progress.
            self.thermalNetwork = None
            return
        if (
            thermalConfig == self._activeThermalConfig
            and self.thermalNetwork is not None
        ):
            return
        self.thermalNetwork = ThermalNetwork(thermalConfig)
        self._activeThermalConfig = thermalConfig
        self._lastThermalMetrics = self.thermalNetwork.GetMetrics()

    def SuspendThermalModel(self) -> Optional[Dict[str, object]]:
        """Suspend self-heating while preserving an exact restorable snapshot.

        Processing details:
            Algorithm: Copy the active configuration, branch temperatures,
            ambient value, elapsed time, and latest diagnostics, then disable
            the network reference and set an explicit suspension flag so every
            temperature application path evaluates the electrical model at its
            fixed reference temperature without producing heat. Reject nested
            suspension because one snapshot must have one matching restore.

        Returns:
            result: Snapshot dictionary, or None when thermal modeling is off.
        """

        self.SynchronizeThermalModel()
        if self._thermalEffectsSuspended:
            raise RuntimeError("thermal model is already suspended")
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
        self._thermalEffectsSuspended = True
        self.thermalNetwork = None
        return snapshot

    def RestoreThermalModel(
        self,
        thermalSnapshot: Optional[Mapping[str, object]],
    ) -> None:
        """Restore thermal state saved before temperature-independent work.

        Processing details:
            Algorithm: Treat None as a no-op. If live configuration was disabled
            during the transaction, clear suspension without reviving the old
            snapshot. Otherwise reconstruct the matching validated network,
            restore branch temperatures, ambient, elapsed time, mutual offset,
            and latest diagnostics without advancing heat. A different enabled
            configuration starts a fresh, topology-compatible network.

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
        currentThermalConfig = self.ResolveThermalConfig()
        if (
            currentThermalConfig is None
            or not currentThermalConfig.enabled
        ):
            # A live disable performed while calibration was running is
            # authoritative. Never revive the older enabled snapshot.
            self.thermalNetwork = None
            self._activeThermalConfig = currentThermalConfig
            self._externalTemperatureOffsetC = 0.0
            self._lastThermalMetrics = {}
            self._thermalEffectsSuspended = False
            return
        if currentThermalConfig != thermalConfig:
            # A different enabled configuration must start its own network;
            # branch states from another topology or parameter set are invalid.
            self._thermalEffectsSuspended = False
            self._activeThermalConfig = None
            self.SynchronizeThermalModel()
            return
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
        self._thermalEffectsSuspended = False

    def ResolveConfiguration(
        self,
    ) -> Tuple[
        str,
        Optional[RappConfig],
        Optional[WienerConfig],
        Optional[GMPConfig],
        Optional[PiecewiseGMPConfig],
        Optional[DohertyConfig],
    ]:
        """Validate and return the currently resolved PA configuration.

        Processing details:
            Algorithm: Resolve values according to state and ChainMap precedence, keeping caller-owned configuration behavior explicit.

        Returns:
            result: Model name followed by optional Rapp, Wiener, GMP,
                piecewise-GMP, and Doherty configurations in deterministic
                order.
        """

        rawModelName = self.parameters["modelName"]
        if not isinstance(rawModelName, str):
            raise TypeError("modelName must be a string")
        normalizedName = rawModelName.strip().lower()
        if normalizedName not in (
            "rapp",
            "wiener",
            "gmp",
            "piecewise_gmp",
            "doherty",
        ):
            raise ValueError(
                "modelName must be 'rapp', 'wiener', 'gmp', "
                "'piecewise_gmp', or 'doherty'"
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
        rawPiecewiseGmpConfig = self.parameters["piecewiseGmpConfig"]
        if rawPiecewiseGmpConfig is not None and not isinstance(
            rawPiecewiseGmpConfig, PiecewiseGMPConfig
        ):
            raise TypeError(
                "piecewiseGmpConfig must be a PiecewiseGMPConfig or None"
            )
        rawDohertyConfig = self.parameters["dohertyConfig"]
        if rawDohertyConfig is not None and not isinstance(
            rawDohertyConfig, DohertyConfig
        ):
            raise TypeError(
                "dohertyConfig must be a DohertyConfig or None"
            )
        FixedPoint(self.width)
        FixedPoint(
            self.width,
            self.parameters["outputFullScaleAmplitude"],
        )
        self.ResolveThermalConfig()
        return (
            normalizedName,
            cast(Optional[RappConfig], rawRappConfig),
            cast(Optional[WienerConfig], rawWienerConfig),
            cast(Optional[GMPConfig], rawGmpConfig),
            cast(Optional[PiecewiseGMPConfig], rawPiecewiseGmpConfig),
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
            piecewiseGmpConfig,
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
        elif normalizedName == "piecewise_gmp":
            selectedModel = PiecewiseGMPPA(
                PiecewiseGMPConfig()
                if piecewiseGmpConfig is None
                else piecewiseGmpConfig
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
            decode them to a normalized floating envelope, apply the most
            recently committed post-DAC calibration drive, evaluate the PA,
            and encode the floating result back to integer-valued public codes.

        Args:
            inputSignal: One-dimensional complex baseband samples supplied to the operation.

        Returns:
            result: Complex128 samples containing raw I/Q codes in fixed mode
                or physical floating samples when ``width`` equals zero.
        """

        self.SynchronizeModel()
        self.SynchronizeThermalModel()
        inputFormat = FixedPoint(self.width)
        outputFormat = FixedPoint(
            self.width, self.outputFullScaleAmplitude
        )
        floatingInput = inputFormat.DecodeComplex(inputSignal)
        floatingOutput, _ = self.ProcessOutputPathsFloating(floatingInput)
        return outputFormat.EncodeComplex(floatingOutput)

    def ProcessOutputPathsFloating(
        self, inputSignal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process normalized samples at the committed calibrated drive.

        Processing details:
            Algorithm: Preserve ``ProcessFloating`` as the raw normalized PA
            kernel, but apply the hidden post-DAC analog drive committed by
            ``PowerCalibration`` before evaluating it. Return independent
            channel and feedback copies because a bare PA has no separate
            receiver path. This protocol lets fixed-point ILC retain the
            calibrated physical operating point after decoding public codes.

        Args:
            inputSignal: Normalized complex samples after the public decoder.

        Returns:
            result: Identical ``(chOut, fbOut)`` floating PA observations.
        """

        self.SynchronizeModel()
        self.SynchronizeThermalModel()
        complexInput = np.asarray(inputSignal, dtype=np.complex128)
        if complexInput.size == 0 or not np.all(np.isfinite(complexInput)):
            raise ValueError("inputSignal must contain finite samples")
        driveScale = np.power(10.0, self._calibrationDriveDb / 20.0)
        floatingOutput = np.asarray(
            self.ProcessFloating(driveScale * complexInput),
            dtype=np.complex128,
        )
        return floatingOutput, floatingOutput.copy()

    def ResolveCalibrationDriveDb(
        self, driveDbPerChain: Sequence[float]
    ) -> float:
        """Validate and return the single-chain analog calibration drive.

        Processing details:
            Algorithm: Require exactly one finite real dB value because this
            facade owns one physical PA. Reject booleans and nested values so
            a calibration adapter cannot silently address the wrong chain.

        Args:
            driveDbPerChain: One-element sequence containing analog drive dB.

        Returns:
            result: Validated scalar analog drive in decibels.
        """

        if isinstance(driveDbPerChain, (str, bytes)):
            raise TypeError("driveDbPerChain must be a one-element sequence")
        driveArray = np.asarray(driveDbPerChain, dtype=object)
        if driveArray.ndim != 1 or driveArray.size != 1:
            raise ValueError(
                "driveDbPerChain must contain exactly one PA drive value"
            )
        driveValue = driveArray[0]
        if (
            not isinstance(
                driveValue,
                (int, float, np.integer, np.floating),
            )
            or isinstance(driveValue, (bool, np.bool_))
            or not np.isfinite(driveValue)
        ):
            raise ValueError(
                "driveDbPerChain must contain one finite real dB value"
            )
        return float(driveValue)

    def SetCalibrationDriveDb(
        self, driveDbPerChain: Sequence[float]
    ) -> None:
        """Commit the hidden post-DAC drive selected by power calibration.

        Processing details:
            Algorithm: Validate the one-chain drive and store it independently
            from the caller's PA coefficient mapping. Later public ``Process``
            calls reproduce the accepted operating point without asking the
            caller to scale fixed-point codes beyond their legal range.

        Args:
            driveDbPerChain: One-element sequence containing analog drive dB.

        Returns:
            result: None. The accepted drive applies to later public calls.
        """

        self._calibrationDriveDb = self.ResolveCalibrationDriveDb(
            driveDbPerChain
        )

    def ProcessCalibrationDrive(
        self,
        inputSignal: np.ndarray,
        driveDbPerChain: Sequence[float],
    ) -> np.ndarray:
        """Evaluate one trial with an explicit post-decode analog drive.

        Processing details:
            Algorithm: Decode the legal public waveform, multiply it by the
            supplied analog drive without changing the committed state, run
            the floating electrical and thermal PA path, and encode the clean
            output once for measurement by ``PowerCalibration``.

        Args:
            inputSignal: Public floating samples or signed fixed-point I/Q codes.
            driveDbPerChain: One-element trial analog-drive sequence in dB.

        Returns:
            result: Public PA output produced by this calibration trial.
        """

        trialDriveDb = self.ResolveCalibrationDriveDb(driveDbPerChain)
        inputFormat = FixedPoint(self.width)
        outputFormat = FixedPoint(
            self.width, self.outputFullScaleAmplitude
        )
        floatingInput = inputFormat.DecodeComplex(inputSignal)
        driveScale = np.power(10.0, trialDriveDb / 20.0)
        floatingOutput = self.ProcessFloating(driveScale * floatingInput)
        return outputFormat.EncodeComplex(floatingOutput)

    def ProcessFloating(self, inputSignal: np.ndarray) -> np.ndarray:
        """Evaluate the PA directly in its normalized floating-point domain.

        Processing details:
            Algorithm: Validate finite normalized complex samples and pass
            them to the active Rapp, Wiener, GMP, or Doherty calculation without
            applying either public fixed-point encoding or the committed
            post-DAC drive. Callers that start from decoded public samples,
            including ILC, use ``ProcessOutputPathsFloating`` instead.

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
            gain and compression drift for one thermal update interval. Return
            the electrical output unchanged when configuration is disabled or
            a power-calibration suspension transaction is active.

        Args:
            baseOutput: Electrical-model samples before temperature drift.
            junctionTemperatureC: Fixed junction temperature in Celsius.

        Returns:
            result: Same-shape temperature-modified complex output samples.
        """

        thermalConfig = self.ResolveThermalConfig()
        complexOutput = np.asarray(baseOutput, dtype=np.complex128)
        if (
            self._thermalEffectsSuspended
            or thermalConfig is None
            or not thermalConfig.enabled
        ):
            return complexOutput
        return self.ApplyTemperatureDriftResolved(
            complexOutput,
            junctionTemperatureC,
            _ThermalRuntime.FromValidatedConfig(thermalConfig),
        )

    def ApplyTemperatureDriftResolved(
        self,
        baseOutput: np.ndarray,
        junctionTemperatureC: float,
        thermalRuntime: _ThermalRuntime,
    ) -> np.ndarray:
        """Apply drift with constants validated by the period operation.

        Processing details:
            Algorithm: Calculate temperature-relative gain, phase, saturation,
            and nonlinear-envelope changes while reusing the validated thermal
            configuration instead of resolving it for this interval.

        Args:
            baseOutput: Electrical-model samples before thermal drift.
            junctionTemperatureC: Interval-start junction temperature in C.
            thermalRuntime: Validated constants shared by the current period.

        Returns:
            result: Temperature-modified complex output samples.
        """

        thermalConfig = thermalRuntime.config
        complexOutput = np.asarray(baseOutput, dtype=np.complex128)
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
        activeMask: Optional[np.ndarray] = None,
    ) -> float:
        """Estimate mean heat power from normalized RF output and efficiency.

        Processing details:
            Algorithm: Map normalized instantaneous envelope power to watts,
            use a caller-supplied full-frame activity mask when available or
            derive a local peak-relative mask for compatibility, calculate
            output-power-dependent efficiency on active samples, and average
            RF loss plus idle bias dissipation over the complete interval.

        Args:
            outputSignal: Temperature-modified normalized PA output segment.
            activeMask: Optional boolean activity classification generated
                once from the complete PA-input waveform.

        Returns:
            result: Mean dissipated heat power in watts for the segment.
        """

        thermalConfig = self.ResolveThermalConfig()
        if thermalConfig is None:
            return 0.0
        return self.EstimateDissipatedPowerWResolved(
            outputSignal,
            activeMask,
            _ThermalRuntime.FromValidatedConfig(thermalConfig),
        )

    def EstimateDissipatedPowerWResolved(
        self,
        outputSignal: np.ndarray,
        activeMask: Optional[np.ndarray],
        thermalRuntime: _ThermalRuntime,
    ) -> float:
        """Estimate interval heat with validated period constants.

        Processing details:
            Algorithm: Map normalized output power into watts, apply the
            selected constant or power-dependent drain efficiency, preserve
            idle-sample heat, and average the unchanged sample-level schedule.

        Args:
            outputSignal: Temperature-modified PA output for one interval.
            activeMask: Optional full-window-derived activity slice.
            thermalRuntime: Validated constants shared by the current period.

        Returns:
            result: Mean dissipated heat power for the interval in watts.
        """

        complexOutput = np.asarray(outputSignal, dtype=np.complex128)
        if complexOutput.ndim != 1 or complexOutput.size == 0:
            raise ValueError("outputSignal must be a nonempty complex vector")
        outputPowerNormalized = np.abs(complexOutput) ** 2
        outputPowerW = thermalRuntime.referencePowerW * outputPowerNormalized
        if activeMask is None:
            peakPower = float(np.max(outputPowerNormalized))
            if peakPower <= np.finfo(float).tiny:
                resolvedActiveMask = np.zeros(
                    complexOutput.size, dtype=bool
                )
            else:
                resolvedActiveMask = (
                    outputPowerNormalized
                    >= thermalRuntime.activeThresholdLinear * peakPower
                )
        else:
            resolvedActiveMask = np.asarray(activeMask, dtype=bool)
            if resolvedActiveMask.shape != complexOutput.shape:
                raise ValueError(
                    "activeMask must match the outputSignal vector shape"
                )
        efficiencyValues = np.full(
            outputPowerW.shape,
            thermalRuntime.minimumDrainEfficiency,
            dtype=float,
        )
        if thermalRuntime.efficiencyModelName == "constant":
            efficiencyValues.fill(thermalRuntime.peakDrainEfficiency)
        else:
            normalizedKneePower = outputPowerW / max(
                thermalRuntime.efficiencyKneePowerW,
                np.finfo(float).tiny,
            )
            efficiencyValues = (
                thermalRuntime.minimumDrainEfficiency
                + (
                    thermalRuntime.peakDrainEfficiency
                    - thermalRuntime.minimumDrainEfficiency
                )
                * normalizedKneePower
                / (1.0 + normalizedKneePower)
            )
        activeDissipation = (
            thermalRuntime.idleDissipatedPowerW
            + outputPowerW
            * (1.0 / efficiencyValues - 1.0)
        )
        dissipatedPowerPerSample = np.where(
            resolvedActiveMask,
            activeDissipation,
            thermalRuntime.idleDissipatedPowerW,
        )
        return float(np.mean(dissipatedPowerPerSample))

    def ProcessThermalFloating(self, inputSignal: np.ndarray) -> np.ndarray:
        """Process one continuous transient data window without an outer gap.

        Processing details:
            Algorithm: Delegate to the general period processor in transient
            mode with a configured duty cycle of one. Direct PaModel users
            retain the historical continuous-waveform behavior, while Channel
            supplies explicit steady-state and scheduled-idle settings.

        Args:
            inputSignal: Normalized finite complex waveform to transmit.

        Returns:
            result: Same-length complex waveform including thermal drift.
        """

        return self.ProcessThermalPeriodFloating(
            inputSignal,
            thermalRunMode="transient",
            thermalDutyCycle=1.0,
            steadyStateToleranceC=1.0e-4,
            maximumSteadyStateIterations=100,
        )

    def BuildThermalActiveMask(
        self, inputSignal: np.ndarray
    ) -> np.ndarray:
        """Classify RF-active samples once for an entire data window.

        Processing details:
            Algorithm: Compare instantaneous PA-input power with one threshold
            derived from the full-window peak. A completely silent waveform
            produces an all-false mask. Reusing this mask for every thermal
            segment prevents a low-level idle segment from becoming active
            merely because it has its own smaller local peak.

        Args:
            inputSignal: Normalized finite PA-input waveform.

        Returns:
            result: Boolean vector marking true RF-active samples.
        """

        thermalConfig = self.ResolveThermalConfig()
        thermalRuntime = (
            None
            if thermalConfig is None
            else _ThermalRuntime.FromValidatedConfig(thermalConfig)
        )
        return self.BuildThermalActiveMaskResolved(
            inputSignal,
            thermalRuntime,
        )

    def BuildThermalActiveMaskResolved(
        self,
        inputSignal: np.ndarray,
        thermalRuntime: Optional[_ThermalRuntime],
    ) -> np.ndarray:
        """Classify activity using the period's cached threshold.

        Processing details:
            Algorithm: Validate the complete input vector, measure its peak
            power once, and compare every sample with the already converted
            peak-relative active threshold.

        Args:
            inputSignal: Complete normalized PA-input data window.
            thermalRuntime: Validated constants, or None when heat is disabled.

        Returns:
            result: Boolean activity mask aligned with the input samples.
        """

        complexInput = np.asarray(inputSignal, dtype=np.complex128)
        if (
            complexInput.ndim != 1
            or complexInput.size == 0
            or not np.all(np.isfinite(complexInput))
        ):
            raise ValueError("inputSignal must be a nonempty finite vector")
        if thermalRuntime is None:
            return np.zeros(complexInput.size, dtype=bool)
        inputPower = np.abs(complexInput) ** 2
        peakPower = float(np.max(inputPower))
        if peakPower <= np.finfo(float).tiny:
            return np.zeros(complexInput.size, dtype=bool)
        threshold = peakPower * thermalRuntime.activeThresholdLinear
        return np.asarray(inputPower >= threshold, dtype=bool)

    def BuildThermalIntervals(
        self, activeMask: np.ndarray
    ) -> Tuple[Tuple[int, int], ...]:
        """Split a data window at thermal limits and activity transitions.

        Processing details:
            Algorithm: Combine regular thermal-update boundaries with every
            true-to-false or false-to-true activity transition. Consequently
            even an idle run shorter than ``thermalUpdateIntervalSamples``
            receives its own idle-power interval and can cool toward the idle
            thermal equilibrium rather than being averaged into RF activity.

        Args:
            activeMask: Full-window boolean RF-activity vector.

        Returns:
            result: Ordered half-open sample-index intervals.
        """

        thermalConfig = self.ResolveThermalConfig()
        if thermalConfig is None:
            raise RuntimeError("thermal model is not configured")
        return self.BuildThermalIntervalsResolved(
            activeMask,
            _ThermalRuntime.FromValidatedConfig(thermalConfig),
        )

    def BuildThermalIntervalsResolved(
        self,
        activeMask: np.ndarray,
        thermalRuntime: _ThermalRuntime,
    ) -> Tuple[Tuple[int, int], ...]:
        """Build unchanged interval boundaries from cached constants.

        Processing details:
            Algorithm: Preserve every configured update boundary and every
            active-to-idle transition, sort their union, and return consecutive
            half-open ranges without merging any thermal interval.

        Args:
            activeMask: Full-window boolean RF-activity classification.
            thermalRuntime: Validated constants shared by the current period.

        Returns:
            result: Ordered half-open interval boundaries for the data window.
        """

        resolvedMask = np.asarray(activeMask, dtype=bool)
        if resolvedMask.ndim != 1 or resolvedMask.size == 0:
            raise ValueError("activeMask must be a nonempty vector")
        intervalLength = thermalRuntime.thermalUpdateIntervalSamples
        regularBoundaries = range(
            intervalLength, resolvedMask.size, intervalLength
        )
        transitionBoundaries = (
            np.flatnonzero(resolvedMask[1:] != resolvedMask[:-1]) + 1
        )
        boundaryValues = sorted(
            {
                0,
                resolvedMask.size,
                *regularBoundaries,
                *(int(value) for value in transitionBoundaries),
            }
        )
        return tuple(
            (startIndex, stopIndex)
            for startIndex, stopIndex in zip(
                boundaryValues[:-1], boundaryValues[1:]
            )
        )

    def SimulateThermalPeriod(
        self,
        baseOutput: np.ndarray,
        activeMask: np.ndarray,
        startingTemperatureRisePerBranchC: np.ndarray,
        externalIdleDurationSec: float,
    ) -> Dict[str, object]:
        """Simulate one complete period without mutating the thermal network.

        Processing details:
            Algorithm: Apply temperature drift at the beginning of each data
            interval, estimate heat with the full-window activity mask, advance
            a local copy of every RC state exactly, and finally append the
            scheduled outer idle interval. The returned waveform contains only
            the caller's data window; idle time changes thermal state but does
            not append samples to the public signal.

        Args:
            baseOutput: Full electrical-model output before thermal drift.
            activeMask: Full-window RF-activity classification.
            startingTemperatureRisePerBranchC: Period-start branch state.
            externalIdleDurationSec: Scheduled idle time after the data window.

        Returns:
            result: Dictionary containing output, heat schedule, branch states,
                energy, and a compact period temperature trace.
        """

        thermalConfig = self.ResolveThermalConfig()
        if thermalConfig is None or self.thermalNetwork is None:
            raise RuntimeError("thermal model is not enabled")
        return self.SimulateThermalPeriodResolved(
            baseOutput,
            activeMask,
            startingTemperatureRisePerBranchC,
            externalIdleDurationSec,
            _ThermalRuntime.FromValidatedConfig(thermalConfig),
        )

    def SimulateThermalPeriodResolved(
        self,
        baseOutput: np.ndarray,
        activeMask: np.ndarray,
        startingTemperatureRisePerBranchC: np.ndarray,
        externalIdleDurationSec: float,
        thermalRuntime: _ThermalRuntime,
        thermalIntervals: Optional[Tuple[Tuple[int, int], ...]] = None,
    ) -> Dict[str, object]:
        """Simulate a period with one validated configuration and schedule.

        Processing details:
            Algorithm: Validate period-level arrays and idle duration, then
            process every original active or idle interval independently while
            reusing config constants and optional prebuilt boundaries. Advance
            only a local branch-state copy and append the unchanged outer-idle
            step so solver probes remain transactional.

        Args:
            baseOutput: Full electrical-model output before thermal drift.
            activeMask: Full-window boolean RF-activity classification.
            startingTemperatureRisePerBranchC: Period-start branch state.
            externalIdleDurationSec: Scheduled idle time after the data window.
            thermalRuntime: Validated constants shared by the current period.
            thermalIntervals: Optional prebuilt boundaries for the same mask.

        Returns:
            result: Output, heat schedule, states, energy, and temperature trace.
        """

        if self.thermalNetwork is None:
            raise RuntimeError("thermal model is not enabled")
        complexBaseOutput = np.asarray(baseOutput, dtype=np.complex128)
        resolvedActiveMask = np.asarray(activeMask, dtype=bool)
        if (
            complexBaseOutput.ndim != 1
            or complexBaseOutput.size == 0
            or resolvedActiveMask.shape != complexBaseOutput.shape
        ):
            raise ValueError(
                "baseOutput and activeMask must be matching nonempty vectors"
            )
        resolvedIdleDurationSec = float(externalIdleDurationSec)
        if (
            not np.isfinite(resolvedIdleDurationSec)
            or resolvedIdleDurationSec < 0.0
        ):
            raise ValueError(
                "externalIdleDurationSec must be finite and nonnegative"
            )
        branchState = np.asarray(
            startingTemperatureRisePerBranchC, dtype=float
        ).copy()
        if branchState.shape != self.thermalNetwork.temperatureRisePerBranchC.shape:
            raise ValueError(
                "startingTemperatureRisePerBranchC has an incompatible shape"
            )
        outputSignal = np.empty_like(
            complexBaseOutput, dtype=np.complex128
        )
        dissipatedPowersW = []
        durationsSec = []
        dataDissipatedEnergyJ = 0.0
        elapsedDataTimeSec = 0.0
        temperatureTraceTimeSec = [0.0]
        temperatureTraceC = [
            float(
                self.thermalNetwork.ambientTemperatureC
                + np.sum(branchState)
                + self._externalTemperatureOffsetC
            )
        ]
        temperatureTraceRfActive = []
        maximumTemperatureC = (
            thermalRuntime.maximumJunctionTemperatureC
        )
        resolvedIntervals = (
            self.BuildThermalIntervalsResolved(
                resolvedActiveMask,
                thermalRuntime,
            )
            if thermalIntervals is None
            else thermalIntervals
        )
        for startIndex, stopIndex in resolvedIntervals:
            junctionTemperatureC = float(
                self.thermalNetwork.ambientTemperatureC
                + np.sum(branchState)
                + self._externalTemperatureOffsetC
            )
            if junctionTemperatureC > maximumTemperatureC:
                raise RuntimeError(
                    "PA junction temperature exceeded "
                    "maximumJunctionTemperatureC"
                )
            outputSegment = self.ApplyTemperatureDriftResolved(
                complexBaseOutput[startIndex:stopIndex],
                junctionTemperatureC,
                thermalRuntime,
            )
            outputSignal[startIndex:stopIndex] = outputSegment
            segmentMask = resolvedActiveMask[startIndex:stopIndex]
            dissipatedPowerW = self.EstimateDissipatedPowerWResolved(
                outputSegment,
                segmentMask,
                thermalRuntime,
            )
            durationSec = (
                (stopIndex - startIndex)
                / thermalRuntime.sampleRateHz
            )
            branchState = (
                self.thermalNetwork.CalculateAdvancedStateResolved(
                    branchState,
                    dissipatedPowerW,
                    durationSec,
                    thermalRuntime,
                )
            )
            elapsedDataTimeSec += durationSec
            dataDissipatedEnergyJ += dissipatedPowerW * durationSec
            dissipatedPowersW.append(dissipatedPowerW)
            durationsSec.append(durationSec)
            temperatureTraceTimeSec.append(elapsedDataTimeSec)
            temperatureTraceC.append(
                float(
                    self.thermalNetwork.ambientTemperatureC
                    + np.sum(branchState)
                    + self._externalTemperatureOffsetC
                )
            )
            temperatureTraceRfActive.append(bool(np.any(segmentMask)))
        dataEndingState = branchState.copy()
        if resolvedIdleDurationSec > 0.0:
            idlePowerW = thermalRuntime.idleDissipatedPowerW
            branchState = (
                self.thermalNetwork.CalculateAdvancedStateResolved(
                    branchState,
                    idlePowerW,
                    resolvedIdleDurationSec,
                    thermalRuntime,
                )
            )
            dissipatedPowersW.append(idlePowerW)
            durationsSec.append(resolvedIdleDurationSec)
            temperatureTraceTimeSec.append(
                elapsedDataTimeSec + resolvedIdleDurationSec
            )
            temperatureTraceC.append(
                float(
                    self.thermalNetwork.ambientTemperatureC
                    + np.sum(branchState)
                    + self._externalTemperatureOffsetC
                )
            )
            temperatureTraceRfActive.append(False)
        if max(temperatureTraceC) > maximumTemperatureC:
            raise RuntimeError(
                "PA junction temperature exceeded maximumJunctionTemperatureC"
            )
        return {
            "outputSignal": outputSignal,
            "dataEndingTemperatureRisePerBranchC": dataEndingState,
            "periodEndingTemperatureRisePerBranchC": branchState,
            "dissipatedPowersW": tuple(dissipatedPowersW),
            "durationsSec": tuple(durationsSec),
            "dataDissipatedEnergyJ": dataDissipatedEnergyJ,
            "periodDissipatedEnergyJ": (
                dataDissipatedEnergyJ
                + thermalRuntime.idleDissipatedPowerW
                * resolvedIdleDurationSec
            ),
            "temperatureTraceTimeSec": tuple(temperatureTraceTimeSec),
            "temperatureTraceC": tuple(temperatureTraceC),
            "temperatureTraceRfActive": tuple(temperatureTraceRfActive),
        }

    def ProcessThermalPeriodFloating(
        self,
        inputSignal: np.ndarray,
        thermalRunMode: str = "steady_state",
        thermalDutyCycle: float = 1.0,
        steadyStateToleranceC: float = 1.0e-4,
        maximumSteadyStateIterations: int = 100,
    ) -> np.ndarray:
        """Process one scheduled thermal period in steady or transient mode.

        Processing details:
            Algorithm: Treat the complete input array as the configured data
            window, derive its outer idle duration from ``thermalDutyCycle``,
            and split its internal active and idle samples. Transient mode
            advances exactly one cycle from live state. Steady-state mode
            repeatedly freezes the temperature-dependent heat trace, solves
            each RC branch's analytic periodic starting state, and verifies
            that every branch returns to that state within tolerance. Solver
            trials never change live elapsed time; only the accepted period is
            committed.

        Args:
            inputSignal: Normalized finite PA-input data-window waveform.
            thermalRunMode: ``"steady_state"`` or ``"transient"``.
            thermalDutyCycle: Data-window duration divided by period duration.
            steadyStateToleranceC: Maximum allowed per-branch closure error.
            maximumSteadyStateIterations: Maximum nonlinear fixed-point steps.

        Returns:
            result: Temperature-modified output for the data window only.
        """

        self.SynchronizeModel()
        self.SynchronizeThermalModel()
        complexInput = np.asarray(inputSignal, dtype=np.complex128)
        if (
            complexInput.ndim != 1
            or complexInput.size == 0
            or not np.all(np.isfinite(complexInput))
        ):
            raise ValueError("inputSignal must be a nonempty finite vector")
        if not isinstance(thermalRunMode, str):
            raise TypeError("thermalRunMode must be a string")
        resolvedRunMode = thermalRunMode.strip().lower()
        if resolvedRunMode not in ("steady_state", "transient"):
            raise ValueError(
                "thermalRunMode must be 'steady_state' or 'transient'"
            )
        if (
            not isinstance(thermalDutyCycle, (int, float))
            or isinstance(thermalDutyCycle, bool)
            or not np.isfinite(thermalDutyCycle)
            or not 0.0 < float(thermalDutyCycle) <= 1.0
        ):
            raise ValueError(
                "thermalDutyCycle must be a finite real number in (0, 1]"
            )
        if (
            not isinstance(steadyStateToleranceC, (int, float))
            or isinstance(steadyStateToleranceC, bool)
            or not np.isfinite(steadyStateToleranceC)
            or float(steadyStateToleranceC) <= 0.0
        ):
            raise ValueError(
                "steadyStateToleranceC must be a finite positive value"
            )
        if (
            not isinstance(maximumSteadyStateIterations, int)
            or isinstance(maximumSteadyStateIterations, bool)
            or maximumSteadyStateIterations < 1
        ):
            raise ValueError(
                "maximumSteadyStateIterations must be a positive integer"
            )
        thermalConfig = self.ResolveThermalConfig()
        if thermalConfig is None or self.thermalNetwork is None:
            return np.asarray(
                self.model.Process(complexInput), dtype=np.complex128
            )
        thermalRuntime = _ThermalRuntime.FromValidatedConfig(thermalConfig)
        baseOutput = np.asarray(
            self.model.Process(complexInput), dtype=np.complex128
        )
        activeMask = self.BuildThermalActiveMaskResolved(
            complexInput,
            thermalRuntime,
        )
        thermalIntervals = self.BuildThermalIntervalsResolved(
            activeMask,
            thermalRuntime,
        )
        signalDurationSec = (
            complexInput.size / thermalRuntime.sampleRateHz
        )
        periodDurationSec = signalDurationSec / float(thermalDutyCycle)
        externalIdleDurationSec = periodDurationSec - signalDurationSec
        periodStartingState = np.asarray(
            self.thermalNetwork.temperatureRisePerBranchC, dtype=float
        ).copy()
        steadyStateConverged = False
        steadyStateIterations = 0
        steadyStateErrorC = 0.0
        simulationResult: Dict[str, object]
        if (
            resolvedRunMode == "steady_state"
            and thermalRuntime.modelName != "static"
        ):
            candidateState = periodStartingState.copy()
            bestErrorC = float("inf")
            for iterationIndex in range(
                1, maximumSteadyStateIterations + 1
            ):
                trialResult = self.SimulateThermalPeriodResolved(
                    baseOutput,
                    activeMask,
                    candidateState,
                    externalIdleDurationSec,
                    thermalRuntime,
                    thermalIntervals,
                )
                solvedState = (
                    self.thermalNetwork.CalculatePeriodicSteadyStateResolved(
                        cast(
                            Sequence[float],
                            trialResult["dissipatedPowersW"],
                        ),
                        cast(
                            Sequence[float],
                            trialResult["durationsSec"],
                        ),
                        thermalRuntime,
                    )
                )
                candidateState = solvedState
                verificationResult = self.SimulateThermalPeriodResolved(
                    baseOutput,
                    activeMask,
                    candidateState,
                    externalIdleDurationSec,
                    thermalRuntime,
                    thermalIntervals,
                )
                endingState = np.asarray(
                    verificationResult[
                        "periodEndingTemperatureRisePerBranchC"
                    ],
                    dtype=float,
                )
                closureErrorC = float(
                    np.max(np.abs(endingState - candidateState))
                )
                bestErrorC = min(bestErrorC, closureErrorC)
                if closureErrorC <= float(steadyStateToleranceC):
                    periodStartingState = candidateState.copy()
                    simulationResult = verificationResult
                    steadyStateConverged = True
                    steadyStateIterations = iterationIndex
                    steadyStateErrorC = closureErrorC
                    break
                candidateState = 0.5 * (
                    candidateState
                    + np.asarray(
                        verificationResult[
                            "periodEndingTemperatureRisePerBranchC"
                        ],
                        dtype=float,
                    )
                )
            else:
                raise RuntimeError(
                    "periodic thermal steady-state solver did not converge "
                    f"within {maximumSteadyStateIterations} iterations; "
                    f"best branch error was {bestErrorC:.6g} C and the "
                    f"allowed tolerance is {float(steadyStateToleranceC):.6g} C"
                )
        else:
            simulationResult = self.SimulateThermalPeriodResolved(
                baseOutput,
                activeMask,
                periodStartingState,
                externalIdleDurationSec,
                thermalRuntime,
                thermalIntervals,
            )
            if resolvedRunMode == "steady_state":
                steadyStateConverged = True
                steadyStateIterations = 0
                steadyStateErrorC = 0.0
        outputSignal = np.asarray(
            simulationResult["outputSignal"], dtype=np.complex128
        )
        dataEndingState = np.asarray(
            simulationResult["dataEndingTemperatureRisePerBranchC"],
            dtype=float,
        )
        periodEndingState = np.asarray(
            simulationResult["periodEndingTemperatureRisePerBranchC"],
            dtype=float,
        )
        self.thermalNetwork.temperatureRisePerBranchC = (
            periodEndingState.copy()
        )
        self.thermalNetwork.elapsedTimeSec += periodDurationSec
        periodStartingJunctionTemperatureC = float(
            self.thermalNetwork.ambientTemperatureC
            + np.sum(periodStartingState)
            + self._externalTemperatureOffsetC
        )
        dataEndingJunctionTemperatureC = float(
            self.thermalNetwork.ambientTemperatureC
            + np.sum(dataEndingState)
            + self._externalTemperatureOffsetC
        )
        periodEndingJunctionTemperatureC = float(
            self.thermalNetwork.ambientTemperatureC
            + np.sum(periodEndingState)
            + self._externalTemperatureOffsetC
        )
        waveformActiveDutyCycle = float(np.mean(activeMask))
        actualDutyCycle = (
            float(thermalDutyCycle) * waveformActiveDutyCycle
        )
        activeOutputPower = (
            float(np.mean(np.abs(outputSignal[activeMask]) ** 2))
            if np.any(activeMask)
            else np.finfo(float).tiny
        )
        dataDissipatedEnergyJ = float(
            simulationResult["dataDissipatedEnergyJ"]
        )
        periodDissipatedEnergyJ = float(
            simulationResult["periodDissipatedEnergyJ"]
        )
        self._lastThermalMetrics = {
            **self.thermalNetwork.GetMetrics(),
            "junctionTemperatureC": periodEndingJunctionTemperatureC,
            "selfHeatingJunctionTemperatureC": (
                self.thermalNetwork.CurrentTemperatureC()
            ),
            "mutualHeatingTemperatureRiseC": (
                self._externalTemperatureOffsetC
            ),
            "startingJunctionTemperatureC": (
                periodStartingJunctionTemperatureC
            ),
            "endingJunctionTemperatureC": dataEndingJunctionTemperatureC,
            "periodStartingJunctionTemperatureC": (
                periodStartingJunctionTemperatureC
            ),
            "dataEndingJunctionTemperatureC": (
                dataEndingJunctionTemperatureC
            ),
            "periodEndingJunctionTemperatureC": (
                periodEndingJunctionTemperatureC
            ),
            "periodStartingTemperatureRisePerBranchC": tuple(
                float(value) for value in periodStartingState
            ),
            "dataEndingTemperatureRisePerBranchC": tuple(
                float(value) for value in dataEndingState
            ),
            "periodEndingTemperatureRisePerBranchC": tuple(
                float(value) for value in periodEndingState
            ),
            "averageDissipatedPowerW": (
                periodDissipatedEnergyJ / periodDurationSec
            ),
            "dataWindowAverageDissipatedPowerW": (
                dataDissipatedEnergyJ / signalDurationSec
            ),
            "activeSampleDutyCycle": waveformActiveDutyCycle,
            "waveformActiveDutyCycle": waveformActiveDutyCycle,
            "configuredDutyCycle": float(thermalDutyCycle),
            "actualDutyCycle": actualDutyCycle,
            "signalDurationSec": signalDurationSec,
            "scheduledIdleDurationSec": externalIdleDurationSec,
            "periodDurationSec": periodDurationSec,
            "thermalRunMode": resolvedRunMode,
            "steadyStateConverged": steadyStateConverged,
            "steadyStateIterations": steadyStateIterations,
            "steadyStateErrorC": steadyStateErrorC,
            "temperatureTraceTimeSec": simulationResult[
                "temperatureTraceTimeSec"
            ],
            "temperatureTraceC": simulationResult["temperatureTraceC"],
            "temperatureTraceRfActive": simulationResult[
                "temperatureTraceRfActive"
            ],
            "outputPowerDbm": (
                float(thermalConfig.referenceOutputPowerDbm)
                + 10.0
                * np.log10(
                    max(activeOutputPower, np.finfo(float).tiny)
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
        return float(np.mean(self.BuildThermalActiveMask(inputSignal)))

    def CalculateActualDutyCycle(
        self,
        inputSignal: np.ndarray,
        thermalDutyCycle: float = 1.0,
    ) -> float:
        """Combine the scheduled data-window duty with measured RF activity.

        Processing details:
            Algorithm: Validate the configured ratio of data-window duration
            to period duration, measure the fraction of active samples inside
            that complete window, and multiply the two independent fractions.
            Internal zero samples therefore reduce actual RF duty without
            changing the caller's configured scheduling duty.

        Args:
            inputSignal: Normalized finite PA-input data-window waveform.
            thermalDutyCycle: Configured data-window fraction of one period.

        Returns:
            result: Actual RF-active fraction of the complete period.
        """

        if (
            not isinstance(thermalDutyCycle, (int, float))
            or isinstance(thermalDutyCycle, bool)
            or not np.isfinite(thermalDutyCycle)
            or not 0.0 < float(thermalDutyCycle) <= 1.0
        ):
            raise ValueError(
                "thermalDutyCycle must be a finite real number in (0, 1]"
            )
        return float(thermalDutyCycle) * self.CalculateActiveDutyCycle(
            inputSignal
        )

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
            **self._lastThermalMetrics,
            **self.thermalNetwork.GetMetrics(),
            "junctionTemperatureC": junctionTemperatureC,
            "selfHeatingJunctionTemperatureC": selfHeatingTemperatureC,
            "mutualHeatingTemperatureRiseC": (
                self._externalTemperatureOffsetC
            ),
            "averageDissipatedPowerW": float(
                thermalConfig.idleDissipatedPowerW
            ),
            "latestExplicitIdleDurationSec": float(idleTimeSec),
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
        self.SynchronizeThermalModel()
        if self.thermalNetwork is None:
            # A disabled thermal model must not accumulate a hidden mutual-
            # heating offset that could alter output after later re-enablement.
            self._externalTemperatureOffsetC = 0.0
            return
        self._externalTemperatureOffsetC = resolvedOffset
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
        thermalConfig = self.ResolveThermalConfig()
        if thermalConfig is None:
            raise RuntimeError(
                "enabled thermal state requires an active ThermalConfig"
            )
        return {
            "enabled": True,
            "sampleRateHz": float(thermalConfig.sampleRateHz),
            "referenceOutputPowerDbm": float(
                thermalConfig.referenceOutputPowerDbm
            ),
            "activePowerThresholdDb": float(
                thermalConfig.activePowerThresholdDb
            ),
            **dict(self._lastThermalMetrics),
        }

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
        outputFullScaleAmplitude: Optional[float] = None,
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
            outputFullScaleAmplitude: Optional physical magnitude represented
                by each PA-output code rail. The default is 2.0.
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
                "outputFullScaleAmplitude": 2.0,
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
        if outputFullScaleAmplitude is not None:
            recognizedOverrides["outputFullScaleAmplitude"] = (
                outputFullScaleAmplitude
            )
        self.parameters: ChainMap[str, object] = ChainMap(
            recognizedOverrides,
            externalParameters,
            self.defaultParameters,
        )
        self.paModels = []
        self._activePaParameterSnapshot = None
        self._calibrationDriveDbPerChain: Tuple[float, ...] = tuple()
        self.lastOutputRmsPerChain: Tuple[float, ...] = tuple()
        self.lastDissipatedPowerWPerChain: Tuple[float, ...] = tuple()
        self._lastMutualHeatingMetrics: Dict[str, object] = {}
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

    @property
    def OutputFullScaleAmplitude(self) -> float:
        """Return the physical magnitude represented by each PA output rail.

        Processing details:
            Algorithm: Resolve the common live MIMO observation scale without
            changing any independently configured PA coefficients.

        Returns:
            result: Positive physical I/Q component full-scale amplitude.
        """

        return float(
            cast(float, self.parameters["outputFullScaleAmplitude"])
        )

    outputFullScaleAmplitude = OutputFullScaleAmplitude

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
        FixedPoint(
            self.width,
            self.parameters["outputFullScaleAmplitude"],
        )
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
        self._calibrationDriveDbPerChain = tuple(
            0.0 for _ in range(self.numTransmitChains)
        )
        self._lastMutualHeatingMetrics = {}

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
            Algorithm: Decode the public matrix of fixed I/Q codes once, apply
            each committed post-DAC calibration drive, process every normalized
            floating column through its PA and power controls, encode every
            result column back to public integer codes, and preserve the
            original vector or matrix orientation.

        Args:
            inputSignal: Complex vector for one chain or matrix shaped samples
                by the configured number of transmit chains.

        Returns:
            result: Processed complex array containing raw I/Q codes in fixed
                mode and physical floating samples in floating mode.
        """

        self.SynchronizeModels()
        inputFormat = FixedPoint(self.width)
        outputFormat = FixedPoint(
            self.width, self.outputFullScaleAmplitude
        )
        complexInput = inputFormat.DecodeComplex(inputSignal)
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
        floatingOutput, _ = self.ProcessOutputPathsFloating(complexInput)
        outputMatrix = outputFormat.EncodeComplex(floatingOutput)
        if inputWasVector and self.numTransmitChains == 1:
            return outputMatrix[:, 0]
        return outputMatrix

    def ProcessOutputPathsFloating(
        self, inputSignal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process every normalized PA input at its committed analog drive.

        Processing details:
            Algorithm: Resolve one hidden post-DAC drive per chain, apply the
            gains to the decoded floating matrix, run the raw PA bank once,
            and duplicate the clean conducted output for channel and feedback
            roles. The raw ``ProcessFloating`` method intentionally remains a
            drive-free kernel for callers that already own the physical scale.

        Args:
            inputSignal: Normalized vector or samples-by-chains matrix after
                public fixed-point decoding.

        Returns:
            result: Identical floating ``(chOut, fbOut)`` observations with the
                same vector or matrix orientation as the input.
        """

        self.SynchronizeModels()
        complexInput = np.asarray(inputSignal, dtype=np.complex128)
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
        driveDbPerChain = self.ResolveCalibrationDriveDbPerChain(
            self._calibrationDriveDbPerChain
        )
        driveScalePerChain = np.power(
            10.0, np.asarray(driveDbPerChain, dtype=float) / 20.0
        )
        floatingOutput = np.asarray(
            self.ProcessFloating(
                inputMatrix * driveScalePerChain.reshape(1, -1)
            ),
            dtype=np.complex128,
        )
        if inputWasVector and self.numTransmitChains == 1:
            floatingOutput = floatingOutput[:, 0]
        return floatingOutput, floatingOutput.copy()

    def ResolveCalibrationDriveDbPerChain(
        self, driveDbPerChain: Sequence[float]
    ) -> Tuple[float, ...]:
        """Validate one hidden analog-drive value per physical PA chain.

        Processing details:
            Algorithm: Flatten a one-dimensional sequence, require its length
            to equal ``numTransmitChains``, and reject boolean or nonfinite
            entries before any exponential amplitude conversion.

        Args:
            driveDbPerChain: Chain-ordered post-DAC drive values in decibels.

        Returns:
            result: Immutable validated drive tuple in physical chain order.
        """

        if isinstance(driveDbPerChain, (str, bytes)):
            raise TypeError("driveDbPerChain must be a numeric sequence")
        driveArray = np.asarray(driveDbPerChain, dtype=object)
        if (
            driveArray.ndim != 1
            or driveArray.size != self.numTransmitChains
        ):
            raise ValueError(
                "driveDbPerChain must contain one value per transmit chain"
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

    def SetCalibrationDriveDb(
        self, driveDbPerChain: Sequence[float]
    ) -> None:
        """Commit post-DAC analog drives selected by power calibration.

        Processing details:
            Algorithm: Validate one value per PA and replace the private drive
            tuple atomically. Public fixed-point waveforms remain legal DAC
            codes while subsequent ``Process`` calls reproduce the calibrated
            operating point on every chain.

        Args:
            driveDbPerChain: Chain-ordered analog drive values in decibels.

        Returns:
            result: None. Accepted drives apply to later public processing.
        """

        self._calibrationDriveDbPerChain = (
            self.ResolveCalibrationDriveDbPerChain(driveDbPerChain)
        )

    def ProcessCalibrationDrive(
        self,
        inputSignal: np.ndarray,
        driveDbPerChain: Sequence[float],
    ) -> np.ndarray:
        """Evaluate a MIMO trial at explicit post-decode analog drives.

        Processing details:
            Algorithm: Decode the public waveform matrix once, apply one trial
            analog gain to every transmit column without changing committed
            state, execute the raw floating PA bank, and encode the clean
            per-chain outputs once for closed-loop power measurement.

        Args:
            inputSignal: Public SISO vector or samples-by-chains matrix.
            driveDbPerChain: One trial analog-drive value per physical PA.

        Returns:
            result: Public clean PA output with the original orientation.
        """

        self.SynchronizeModels()
        inputFormat = FixedPoint(self.width)
        outputFormat = FixedPoint(
            self.width, self.outputFullScaleAmplitude
        )
        floatingInput = inputFormat.DecodeComplex(inputSignal)
        inputWasVector = floatingInput.ndim == 1
        inputMatrix = (
            floatingInput.reshape(-1, 1)
            if inputWasVector
            else floatingInput
        )
        if (
            inputMatrix.ndim != 2
            or inputMatrix.shape[0] == 0
            or inputMatrix.shape[1] != self.numTransmitChains
            or not np.all(np.isfinite(inputMatrix))
        ):
            raise ValueError(
                "inputSignal must have one finite column per transmit chain"
            )
        resolvedDriveDb = self.ResolveCalibrationDriveDbPerChain(
            driveDbPerChain
        )
        driveScalePerChain = np.power(
            10.0, np.asarray(resolvedDriveDb, dtype=float) / 20.0
        )
        floatingOutput = self.ProcessFloating(
            inputMatrix * driveScalePerChain.reshape(1, -1)
        )
        publicOutput = outputFormat.EncodeComplex(floatingOutput)
        if inputWasVector and self.numTransmitChains == 1:
            return np.asarray(publicOutput)[:, 0]
        return np.asarray(publicOutput, dtype=np.complex128)

    def ProcessFloating(
        self, inputSignal: np.ndarray
    ) -> np.ndarray:
        """Evaluate every PA chain without public fixed-point conversion.

        Processing details:
            Algorithm: Validate a normalized samples-by-chains matrix, call
            each independently configured PA's drive-free floating processing
            path, update the most recent per-chain RMS diagnostics, and preserve
            a SISO vector only when the configured PA bank has one chain.
            Decoded public samples use ``ProcessOutputPathsFloating`` so the
            committed per-chain post-DAC drives are not lost.

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

    def ProcessThermalPeriodFloating(
        self,
        inputSignal: np.ndarray,
        thermalRunMode: str = "steady_state",
        thermalDutyCycle: float = 1.0,
        steadyStateToleranceC: float = 1.0e-4,
        maximumSteadyStateIterations: int = 100,
    ) -> np.ndarray:
        """Process all PA chains over one common scheduled thermal period.

        Processing details:
            Algorithm: Apply each chain's legacy input scaling, delegate the
            self-heating period to its PaModel, then apply legacy output
            scaling. In steady-state mode an enabled mutual C/W matrix is
            included in an outer fixed-point loop. Every outer probe restores
            the original thermal snapshots, so only the accepted common
            period advances elapsed time. Transient mode preserves the causal
            one-period-late mutual-heating update.

        Args:
            inputSignal: Normalized vector or samples-by-chains PA input.
            thermalRunMode: ``"steady_state"`` or ``"transient"``.
            thermalDutyCycle: Common data-window fraction of one period.
            steadyStateToleranceC: Maximum thermal fixed-point error in C.
            maximumSteadyStateIterations: Maximum self and mutual iterations.

        Returns:
            result: Same-orientation floating PA-bank output data window.
        """

        self.SynchronizeModels()
        complexInput = np.asarray(inputSignal, dtype=np.complex128)
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
        if not isinstance(thermalRunMode, str):
            raise TypeError("thermalRunMode must be a string")
        resolvedRunMode = thermalRunMode.strip().lower()
        if resolvedRunMode not in ("steady_state", "transient"):
            raise ValueError(
                "thermalRunMode must be 'steady_state' or 'transient'"
            )
        if (
            not isinstance(thermalDutyCycle, (int, float))
            or isinstance(thermalDutyCycle, bool)
            or not np.isfinite(thermalDutyCycle)
            or not 0.0 < float(thermalDutyCycle) <= 1.0
        ):
            raise ValueError(
                "thermalDutyCycle must be a finite real number in (0, 1]"
            )
        if (
            not isinstance(steadyStateToleranceC, (int, float))
            or isinstance(steadyStateToleranceC, bool)
            or not np.isfinite(steadyStateToleranceC)
            or float(steadyStateToleranceC) <= 0.0
        ):
            raise ValueError(
                "steadyStateToleranceC must be a finite positive value"
            )
        if (
            not isinstance(maximumSteadyStateIterations, int)
            or isinstance(maximumSteadyStateIterations, bool)
            or maximumSteadyStateIterations < 1
        ):
            raise ValueError(
                "maximumSteadyStateIterations must be a positive integer"
            )
        enabledThermalSampleRatesHz = tuple(
            float(thermalConfig.sampleRateHz)
            for paModel in self.paModels
            for thermalConfig in (paModel.ResolveThermalConfig(),)
            if thermalConfig is not None and thermalConfig.enabled
        )
        if enabledThermalSampleRatesHz and not all(
            np.isclose(
                sampleRateHz,
                enabledThermalSampleRatesHz[0],
                rtol=1.0e-12,
                atol=0.0,
            )
            for sampleRateHz in enabledThermalSampleRatesHz[1:]
        ):
            raise ValueError(
                "all enabled MIMO ThermalConfig.sampleRateHz values must "
                "match so every chain advances one common physical period"
            )
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
        couplingMatrix = self.ResolveThermalCouplingMatrix()
        hasMutualHeating = bool(np.any(couplingMatrix > 0.0))
        # One MIMO period is an atomic thermal transaction. A later chain can
        # fail after an earlier chain has already advanced, so retain every
        # chain state even when the mutual-heating matrix is disabled.
        thermalSnapshots: Tuple[
            Optional[Dict[str, object]], ...
        ] = self.SuspendThermalModel()
        self.RestoreThermalModel(thermalSnapshots)
        previousOutputRmsPerChain = tuple(self.lastOutputRmsPerChain)
        previousDissipatedPowerWPerChain = tuple(
            self.lastDissipatedPowerWPerChain
        )
        previousMutualHeatingMetrics = dict(
            self._lastMutualHeatingMetrics
        )
        candidateOffsetsC = np.asarray(
            [
                float(
                    paModel.GetThermalMetrics().get(
                        "mutualHeatingTemperatureRiseC", 0.0
                    )
                )
                for paModel in self.paModels
            ],
            dtype=float,
        )
        if not hasMutualHeating:
            # A live configuration update can remove a previously active
            # coupling matrix. Clear the stale neighbor-temperature offsets
            # before this period so disabled mutual heating has no memory.
            candidateOffsetsC = np.zeros(
                self.numTransmitChains, dtype=float
            )
        outputColumns = []
        outputRmsValues = []
        mutualHeatingIterations = 0
        mutualHeatingErrorC = 0.0
        iterationLimit = (
            maximumSteadyStateIterations
            if resolvedRunMode == "steady_state" and hasMutualHeating
            else 1
        )
        try:
            for mutualIterationIndex in range(1, iterationLimit + 1):
                if thermalSnapshots is not None:
                    self.RestoreThermalModel(thermalSnapshots)
                    for paModel, temperatureOffsetC in zip(
                        self.paModels, candidateOffsetsC
                    ):
                        paModel.SetExternalTemperatureOffsetC(
                            float(temperatureOffsetC)
                        )
                outputColumns = []
                outputRmsValues = []
                for chainIndex, paModel in enumerate(self.paModels):
                    inputScale = 10.0 ** (
                        float(inputPowerDbValues[chainIndex]) / 20.0
                    )
                    chainOutput = paModel.ProcessThermalPeriodFloating(
                        inputScale * inputMatrix[:, chainIndex],
                        thermalRunMode=resolvedRunMode,
                        thermalDutyCycle=thermalDutyCycle,
                        steadyStateToleranceC=steadyStateToleranceC,
                        maximumSteadyStateIterations=(
                            maximumSteadyStateIterations
                        ),
                    )
                    outputScale = 10.0 ** (
                        float(outputPowerDbValues[chainIndex]) / 20.0
                    )
                    chainOutput = outputScale * chainOutput
                    targetOutputRms = targetOutputRmsValues[chainIndex]
                    targetOutputPowerDbm = targetOutputPowerDbmValues[
                        chainIndex
                    ]
                    if targetOutputPowerDbm is not None:
                        targetOutputRms = PowerCalibration(
                            loadResistanceOhm=self.parameters[
                                "loadResistanceOhm"
                            ],
                            maximumOutputPowerDbm=self.parameters[
                                "maximumOutputPowerDbm"
                            ],
                        ).DbmToRms(targetOutputPowerDbm)
                    if targetOutputRms is not None:
                        currentRms = float(
                            np.sqrt(np.mean(np.abs(chainOutput) ** 2))
                        )
                        if currentRms <= np.finfo(float).tiny:
                            raise ValueError(
                                "cannot set target RMS on a zero-power PA output"
                            )
                        chainOutput = (
                            float(targetOutputRms)
                            * chainOutput
                            / currentRms
                        )
                    outputColumns.append(
                        np.asarray(chainOutput, dtype=np.complex128)
                    )
                    outputRmsValues.append(
                        float(
                            np.sqrt(np.mean(np.abs(chainOutput) ** 2))
                        )
                    )
                self.lastDissipatedPowerWPerChain = tuple(
                    float(
                        paModel.GetThermalMetrics().get(
                            "averageDissipatedPowerW", 0.0
                        )
                    )
                    for paModel in self.paModels
                )
                if not (
                    resolvedRunMode == "steady_state"
                    and hasMutualHeating
                ):
                    break
                solvedOffsetsC = couplingMatrix @ np.asarray(
                    self.lastDissipatedPowerWPerChain, dtype=float
                )
                mutualHeatingErrorC = float(
                    np.max(np.abs(solvedOffsetsC - candidateOffsetsC))
                )
                mutualHeatingIterations = mutualIterationIndex
                if mutualHeatingErrorC <= float(steadyStateToleranceC):
                    # The output, traces, and period-ending metrics were all
                    # evaluated with candidateOffsetsC. Keep that accepted
                    # state intact; solvedOffsetsC is only the fixed-point
                    # residual check and may differ by as much as tolerance.
                    break
                candidateOffsetsC = 0.5 * (
                    candidateOffsetsC + solvedOffsetsC
                )
            else:
                raise RuntimeError(
                    "MIMO mutual-heating steady-state solver did not "
                    f"converge within {maximumSteadyStateIterations} "
                    f"iterations; final error was "
                    f"{mutualHeatingErrorC:.6g} C and the allowed tolerance "
                    f"is {float(steadyStateToleranceC):.6g} C"
                )
        except Exception:
            self.RestoreThermalModel(thermalSnapshots)
            self.lastOutputRmsPerChain = previousOutputRmsPerChain
            self.lastDissipatedPowerWPerChain = (
                previousDissipatedPowerWPerChain
            )
            self._lastMutualHeatingMetrics = (
                previousMutualHeatingMetrics
            )
            raise
        outputMatrix = np.column_stack(outputColumns)
        self.lastOutputRmsPerChain = tuple(outputRmsValues)
        if resolvedRunMode == "transient":
            self.UpdateMutualHeating()
        self._lastMutualHeatingMetrics = {
            "steadyStateConverged": (
                resolvedRunMode == "steady_state"
                and (
                    not hasMutualHeating
                    or mutualHeatingErrorC
                    <= float(steadyStateToleranceC)
                )
            ),
            "steadyStateIterations": mutualHeatingIterations,
            "steadyStateErrorC": mutualHeatingErrorC,
        }
        if inputWasVector and self.numTransmitChains == 1:
            return outputMatrix[:, 0]
        return outputMatrix

    def CalculateActualDutyCycle(
        self,
        inputSignal: np.ndarray,
        thermalDutyCycle: float = 1.0,
    ) -> Tuple[float, ...]:
        """Return one complete-period RF duty fraction per PA chain.

        Processing details:
            Algorithm: Validate the normalized samples-by-chains matrix and
            delegate full-window activity detection to each physical PaModel,
            multiplying every result by the common scheduling duty cycle.

        Args:
            inputSignal: Normalized PA-input vector or matrix.
            thermalDutyCycle: Configured data-window fraction of one period.

        Returns:
            result: Chain-ordered actual RF duty-cycle tuple.
        """

        self.SynchronizeModels()
        complexInput = np.asarray(inputSignal, dtype=np.complex128)
        inputMatrix = (
            complexInput.reshape(-1, 1)
            if complexInput.ndim == 1
            else complexInput
        )
        if (
            inputMatrix.ndim != 2
            or inputMatrix.shape[0] == 0
            or inputMatrix.shape[1] != self.numTransmitChains
            or not np.all(np.isfinite(inputMatrix))
        ):
            raise ValueError(
                "inputSignal must have one finite column per transmit chain"
            )
        return tuple(
            paModel.CalculateActualDutyCycle(
                inputMatrix[:, chainIndex], thermalDutyCycle
            )
            for chainIndex, paModel in enumerate(self.paModels)
        )

    def ProcessChain(
        self, inputSignal: np.ndarray, chainIndex: int
    ) -> np.ndarray:
        """Process a vector through one selected PA and power calibration.

        Processing details:
            Algorithm: Decode raw external codes, apply the selected chain's
            committed post-DAC calibration drive, floating PA and power
            controls, then encode raw external codes.

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
        inputFormat = FixedPoint(self.width)
        outputFormat = FixedPoint(
            self.width, self.outputFullScaleAmplitude
        )
        complexInput = inputFormat.DecodeComplex(inputSignal)
        if complexInput.ndim != 1 or complexInput.size == 0:
            raise ValueError("inputSignal must be a nonempty vector")
        if not np.all(np.isfinite(complexInput)):
            raise ValueError("inputSignal must contain finite samples")
        driveDbPerChain = self.ResolveCalibrationDriveDbPerChain(
            self._calibrationDriveDbPerChain
        )
        driveScale = np.power(
            10.0, driveDbPerChain[chainIndex] / 20.0
        )
        chainOutput = self.ProcessChainFloating(
            driveScale * complexInput, chainIndex
        )
        return outputFormat.EncodeComplex(chainOutput)

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
        thermalSnapshots = []
        try:
            for paModel in self.paModels:
                thermalSnapshots.append(paModel.SuspendThermalModel())
        except Exception:
            # A later chain can reject snapshot creation. Restore every chain
            # already suspended so the transaction never leaves a partial
            # MIMO bank in temperature-independent mode.
            for restoredPaModel, thermalSnapshot in zip(
                self.paModels, thermalSnapshots
            ):
                restoredPaModel.RestoreThermalModel(thermalSnapshot)
            raise
        return tuple(thermalSnapshots)

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
            ),
            "mutualHeating": dict(self._lastMutualHeatingMetrics),
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
        self._calibrationDriveDbPerChain: Tuple[float, ...] = tuple()

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

    @property
    def OutputFullScaleAmplitude(self) -> float:
        """Forward the wrapped PA output-code physical full scale.

        Processing details:
            Algorithm: Read the wrapped protocol attribute and retain a scale
            of one for third-party normalized-output PA objects.

        Returns:
            result: Positive physical I/Q component full-scale amplitude.
        """

        rawFullScaleAmplitude = getattr(
            self.paModel, "outputFullScaleAmplitude", 1.0
        )
        return FixedPoint(
            self.width, rawFullScaleAmplitude
        ).fullScaleAmplitude

    outputFullScaleAmplitude = OutputFullScaleAmplitude

    def Process(self, inputSignal: np.ndarray) -> np.ndarray:
        """Apply the base PA and then add its conjugate image component.

        Processing details:
            Algorithm: Execute the configured signal-processing path, preserve sample alignment, and return the complete downstream result.

        Args:
            inputSignal: One-dimensional complex baseband samples supplied to the operation.

        Returns:
            result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
        """

        inputFormat = FixedPoint(self.width)
        outputFormat = FixedPoint(
            self.width, self.outputFullScaleAmplitude
        )
        floatingInput = inputFormat.DecodeComplex(inputSignal)
        floatingOutput, _ = self.ProcessOutputPathsFloating(floatingInput)
        return outputFormat.EncodeComplex(floatingOutput)

    def SetCalibrationDriveDb(
        self, driveDbPerChain: Sequence[float]
    ) -> None:
        """Commit calibrated post-DAC drive on the wrapped physical PA.

        Processing details:
            Algorithm: Validate and retain the complete chain-ordered sequence,
            then forward it when the wrapped plant exposes the paired analog-
            drive protocol. Otherwise the facade applies this fallback drive
            before the wrapped raw or public processor. A delegated drive is
            never applied a second time by the facade.

        Args:
            driveDbPerChain: One finite analog-drive value per wrapped PA chain.

        Returns:
            result: None. The wrapped plant or fallback facade owns the drive.
        """

        if isinstance(driveDbPerChain, (str, bytes)):
            raise TypeError("driveDbPerChain must be a numeric sequence")
        driveArray = np.asarray(driveDbPerChain, dtype=object)
        if driveArray.ndim != 1 or driveArray.size == 0:
            raise ValueError("driveDbPerChain must be a nonempty sequence")
        resolvedDriveValues = []
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
            resolvedDriveValues.append(float(driveValue))
        resolvedDriveDb = tuple(resolvedDriveValues)
        trialMethod = getattr(self.paModel, "ProcessCalibrationDrive", None)
        commitMethod = getattr(self.paModel, "SetCalibrationDriveDb", None)
        if callable(trialMethod) != callable(commitMethod):
            raise TypeError(
                "wrapped PA must expose both ProcessCalibrationDrive and "
                "SetCalibrationDriveDb, or neither"
            )
        self._calibrationDriveDbPerChain = resolvedDriveDb
        if callable(commitMethod):
            commitMethod(resolvedDriveDb)

    def ProcessCalibrationDrive(
        self,
        inputSignal: np.ndarray,
        driveDbPerChain: Sequence[float],
    ) -> np.ndarray:
        """Evaluate one wrapped-PA calibration trial before output IQ mapping.

        Processing details:
            Algorithm: Delegate public-code decoding, explicit trial drive, PA
            evaluation, and encoding when the wrapped calibration protocol is
            available. Otherwise decode locally and apply the trial drive before
            the wrapped raw processor. In both cases apply the output direct/
            image mapping and encode the measurement without committing state.

        Args:
            inputSignal: Public floating samples or fixed-point I/Q codes.
            driveDbPerChain: Trial analog-drive values in physical chain order.

        Returns:
            result: Public IQ-imbalanced output of the uncommitted trial.
        """

        inputFormat = FixedPoint(self.width)
        outputFormat = FixedPoint(
            self.width, self.outputFullScaleAmplitude
        )
        trialMethod = getattr(self.paModel, "ProcessCalibrationDrive", None)
        commitMethod = getattr(self.paModel, "SetCalibrationDriveDb", None)
        if callable(trialMethod) != callable(commitMethod):
            raise TypeError(
                "wrapped PA must expose both ProcessCalibrationDrive and "
                "SetCalibrationDriveDb, or neither"
            )
        if callable(trialMethod):
            rawPaOutput = trialMethod(inputSignal, driveDbPerChain)
            floatingPaOutput = outputFormat.DecodeComplex(rawPaOutput)
            floatingOutput = np.asarray(
                self.directCoefficient * floatingPaOutput
                + self.imageCoefficient * np.conj(floatingPaOutput),
                dtype=np.complex128,
            )
        else:
            floatingInput = inputFormat.DecodeComplex(inputSignal)
            driveArray = np.asarray(driveDbPerChain, dtype=float).reshape(-1)
            inputMatrix = (
                floatingInput.reshape(-1, 1)
                if floatingInput.ndim == 1
                else floatingInput
            )
            if (
                inputMatrix.ndim != 2
                or driveArray.size != inputMatrix.shape[1]
                or not np.all(np.isfinite(driveArray))
            ):
                raise ValueError(
                    "driveDbPerChain must contain one finite value per chain"
                )
            drivenInput = inputMatrix * np.power(
                10.0, driveArray.reshape(1, -1) / 20.0
            )
            if floatingInput.ndim == 1:
                drivenInput = drivenInput[:, 0]
            floatingOutput = self.ProcessFloating(drivenInput)
        return outputFormat.EncodeComplex(floatingOutput)

    def ProcessOutputPathsFloating(
        self, inputSignal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Preserve committed drive and both observations through the wrapper.

        Processing details:
            Algorithm: Prefer the wrapped plant's committed-drive floating
            dual-output protocol. When absent, apply this facade's validated
            fallback drive before the raw or public wrapped processor. Then
            apply the same output IQ transformation independently to channel
            and feedback observations; a single output is duplicated.

        Args:
            inputSignal: Normalized complex samples after public decoding.

        Returns:
            result: IQ-imbalanced floating ``(chOut, fbOut)`` observations.
        """

        complexInput = np.asarray(inputSignal, dtype=np.complex128)
        outputPathsProcessor = getattr(
            self.paModel, "ProcessOutputPathsFloating", None
        )
        if callable(outputPathsProcessor):
            paChannelOutput, paFeedbackOutput = outputPathsProcessor(
                complexInput
            )
        else:
            drivenInput = complexInput
            if self._calibrationDriveDbPerChain:
                inputMatrix = (
                    complexInput.reshape(-1, 1)
                    if complexInput.ndim == 1
                    else complexInput
                )
                if (
                    inputMatrix.ndim != 2
                    or inputMatrix.shape[1]
                    != len(self._calibrationDriveDbPerChain)
                ):
                    raise ValueError(
                        "committed calibration drive must contain one value "
                        "per input chain"
                    )
                drivenMatrix = inputMatrix * np.power(
                    10.0,
                    np.asarray(
                        self._calibrationDriveDbPerChain, dtype=float
                    ).reshape(1, -1)
                    / 20.0,
                )
                drivenInput = (
                    drivenMatrix[:, 0]
                    if complexInput.ndim == 1
                    else drivenMatrix
                )
            floatingProcessor = getattr(self.paModel, "ProcessFloating", None)
            if callable(floatingProcessor):
                paChannelOutput = floatingProcessor(drivenInput)
            else:
                inputFormat = FixedPoint(self.width)
                outputFormat = FixedPoint(
                    self.width, self.outputFullScaleAmplitude
                )
                publicOutput = self.paModel.Process(
                    inputFormat.EncodeComplex(drivenInput)
                )
                paChannelOutput = outputFormat.DecodeComplex(publicOutput)
            paFeedbackOutput = np.asarray(
                paChannelOutput, dtype=np.complex128
            ).copy()

        complexChannelOutput = np.asarray(
            paChannelOutput, dtype=np.complex128
        )
        complexFeedbackOutput = np.asarray(
            paFeedbackOutput, dtype=np.complex128
        )
        return (
            np.asarray(
                self.directCoefficient * complexChannelOutput
                + self.imageCoefficient * np.conj(complexChannelOutput),
                dtype=np.complex128,
            ),
            np.asarray(
                self.directCoefficient * complexFeedbackOutput
                + self.imageCoefficient * np.conj(complexFeedbackOutput),
                dtype=np.complex128,
            ),
        )

    def ProcessFloating(self, inputSignal: np.ndarray) -> np.ndarray:
        """Apply PA nonlinearity and IQ imbalance in floating-point units.

        Processing details:
            Algorithm: Evaluate the wrapped PA's raw drive-free kernel without
            public code conversion when supported, then combine direct and
            conjugated image paths. Decoded public samples and ILC use
            ``ProcessOutputPathsFloating`` to preserve committed analog drive.

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
            inputFormat = FixedPoint(self.width)
            outputFormat = FixedPoint(
                self.width, self.outputFullScaleAmplitude
            )
            publicOutput = self.paModel.Process(
                inputFormat.EncodeComplex(complexInput)
            )
            paOutput = outputFormat.DecodeComplex(publicOutput)
        return np.asarray(
            self.directCoefficient * paOutput
            + self.imageCoefficient * np.conj(paOutput),
            dtype=np.complex128,
        )

    def ProcessThermalPeriodFloating(
        self,
        inputSignal: np.ndarray,
        thermalRunMode: str = "steady_state",
        thermalDutyCycle: float = 1.0,
        steadyStateToleranceC: float = 1.0e-4,
        maximumSteadyStateIterations: int = 100,
    ) -> np.ndarray:
        """Preserve periodic thermal scheduling through the IQ wrapper.

        Processing details:
            Algorithm: Delegate the complete self-heating period to the
            wrapped PA and apply the widely-linear output transformation only
            after the physical PA has generated its temperature-dependent
            waveform. This keeps heat generation at the PA-output reference
            plane while retaining the wrapper's observable IQ imbalance.

        Args:
            inputSignal: Normalized floating PA-input data window.
            thermalRunMode: ``"steady_state"`` or ``"transient"``.
            thermalDutyCycle: Data-window fraction of the complete period.
            steadyStateToleranceC: Allowed periodic closure error in C.
            maximumSteadyStateIterations: Maximum thermal solver iterations.

        Returns:
            result: Temperature-aware, IQ-imbalanced floating waveform.
        """

        thermalProcessor = getattr(
            self.paModel, "ProcessThermalPeriodFloating", None
        )
        if not callable(thermalProcessor):
            return self.ProcessFloating(inputSignal)
        paOutput = thermalProcessor(
            inputSignal,
            thermalRunMode=thermalRunMode,
            thermalDutyCycle=thermalDutyCycle,
            steadyStateToleranceC=steadyStateToleranceC,
            maximumSteadyStateIterations=maximumSteadyStateIterations,
        )
        return np.asarray(
            self.directCoefficient * paOutput
            + self.imageCoefficient * np.conj(paOutput),
            dtype=np.complex128,
        )

    def SuspendThermalModel(self) -> object:
        """Suspend a wrapped thermal PA for cold power-calibration trials.

        Processing details:
            Algorithm: Forward the transactional snapshot request when the
            wrapped PA supports thermal state; otherwise return None so the
            paired restore operation remains a harmless no-op.

        Returns:
            result: Opaque wrapped-PA thermal snapshot or None.
        """

        suspendMethod = getattr(self.paModel, "SuspendThermalModel", None)
        restoreMethod = getattr(self.paModel, "RestoreThermalModel", None)
        if callable(suspendMethod) != callable(restoreMethod):
            raise TypeError(
                "wrapped PA thermal transaction must expose both "
                "SuspendThermalModel and RestoreThermalModel, or neither"
            )
        return suspendMethod() if callable(suspendMethod) else None

    def RestoreThermalModel(self, thermalSnapshot: object) -> None:
        """Restore a wrapped PA thermal snapshot without advancing time.

        Processing details:
            Algorithm: Forward a non-None opaque snapshot to the wrapped PA.
            A None snapshot represents a wrapped PA without thermal support.

        Args:
            thermalSnapshot: Value returned by ``SuspendThermalModel``.

        Returns:
            result: None. The prior wrapped thermal state is restored.
        """

        if thermalSnapshot is None:
            return
        restoreMethod = getattr(self.paModel, "RestoreThermalModel", None)
        if not callable(restoreMethod):
            raise TypeError(
                "wrapped thermal PA must expose RestoreThermalModel"
            )
        restoreMethod(thermalSnapshot)

    def GetThermalMetrics(self) -> Dict[str, object]:
        """Return the wrapped PA's latest thermal diagnostics.

        Processing details:
            Algorithm: Forward the read-only metrics protocol and copy its
            mapping so the IQ wrapper cannot hide enabled periodic scheduling.

        Returns:
            result: Thermal metrics dictionary or an explicit disabled flag.
        """

        metricsMethod = getattr(self.paModel, "GetThermalMetrics", None)
        if not callable(metricsMethod):
            return {"enabled": False}
        thermalMetrics = metricsMethod()
        if not isinstance(thermalMetrics, Mapping):
            raise TypeError(
                "wrapped PA GetThermalMetrics must return a mapping"
            )
        return dict(thermalMetrics)

    def CalculateActualDutyCycle(
        self,
        inputSignal: np.ndarray,
        thermalDutyCycle: float = 1.0,
    ) -> object:
        """Forward actual RF-duty observation to the wrapped physical PA.

        Processing details:
            Algorithm: Reuse the wrapped PA's own activity threshold and
            reference plane because output IQ imbalance does not alter the
            PA-input activity schedule or its heat-generation duty cycle.

        Args:
            inputSignal: Normalized floating PA-input data window.
            thermalDutyCycle: Data-window fraction of the complete period.

        Returns:
            result: Scalar or per-chain complete-period RF duty fraction.
        """

        dutyMethod = getattr(
            self.paModel, "CalculateActualDutyCycle", None
        )
        if not callable(dutyMethod):
            raise TypeError(
                "wrapped thermal PA must expose CalculateActualDutyCycle"
            )
        return dutyMethod(inputSignal, thermalDutyCycle)

    def ResetThermalState(
        self,
        junctionTemperatureC: Optional[object] = None,
        ambientTemperatureC: Optional[float] = None,
    ) -> None:
        """Reset the wrapped PA's optional thermal state.

        Processing details:
            Algorithm: Forward the explicit reset request without changing IQ
            coefficients, and reject wrappers around PAs lacking that protocol.

        Args:
            junctionTemperatureC: Optional scalar or per-chain start value.
            ambientTemperatureC: Optional thermal-boundary temperature in C.

        Returns:
            result: None. Wrapped thermal state is reset.
        """

        resetMethod = getattr(self.paModel, "ResetThermalState", None)
        if not callable(resetMethod):
            raise TypeError("wrapped PA does not support ResetThermalState")
        resetMethod(junctionTemperatureC, ambientTemperatureC)

    def AdvanceIdle(self, idleTimeSec: float) -> object:
        """Advance wrapped thermal state through an additional idle gap.

        Processing details:
            Algorithm: Forward the physical duration to the wrapped PA; the IQ
            output transformation has no independent heat state to advance.

        Args:
            idleTimeSec: Nonnegative additional idle duration in seconds.

        Returns:
            result: Wrapped PA junction temperature result.
        """

        advanceMethod = getattr(self.paModel, "AdvanceIdle", None)
        if not callable(advanceMethod):
            raise TypeError("wrapped PA does not support AdvanceIdle")
        return advanceMethod(idleTimeSec)

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
    nonlinearScale: float = 0.135,
) -> Tuple[
    Dict[Tuple[int, int], complex],
    Dict[Tuple[int, int, int], complex],
    Dict[Tuple[int, int, int], complex],
]:
    """Create monotonic static GMP coefficients with mild dynamic memory.

    Processing details:
        Algorithm: Select a Rapp-like per-order steady-state target fitted over
        normalized amplitudes from zero through two, scale every nonlinear
        steady-state order by the configured default strength, generate small
        causal main and envelope-cross memory tails, and solve the zero-delay
        main coefficient so every same-order coefficient still sums to its
        static target. Memory depth therefore changes transient and frequency
        behavior without duplicating compression in a constant-envelope
        plateau.

    Args:
        nonlinearOrders: Positive odd polynomial orders included in the model.
        memoryDepth: Number of causal sample delays included in the model.
        crossMemoryDepth: Number of envelope cross-delays included in the GMP model.
        nonlinearScale: Strength applied to orders above one. Zero selects the
            linear-memory floor and one selects the full reference fit.

    Returns:
        result: Tuple[Dict[Tuple[int, int], complex], Dict[Tuple[int, int, int], complex], Dict[Tuple[int, int, int], complex]]. The computed value described by the summary, with documented units, shape, and normalization.
    """

    if (
        not isinstance(nonlinearScale, (int, float))
        or isinstance(nonlinearScale, bool)
        or not np.isfinite(nonlinearScale)
        or not 0.0 <= float(nonlinearScale) <= 1.0
    ):
        raise ValueError(
            "nonlinearScale must be finite and between zero and one"
        )

    # These reference coefficients were fitted over 0 <= |x| <= 2 to a
    # bounded Rapp-like AM-AM curve with mild AM-PM conversion.  The public
    # default uses 13.5 percent of the nonlinear orders so a 20 dBm Wi-Fi
    # operating point remains moderately distorted instead of entering deep
    # compression.  Scaling toward zero preserves the reference curve's
    # monotonicity and the first-order electrical-memory floor.
    referenceSteadyStateCoefficient = {
        1: 1.261692 + 0.014052j,
        3: -0.291144 + 0.054204j,
        5: 0.031812 - 0.022452j,
        7: -0.000168 + 0.002784j,
    }
    selectedOrders = tuple(dict.fromkeys(int(order) for order in nonlinearOrders))
    steadyStateCoefficients = {
        nonlinearOrder: (
            referenceSteadyStateCoefficient.get(
                nonlinearOrder,
                0.0 + 0.0j,
            )
            * (
                1.0
                if nonlinearOrder == 1
                else float(nonlinearScale)
            )
        )
        for nonlinearOrder in selectedOrders
    }

    if 1 in steadyStateCoefficients:
        # Removing stabilizing higher orders from a fitted polynomial can make
        # the remaining subset fold back.  Reduce all nonlinear targets by one
        # common factor only when the requested default order subset needs it.
        # The complete (1, 3, 5, 7) default passes unchanged with factor one.
        validationAmplitude = np.linspace(0.0, 2.0, 4097)
        fullScaleOutput = sum(
            coefficient * validationAmplitude**nonlinearOrder
            for nonlinearOrder, coefficient in (
                steadyStateCoefficients.items()
            )
            if coefficient != 0.0 + 0.0j
        )
        fullScaleIsMonotonic = bool(
            np.all(np.diff(np.abs(fullScaleOutput)) >= -1.0e-12)
        )
        nonlinearScale = 1.0
        if not fullScaleIsMonotonic:
            lowerScale = 0.0
            upperScale = 1.0
            for _ in range(56):
                candidateScale = 0.5 * (lowerScale + upperScale)
                candidateOutput = sum(
                    coefficient
                    * (
                        1.0
                        if nonlinearOrder == 1
                        else candidateScale
                    )
                    * validationAmplitude**nonlinearOrder
                    for nonlinearOrder, coefficient in (
                        steadyStateCoefficients.items()
                    )
                    if coefficient != 0.0 + 0.0j
                )
                isMonotonic = bool(
                    np.all(
                        np.diff(np.abs(candidateOutput)) >= -1.0e-12
                    )
                )
                if isMonotonic:
                    lowerScale = candidateScale
                else:
                    upperScale = candidateScale
            nonlinearScale = 0.98 * lowerScale
        steadyStateCoefficients = {
            nonlinearOrder: (
                coefficient
                if nonlinearOrder == 1
                else nonlinearScale * coefficient
            )
            for nonlinearOrder, coefficient in (
                steadyStateCoefficients.items()
            )
        }
    elif len(selectedOrders) > 0:
        # A default PA without a first-order path has no meaningful small-signal
        # gain.  Keep its lowest requested basis monotonic and leave all higher
        # default targets at zero; measured custom dictionaries are unaffected.
        minimumOrder = min(selectedOrders)
        steadyStateCoefficients = {
            nonlinearOrder: (
                1.0 + 0.0j
                if nonlinearOrder == minimumOrder
                else 0.0 + 0.0j
            )
            for nonlinearOrder in selectedOrders
        }
    mainCoefficients: Dict[Tuple[int, int], complex] = {}
    laggingCoefficients: Dict[Tuple[int, int, int], complex] = {}
    leadingCoefficients: Dict[Tuple[int, int, int], complex] = {}

    for nonlinearOrder in selectedOrders:
        targetCoefficient = steadyStateCoefficients[nonlinearOrder]
        if targetCoefficient == 0.0 + 0.0j:
            continue

        # Delayed main-branch coefficients model a small dynamic residual.  A
        # six-percent first tail is large enough to expose electrical memory in
        # two-tone and wideband tests, but it avoids the former implementation's
        # 34-percent repeated compression term that made a constant high-level
        # run collapse after its first sample.
        for memoryIndex in range(1, memoryDepth):
            if nonlinearOrder == 1:
                delayedCoefficient = (0.045 - 0.020j) * (
                    (-0.45) ** (memoryIndex - 1)
                )
            else:
                delayedCoefficient = targetCoefficient * (
                    0.06**memoryIndex
                ) * np.exp(-1j * 0.18 * memoryIndex)
            mainCoefficients[(nonlinearOrder, memoryIndex)] = (
                delayedCoefficient
            )

        if nonlinearOrder > 1 and targetCoefficient != 0.0 + 0.0j:
            for memoryIndex in range(memoryDepth):
                for crossIndex in range(1, crossMemoryDepth + 1):
                    crossDecay = (0.22**memoryIndex) * (
                        0.42**crossIndex
                    )
                    laggingCoefficients[
                        (nonlinearOrder, memoryIndex, crossIndex)
                    ] = (
                        targetCoefficient
                        * (-0.060 + 0.025j)
                        * crossDecay
                    )
                    leadingCoefficients[
                        (nonlinearOrder, memoryIndex, crossIndex)
                    ] = (
                        targetCoefficient
                        * (0.040 - 0.018j)
                        * crossDecay
                    )

        # A constant complex envelope makes every basis of the same order
        # identical.  Set the zero-delay coefficient to the residual required
        # for their total to equal targetCoefficient.  Consequently changing
        # memoryDepth or crossMemoryDepth changes dynamics without changing the
        # settled static gain curve.
        delayedMainSum = sum(
            coefficient
            for (order, memoryIndex), coefficient in mainCoefficients.items()
            if order == nonlinearOrder and memoryIndex > 0
        )
        laggingSum = sum(
            coefficient
            for (order, _, _), coefficient in laggingCoefficients.items()
            if order == nonlinearOrder
        )
        leadingSum = sum(
            coefficient
            for (order, _, _), coefficient in leadingCoefficients.items()
            if order == nonlinearOrder
        )
        mainCoefficients[(nonlinearOrder, 0)] = (
            targetCoefficient
            - delayedMainSum
            - laggingSum
            - leadingSum
        )

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
