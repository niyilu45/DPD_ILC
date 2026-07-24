"""Power-amplifier behavioral models used by the DPD-ILC simulation.

Callers construct ``PaModel`` with ``modelName="wiener"`` or
``modelName="gmp"`` and then call ``Process``. Two nonlinear model families
are provided internally:

* ``WienerPA`` applies a linear memory filter followed by a smooth Rapp
  AM-AM characteristic and a saturating AM-PM characteristic.
* ``GMPPA`` implements the generalized memory polynomial main, lagging,
  and leading cross terms described in the project theory document.

``PaModel`` accepts one complex stream. ``MimoPaModel`` owns one independent
``PaModel`` per transmit chain, accepts a samples-by-chains matrix, and applies
independent input drive and output-power calibration on every chain.
"""

from collections import ChainMap
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, cast

import numpy as np


class PowerCalibration:
    """Convert PA complex-envelope RMS voltage and absolute dBm power.

    The project uses the explicit convention that the RMS magnitude of a
    complex baseband waveform is the RMS voltage delivered to the configured
    resistive port. Under this convention ``P = Vrms**2 / R``. Keeping the
    resistance in one object prevents command-line, benchmark, and plotting
    code from applying inconsistent power scales.
    """

    def __init__(
        self,
        loadResistanceOhm: Optional[float] = None,
        parameters: Optional[Mapping[str, object]] = None,
        **parameterOverrides: object,
    ) -> None:
        """Initialize a live ChainMap-backed RF power calibration.

        Processing details:
            Algorithm: Define the standard 50-ohm default inside this
            constructor, layer caller values ahead of it, and validate the
            resolved resistance before any logarithmic conversion.

        Args:
            loadResistanceOhm: Optional resistive port value in ohms.
            parameters: Optional caller-owned mapping of calibration values.
            parameterOverrides: Highest-priority local calibration overrides.

        Returns:
            result: None. The converter is ready for dBm/RMS transformations.
        """

        self.defaultParameters: Mapping[str, object] = MappingProxyType(
            {"loadResistanceOhm": 50.0}
        )
        directOverrides = dict(parameterOverrides)
        if loadResistanceOhm is not None:
            directOverrides["loadResistanceOhm"] = loadResistanceOhm
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping or None")
        externalParameters = {} if parameters is None else parameters
        self.parameters: ChainMap[str, object] = ChainMap(
            directOverrides,
            externalParameters,
            self.defaultParameters,
        )
        self.Validate()

    @property
    def LoadResistanceOhm(self) -> float:
        """Return the resolved resistive PA port value in ohms.

        Processing details:
            Algorithm: Read the highest-priority ChainMap value after
            constructor and update validation.

        Returns:
            result: Positive finite resistance in ohms.
        """

        return float(cast(float, self.parameters["loadResistanceOhm"]))

    loadResistanceOhm = LoadResistanceOhm

    def GetParameters(self) -> Dict[str, object]:
        """Return a flattened calibration parameter snapshot.

        Processing details:
            Algorithm: Resolve all ChainMap layers without changing the live
            caller-owned mapping.

        Returns:
            result: Ordinary dictionary containing the resolved resistance.
        """

        return dict(self.parameters)

    def UpdateParameters(self, **parameterOverrides: object) -> None:
        """Apply and validate high-priority calibration overrides.

        Processing details:
            Algorithm: Update the local ChainMap layer transactionally and
            restore the previous values if validation fails.

        Args:
            parameterOverrides: Local calibration values to replace.

        Returns:
            result: None. The active converter is updated in place.
        """

        previousOverrides = dict(self.parameters.maps[0])
        self.parameters.maps[0].update(parameterOverrides)
        try:
            self.Validate()
        except (TypeError, ValueError):
            self.parameters.maps[0].clear()
            self.parameters.maps[0].update(previousOverrides)
            raise

    def Validate(self) -> None:
        """Validate the resolved resistance and reject unknown settings.

        Processing details:
            Algorithm: Check the exact supported key set, numeric type,
            finiteness, and positive physical domain.

        Returns:
            result: None. Invalid calibration raises an exception.
        """

        unknownParameters = set(self.parameters).difference(
            self.defaultParameters
        )
        if unknownParameters:
            unknownNames = ", ".join(
                sorted(str(parameterName) for parameterName in unknownParameters)
            )
            raise TypeError(
                f"unknown PowerCalibration parameters: {unknownNames}"
            )
        resistanceValue = self.parameters["loadResistanceOhm"]
        if (
            not isinstance(resistanceValue, (int, float))
            or isinstance(resistanceValue, bool)
            or not np.isfinite(resistanceValue)
            or resistanceValue <= 0.0
        ):
            raise ValueError(
                "loadResistanceOhm must be finite and positive"
            )

    def DbmToRms(self, inputPowerDbm: float) -> float:
        """Convert absolute port power in dBm to complex-envelope RMS volts.

        Processing details:
            Algorithm: Convert dBm to watts with the one-milliwatt reference,
            multiply by resistance, and take the positive RMS square root.

        Args:
            inputPowerDbm: Absolute available power in dBm.

        Returns:
            result: Positive RMS voltage used to scale a unit-RMS waveform.
        """

        if (
            not isinstance(inputPowerDbm, (int, float))
            or isinstance(inputPowerDbm, bool)
            or not np.isfinite(inputPowerDbm)
        ):
            raise ValueError("inputPowerDbm must be finite")
        # Compute the voltage directly with a 20-log amplitude exponent. This
        # is algebraically identical to converting through watts, while NumPy
        # lets the explicit finite-range check handle overflow and underflow.
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            rmsVoltage = (
                np.sqrt(1.0e-3 * self.loadResistanceOhm)
                * np.power(10.0, float(inputPowerDbm) / 20.0)
            )
        if not np.isfinite(rmsVoltage) or rmsVoltage <= 0.0:
            raise ValueError("inputPowerDbm is outside the numeric range")
        return float(rmsVoltage)

    def RmsToDbm(self, signalRms: float) -> float:
        """Convert complex-envelope RMS volts to absolute port power in dBm.

        Processing details:
            Algorithm: Divide squared RMS voltage by resistance to obtain
            watts, normalize by one milliwatt, and take ten-base logarithms.

        Args:
            signalRms: Positive complex-envelope RMS voltage.

        Returns:
            result: Absolute resistive-port power in dBm.
        """

        if (
            not isinstance(signalRms, (int, float))
            or isinstance(signalRms, bool)
            or not np.isfinite(signalRms)
            or signalRms <= 0.0
        ):
            raise ValueError("signalRms must be finite and positive")
        # The logarithmic form avoids squaring a very large finite voltage.
        return float(
            20.0 * np.log10(float(signalRms))
            - 10.0 * np.log10(self.loadResistanceOhm * 1.0e-3)
        )


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


class PaModel:
    """Configure and operate one Wiener or GMP nonlinear PA model.

    The facade gives every caller the same object-oriented construction and
    processing interface while retaining the dedicated model implementations.

    Example:
        ``paModel = PaModel(modelName="wiener")``
        ``outputSignal = paModel.Process(inputSignal)``
    """

    def __init__(
        self,
        modelName: Optional[str] = None,
        wienerConfig: Optional[WienerConfig] = None,
        gmpConfig: Optional[GMPConfig] = None,
        parameters: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Initialize the PA facade and select its active model family.

        Processing details:
            Algorithm: Define immutable PA defaults inside this constructor,
            then layer direct arguments and the caller-owned mapping ahead of
            them so callers provide only values they intend to override.

        Args:
            modelName: Selected PA model family name.
            wienerConfig: Optional Wiener configuration; None selects built-in values.
            gmpConfig: Optional GMP configuration; None selects built-in values.
            parameters: Optional external mapping layered ahead of the built-in defaults.

        Returns:
            result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
        """
        self.defaultParameters: Mapping[str, object] = MappingProxyType(
            {
                "modelName": "wiener",
                "wienerConfig": None,
                "gmpConfig": None,
            }
        )
        parameterOverrides: Dict[str, object] = {}
        if modelName is not None:
            parameterOverrides["modelName"] = modelName
        if wienerConfig is not None:
            parameterOverrides["wienerConfig"] = wienerConfig
        if gmpConfig is not None:
            parameterOverrides["gmpConfig"] = gmpConfig
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping or None")
        externalParameters = {} if parameters is None else parameters
        self.parameters: ChainMap[str, object] = ChainMap(
            parameterOverrides,
            externalParameters,
            self.defaultParameters,
        )
        self.model = None
        self._activeConfiguration: Optional[
            Tuple[str, Optional[WienerConfig], Optional[GMPConfig]]
        ] = None
        self.SynchronizeModel()

    @property
    def ModelName(self) -> str:
        """Return the normalized model name resolved by the ChainMap.

        Processing details:
            Algorithm: Carry out the described operation using validated inputs, explicit array-shape handling, and deterministic project conventions.

        Returns:
            result: str. The computed value described by the summary, with documented units, shape, and normalization.
        """

        normalizedName, _, _ = self.ResolveConfiguration()
        return normalizedName

    modelName = ModelName

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

        previousOverrides = dict(self.parameters.maps[0])
        self.parameters.maps[0].update(parameterOverrides)
        try:
            self.SynchronizeModel()
        except (TypeError, ValueError):
            self.parameters.maps[0].clear()
            self.parameters.maps[0].update(previousOverrides)
            self.SynchronizeModel()
            raise

    def ResolveConfiguration(
        self,
    ) -> Tuple[str, Optional[WienerConfig], Optional[GMPConfig]]:
        """Validate and return the currently resolved PA configuration.

        Processing details:
            Algorithm: Resolve values according to state and ChainMap precedence, keeping caller-owned configuration behavior explicit.

        Returns:
            result: Tuple[str, Optional[WienerConfig], Optional[GMPConfig]]. The computed value described by the summary, with documented units, shape, and normalization.
        """

        unknownParameters = set(self.parameters).difference(
            self.defaultParameters
        )
        if unknownParameters:
            unknownNames = ", ".join(
                sorted(str(parameterName) for parameterName in unknownParameters)
            )
            raise TypeError(f"unknown PaModel parameters: {unknownNames}")
        rawModelName = self.parameters["modelName"]
        if not isinstance(rawModelName, str):
            raise TypeError("modelName must be a string")
        normalizedName = rawModelName.strip().lower()
        if normalizedName not in ("wiener", "gmp"):
            raise ValueError("modelName must be either 'wiener' or 'gmp'")

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
        return (
            normalizedName,
            cast(Optional[WienerConfig], rawWienerConfig),
            cast(Optional[GMPConfig], rawGmpConfig),
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
        normalizedName, wienerConfig, gmpConfig = selectedConfiguration
        if normalizedName == "wiener":
            selectedModel = WienerPA(
                WienerConfig() if wienerConfig is None else wienerConfig
            )
        else:
            selectedModel = GMPPA(
                GMPConfig() if gmpConfig is None else gmpConfig
            )
        self.model = selectedModel
        self._activeConfiguration = selectedConfiguration

    def Process(self, inputSignal: np.ndarray) -> np.ndarray:
        """Pass a complex waveform through the configured PA model.

        Processing details:
            Algorithm: Execute the configured signal-processing path, preserve sample alignment, and return the complete downstream result.

        Args:
            inputSignal: One-dimensional complex baseband samples supplied to the operation.

        Returns:
            result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
        """

        self.SynchronizeModel()
        return self.model.Process(inputSignal)

    def SmallSignalGain(self) -> complex:
        """Return the configured model's DC small-signal complex gain.

        Processing details:
            Algorithm: Carry out the described operation using validated inputs, explicit array-shape handling, and deterministic project conventions.

        Returns:
            result: complex. The computed value described by the summary, with documented units, shape, and normalization.
        """

        self.SynchronizeModel()
        return self.model.SmallSignalGain()


class MimoPaModel:
    """Operate independent nonlinear PA models on all transmit chains.

    Each chain can select its own Wiener/GMP configuration, input drive,
    relative output power, and optional absolute output-power target in dBm.
    A legacy RMS-voltage target remains available for compatibility. The class
    intentionally does not add electrical crosstalk; a coupled MIMO PA can be
    introduced later without changing the samples-by-chains interface.
    """

    def __init__(
        self,
        parameters: Optional[Mapping[str, object]] = None,
        **parameterOverrides: object,
    ) -> None:
        """Initialize all chain models with internal default parameters.

        Processing details:
            Algorithm: Define immutable defaults locally, layer caller
            overrides with ``ChainMap``, validate every per-chain sequence,
            and construct one independent ``PaModel`` for each transmit chain.

        Args:
            parameters: Optional caller-owned mapping containing overrides.
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
            }
        )
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping or None")
        externalParameters = {} if parameters is None else parameters
        self.parameters: ChainMap[str, object] = ChainMap(
            dict(parameterOverrides),
            externalParameters,
            self.defaultParameters,
        )
        self.paModels = []
        self._activePaParameterSnapshot = None
        self.lastOutputRmsPerChain: Tuple[float, ...] = tuple()
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

        previousOverrides = dict(self.parameters.maps[0])
        self.parameters.maps[0].update(parameterOverrides)
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
            Algorithm: Reject unknown keys, validate the positive chain count,
            resolve all per-chain sequences, validate the resistive port, and
            reject conflicting legacy RMS and absolute dBm targets.

        Returns:
            result: None. Invalid settings raise descriptive exceptions.
        """

        unknownParameters = set(self.parameters).difference(
            self.defaultParameters
        )
        if unknownParameters:
            unknownNames = ", ".join(
                sorted(str(parameterName) for parameterName in unknownParameters)
            )
            raise TypeError(f"unknown MimoPaModel parameters: {unknownNames}")
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
            loadResistanceOhm=self.parameters["loadResistanceOhm"]
        )
        for targetPowerDbm in targetOutputPowerDbmValues:
            if targetPowerDbm is not None:
                powerCalibration.DbmToRms(targetPowerDbm)

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
            PaModel(parameters=chainParameters)
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
            Algorithm: Apply per-chain input dB drive, nonlinear PA processing,
            relative output dB calibration, and optional absolute dBm scaling
            after conversion to an RMS voltage; stack outputs with the original
            samples-by-chains orientation.

        Args:
            inputSignal: Complex vector for one chain or matrix shaped samples
                by the configured number of transmit chains.

        Returns:
            result: Processed complex array with the same dimensionality.
        """

        self.SynchronizeModels()
        complexInput = np.asarray(inputSignal, dtype=np.complex128)
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
            chainOutput = self.ProcessChain(
                complexInput[:, chainIndex], chainIndex
            )
            outputColumns.append(chainOutput)
            outputRmsValues.append(
                float(np.sqrt(np.mean(np.abs(chainOutput) ** 2)))
            )
        outputMatrix = np.column_stack(outputColumns)
        self.lastOutputRmsPerChain = tuple(outputRmsValues)
        if inputWasVector and self.numTransmitChains == 1:
            return outputMatrix[:, 0]
        return outputMatrix

    def ProcessChain(
        self, inputSignal: np.ndarray, chainIndex: int
    ) -> np.ndarray:
        """Process a vector through one selected PA and power calibration.

        Processing details:
            Algorithm: Apply this chain's input drive, nonlinear model,
            relative output calibration, and optional absolute RMS target.
            This method is also the independent plant used by per-chain ILC.

        Args:
            inputSignal: One-dimensional complex samples for one RF chain.
            chainIndex: Zero-based physical PA index.

        Returns:
            result: Processed one-dimensional complex samples.
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
        chainOutput = self.paModels[chainIndex].Process(
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
                loadResistanceOhm=self.parameters["loadResistanceOhm"]
            ).DbmToRms(targetOutputPowerDbm)
        if targetOutputRms is not None:
            currentRms = np.sqrt(np.mean(np.abs(chainOutput) ** 2))
            if currentRms <= np.finfo(float).tiny:
                raise ValueError(
                    "cannot set target RMS on a zero-power PA output"
                )
            chainOutput = float(targetOutputRms) * chainOutput / currentRms
        return chainOutput

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
            loadResistanceOhm=self.parameters["loadResistanceOhm"]
        )
        return tuple(
            powerCalibration.RmsToDbm(outputRms)
            for outputRms in self.lastOutputRmsPerChain
        )


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

    def Process(self, inputSignal: np.ndarray) -> np.ndarray:
        """Apply the base PA and then add its conjugate image component.

        Processing details:
            Algorithm: Execute the configured signal-processing path, preserve sample alignment, and return the complete downstream result.

        Args:
            inputSignal: One-dimensional complex baseband samples supplied to the operation.

        Returns:
            result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
        """

        paOutput = self.paModel.Process(inputSignal)
        return (
            self.directCoefficient * paOutput
            + self.imageCoefficient * np.conj(paOutput)
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
