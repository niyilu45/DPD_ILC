"""Power-amplifier behavioral models used by the DPD-ILC simulation.

Callers construct ``PaModel`` with ``modelName="wiener"`` or
``modelName="gmp"`` and then call ``Process``. Two nonlinear model families
are provided internally:

* ``WienerPA`` applies a linear memory filter followed by a smooth Rapp
  AM-AM characteristic and a saturating AM-PM characteristic.
* ``GMPPA`` implements the generalized memory polynomial main, lagging,
  and leading cross terms described in the project theory document.

Every model accepts and returns a one-dimensional complex baseband array.
"""

from collections import ChainMap
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Sequence, Tuple, cast

import numpy as np


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
        """Reject nonphysical settings before processing a waveform."""

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

        complexInput = _AsComplexVector(inputSignal)
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
        """Return the DC small-signal gain of the linear Wiener cascade."""

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
        """Validate order and memory dimensions used by the GMP expansion."""

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
        config.Validate()
        self.config = config
        defaultMain, defaultLagging, defaultLeading = _DefaultGmpCoefficients(
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
        """Evaluate the main, lagging, and leading GMP basis expansions."""

        complexInput = _AsComplexVector(inputSignal)
        outputSignal = np.zeros_like(complexInput)

        # Main branch: x[n-m] * |x[n-m]|^(p-1).
        for (nonlinearOrder, memoryIndex), coefficient in self.mainCoefficients.items():
            delayedSignal = _DelaySignal(complexInput, memoryIndex)
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
            carrierSignal = _DelaySignal(complexInput, memoryIndex)
            envelopeSignal = _DelaySignal(
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
            carrierSignal = _DelaySignal(
                complexInput, memoryIndex + crossIndex
            )
            envelopeSignal = _DelaySignal(complexInput, memoryIndex)
            outputSignal += (
                coefficient
                * carrierSignal
                * np.abs(envelopeSignal) ** (nonlinearOrder - 1)
            )
        return outputSignal

    def SmallSignalGain(self) -> complex:
        """Return the DC gain contributed by all first-order main terms."""

        return sum(
            coefficient
            for (nonlinearOrder, _), coefficient in self.mainCoefficients.items()
            if nonlinearOrder == 1
        )


paModelDefaultParameters: Mapping[str, object] = MappingProxyType(
    {
        "modelName": "wiener",
        "wienerConfig": None,
        "gmpConfig": None,
    }
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
            paModelDefaultParameters,
        )
        self.model = None
        self._activeConfiguration: Optional[
            Tuple[str, Optional[WienerConfig], Optional[GMPConfig]]
        ] = None
        self._SynchronizeModel()

    @property
    def modelName(self) -> str:
        """Return the normalized model name resolved by the ChainMap."""

        normalizedName, _, _ = self._ResolveConfiguration()
        return normalizedName

    def GetParameters(self) -> Dict[str, object]:
        """Return a flattened snapshot of all resolved PA parameters."""

        return dict(self.parameters)

    def UpdateParameters(self, **parameterOverrides: object) -> None:
        """Apply validated high-priority PA configuration overrides."""

        previousOverrides = dict(self.parameters.maps[0])
        self.parameters.maps[0].update(parameterOverrides)
        try:
            self._SynchronizeModel()
        except (TypeError, ValueError):
            self.parameters.maps[0].clear()
            self.parameters.maps[0].update(previousOverrides)
            self._SynchronizeModel()
            raise

    def _ResolveConfiguration(
        self,
    ) -> Tuple[str, Optional[WienerConfig], Optional[GMPConfig]]:
        """Validate and return the currently resolved PA configuration."""

        unknownParameters = set(self.parameters).difference(
            paModelDefaultParameters
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

    def _SynchronizeModel(self) -> None:
        """Rebuild the PA when a live external parameter mapping changes."""

        selectedConfiguration = self._ResolveConfiguration()
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
        """Pass a complex waveform through the configured PA model."""

        self._SynchronizeModel()
        return self.model.Process(inputSignal)

    def SmallSignalGain(self) -> complex:
        """Return the configured model's DC small-signal complex gain."""

        self._SynchronizeModel()
        return self.model.SmallSignalGain()


class IQImbalancePA:
    """Wrap any PA with a widely-linear output IQ-imbalance model."""

    def __init__(
        self,
        paModel,
        directCoefficient: complex = 1.0 + 0.0j,
        imageCoefficient: complex = 0.045 * np.exp(1j * 0.35),
    ) -> None:
        self.paModel = paModel
        self.directCoefficient = complex(directCoefficient)
        self.imageCoefficient = complex(imageCoefficient)

    def Process(self, inputSignal: np.ndarray) -> np.ndarray:
        """Apply the base PA and then add its conjugate image component."""

        paOutput = self.paModel.Process(inputSignal)
        return (
            self.directCoefficient * paOutput
            + self.imageCoefficient * np.conj(paOutput)
        )

    def SmallSignalGain(self) -> complex:
        """Return the direct-path small-signal gain of the wrapped PA."""

        return self.directCoefficient * self.paModel.SmallSignalGain()


def _AsComplexVector(inputSignal: np.ndarray) -> np.ndarray:
    """Convert input to a finite one-dimensional complex array."""

    complexInput = np.asarray(inputSignal, dtype=np.complex128)
    if complexInput.ndim != 1:
        raise ValueError("inputSignal must be one-dimensional")
    if not np.all(np.isfinite(complexInput)):
        raise ValueError("inputSignal contains NaN or infinite values")
    return complexInput


def _DelaySignal(inputSignal: np.ndarray, sampleDelay: int) -> np.ndarray:
    """Apply a causal integer delay without changing the array length."""

    if sampleDelay < 0:
        raise ValueError("sampleDelay cannot be negative")
    if sampleDelay == 0:
        return inputSignal
    delayedSignal = np.zeros_like(inputSignal)
    if sampleDelay < inputSignal.size:
        delayedSignal[sampleDelay:] = inputSignal[:-sampleDelay]
    return delayedSignal


def _DefaultGmpCoefficients(
    nonlinearOrders: Sequence[int],
    memoryDepth: int,
    crossMemoryDepth: int,
) -> Tuple[
    Dict[Tuple[int, int], complex],
    Dict[Tuple[int, int, int], complex],
    Dict[Tuple[int, int, int], complex],
]:
    """Create stable default coefficients with compression and memory effects."""

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
    """Add complex white Gaussian feedback noise at the requested SNR."""

    complexInput = _AsComplexVector(inputSignal)
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
