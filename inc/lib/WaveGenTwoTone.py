"""Object-oriented complex-baseband two-tone waveform generation."""

from collections import ChainMap
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Tuple, cast

import numpy as np

# Cross-package imports support both repository-root and ``inc``-root imports.
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


@dataclass(frozen=True)
class TwoToneWaveform:
    """Store generated samples and the physical two-tone configuration."""

    samples: np.ndarray
    sampleRateHz: float
    toneFrequenciesHz: Tuple[float, float]
    toneAmplitudes: Tuple[float, float]
    tonePhasesDegrees: Tuple[float, float]
    numSamples: int
    rmsLevel: float
    width: int
    ilcBandwidthHz: float

    def IntermodulationFrequencies(
        self, nonlinearOrder: int
    ) -> Tuple[float, float]:
        """Return the lower and upper odd-order intermodulation frequencies.

        Processing details:
            Algorithm: For odd order ``p``, evaluate
            ``((p+1)/2)f1-((p-1)/2)f2`` and its upper-side counterpart using
            the sorted fundamental frequencies retained in this metadata.

        Args:
            nonlinearOrder: Odd intermodulation order from three upward.

        Returns:
            result: Lower-side and upper-side product frequencies in hertz.
        """

        if (
            not isinstance(nonlinearOrder, int)
            or isinstance(nonlinearOrder, bool)
            or nonlinearOrder < 3
            or nonlinearOrder % 2 == 0
        ):
            raise ValueError(
                "nonlinearOrder must be an odd integer no smaller than three"
            )
        lowerToneHz, upperToneHz = self.toneFrequenciesHz
        outerCoefficient = (nonlinearOrder + 1) // 2
        innerCoefficient = (nonlinearOrder - 1) // 2
        return (
            outerCoefficient * lowerToneHz
            - innerCoefficient * upperToneHz,
            outerCoefficient * upperToneHz
            - innerCoefficient * lowerToneHz,
        )


class WaveGenTwoTone:
    """Generate configurable floating-point or fixed-point two-tone samples."""

    def __init__(
        self,
        parameters: Optional[Mapping[str, object]] = None,
        width: Optional[int] = None,
        **parameterOverrides: object,
    ) -> None:
        """Initialize live ChainMap-backed two-tone generation parameters.

        Processing details:
            Algorithm: Keep immutable defaults inside the class, place direct
            overrides and a live caller mapping ahead of them, ignore unknown
            keys with warnings, and validate all resolved physical settings.

        Args:
            parameters: Optional caller-owned mapping layered ahead of defaults.
            width: Optional public I/Q component width; zero selects floating
                samples and a positive value selects signed integer codes.
            parameterOverrides: Highest-priority recognized parameter values.

        Returns:
            result: None. A validated reusable generator is initialized.
        """

        self.defaultParameters: Mapping[str, object] = MappingProxyType(
            {
                "sampleRateHz": 100.0e6,
                "toneFrequenciesHz": (-2.0e6, 2.0e6),
                "toneAmplitudes": (1.0, 1.0),
                "tonePhasesDegrees": (0.0, 0.0),
                "numSamples": 32768,
                "rmsLevel": 0.5,
                "width": 16,
                "ilcBandwidthHz": None,
            }
        )
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping or None")
        directOverrides = dict(parameterOverrides)
        if width is not None:
            directOverrides["width"] = width
        externalParameters: Mapping[str, object] = (
            {}
            if parameters is None
            else RecognizedParameterView(
                parameters,
                self.defaultParameters,
                "WaveGenTwoTone",
            )
        )
        recognizedOverrides = FilterRecognizedParameters(
            directOverrides,
            self.defaultParameters,
            "WaveGenTwoTone",
        )
        self.parameters: ChainMap[str, object] = ChainMap(
            recognizedOverrides,
            externalParameters,
            self.defaultParameters,
        )
        self.ValidateParameters()

    @property
    def Width(self) -> int:
        """Return the configured public I/Q component width.

        Processing details:
            Algorithm: Read the validated resolved value from the ChainMap
            without copying or mutating the caller-owned parameter mapping.

        Returns:
            result: Zero for floating mode or a positive signed-code width.
        """

        return cast(int, self.parameters["width"])

    width = Width

    def GetParameters(self) -> Dict[str, object]:
        """Return a flattened snapshot of all resolved generation settings.

        Processing details:
            Algorithm: Resolve the ChainMap layers into an independent normal
            dictionary so callers cannot mutate generator state through it.

        Returns:
            result: Dictionary containing every supported parameter.
        """

        return dict(self.parameters)

    def UpdateParameters(self, **parameterOverrides: object) -> None:
        """Apply recognized high-priority parameter updates transactionally.

        Processing details:
            Algorithm: Filter unknown keys with a warning, update the local
            ChainMap layer, validate the complete configuration, and restore
            the previous layer if any recognized value is invalid.

        Args:
            parameterOverrides: Recognized values to replace for later calls.

        Returns:
            result: None. Valid changes remain active in this generator.
        """

        recognizedOverrides = FilterRecognizedParameters(
            parameterOverrides,
            self.defaultParameters,
            "WaveGenTwoTone.UpdateParameters",
        )
        previousOverrides = dict(self.parameters.maps[0])
        self.parameters.maps[0].update(recognizedOverrides)
        try:
            self.ValidateParameters()
        except (TypeError, ValueError):
            self.parameters.maps[0].clear()
            self.parameters.maps[0].update(previousOverrides)
            raise

    def ResolvePair(
        self, parameterName: str, requirePositive: bool
    ) -> Tuple[float, float]:
        """Resolve one finite numeric two-element configuration sequence.

        Processing details:
            Algorithm: Reject text and non-sequence values, convert exactly
            two real entries to floats, then enforce finiteness and an optional
            strict-positive domain before returning an immutable pair.

        Args:
            parameterName: Name of the supported pair-valued parameter.
            requirePositive: Whether both resolved values must exceed zero.

        Returns:
            result: Validated two-element floating-point tuple.
        """

        parameterValue = self.parameters[parameterName]
        if (
            isinstance(parameterValue, (str, bytes))
            or not isinstance(
                parameterValue,
                (list, tuple, np.ndarray),
            )
            or len(parameterValue) != 2
        ):
            raise ValueError(f"{parameterName} must contain exactly two values")
        resolvedValues = []
        for pairValue in parameterValue:
            if (
                not isinstance(pairValue, (int, float))
                or isinstance(pairValue, bool)
                or not np.isfinite(pairValue)
                or (requirePositive and float(pairValue) <= 0.0)
            ):
                domainText = "finite and positive" if requirePositive else "finite"
                raise ValueError(
                    f"every {parameterName} value must be {domainText}"
                )
            resolvedValues.append(float(pairValue))
        return (resolvedValues[0], resolvedValues[1])

    def ResolveIntermodulationFrequencies(
        self, nonlinearOrder: int
    ) -> Tuple[float, float]:
        """Calculate odd-order products from the configured fundamentals.

        Processing details:
            Algorithm: Sort the two configured tones, construct temporary
            metadata-free odd-order coefficients, and calculate the lower and
            upper products without generating a waveform.

        Args:
            nonlinearOrder: Odd intermodulation order from three upward.

        Returns:
            result: Lower-side and upper-side frequencies in hertz.
        """

        if (
            not isinstance(nonlinearOrder, int)
            or isinstance(nonlinearOrder, bool)
            or nonlinearOrder < 3
            or nonlinearOrder % 2 == 0
        ):
            raise ValueError(
                "nonlinearOrder must be an odd integer no smaller than three"
            )
        lowerToneHz, upperToneHz = sorted(
            self.ResolvePair("toneFrequenciesHz", False)
        )
        outerCoefficient = (nonlinearOrder + 1) // 2
        innerCoefficient = (nonlinearOrder - 1) // 2
        return (
            outerCoefficient * lowerToneHz
            - innerCoefficient * upperToneHz,
            outerCoefficient * upperToneHz
            - innerCoefficient * lowerToneHz,
        )

    def ResolveIlcBandwidthHz(self) -> float:
        """Return the requested ILC update bandwidth in hertz.

        Processing details:
            Algorithm: Use an explicit positive caller value when supplied;
            otherwise span both seventh-order products with ten percent guard
            so frequency-domain ILC can synthesize cancellation components.

        Returns:
            result: Positive two-sided channel bandwidth in hertz.
        """

        configuredBandwidthHz = self.parameters["ilcBandwidthHz"]
        if configuredBandwidthHz is not None:
            if (
                not isinstance(configuredBandwidthHz, (int, float))
                or isinstance(configuredBandwidthHz, bool)
                or not np.isfinite(configuredBandwidthHz)
                or float(configuredBandwidthHz) <= 0.0
            ):
                raise ValueError(
                    "ilcBandwidthHz must be finite and positive or None"
                )
            return float(configuredBandwidthHz)
        im7FrequenciesHz = self.ResolveIntermodulationFrequencies(7)
        maximumFrequencyHz = max(
            abs(frequencyHz) for frequencyHz in im7FrequenciesHz
        )
        return 2.2 * maximumFrequencyHz

    def ValidateParameters(self) -> None:
        """Validate tones, sampling, length, scaling, bandwidth, and width.

        Processing details:
            Algorithm: Check finite physical scalars and pairs, require distinct
            ordered fundamentals, verify both seventh-order products lie below
            complex Nyquist, require the ILC bandwidth to fit the sample rate,
            and instantiate ``FixedPoint`` to validate the external width.

        Returns:
            result: None. Invalid settings raise a descriptive exception.
        """

        sampleRateHz = self.parameters["sampleRateHz"]
        if (
            not isinstance(sampleRateHz, (int, float))
            or isinstance(sampleRateHz, bool)
            or not np.isfinite(sampleRateHz)
            or float(sampleRateHz) <= 0.0
        ):
            raise ValueError("sampleRateHz must be finite and positive")
        toneFrequenciesHz = self.ResolvePair("toneFrequenciesHz", False)
        if toneFrequenciesHz[0] == toneFrequenciesHz[1]:
            raise ValueError("toneFrequenciesHz must contain distinct values")
        self.ResolvePair("toneAmplitudes", True)
        self.ResolvePair("tonePhasesDegrees", False)
        numSamples = self.parameters["numSamples"]
        if (
            not isinstance(numSamples, int)
            or isinstance(numSamples, bool)
            or numSamples < 64
        ):
            raise ValueError("numSamples must be an integer no smaller than 64")
        rmsLevel = self.parameters["rmsLevel"]
        if (
            not isinstance(rmsLevel, (int, float))
            or isinstance(rmsLevel, bool)
            or not np.isfinite(rmsLevel)
            or not 0.0 < float(rmsLevel) <= 1.0
        ):
            raise ValueError("rmsLevel must be finite and in the interval (0, 1]")
        nyquistHz = 0.5 * float(sampleRateHz)
        for nonlinearOrder in (3, 5, 7):
            productFrequenciesHz = self.ResolveIntermodulationFrequencies(
                nonlinearOrder
            )
            if any(
                abs(frequencyHz) >= nyquistHz
                for frequencyHz in productFrequenciesHz
            ):
                raise ValueError(
                    f"IM{nonlinearOrder} products must lie inside complex Nyquist"
                )
        ilcBandwidthHz = self.ResolveIlcBandwidthHz()
        if ilcBandwidthHz >= float(sampleRateHz):
            raise ValueError("ilcBandwidthHz must be smaller than sampleRateHz")
        FixedPoint(cast(int, self.parameters["width"]))

    def Generate(self) -> TwoToneWaveform:
        """Generate one normalized complex-baseband two-tone waveform.

        Processing details:
            Algorithm: Sum two independently configured complex exponentials,
            normalize their finite-record RMS to ``rmsLevel``, encode only at
            the public fixed-point boundary, and return samples with immutable
            metadata needed by IM3, IM5, IM7, and ILC analysis.

        Returns:
            result: Metadata-rich floating or integer-code two-tone waveform.
        """

        self.ValidateParameters()
        sampleRateHz = float(cast(float, self.parameters["sampleRateHz"]))
        numSamples = cast(int, self.parameters["numSamples"])
        toneFrequenciesHz = self.ResolvePair("toneFrequenciesHz", False)
        toneAmplitudes = self.ResolvePair("toneAmplitudes", True)
        tonePhasesDegrees = self.ResolvePair("tonePhasesDegrees", False)
        sampleTimes = np.arange(numSamples, dtype=float) / sampleRateHz
        floatingSamples = np.zeros(numSamples, dtype=np.complex128)
        for (
            toneFrequencyHz,
            toneAmplitude,
            tonePhaseDegrees,
        ) in zip(
            toneFrequenciesHz,
            toneAmplitudes,
            tonePhasesDegrees,
        ):
            floatingSamples += toneAmplitude * np.exp(
                1j
                * (
                    2.0 * np.pi * toneFrequencyHz * sampleTimes
                    + np.deg2rad(tonePhaseDegrees)
                )
            )
        waveformRms = float(np.sqrt(np.mean(np.abs(floatingSamples) ** 2)))
        if waveformRms <= np.finfo(float).tiny:
            raise ValueError("configured tones cancel over the generated record")
        rmsLevel = float(cast(float, self.parameters["rmsLevel"]))
        floatingSamples *= rmsLevel / waveformRms
        publicSamples = FixedPoint(self.width).EncodeComplex(floatingSamples)
        sortedFrequenciesHz = tuple(sorted(toneFrequenciesHz))
        sortedIndices = tuple(
            toneFrequenciesHz.index(frequencyHz)
            for frequencyHz in sortedFrequenciesHz
        )
        return TwoToneWaveform(
            samples=publicSamples,
            sampleRateHz=sampleRateHz,
            toneFrequenciesHz=(
                float(sortedFrequenciesHz[0]),
                float(sortedFrequenciesHz[1]),
            ),
            toneAmplitudes=(
                toneAmplitudes[sortedIndices[0]],
                toneAmplitudes[sortedIndices[1]],
            ),
            tonePhasesDegrees=(
                tonePhasesDegrees[sortedIndices[0]],
                tonePhasesDegrees[sortedIndices[1]],
            ),
            numSamples=numSamples,
            rmsLevel=rmsLevel,
            width=self.width,
            ilcBandwidthHz=self.ResolveIlcBandwidthHz(),
        )
