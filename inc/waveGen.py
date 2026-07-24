"""Generate IEEE 802.11ac VHT, 802.11ax HE, or 802.11be EHT waveforms.

Callers first construct ``GenWifi`` with the requested frame format, bandwidth,
sample rate, MCS, guard interval, and packet length, and then call ``Generate``.
The class creates a single-user full-band complex baseband packet with the
appropriate VHT, HE-SU, or EHT field order, OFDM numerology, and MCS table.
Standard names such as ``11ac`` are accepted as aliases of PHY names such as
``VHT``.

DPD characterization only requires statistically representative coded
symbols. Therefore, this module creates randomized post-FEC stimulus bits
instead of implementing an LDPC encoder and decoder. The generated waveform
is a Wi-Fi PHY stimulus, not a bit-exact protocol conformance implementation.
"""

from collections import ChainMap
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Tuple, cast

import numpy as np


@dataclass(frozen=True)
class MCSInfo:
    """Describe the modulation and nominal coding parameters of one Wi-Fi MCS."""

    index: int
    modulation: str
    qamOrder: int
    codeRate: float
    bitsPerSubcarrier: int


def NormalizeFrameFormat(frameFormat: str) -> str:
    """Convert a standard-generation or PHY-format name to VHT, HE, or EHT.

    Processing details:
        Algorithm: Strip surrounding whitespace, compare case-insensitively
        against the documented alias table, and return one canonical PHY name
        for all subsequent validation and waveform construction.

    Args:
        frameFormat: Input such as VHT, HE, EHT, 11ac, 11ax, or 11be.

    Returns:
        result: Canonical string equal to VHT, HE, or EHT.
    """

    frameFormatAliases: Mapping[str, str] = MappingProxyType(
        {
            "VHT": "VHT",
            "11AC": "VHT",
            "802.11AC": "VHT",
            "HE": "HE",
            "11AX": "HE",
            "802.11AX": "HE",
            "EHT": "EHT",
            "11BE": "EHT",
            "802.11BE": "EHT",
        }
    )
    if not isinstance(frameFormat, str):
        raise TypeError("frameFormat must be a string")
    normalizedInput = frameFormat.strip().upper()
    if normalizedInput not in frameFormatAliases:
        raise ValueError(
            "frameFormat must be VHT/11ac, HE/11ax, or EHT/11be"
        )
    return frameFormatAliases[normalizedInput]


class GenWifi:
    """Configure and generate a SISO or MIMO VHT, HE, or EHT waveform.

    A ``ChainMap`` resolves explicit keyword overrides first, a caller-owned
    parameter mapping second, and immutable constructor-internal defaults
    last. Changes to the caller-owned mapping remain visible until a
    higher-priority keyword override shadows the same key.

    Example:
        ``wifiGenerator = GenWifi(frameFormat="11be", bandwidthMhz=80, mcs=11)``
        ``waveform = wifiGenerator.Generate()``
    """

    def __init__(
        self,
        parameters: Optional[Mapping[str, object]] = None,
        **parameterOverrides: object,
    ) -> None:
        """Initialize a VHT/HE/EHT waveform generator with ChainMap defaults.

        Processing details:
            Algorithm: Define immutable defaults inside the constructor, then
            layer direct overrides and the caller-owned mapping ahead of those
            defaults without requiring callers to import or repeat them.

        Args:
            parameters: Optional external mapping layered ahead of the built-in defaults.
            parameterOverrides: Highest-priority keyword values applied to the local ChainMap layer.

        Returns:
            result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
        """
        self.defaultParameters: Mapping[str, object] = MappingProxyType(
            {
                "frameFormat": "EHT",
                "bandwidthMhz": 80,
                "mcs": 9,
                "numDataSymbols": 20,
                "guardIntervalUs": 0.8,
                "sampleRateHz": None,
                "oversampling": 4,
                "seed": 7,
                "numTransmitAntennas": 1,
                "numSpatialStreams": 1,
                "spatialMapping": "direct",
                "spatialMappingMatrix": None,
                "cyclicShiftEnabled": True,
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
        self.Validate()

    def ResolveMcsTable(
        self, frameFormat: str
    ) -> Mapping[int, MCSInfo]:
        """Build the immutable MCS table for one normalized PHY format.

        Processing details:
            Algorithm: Define the complete EHT modulation/rate mapping inside
            this method, normalize the requested format, and return only the
            index range standardized by VHT, HE, or EHT. No module-level
            table or mutable global configuration is retained.

        Args:
            frameFormat: PHY name or equivalent 802.11 generation alias.

        Returns:
            result: Immutable mapping from MCS index to modulation metadata.
        """

        normalizedFormat = NormalizeFrameFormat(frameFormat)
        ehtMcsTable: Mapping[int, MCSInfo] = MappingProxyType(
            {
                0: MCSInfo(0, "BPSK", 2, 1.0 / 2.0, 1),
                1: MCSInfo(1, "QPSK", 4, 1.0 / 2.0, 2),
                2: MCSInfo(2, "QPSK", 4, 3.0 / 4.0, 2),
                3: MCSInfo(3, "16-QAM", 16, 1.0 / 2.0, 4),
                4: MCSInfo(4, "16-QAM", 16, 3.0 / 4.0, 4),
                5: MCSInfo(5, "64-QAM", 64, 2.0 / 3.0, 6),
                6: MCSInfo(6, "64-QAM", 64, 3.0 / 4.0, 6),
                7: MCSInfo(7, "64-QAM", 64, 5.0 / 6.0, 6),
                8: MCSInfo(8, "256-QAM", 256, 3.0 / 4.0, 8),
                9: MCSInfo(9, "256-QAM", 256, 5.0 / 6.0, 8),
                10: MCSInfo(10, "1024-QAM", 1024, 3.0 / 4.0, 10),
                11: MCSInfo(11, "1024-QAM", 1024, 5.0 / 6.0, 10),
                12: MCSInfo(12, "4096-QAM", 4096, 3.0 / 4.0, 12),
                13: MCSInfo(13, "4096-QAM", 4096, 5.0 / 6.0, 12),
            }
        )
        maximumMcsByFormat: Mapping[str, int] = MappingProxyType(
            {"VHT": 9, "HE": 11, "EHT": 13}
        )
        maximumMcs = maximumMcsByFormat[normalizedFormat]
        return MappingProxyType(
            {
                mcsIndex: mcsInfo
                for mcsIndex, mcsInfo in ehtMcsTable.items()
                if mcsIndex <= maximumMcs
            }
        )

    @property
    def FrameFormat(self) -> str:
        """Return the currently resolved frame format.

        Processing details:
            Algorithm: Carry out the described operation using validated inputs, explicit array-shape handling, and deterministic project conventions.

        Returns:
            result: str. The computed value described by the summary, with documented units, shape, and normalization.
        """

        return NormalizeFrameFormat(cast(str, self.parameters["frameFormat"]))

    frameFormat = FrameFormat

    @property
    def BandwidthMhz(self) -> int:
        """Return the currently resolved channel bandwidth.

        Processing details:
            Algorithm: Carry out the described operation using validated inputs, explicit array-shape handling, and deterministic project conventions.

        Returns:
            result: int. The computed value described by the summary, with documented units, shape, and normalization.
        """

        return cast(int, self.parameters["bandwidthMhz"])

    bandwidthMhz = BandwidthMhz

    @property
    def Mcs(self) -> int:
        """Return the currently resolved MCS index.

        Processing details:
            Algorithm: Carry out the described operation using validated inputs, explicit array-shape handling, and deterministic project conventions.

        Returns:
            result: int. The computed value described by the summary, with documented units, shape, and normalization.
        """

        return cast(int, self.parameters["mcs"])

    mcs = Mcs

    @property
    def NumDataSymbols(self) -> int:
        """Return the currently resolved data-symbol count.

        Processing details:
            Algorithm: Carry out the described operation using validated inputs, explicit array-shape handling, and deterministic project conventions.

        Returns:
            result: int. The computed value described by the summary, with documented units, shape, and normalization.
        """

        return cast(int, self.parameters["numDataSymbols"])

    numDataSymbols = NumDataSymbols

    @property
    def GuardIntervalUs(self) -> float:
        """Return the currently resolved guard interval in microseconds.

        Processing details:
            Algorithm: Carry out the described operation using validated inputs, explicit array-shape handling, and deterministic project conventions.

        Returns:
            result: float. The computed value described by the summary, with documented units, shape, and normalization.
        """

        return cast(float, self.parameters["guardIntervalUs"])

    guardIntervalUs = GuardIntervalUs

    @property
    def SampleRateHz(self) -> float:
        """Return the user-selected or backward-compatible sample rate.

        Processing details:
            Algorithm: Use ``sampleRateHz`` whenever the caller supplies it.
            Otherwise multiply channel bandwidth by the legacy integer
            ``oversampling`` value so existing call sites retain their
            previous waveform lengths.

        Returns:
            result: Positive complex-baseband sampling rate in hertz.
        """

        configuredSampleRate = self.parameters["sampleRateHz"]
        if configuredSampleRate is not None:
            return float(cast(float, configuredSampleRate))
        return (
            float(self.bandwidthMhz)
            * 1.0e6
            * float(cast(int, self.parameters["oversampling"]))
        )

    sampleRateHz = SampleRateHz

    @property
    def Oversampling(self) -> float:
        """Return the derived sample-rate-to-bandwidth ratio.

        Processing details:
            Algorithm: Divide the authoritative resolved sample rate by the
            configured channel bandwidth. This value is metadata and may be
            noninteger when ``sampleRateHz`` directly selects a compatible
            OFDM clock.

        Returns:
            result: Dimensionless effective oversampling ratio.
        """

        bandwidthHz = float(self.bandwidthMhz) * 1.0e6
        return self.sampleRateHz / bandwidthHz

    oversampling = Oversampling

    @property
    def Seed(self) -> int:
        """Return the currently resolved random seed.

        Processing details:
            Algorithm: Carry out the described operation using validated inputs, explicit array-shape handling, and deterministic project conventions.

        Returns:
            result: int. The computed value described by the summary, with documented units, shape, and normalization.
        """

        return cast(int, self.parameters["seed"])

    seed = Seed

    @property
    def NumTransmitAntennas(self) -> int:
        """Return the configured number of physical transmit chains.

        Processing details:
            Algorithm: Resolve the integer from the active parameter layers;
            validation guarantees the format-specific antenna bound.

        Returns:
            result: Number of transmit antennas in the generated waveform.
        """

        return cast(int, self.parameters["numTransmitAntennas"])

    numTransmitAntennas = NumTransmitAntennas

    @property
    def NumSpatialStreams(self) -> int:
        """Return the configured number of independently modulated streams.

        Processing details:
            Algorithm: Resolve the integer from the active parameter layers;
            validation guarantees it does not exceed transmit dimensions.

        Returns:
            result: Number of VHT, HE, or EHT spatial streams.
        """

        return cast(int, self.parameters["numSpatialStreams"])

    numSpatialStreams = NumSpatialStreams

    @property
    def SpatialMapping(self) -> str:
        """Return the normalized spatial-mapping mode name.

        Processing details:
            Algorithm: Strip whitespace and lowercase the configured mode so
            downstream matrix construction has one canonical representation.

        Returns:
            result: ``direct``, ``dft``, or ``custom``.
        """

        return cast(str, self.parameters["spatialMapping"]).strip().lower()

    spatialMapping = SpatialMapping

    @property
    def SpatialMappingMatrix(self) -> Optional[np.ndarray]:
        """Return a copy of the optional caller-supplied mapping matrix.

        Processing details:
            Algorithm: Preserve ``None`` for automatic mapping; otherwise
            convert the configured matrix to complex128 and return a copy.

        Returns:
            result: Optional matrix shaped transmit antennas by streams.
        """

        rawMatrix = self.parameters["spatialMappingMatrix"]
        if rawMatrix is None:
            return None
        return np.asarray(rawMatrix, dtype=np.complex128).copy()

    spatialMappingMatrix = SpatialMappingMatrix

    @property
    def CyclicShiftEnabled(self) -> bool:
        """Return whether per-chain cyclic shift diversity is enabled.

        Processing details:
            Algorithm: Read the validated Boolean without changing external
            live-mapping behavior.

        Returns:
            result: True when frequency-dependent CSD phases are applied.
        """

        return cast(bool, self.parameters["cyclicShiftEnabled"])

    cyclicShiftEnabled = CyclicShiftEnabled

    def GetParameters(self) -> Dict[str, object]:
        """Return a flattened snapshot of all resolved parameters.

        Processing details:
            Algorithm: Resolve values according to state and ChainMap precedence, keeping caller-owned configuration behavior explicit.

        Returns:
            result: Dict[str, object]. The computed value described by the summary, with documented units, shape, and normalization.
        """

        resolvedParameters = dict(self.parameters)
        resolvedParameters["sampleRateHz"] = self.sampleRateHz
        resolvedParameters["oversampling"] = self.oversampling
        return resolvedParameters

    def UpdateParameters(self, **parameterOverrides: object) -> None:
        """Apply validated high-priority overrides to this generator.

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
            self.Validate()
        except (TypeError, ValueError):
            self.parameters.maps[0].clear()
            self.parameters.maps[0].update(previousOverrides)
            raise

    def Validate(self) -> None:
        """Validate every option before allocating potentially large arrays.

        Processing details:
            Algorithm: Evaluate every documented constraint in deterministic order and stop at the first invalid condition without changing valid state.

        Returns:
            result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
        """

        unknownParameters = set(self.parameters).difference(
            self.defaultParameters
        )
        if unknownParameters:
            unknownNames = ", ".join(
                sorted(str(parameterName) for parameterName in unknownParameters)
            )
            raise TypeError(f"unknown GenWifi parameters: {unknownNames}")
        normalizedFormat = NormalizeFrameFormat(
            cast(str, self.parameters["frameFormat"])
        )
        supportedBandwidths = (20, 40, 80, 160)
        guardIntervalsByFormat: Mapping[
            str, Tuple[float, ...]
        ] = MappingProxyType(
            {
                "VHT": (0.4, 0.8),
                "HE": (0.8, 1.6, 3.2),
                "EHT": (0.8, 1.6, 3.2),
            }
        )
        if not isinstance(self.bandwidthMhz, int) or isinstance(
            self.bandwidthMhz, bool
        ):
            raise TypeError("bandwidthMhz must be an integer")
        if self.bandwidthMhz not in supportedBandwidths:
            raise ValueError("bandwidthMhz must be one of 20, 40, 80, 160")
        selectedMcsTable = self.ResolveMcsTable(normalizedFormat)
        if not isinstance(self.mcs, int) or isinstance(self.mcs, bool):
            raise TypeError("mcs must be an integer")
        if self.mcs not in selectedMcsTable:
            maximumMcs = max(selectedMcsTable)
            raise ValueError(
                f"{normalizedFormat} MCS must be an integer from 0 through {maximumMcs}"
            )
        if not isinstance(self.numDataSymbols, int) or isinstance(
            self.numDataSymbols, bool
        ):
            raise TypeError("numDataSymbols must be an integer")
        if self.numDataSymbols < 1:
            raise ValueError("numDataSymbols must be positive")
        if not isinstance(self.guardIntervalUs, (int, float)) or isinstance(
            self.guardIntervalUs, bool
        ):
            raise TypeError("guardIntervalUs must be numeric")
        supportedGuardIntervals = guardIntervalsByFormat[normalizedFormat]
        if self.guardIntervalUs not in supportedGuardIntervals:
            intervalText = ", ".join(
                f"{intervalValue:g}" for intervalValue in supportedGuardIntervals
            )
            raise ValueError(
                f"{normalizedFormat} guardIntervalUs must be one of {intervalText}"
            )
        rawOversampling = self.parameters["oversampling"]
        if (
            not isinstance(rawOversampling, int)
            or isinstance(rawOversampling, bool)
            or rawOversampling < 1
        ):
            raise ValueError("oversampling must be a positive integer")
        rawSampleRateHz = self.parameters["sampleRateHz"]
        if rawSampleRateHz is not None and (
            not isinstance(rawSampleRateHz, (int, float))
            or isinstance(rawSampleRateHz, bool)
            or not np.isfinite(rawSampleRateHz)
            or rawSampleRateHz <= 0.0
        ):
            raise ValueError("sampleRateHz must be finite and positive or None")
        bandwidthHz = float(self.bandwidthMhz) * 1.0e6
        sampleRateHz = self.sampleRateHz
        if sampleRateHz < bandwidthHz:
            raise ValueError(
                "sampleRateHz must be at least the channel bandwidth"
            )

        # Direct sample-rate selection must preserve exact integer sample
        # counts for every OFDM timing interval used by this PHY. This allows
        # noninteger bandwidth ratios, such as 30 MHz for a 20 MHz VHT frame,
        # without silently rounding the nominal subcarrier spacing or GI.
        usefulDurationUs = 3.2 if normalizedFormat == "VHT" else 12.8
        requiredDurationsUs = [
            ("legacy useful symbol", 3.2),
            ("legacy guard interval", 0.8),
            ("data useful symbol", usefulDurationUs),
            ("configured guard interval", float(self.guardIntervalUs)),
        ]
        if normalizedFormat in ("HE", "EHT") and self.guardIntervalUs == 1.6:
            requiredDurationsUs.append(("2x LTF useful symbol", 6.4))
        for durationName, durationUs in requiredDurationsUs:
            sampleCount = sampleRateHz * durationUs * 1.0e-6
            if not np.isclose(
                sampleCount,
                round(sampleCount),
                rtol=0.0,
                atol=1.0e-8,
            ):
                raise ValueError(
                    "sampleRateHz is incompatible with the "
                    f"{durationName} duration; choose a rate that gives an "
                    "integer sample count"
                )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        maximumSpatialStreams = 8
        for parameterName, parameterValue in (
            ("numTransmitAntennas", self.numTransmitAntennas),
            ("numSpatialStreams", self.numSpatialStreams),
        ):
            if not isinstance(parameterValue, int) or isinstance(
                parameterValue, bool
            ):
                raise TypeError(f"{parameterName} must be an integer")
            if parameterValue < 1 or parameterValue > maximumSpatialStreams:
                raise ValueError(
                    f"{normalizedFormat} {parameterName} must be from 1 "
                    f"through {maximumSpatialStreams}"
                )
        if self.numSpatialStreams > self.numTransmitAntennas:
            raise ValueError(
                "numSpatialStreams cannot exceed numTransmitAntennas"
            )
        if self.spatialMapping not in ("direct", "dft", "custom"):
            raise ValueError(
                "spatialMapping must be 'direct', 'dft', or 'custom'"
            )
        rawSpatialMappingMatrix = self.parameters["spatialMappingMatrix"]
        if self.spatialMapping == "custom":
            if rawSpatialMappingMatrix is None:
                raise ValueError(
                    "custom spatialMapping requires spatialMappingMatrix"
                )
            mappingMatrix = np.asarray(
                rawSpatialMappingMatrix, dtype=np.complex128
            )
            expectedShape = (
                self.numTransmitAntennas,
                self.numSpatialStreams,
            )
            if mappingMatrix.shape != expectedShape:
                raise ValueError(
                    "spatialMappingMatrix must have shape "
                    f"{expectedShape}"
                )
            if not np.all(np.isfinite(mappingMatrix)):
                raise ValueError(
                    "spatialMappingMatrix contains NaN or infinite values"
                )
            gramMatrix = mappingMatrix.conj().T @ mappingMatrix
            if not np.allclose(
                gramMatrix,
                np.eye(self.numSpatialStreams),
                rtol=1.0e-7,
                atol=1.0e-9,
            ):
                raise ValueError(
                    "spatialMappingMatrix columns must be orthonormal"
                )
        elif rawSpatialMappingMatrix is not None:
            raise ValueError(
                "spatialMappingMatrix is only valid for custom mapping"
            )
        if not isinstance(self.cyclicShiftEnabled, bool):
            raise TypeError("cyclicShiftEnabled must be boolean")

    def GetMcsInfo(self) -> MCSInfo:
        """Return the MCS record selected by this generator instance.

        Processing details:
            Algorithm: Resolve values according to state and ChainMap precedence, keeping caller-owned configuration behavior explicit.

        Returns:
            result: MCSInfo. The computed value described by the summary, with documented units, shape, and normalization.
        """

        self.Validate()
        return self.ResolveMcsTable(self.frameFormat)[self.mcs]

    def Generate(self) -> "WifiWaveform":
        """Generate one configured VHT, HE, or EHT packet and its metadata.

        Processing details:
            Algorithm: Generate deterministic complex-baseband data from validated settings while preserving sample-rate and energy conventions.

        Returns:
            result: 'WifiWaveform'. The computed value described by the summary, with documented units, shape, and normalization.
        """

        return GenerateWifiWaveform(self)


@dataclass
class WifiWaveform:
    """Return waveform samples and metadata required for EVM demodulation."""

    samples: np.ndarray
    sampleRateHz: float
    bandwidthHz: float
    fftLength: int
    cpLength: int
    oversampling: float
    activeSubcarriers: np.ndarray
    dataSubcarriers: np.ndarray
    pilotSubcarriers: np.ndarray
    referenceDataSymbols: np.ndarray
    fieldSlices: Dict[str, slice]
    dataSymbolStarts: np.ndarray
    symbolLength: int
    mcsInfo: MCSInfo
    normalizationScale: float
    codedBitsPerSymbol: int
    informationBitsPerSymbol: int
    frameFormat: str
    dataFieldName: str
    formatName: str
    numTransmitAntennas: int
    numSpatialStreams: int
    spatialMapping: str
    spatialMappingMatrix: np.ndarray
    cyclicShiftsSeconds: np.ndarray
    ltfSymbolCount: int


def ActiveTones(
    bandwidthMhz: int,
    frameFormat: str = "EHT",
) -> np.ndarray:
    """Return centered active-subcarrier indices for one Wi-Fi PHY format.

    Processing details:
        Algorithm: Carry out the described operation using validated inputs, explicit array-shape handling, and deterministic project conventions.

    Args:
        bandwidthMhz: Configured Wi-Fi channel bandwidth in megahertz.
        frameFormat: PHY name or equivalent standard-generation alias.

    Returns:
        result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
    """

    normalizedFormat = NormalizeFrameFormat(frameFormat)
    vhtActiveToneCount: Mapping[int, int] = MappingProxyType(
        {20: 56, 40: 114, 80: 242, 160: 484}
    )
    heEhtActiveToneCount: Mapping[int, int] = MappingProxyType(
        {20: 242, 40: 484, 80: 996, 160: 1992}
    )
    if bandwidthMhz not in heEhtActiveToneCount:
        raise ValueError("bandwidthMhz must be one of 20, 40, 80, 160")

    if normalizedFormat == "VHT":
        if bandwidthMhz == 20:
            toneIndices = np.r_[np.arange(-28, 0), np.arange(1, 29)]
        elif bandwidthMhz == 40:
            toneIndices = np.r_[np.arange(-58, -1), np.arange(2, 59)]
        elif bandwidthMhz == 80:
            toneIndices = np.r_[np.arange(-122, -1), np.arange(2, 123)]
        else:
            # A 160 MHz VHT channel is represented as two 80 MHz tone plans
            # separated by 256 subcarrier bins on the 312.5 kHz grid.
            localTones = ActiveTones(80, "VHT")
            toneIndices = np.r_[localTones - 128, localTones + 128]
        expectedToneCount = vhtActiveToneCount[bandwidthMhz]
    elif bandwidthMhz == 20:
        toneIndices = np.r_[np.arange(-122, -1), np.arange(2, 123)]
        expectedToneCount = heEhtActiveToneCount[bandwidthMhz]
    elif bandwidthMhz == 40:
        toneIndices = np.r_[np.arange(-244, -2), np.arange(3, 245)]
        expectedToneCount = heEhtActiveToneCount[bandwidthMhz]
    elif bandwidthMhz == 80:
        toneIndices = np.r_[np.arange(-500, -2), np.arange(3, 501)]
        expectedToneCount = heEhtActiveToneCount[bandwidthMhz]
    else:
        # A full-band 160 MHz allocation consists of two 996-tone RUs whose
        # centers are separated by 1024 HE/EHT subcarrier spacings.
        ru996Tones = np.r_[np.arange(-500, -2), np.arange(3, 501)]
        toneIndices = np.r_[ru996Tones - 512, ru996Tones + 512]
        expectedToneCount = heEhtActiveToneCount[bandwidthMhz]

    if toneIndices.size != expectedToneCount:
        raise RuntimeError("internal Wi-Fi active-tone construction error")
    return toneIndices.astype(np.int32)


def PilotTones(
    activeTones: np.ndarray,
    bandwidthMhz: int,
    frameFormat: str = "EHT",
) -> np.ndarray:
    """Build a format-specific symmetric full-band Wi-Fi pilot pattern.

    Processing details:
        Algorithm: Generate deterministic complex-baseband data from validated settings while preserving sample-rate and energy conventions.

    Args:
        activeTones: Centered indices of all active allocation subcarriers.
        bandwidthMhz: Configured Wi-Fi channel bandwidth in megahertz.
        frameFormat: PHY name or equivalent standard-generation alias.

    Returns:
        result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
    """

    normalizedFormat = NormalizeFrameFormat(frameFormat)
    if normalizedFormat == "VHT":
        if bandwidthMhz == 20:
            pilotTones = np.array([-21, -7, 7, 21], dtype=np.int32)
        elif bandwidthMhz == 40:
            pilotTones = np.array(
                [-53, -25, -11, 11, 25, 53], dtype=np.int32
            )
        elif bandwidthMhz == 80:
            pilotTones = np.array(
                [-103, -75, -39, -11, 11, 39, 75, 103],
                dtype=np.int32,
            )
        elif bandwidthMhz == 160:
            localPilotTones = PilotTones(
                ActiveTones(80, "VHT"), 80, "VHT"
            )
            pilotTones = np.r_[
                localPilotTones - 128,
                localPilotTones + 128,
            ].astype(np.int32)
        else:
            raise ValueError(
                "bandwidthMhz must be one of 20, 40, 80, 160"
            )
        if not np.all(np.isin(pilotTones, activeTones)):
            raise RuntimeError("internal VHT pilot-tone construction error")
        return pilotTones

    if bandwidthMhz == 20:
        return np.array(
            [-116, -90, -48, -22, 22, 48, 90, 116], dtype=np.int32
        )

    if bandwidthMhz == 40:
        return np.array(
            [
                -238,
                -212,
                -170,
                -144,
                -102,
                -76,
                -34,
                -8,
                8,
                34,
                76,
                102,
                144,
                170,
                212,
                238,
            ],
            dtype=np.int32,
        )

    if bandwidthMhz == 160:
        localActiveTones = ActiveTones(80, normalizedFormat)
        localPilotTones = PilotTones(
            localActiveTones, 80, normalizedFormat
        )
        return np.r_[localPilotTones - 512, localPilotTones + 512].astype(
            np.int32
        )

    # For the 996-tone RU, select eight negative-frequency pilot locations
    # and mirror them. This preserves symmetry, edge clearance, and the EHT
    # choice of sixteen pilot tones across the full 80 MHz allocation.
    negativeTones = activeTones[activeTones < 0]
    pilotPositions = np.linspace(20, negativeTones.size - 21, 8).round().astype(int)
    lowerPilots = negativeTones[pilotPositions]
    return np.r_[lowerPilots, -lowerPilots[::-1]].astype(np.int32)


def GrayToBinary(grayValues: np.ndarray) -> np.ndarray:
    """Convert vectorized Gray-coded integers into natural binary integers.

    Processing details:
        Algorithm: Carry out the described operation using validated inputs, explicit array-shape handling, and deterministic project conventions.

    Args:
        grayValues: Gray-coded labels converted into natural binary indices.

    Returns:
        result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
    """

    binaryValues = grayValues.copy()
    bitShift = 1
    while bitShift <= 32:
        binaryValues ^= binaryValues >> bitShift
        bitShift <<= 1
    return binaryValues


def QamModulate(bits: np.ndarray, qamOrder: int) -> np.ndarray:
    """Map bits to unit-average-power Gray-coded BPSK or square QAM symbols.

    Processing details:
        Algorithm: Generate deterministic complex-baseband data from validated settings while preserving sample-rate and energy conventions.

    Args:
        bits: Binary payload values mapped into modulation symbols.
        qamOrder: Constellation order used for BPSK or square QAM mapping.

    Returns:
        result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
    """

    flattenedBits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    bitsPerSymbol = int(np.log2(qamOrder))
    if flattenedBits.size % bitsPerSymbol:
        raise ValueError("number of bits must be a multiple of log2(qamOrder)")

    bitGroups = flattenedBits.reshape(-1, bitsPerSymbol)
    if qamOrder == 2:
        return (1.0 - 2.0 * bitGroups[:, 0].astype(float)).astype(np.complex128)

    # The first half of each bit group controls I and the second half controls
    # Q. Gray-to-binary conversion produces the corresponding PAM level.
    axisBits = bitsPerSymbol // 2
    binaryWeights = (1 << np.arange(axisBits - 1, -1, -1)).astype(np.int64)
    grayI = bitGroups[:, :axisBits].dot(binaryWeights)
    grayQ = bitGroups[:, axisBits:].dot(binaryWeights)
    binaryI = GrayToBinary(grayI)
    binaryQ = GrayToBinary(grayQ)
    constellationSide = int(np.sqrt(qamOrder))
    levelI = 2.0 * binaryI - (constellationSide - 1)
    levelQ = 2.0 * binaryQ - (constellationSide - 1)
    normalization = np.sqrt((2.0 / 3.0) * (qamOrder - 1))
    return (levelI + 1j * levelQ) / normalization


def PilotSequence(sequenceLength: int, randomGenerator: np.random.Generator) -> np.ndarray:
    """Create a deterministic, seed-controlled BPSK pilot sequence.

    Processing details:
        Algorithm: Generate deterministic complex-baseband data from validated settings while preserving sample-rate and energy conventions.

    Args:
        sequenceLength: Number of deterministic pilot symbols to create.
        randomGenerator: NumPy random generator that makes results reproducible.

    Returns:
        result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
    """

    return (
        1.0 - 2.0 * randomGenerator.integers(0, 2, sequenceLength)
    ).astype(np.complex128)


def OfdmSymbol(
    fftLength: int,
    cpLength: int,
    subcarriers: np.ndarray,
    subcarrierValues: np.ndarray,
) -> np.ndarray:
    """Create one energy-normalized OFDM symbol with a cyclic prefix.

    Processing details:
        Algorithm: Generate deterministic complex-baseband data from validated settings while preserving sample-rate and energy conventions.

    Args:
        fftLength: Number of time samples and frequency bins in the OFDM transform.
        cpLength: Number of cyclic-prefix samples prepended to the useful symbol.
        subcarriers: Centered indices receiving the supplied frequency-domain values.
        subcarrierValues: Complex symbols mapped onto the selected subcarriers.

    Returns:
        result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
    """

    frequencyGrid = np.zeros(fftLength, dtype=np.complex128)
    frequencyGrid[np.mod(subcarriers, fftLength)] = subcarrierValues
    usefulSamples = np.fft.ifft(frequencyGrid) * np.sqrt(fftLength)
    return np.r_[usefulSamples[-cpLength:], usefulSamples]


def TrainingField(
    symbolCount: int,
    legacyFftLength: int,
    subchannelCount: int,
    randomGenerator: np.random.Generator,
    fieldNumber: int,
) -> np.ndarray:
    """Generate deterministic 4 us Wi-Fi training or signaling symbols.

    Processing details:
        Algorithm: Generate deterministic complex-baseband data from validated settings while preserving sample-rate and energy conventions.

    Args:
        symbolCount: Number of OFDM symbols generated for the selected field.
        legacyFftLength: Bonded legacy-compatible FFT length at the active sample rate.
        subchannelCount: Number of bonded 20 MHz legacy subchannels.
        randomGenerator: NumPy random generator that makes results reproducible.
        fieldNumber: Stable field identifier used to derive deterministic random seeds.

    Returns:
        result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
    """

    legacyCpLength = legacyFftLength // 4
    subchannelCenters = (
        np.arange(subchannelCount) - (subchannelCount - 1) / 2.0
    ) * 64
    toneList = []
    for subchannelCenter in subchannelCenters.astype(int):
        localTones = np.r_[np.arange(-26, 0), np.arange(1, 27)]
        toneList.extend((subchannelCenter + localTones).tolist())
    toneIndices = np.asarray(toneList, dtype=np.int32)

    outputSymbols = []
    for symbolIndex in range(symbolCount):
        # Derive an independent repeatable generator for every preamble field
        # so changing the payload length does not change its training symbols.
        localSeed = (
            int(randomGenerator.integers(0, 2**31 - 1))
            + 257 * fieldNumber
            + symbolIndex
        )
        localGenerator = np.random.default_rng(localSeed)
        toneValues = PilotSequence(toneIndices.size, localGenerator)
        outputSymbols.append(
            OfdmSymbol(
                legacyFftLength, legacyCpLength, toneIndices, toneValues
            )
        )
    return np.concatenate(outputSymbols)


def BuildSpatialMappingMatrix(config: GenWifi) -> np.ndarray:
    """Build the constant orthonormal spatial mapping matrix.

    Processing details:
        Algorithm: Use direct stream-to-chain placement, a normalized partial
        DFT for uniform spatial expansion, or the validated custom matrix.

    Args:
        config: Validated waveform configuration containing MIMO dimensions.

    Returns:
        result: Complex matrix shaped transmit antennas by spatial streams.
    """

    numTransmitAntennas = config.numTransmitAntennas
    numSpatialStreams = config.numSpatialStreams
    if config.spatialMapping == "direct":
        mappingMatrix = np.zeros(
            (numTransmitAntennas, numSpatialStreams),
            dtype=np.complex128,
        )
        mappingMatrix[:numSpatialStreams, :] = np.eye(numSpatialStreams)
        return mappingMatrix
    if config.spatialMapping == "dft":
        antennaIndices = np.arange(numTransmitAntennas)[:, None]
        streamIndices = np.arange(numSpatialStreams)[None, :]
        return np.exp(
            -1j
            * 2.0
            * np.pi
            * antennaIndices
            * streamIndices
            / float(numTransmitAntennas)
        ) / np.sqrt(float(numTransmitAntennas))
    customMatrix = config.spatialMappingMatrix
    if customMatrix is None:
        raise RuntimeError("validated custom spatial mapping matrix is missing")
    return customMatrix


def GetLtfSymbolCount(frameFormat: str, numSpatialStreams: int) -> int:
    """Return the training-symbol count required by the spatial dimensions.

    Processing details:
        Algorithm: Apply the standard 1,2,4,6,8 training-dimension
        progression through at most eight VHT/HE/EHT spatial streams.

    Args:
        frameFormat: Canonical VHT, HE, or EHT name.
        numSpatialStreams: Number of independently trained spatial streams.

    Returns:
        result: Number of orthogonally coded format-specific LTF symbols.
    """

    normalizedFormat = NormalizeFrameFormat(frameFormat)
    if numSpatialStreams <= 1:
        return 1
    if numSpatialStreams <= 2:
        return 2
    if numSpatialStreams <= 4:
        return 4
    if numSpatialStreams <= 6:
        return 6
    if numSpatialStreams <= 8:
        return 8
    raise ValueError(
        f"{normalizedFormat} supports at most eight spatial streams"
    )


def BuildLtfTrainingMatrix(
    numSpatialStreams: int, ltfSymbolCount: int
) -> np.ndarray:
    """Create orthogonal phase codes across format-specific LTF symbols.

    Processing details:
        Algorithm: Select the first spatial-stream rows of a DFT matrix whose
        columns index LTF symbols. The rows are mutually orthogonal and allow
        the receiver to separate every trained spatial dimension.

    Args:
        numSpatialStreams: Number of spatial streams to train.
        ltfSymbolCount: Number of available LTF OFDM symbols.

    Returns:
        result: Complex matrix shaped streams by LTF symbols.
    """

    if ltfSymbolCount < numSpatialStreams:
        raise ValueError("ltfSymbolCount cannot be smaller than streams")
    streamIndices = np.arange(numSpatialStreams)[:, None]
    symbolIndices = np.arange(ltfSymbolCount)[None, :]
    return np.exp(
        -1j
        * 2.0
        * np.pi
        * streamIndices
        * symbolIndices
        / float(ltfSymbolCount)
    )


def GetCyclicShifts(config: GenWifi) -> np.ndarray:
    """Return per-chain cyclic shifts in seconds.

    Processing details:
        Algorithm: Select the configured number of WLAN shift values and
        replace them with zeros when cyclic shift diversity is disabled.

    Args:
        config: Validated MIMO waveform configuration.

    Returns:
        result: Floating-point vector of shifts in seconds.
    """

    cyclicShiftNanoseconds = np.asarray(
        (
            0.0,
            -400.0,
            -200.0,
            -600.0,
            -350.0,
            -650.0,
            -100.0,
            -750.0,
        ),
        dtype=float,
    )
    selectedShifts = cyclicShiftNanoseconds[
        : config.numTransmitAntennas
    ].copy()
    if not config.cyclicShiftEnabled:
        selectedShifts.fill(0.0)
    return selectedShifts * 1.0e-9


def BuildCsdPhaseMatrix(
    subcarrierIndices: np.ndarray,
    subcarrierSpacingHz: float,
    cyclicShiftsSeconds: np.ndarray,
) -> np.ndarray:
    """Build frequency-dependent CSD phases for all tones and chains.

    Processing details:
        Algorithm: Evaluate ``exp(-j*2*pi*k*deltaF*tau)`` for every centered
        subcarrier index and transmit-chain cyclic shift.

    Args:
        subcarrierIndices: Centered signed OFDM tone indices.
        subcarrierSpacingHz: Tone spacing in hertz.
        cyclicShiftsSeconds: Per-chain cyclic shifts in seconds.

    Returns:
        result: Matrix shaped tones by transmit antennas.
    """

    toneFrequencies = (
        np.asarray(subcarrierIndices, dtype=float).reshape(-1, 1)
        * float(subcarrierSpacingHz)
    )
    shifts = np.asarray(cyclicShiftsSeconds, dtype=float).reshape(1, -1)
    return np.exp(-1j * 2.0 * np.pi * toneFrequencies * shifts)


def SpatialMapTones(
    streamValues: np.ndarray,
    subcarrierIndices: np.ndarray,
    spatialMappingMatrix: np.ndarray,
    subcarrierSpacingHz: float,
    cyclicShiftsSeconds: np.ndarray,
) -> np.ndarray:
    """Map spatial-stream tone values onto physical transmit chains.

    Processing details:
        Algorithm: Apply ``x[k] = Q*s[k]`` on every tone and multiply each
        antenna by its frequency-dependent cyclic-shift phase.

    Args:
        streamValues: Complex matrix shaped tones by spatial streams.
        subcarrierIndices: Centered signed tone indices matching the rows.
        spatialMappingMatrix: Orthonormal matrix shaped antennas by streams.
        subcarrierSpacingHz: OFDM tone spacing in hertz.
        cyclicShiftsSeconds: Per-chain CSD shifts in seconds.

    Returns:
        result: Complex tone matrix shaped tones by transmit antennas.
    """

    complexStreams = np.asarray(streamValues, dtype=np.complex128)
    if complexStreams.ndim != 2:
        raise ValueError("streamValues must be a two-dimensional matrix")
    if complexStreams.shape[0] != np.asarray(subcarrierIndices).size:
        raise ValueError("streamValues and subcarrierIndices must align")
    if complexStreams.shape[1] != spatialMappingMatrix.shape[1]:
        raise ValueError("streamValues do not match spatial stream count")
    antennaValues = complexStreams @ spatialMappingMatrix.T
    csdPhases = BuildCsdPhaseMatrix(
        subcarrierIndices,
        subcarrierSpacingHz,
        cyclicShiftsSeconds,
    )
    return antennaValues * csdPhases


def BuildMimoOfdmSymbol(
    fftLength: int,
    cpLength: int,
    subcarrierIndices: np.ndarray,
    streamValues: np.ndarray,
    spatialMappingMatrix: np.ndarray,
    subcarrierSpacingHz: float,
    cyclicShiftsSeconds: np.ndarray,
) -> np.ndarray:
    """Generate one CP-OFDM symbol for every transmit antenna.

    Processing details:
        Algorithm: Spatially map stream values tone by tone, invoke the SISO
        OFDM modulator independently on every mapped antenna column, and stack
        time samples along a common first dimension.

    Args:
        fftLength: Oversampled IFFT length.
        cpLength: Cyclic-prefix length in samples.
        subcarrierIndices: Centered signed allocated tones.
        streamValues: Values shaped tones by spatial streams.
        spatialMappingMatrix: Orthonormal antennas-by-streams mapping.
        subcarrierSpacingHz: Tone spacing in hertz.
        cyclicShiftsSeconds: Per-chain cyclic shifts in seconds.

    Returns:
        result: Time-domain matrix shaped samples by transmit antennas.
    """

    antennaValues = SpatialMapTones(
        streamValues,
        subcarrierIndices,
        spatialMappingMatrix,
        subcarrierSpacingHz,
        cyclicShiftsSeconds,
    )
    antennaSymbols = [
        OfdmSymbol(
            fftLength,
            cpLength,
            subcarrierIndices,
            antennaValues[:, antennaIndex],
        )
        for antennaIndex in range(spatialMappingMatrix.shape[0])
    ]
    return np.column_stack(antennaSymbols)


def MapCommonFieldToAntennas(
    fieldSamples: np.ndarray,
    sampleRateHz: float,
    cyclicShiftsSeconds: np.ndarray,
) -> np.ndarray:
    """Replicate one non-spatial field across antennas with CSD.

    Processing details:
        Algorithm: Split the field into its 4-us legacy-rate OFDM symbols,
        apply each cyclic shift independently as a frequency-domain circular
        shift per symbol, and preserve total power across antenna copies.

    Args:
        fieldSamples: One-dimensional common-field complex waveform.
        sampleRateHz: Complex sample rate in samples per second.
        cyclicShiftsSeconds: Per-chain cyclic shifts in seconds.

    Returns:
        result: Common-field matrix shaped samples by transmit antennas.
    """

    complexField = np.asarray(fieldSamples, dtype=np.complex128).reshape(-1)
    legacySymbolLength = int(round(4.0e-6 * float(sampleRateHz)))
    if (
        legacySymbolLength <= 0
        or complexField.size % legacySymbolLength != 0
    ):
        raise ValueError(
            "fieldSamples must contain complete 4-us common-field symbols"
        )
    fieldSymbols = complexField.reshape(-1, legacySymbolLength)
    fieldSpectrum = np.fft.fft(fieldSymbols, axis=1)
    frequencyBins = np.fft.fftfreq(
        legacySymbolLength, d=1.0 / float(sampleRateHz)
    )
    antennaFields = []
    antennaScale = 1.0 / np.sqrt(float(cyclicShiftsSeconds.size))
    for shiftSeconds in cyclicShiftsSeconds:
        shiftedField = np.fft.ifft(
            fieldSpectrum
            * np.exp(
                -1j * 2.0 * np.pi * frequencyBins * shiftSeconds
            )[None, :],
            axis=1,
        ).reshape(-1)
        antennaFields.append(antennaScale * shiftedField)
    return np.column_stack(antennaFields)


def GenerateWifiWaveform(config: GenWifi) -> WifiWaveform:
    """Generate one VHT, HE, or EHT packet for a validated generator.

    Args:
        config: The constructed Wi-Fi generator and all of its PHY settings.

    Returns:
        A ``WifiWaveform`` containing unit-RMS samples, field boundaries,
        subcarrier allocations, and the transmitted data constellation.
    """

    config.Validate()
    normalizedFormat = config.frameFormat
    randomGenerator = np.random.default_rng(config.seed)
    mcsInfo = config.GetMcsInfo()
    bandwidthHz = float(config.bandwidthMhz) * 1e6
    vhtBaseFftLength: Mapping[int, int] = MappingProxyType(
        {20: 64, 40: 128, 80: 256, 160: 512}
    )
    heEhtBaseFftLength: Mapping[int, int] = MappingProxyType(
        {20: 256, 40: 512, 80: 1024, 160: 2048}
    )
    vhtPilotToneCount: Mapping[int, int] = MappingProxyType(
        {20: 4, 40: 6, 80: 8, 160: 16}
    )
    heEhtPilotToneCount: Mapping[int, int] = MappingProxyType(
        {20: 8, 40: 16, 80: 16, 160: 32}
    )
    if normalizedFormat == "VHT":
        baseFftLength = vhtBaseFftLength[config.bandwidthMhz]
        expectedPilotCount = vhtPilotToneCount[config.bandwidthMhz]
    else:
        baseFftLength = heEhtBaseFftLength[config.bandwidthMhz]
        expectedPilotCount = heEhtPilotToneCount[config.bandwidthMhz]
    sampleRateHz = config.sampleRateHz
    fftLength = int(
        round(baseFftLength * sampleRateHz / bandwidthHz)
    )
    cpLength = int(round(config.guardIntervalUs * 1e-6 * sampleRateHz))
    subcarrierSpacingHz = sampleRateHz / float(fftLength)
    spatialMappingMatrix = BuildSpatialMappingMatrix(config)
    cyclicShiftsSeconds = GetCyclicShifts(config)
    ltfSymbolCount = GetLtfSymbolCount(
        normalizedFormat, config.numSpatialStreams
    )

    activeSubcarriers = ActiveTones(
        config.bandwidthMhz, normalizedFormat
    )
    pilotSubcarriers = PilotTones(
        activeSubcarriers,
        config.bandwidthMhz,
        normalizedFormat,
    )
    if pilotSubcarriers.size != expectedPilotCount:
        raise RuntimeError("internal Wi-Fi pilot tone construction error")
    dataSubcarriers = activeSubcarriers[
        ~np.isin(activeSubcarriers, pilotSubcarriers)
    ]

    packetFields = []
    fieldSlices: Dict[str, slice] = {}
    sampleCursor = 0

    def AppendField(fieldName: str, fieldSamples: np.ndarray) -> None:
        """Append a field and record its half-open sample interval.

        Processing details:
            Algorithm: Carry out the described operation using validated inputs, explicit array-shape handling, and deterministic project conventions.

        Args:
            fieldName: Caller-supplied value consumed according to the function contract.
            fieldSamples: Caller-supplied value consumed according to the function contract.

        Returns:
            result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
        """

        nonlocal sampleCursor
        complexSamples = np.asarray(fieldSamples, dtype=np.complex128)
        if complexSamples.ndim == 1:
            complexSamples = complexSamples.reshape(-1, 1)
        if (
            complexSamples.ndim != 2
            or complexSamples.shape[1] != config.numTransmitAntennas
        ):
            raise ValueError(
                "fieldSamples must have one column per transmit antenna"
            )
        packetFields.append(complexSamples)
        fieldSlices[fieldName] = slice(
            sampleCursor, sampleCursor + complexSamples.shape[0]
        )
        sampleCursor += complexSamples.shape[0]

    legacyFftLength = int(round(3.2e-6 * sampleRateHz))
    legacySubchannelCount = config.bandwidthMhz // 20
    # All three PHY generations start with a legacy-compatible prefix. VHT
    # then uses its VHT signaling fields, while HE/EHT add RL-SIG and their
    # newer format-specific signaling fields.
    if normalizedFormat == "EHT":
        preambleSpecification: Tuple[Tuple[str, int], ...] = (
            ("L-STF", 2),
            ("L-LTF", 2),
            ("L-SIG", 1),
            ("RL-SIG", 1),
            ("U-SIG", 2),
            ("EHT-SIG", 1),
        )
        shortTrainingFieldName = "EHT-STF"
        longTrainingFieldName = "EHT-LTF"
        dataFieldName = "EHT-Data"
        formatName = "EHT-MU non-OFDMA single-user"
    elif normalizedFormat == "HE":
        preambleSpecification = (
            ("L-STF", 2),
            ("L-LTF", 2),
            ("L-SIG", 1),
            ("RL-SIG", 1),
            ("HE-SIG-A", 2),
        )
        shortTrainingFieldName = "HE-STF"
        longTrainingFieldName = "HE-LTF"
        dataFieldName = "HE-Data"
        formatName = "HE-SU single-user"
    else:
        preambleSpecification = (
            ("L-STF", 2),
            ("L-LTF", 2),
            ("L-SIG", 1),
            ("VHT-SIG-A", 2),
        )
        shortTrainingFieldName = "VHT-STF"
        longTrainingFieldName = "VHT-LTF"
        dataFieldName = "VHT-Data"
        formatName = "VHT single-user (IEEE 802.11ac)"
    for fieldNumber, (fieldName, symbolCount) in enumerate(
        preambleSpecification
    ):
        fieldSamples = TrainingField(
            symbolCount,
            legacyFftLength,
            legacySubchannelCount,
            randomGenerator,
            fieldNumber,
        )
        AppendField(
            fieldName,
            MapCommonFieldToAntennas(
                fieldSamples,
                sampleRateHz,
                cyclicShiftsSeconds,
            ),
        )

    # VHT-STF, HE-STF, and EHT-STF are 4 us fields. A legacy-rate OFDM
    # construction preserves that duration and a repeatable wideband stimulus.
    shortTrainingSamples = TrainingField(
            1,
            legacyFftLength,
            legacySubchannelCount,
            randomGenerator,
            len(preambleSpecification),
        )
    AppendField(
        shortTrainingFieldName,
        MapCommonFieldToAntennas(
            shortTrainingSamples,
            sampleRateHz,
            cyclicShiftsSeconds,
        ),
    )

    # Every VHT-LTF uses a 3.2 us useful symbol and a 0.8 us guard interval;
    # multiple spatial dimensions add orthogonally coded LTF symbols.
    # HE/EHT use format-specific 2x or 4x LTF constructions.
    if normalizedFormat == "VHT":
        longTrainingFftLength = fftLength
        longTrainingSubcarriers = activeSubcarriers
        longTrainingGuardIntervalUs = 0.8
    elif config.guardIntervalUs == 1.6:
        longTrainingFftLength = fftLength // 2
        evenToneMask = np.mod(activeSubcarriers, 2) == 0
        longTrainingSubcarriers = activeSubcarriers[evenToneMask] // 2
        longTrainingGuardIntervalUs = config.guardIntervalUs
    else:
        longTrainingFftLength = fftLength
        longTrainingSubcarriers = activeSubcarriers
        longTrainingGuardIntervalUs = config.guardIntervalUs
    longTrainingCpLength = int(
        round(longTrainingGuardIntervalUs * 1e-6 * sampleRateHz)
    )
    longTrainingValues = PilotSequence(
        longTrainingSubcarriers.size, randomGenerator
    )
    ltfTrainingMatrix = BuildLtfTrainingMatrix(
        config.numSpatialStreams, ltfSymbolCount
    )
    longTrainingSymbols = []
    longTrainingSubcarrierSpacingHz = (
        sampleRateHz / float(longTrainingFftLength)
    )
    for ltfSymbolIndex in range(ltfSymbolCount):
        ltfStreamValues = (
            longTrainingValues[:, None]
            * ltfTrainingMatrix[:, ltfSymbolIndex][None, :]
        )
        longTrainingSymbols.append(
            BuildMimoOfdmSymbol(
                longTrainingFftLength,
                longTrainingCpLength,
                longTrainingSubcarriers,
                ltfStreamValues,
                spatialMappingMatrix,
                longTrainingSubcarrierSpacingHz,
                cyclicShiftsSeconds,
            )
        )
    AppendField(
        longTrainingFieldName,
        np.concatenate(longTrainingSymbols, axis=0),
    )

    # VHT-SIG-B follows VHT-LTF and occupies one 4 us OFDM symbol. HE and EHT
    # carry equivalent signaling earlier in their respective preambles.
    if normalizedFormat == "VHT":
        vhtSigBSamples = TrainingField(
                1,
                legacyFftLength,
                legacySubchannelCount,
                randomGenerator,
                len(preambleSpecification) + 1,
            )
        AppendField(
            "VHT-SIG-B",
            MapCommonFieldToAntennas(
                vhtSigBSamples,
                sampleRateHz,
                cyclicShiftsSeconds,
            ),
        )

    bitsPerSubcarrier = mcsInfo.bitsPerSubcarrier
    totalCodedBits = (
        config.numDataSymbols
        * dataSubcarriers.size
        * config.numSpatialStreams
        * bitsPerSubcarrier
    )
    codedBits = randomGenerator.integers(
        0, 2, totalCodedBits, dtype=np.uint8
    )
    qamSymbols = QamModulate(codedBits, mcsInfo.qamOrder).reshape(
        config.numDataSymbols,
        dataSubcarriers.size,
        config.numSpatialStreams,
    )

    dataStart = sampleCursor
    dataSymbolStarts = []
    dataTimeSymbols = []
    symbolLength = fftLength + cpLength
    for symbolIndex in range(config.numDataSymbols):
        pilotValues = PilotSequence(
            pilotSubcarriers.size * config.numSpatialStreams,
            randomGenerator,
        ).reshape(pilotSubcarriers.size, config.numSpatialStreams)
        subcarrierValues = np.concatenate(
            (qamSymbols[symbolIndex], pilotValues), axis=0
        )
        mappedSubcarriers = np.r_[dataSubcarriers, pilotSubcarriers]
        dataSymbolStarts.append(dataStart + symbolIndex * symbolLength)
        dataTimeSymbols.append(
            BuildMimoOfdmSymbol(
                fftLength,
                cpLength,
                mappedSubcarriers,
                subcarrierValues,
                spatialMappingMatrix,
                subcarrierSpacingHz,
                cyclicShiftsSeconds,
            )
        )
    dataSamples = np.concatenate(dataTimeSymbols, axis=0)
    AppendField(dataFieldName, dataSamples)
    if fieldSlices[dataFieldName].start != dataStart:
        raise RuntimeError("internal Wi-Fi data field offset error")

    packetSamples = np.concatenate(packetFields, axis=0)
    packetRms = np.sqrt(
        np.mean(np.sum(np.abs(packetSamples) ** 2, axis=1))
    )
    normalizationScale = 1.0 / max(packetRms, np.finfo(float).tiny)
    packetSamples *= normalizationScale

    codedBitsPerSymbol = (
        dataSubcarriers.size
        * bitsPerSubcarrier
        * config.numSpatialStreams
    )
    informationBitsPerSymbol = int(
        np.floor(codedBitsPerSymbol * mcsInfo.codeRate)
    )
    outputSamples = (
        packetSamples[:, 0]
        if config.numTransmitAntennas == 1
        else packetSamples
    )
    outputQamSymbols = (
        qamSymbols[:, :, 0]
        if config.numSpatialStreams == 1
        else qamSymbols
    ) * normalizationScale
    formatName = (
        f"{formatName}; {config.numSpatialStreams} spatial stream(s), "
        f"{config.numTransmitAntennas} transmit chain(s)"
    )
    return WifiWaveform(
        samples=outputSamples,
        sampleRateHz=sampleRateHz,
        bandwidthHz=bandwidthHz,
        fftLength=fftLength,
        cpLength=cpLength,
        oversampling=config.oversampling,
        activeSubcarriers=activeSubcarriers,
        dataSubcarriers=dataSubcarriers,
        pilotSubcarriers=pilotSubcarriers,
        referenceDataSymbols=outputQamSymbols,
        fieldSlices=fieldSlices,
        dataSymbolStarts=np.asarray(dataSymbolStarts, dtype=np.int64),
        symbolLength=symbolLength,
        mcsInfo=mcsInfo,
        normalizationScale=normalizationScale,
        codedBitsPerSymbol=codedBitsPerSymbol,
        informationBitsPerSymbol=informationBitsPerSymbol,
        frameFormat=normalizedFormat,
        dataFieldName=dataFieldName,
        formatName=formatName,
        numTransmitAntennas=config.numTransmitAntennas,
        numSpatialStreams=config.numSpatialStreams,
        spatialMapping=config.spatialMapping,
        spatialMappingMatrix=spatialMappingMatrix,
        cyclicShiftsSeconds=cyclicShiftsSeconds,
        ltfSymbolCount=ltfSymbolCount,
    )
