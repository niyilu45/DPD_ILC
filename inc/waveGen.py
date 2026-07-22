"""Generate IEEE 802.11ac VHT, 802.11ax HE, or 802.11be EHT waveforms.

Callers first construct ``GenWifi`` with the requested frame format, bandwidth,
MCS, guard interval, and packet length, and then call ``Generate``. The class
creates a single-user full-band complex baseband packet with the appropriate
VHT, HE-SU, or EHT field order, OFDM numerology, and MCS table. Standard names
such as ``11ac`` are accepted as aliases of the PHY names such as ``VHT``.

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


# EHT MCS 0 through 13 cover BPSK through 4096-QAM.
ehtMcsTable: Mapping[int, MCSInfo] = {
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

# HE uses the same modulation and rate mapping through MCS 11. MCS 12 and 13
# are EHT-only because 4096-QAM was introduced by IEEE 802.11be.
heMcsTable: Mapping[int, MCSInfo] = {
    mcsIndex: mcsInfo
    for mcsIndex, mcsInfo in ehtMcsTable.items()
    if mcsIndex <= 11
}

# IEEE 802.11ac VHT defines MCS 0 through 9 for the single-stream modulation
# mapping used by this project. The numerical mappings match HE/EHT through
# 256-QAM, while later PHY generations add the higher MCS values.
vhtMcsTable: Mapping[int, MCSInfo] = {
    mcsIndex: mcsInfo
    for mcsIndex, mcsInfo in ehtMcsTable.items()
    if mcsIndex <= 9
}


# Inputs are normalized to one canonical PHY name so every downstream module
# sees only VHT, HE, or EHT. Both common shortened standard names and formal
# IEEE amendment names are accepted without case sensitivity.
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


# VHT uses 312.5 kHz data-subcarrier spacing. HE and EHT use 78.125 kHz,
# therefore the newer formats require four times as many IFFT bins at a given
# channel bandwidth before optional oversampling is applied.
vhtBaseFftLength = {20: 64, 40: 128, 80: 256, 160: 512}
heEhtBaseFftLength = {20: 256, 40: 512, 80: 1024, 160: 2048}
vhtActiveToneCount = {20: 56, 40: 114, 80: 242, 160: 484}
heEhtActiveToneCount = {20: 242, 40: 484, 80: 996, 160: 1992}
vhtPilotToneCount = {20: 4, 40: 6, 80: 8, 160: 16}
heEhtPilotToneCount = {20: 8, 40: 16, 80: 16, 160: 32}
mcsTableByFormat: Mapping[str, Mapping[int, MCSInfo]] = MappingProxyType(
    {
        "VHT": vhtMcsTable,
        "HE": heMcsTable,
        "EHT": ehtMcsTable,
    }
)
guardIntervalsByFormat: Mapping[str, Tuple[float, ...]] = MappingProxyType(
    {
        "VHT": (0.4, 0.8),
        "HE": (0.8, 1.6, 3.2),
        "EHT": (0.8, 1.6, 3.2),
    }
)


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

    if not isinstance(frameFormat, str):
        raise TypeError("frameFormat must be a string")
    normalizedInput = frameFormat.strip().upper()
    if normalizedInput not in frameFormatAliases:
        raise ValueError(
            "frameFormat must be VHT/11ac, HE/11ax, or EHT/11be"
        )
    return frameFormatAliases[normalizedInput]


genWifiDefaultParameters: Mapping[str, object] = MappingProxyType(
    {
        "frameFormat": "EHT",
        "bandwidthMhz": 80,
        "mcs": 9,
        "numDataSymbols": 20,
        "guardIntervalUs": 0.8,
        "oversampling": 4,
        "seed": 7,
    }
)


class GenWifi:
    """Configure and generate a single-user VHT, HE, or EHT Wi-Fi waveform.

    A ``ChainMap`` resolves explicit keyword overrides first, a caller-owned
    parameter mapping second, and immutable module defaults last. Changes to
    the caller-owned mapping remain visible until a higher-priority keyword
    override shadows the same key.

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
            Algorithm: Carry out the described operation using validated inputs, explicit array-shape handling, and deterministic project conventions.

        Args:
            parameters: Optional external mapping layered ahead of the built-in defaults.
            parameterOverrides: Highest-priority keyword values applied to the local ChainMap layer.

        Returns:
            result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
        """
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping or None")
        externalParameters = {} if parameters is None else parameters
        self.parameters: ChainMap[str, object] = ChainMap(
            dict(parameterOverrides),
            externalParameters,
            genWifiDefaultParameters,
        )
        self.Validate()

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
    def Oversampling(self) -> int:
        """Return the currently resolved oversampling factor.

        Processing details:
            Algorithm: Carry out the described operation using validated inputs, explicit array-shape handling, and deterministic project conventions.

        Returns:
            result: int. The computed value described by the summary, with documented units, shape, and normalization.
        """

        return cast(int, self.parameters["oversampling"])

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

    def GetParameters(self) -> Dict[str, object]:
        """Return a flattened snapshot of all resolved parameters.

        Processing details:
            Algorithm: Resolve values according to state and ChainMap precedence, keeping caller-owned configuration behavior explicit.

        Returns:
            result: Dict[str, object]. The computed value described by the summary, with documented units, shape, and normalization.
        """

        return dict(self.parameters)

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
            genWifiDefaultParameters
        )
        if unknownParameters:
            unknownNames = ", ".join(
                sorted(str(parameterName) for parameterName in unknownParameters)
            )
            raise TypeError(f"unknown GenWifi parameters: {unknownNames}")
        normalizedFormat = NormalizeFrameFormat(
            cast(str, self.parameters["frameFormat"])
        )
        if not isinstance(self.bandwidthMhz, int) or isinstance(
            self.bandwidthMhz, bool
        ):
            raise TypeError("bandwidthMhz must be an integer")
        if self.bandwidthMhz not in heEhtBaseFftLength:
            raise ValueError("bandwidthMhz must be one of 20, 40, 80, 160")
        selectedMcsTable = mcsTableByFormat[normalizedFormat]
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
        if (
            not isinstance(self.oversampling, int)
            or isinstance(self.oversampling, bool)
            or self.oversampling < 1
        ):
            raise ValueError("oversampling must be a positive integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")

    def GetMcsInfo(self) -> MCSInfo:
        """Return the MCS record selected by this generator instance.

        Processing details:
            Algorithm: Resolve values according to state and ChainMap precedence, keeping caller-owned configuration behavior explicit.

        Returns:
            result: MCSInfo. The computed value described by the summary, with documented units, shape, and normalization.
        """

        self.Validate()
        return mcsTableByFormat[self.frameFormat][self.mcs]

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
    oversampling: int
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
    if bandwidthMhz not in heEhtBaseFftLength:
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
    oversampling: int,
    randomGenerator: np.random.Generator,
    fieldNumber: int,
) -> np.ndarray:
    """Generate deterministic 4 us Wi-Fi training or signaling symbols.

    Processing details:
        Algorithm: Generate deterministic complex-baseband data from validated settings while preserving sample-rate and energy conventions.

    Args:
        symbolCount: Number of OFDM symbols generated for the selected field.
        legacyFftLength: Bonded legacy-compatible FFT length at the active sample rate.
        oversampling: Integer sample-rate expansion relative to channel bandwidth.
        randomGenerator: NumPy random generator that makes results reproducible.
        fieldNumber: Stable field identifier used to derive deterministic random seeds.

    Returns:
        result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
    """

    legacyCpLength = legacyFftLength // 4
    # Oversampling enlarges the IFFT without creating additional 20 MHz
    # subchannels. Divide it out before constructing bonded legacy segments.
    subchannelCount = legacyFftLength // (64 * oversampling)
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
    if normalizedFormat == "VHT":
        baseFftLength = vhtBaseFftLength[config.bandwidthMhz]
        expectedPilotCount = vhtPilotToneCount[config.bandwidthMhz]
    else:
        baseFftLength = heEhtBaseFftLength[config.bandwidthMhz]
        expectedPilotCount = heEhtPilotToneCount[config.bandwidthMhz]
    fftLength = baseFftLength * config.oversampling
    sampleRateHz = bandwidthHz * config.oversampling
    cpLength = int(round(config.guardIntervalUs * 1e-6 * sampleRateHz))

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
        packetFields.append(complexSamples)
        fieldSlices[fieldName] = slice(
            sampleCursor, sampleCursor + complexSamples.size
        )
        sampleCursor += complexSamples.size

    legacyFftLength = (
        64 * (config.bandwidthMhz // 20) * config.oversampling
    )
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
            config.oversampling,
            randomGenerator,
            fieldNumber,
        )
        AppendField(fieldName, fieldSamples)

    # VHT-STF, HE-STF, and EHT-STF are 4 us fields. A legacy-rate OFDM
    # construction preserves that duration and a repeatable wideband stimulus.
    AppendField(
        shortTrainingFieldName,
        TrainingField(
            1,
            legacyFftLength,
            config.oversampling,
            randomGenerator,
            len(preambleSpecification),
        ),
    )

    # A single-stream VHT-LTF always uses a 3.2 us useful symbol and a 0.8 us
    # guard interval. HE/EHT use format-specific 2x or 4x LTF constructions.
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
    AppendField(
        longTrainingFieldName,
        OfdmSymbol(
            longTrainingFftLength,
            longTrainingCpLength,
            longTrainingSubcarriers,
            longTrainingValues,
        ),
    )

    # VHT-SIG-B follows VHT-LTF and occupies one 4 us OFDM symbol. HE and EHT
    # carry equivalent signaling earlier in their respective preambles.
    if normalizedFormat == "VHT":
        AppendField(
            "VHT-SIG-B",
            TrainingField(
                1,
                legacyFftLength,
                config.oversampling,
                randomGenerator,
                len(preambleSpecification) + 1,
            ),
        )

    bitsPerSubcarrier = mcsInfo.bitsPerSubcarrier
    totalCodedBits = (
        config.numDataSymbols * dataSubcarriers.size * bitsPerSubcarrier
    )
    codedBits = randomGenerator.integers(
        0, 2, totalCodedBits, dtype=np.uint8
    )
    qamSymbols = QamModulate(codedBits, mcsInfo.qamOrder).reshape(
        config.numDataSymbols, dataSubcarriers.size
    )

    dataStart = sampleCursor
    dataSymbolStarts = []
    dataTimeSymbols = []
    symbolLength = fftLength + cpLength
    for symbolIndex in range(config.numDataSymbols):
        pilotValues = PilotSequence(pilotSubcarriers.size, randomGenerator)
        subcarrierValues = np.r_[qamSymbols[symbolIndex], pilotValues]
        mappedSubcarriers = np.r_[dataSubcarriers, pilotSubcarriers]
        dataSymbolStarts.append(dataStart + symbolIndex * symbolLength)
        dataTimeSymbols.append(
            OfdmSymbol(
                fftLength,
                cpLength,
                mappedSubcarriers,
                subcarrierValues,
            )
        )
    dataSamples = np.concatenate(dataTimeSymbols)
    AppendField(dataFieldName, dataSamples)
    if fieldSlices[dataFieldName].start != dataStart:
        raise RuntimeError("internal Wi-Fi data field offset error")

    packetSamples = np.concatenate(packetFields)
    packetRms = np.sqrt(np.mean(np.abs(packetSamples) ** 2))
    normalizationScale = 1.0 / max(packetRms, np.finfo(float).tiny)
    packetSamples *= normalizationScale

    codedBitsPerSymbol = dataSubcarriers.size * bitsPerSubcarrier
    informationBitsPerSymbol = int(
        np.floor(codedBitsPerSymbol * mcsInfo.codeRate)
    )
    return WifiWaveform(
        samples=packetSamples,
        sampleRateHz=sampleRateHz,
        bandwidthHz=bandwidthHz,
        fftLength=fftLength,
        cpLength=cpLength,
        oversampling=config.oversampling,
        activeSubcarriers=activeSubcarriers,
        dataSubcarriers=dataSubcarriers,
        pilotSubcarriers=pilotSubcarriers,
        referenceDataSymbols=qamSymbols,
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
    )
