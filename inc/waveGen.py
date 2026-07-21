"""Generate IEEE 802.11ax HE or IEEE 802.11be EHT Wi-Fi waveforms.

Callers first construct ``GenWifi`` with the requested frame format, bandwidth,
MCS, guard interval, and packet length, and then call ``Generate``. The class
creates a single-user full-band complex baseband packet with the appropriate
HE-SU or EHT-MU field order and MCS table.

DPD characterization only requires statistically representative coded
symbols. Therefore, this module creates randomized post-FEC stimulus bits
instead of implementing an LDPC encoder and decoder. The generated waveform
is a Wi-Fi PHY stimulus, not a bit-exact protocol conformance implementation.
"""

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

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


# HE and EHT both use 78.125 kHz data subcarrier spacing at these bandwidths.
_baseFftLength = {20: 256, 40: 512, 80: 1024, 160: 2048}
_ruToneCount = {20: 242, 40: 484, 80: 996, 160: 1992}
_pilotToneCount = {20: 8, 40: 16, 80: 16, 160: 32}


@dataclass(frozen=True)
class GenWifi:
    """Configure and generate a single-user HE or EHT Wi-Fi waveform.

    Example:
        ``wifiGenerator = GenWifi(frameFormat="EHT", bandwidthMhz=80, mcs=11)``
        ``waveform = wifiGenerator.Generate()``
    """

    frameFormat: str = "EHT"
    bandwidthMhz: int = 80
    mcs: int = 11
    numDataSymbols: int = 20
    guardIntervalUs: float = 0.8
    oversampling: int = 4
    seed: int = 7

    def Validate(self) -> None:
        """Validate every option before allocating potentially large arrays."""

        normalizedFormat = self.frameFormat.strip().upper()
        if normalizedFormat not in ("EHT", "HE"):
            raise ValueError("frameFormat must be either 'EHT' or 'HE'")
        if self.bandwidthMhz not in _baseFftLength:
            raise ValueError("bandwidthMhz must be one of 20, 40, 80, 160")
        selectedMcsTable = (
            ehtMcsTable if normalizedFormat == "EHT" else heMcsTable
        )
        if self.mcs not in selectedMcsTable:
            maximumMcs = 13 if normalizedFormat == "EHT" else 11
            raise ValueError(
                f"{normalizedFormat} MCS must be an integer from 0 through {maximumMcs}"
            )
        if self.numDataSymbols < 1:
            raise ValueError("numDataSymbols must be positive")
        if self.guardIntervalUs not in (0.8, 1.6, 3.2):
            raise ValueError("guardIntervalUs must be 0.8, 1.6, or 3.2")
        if not isinstance(self.oversampling, int) or self.oversampling < 1:
            raise ValueError("oversampling must be a positive integer")

    def GetMcsInfo(self) -> MCSInfo:
        """Return the MCS record selected by this generator instance."""

        self.Validate()
        normalizedFormat = self.frameFormat.strip().upper()
        selectedMcsTable = (
            ehtMcsTable if normalizedFormat == "EHT" else heMcsTable
        )
        return selectedMcsTable[self.mcs]

    def Generate(self) -> "WifiWaveform":
        """Generate one configured HE or EHT packet and its metadata."""

        return _GenerateWifiWaveform(self)


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


def _ActiveTones(bandwidthMhz: int) -> np.ndarray:
    """Return centered subcarrier indices for a full-band HE/EHT resource unit."""

    if bandwidthMhz == 20:
        toneIndices = np.r_[np.arange(-122, -1), np.arange(2, 123)]
    elif bandwidthMhz == 40:
        toneIndices = np.r_[np.arange(-244, -2), np.arange(3, 245)]
    elif bandwidthMhz == 80:
        toneIndices = np.r_[np.arange(-500, -2), np.arange(3, 501)]
    else:
        # A full-band 160 MHz allocation consists of two 996-tone RUs whose
        # centers are separated by 1024 HE/EHT subcarrier spacings.
        ru996Tones = np.r_[np.arange(-500, -2), np.arange(3, 501)]
        toneIndices = np.r_[ru996Tones - 512, ru996Tones + 512]

    if toneIndices.size != _ruToneCount[bandwidthMhz]:
        raise RuntimeError("internal Wi-Fi RU tone construction error")
    return toneIndices.astype(np.int32)


def _PilotTones(activeTones: np.ndarray, bandwidthMhz: int) -> np.ndarray:
    """Build a symmetric full-band pilot pattern for HE/EHT packets."""

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
        localActiveTones = _ActiveTones(80)
        localPilotTones = _PilotTones(localActiveTones, 80)
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


def _GrayToBinary(grayValues: np.ndarray) -> np.ndarray:
    """Convert vectorized Gray-coded integers into natural binary integers."""

    binaryValues = grayValues.copy()
    bitShift = 1
    while bitShift <= 32:
        binaryValues ^= binaryValues >> bitShift
        bitShift <<= 1
    return binaryValues


def QamModulate(bits: np.ndarray, qamOrder: int) -> np.ndarray:
    """Map bits to unit-average-power Gray-coded BPSK or square QAM symbols."""

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
    binaryI = _GrayToBinary(grayI)
    binaryQ = _GrayToBinary(grayQ)
    constellationSide = int(np.sqrt(qamOrder))
    levelI = 2.0 * binaryI - (constellationSide - 1)
    levelQ = 2.0 * binaryQ - (constellationSide - 1)
    normalization = np.sqrt((2.0 / 3.0) * (qamOrder - 1))
    return (levelI + 1j * levelQ) / normalization


def _PilotSequence(sequenceLength: int, randomGenerator: np.random.Generator) -> np.ndarray:
    """Create a deterministic, seed-controlled BPSK pilot sequence."""

    return (
        1.0 - 2.0 * randomGenerator.integers(0, 2, sequenceLength)
    ).astype(np.complex128)


def _OfdmSymbol(
    fftLength: int,
    cpLength: int,
    subcarriers: np.ndarray,
    subcarrierValues: np.ndarray,
) -> np.ndarray:
    """Create one energy-normalized OFDM symbol with a cyclic prefix."""

    frequencyGrid = np.zeros(fftLength, dtype=np.complex128)
    frequencyGrid[np.mod(subcarriers, fftLength)] = subcarrierValues
    usefulSamples = np.fft.ifft(frequencyGrid) * np.sqrt(fftLength)
    return np.r_[usefulSamples[-cpLength:], usefulSamples]


def _TrainingField(
    symbolCount: int,
    legacyFftLength: int,
    oversampling: int,
    randomGenerator: np.random.Generator,
    fieldNumber: int,
) -> np.ndarray:
    """Generate deterministic 4 us HE/EHT training or signaling symbols."""

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
        toneValues = _PilotSequence(toneIndices.size, localGenerator)
        outputSymbols.append(
            _OfdmSymbol(
                legacyFftLength, legacyCpLength, toneIndices, toneValues
            )
        )
    return np.concatenate(outputSymbols)


def _GenerateWifiWaveform(config: GenWifi) -> WifiWaveform:
    """Generate one HE or EHT packet for a validated ``GenWifi`` instance.

    Args:
        config: The constructed Wi-Fi generator and all of its PHY settings.

    Returns:
        A ``WifiWaveform`` containing unit-RMS samples, field boundaries,
        subcarrier allocations, and the transmitted data constellation.
    """

    config.Validate()
    normalizedFormat = config.frameFormat.strip().upper()
    randomGenerator = np.random.default_rng(config.seed)
    mcsInfo = config.GetMcsInfo()
    bandwidthHz = float(config.bandwidthMhz) * 1e6
    baseFftLength = _baseFftLength[config.bandwidthMhz]
    fftLength = baseFftLength * config.oversampling
    sampleRateHz = bandwidthHz * config.oversampling
    cpLength = int(round(config.guardIntervalUs * 1e-6 * sampleRateHz))

    activeSubcarriers = _ActiveTones(config.bandwidthMhz)
    pilotSubcarriers = _PilotTones(activeSubcarriers, config.bandwidthMhz)
    if pilotSubcarriers.size != _pilotToneCount[config.bandwidthMhz]:
        raise RuntimeError("internal Wi-Fi pilot tone construction error")
    dataSubcarriers = activeSubcarriers[
        ~np.isin(activeSubcarriers, pilotSubcarriers)
    ]

    packetFields = []
    fieldSlices: Dict[str, slice] = {}
    sampleCursor = 0

    def AppendField(fieldName: str, fieldSamples: np.ndarray) -> None:
        """Append a field and record its half-open sample interval."""

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
    # HE-SU and EHT-MU share the legacy-compatible prefix but use different
    # signaling fields after RL-SIG.
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
    else:
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
    for fieldNumber, (fieldName, symbolCount) in enumerate(
        preambleSpecification
    ):
        fieldSamples = _TrainingField(
            symbolCount,
            legacyFftLength,
            config.oversampling,
            randomGenerator,
            fieldNumber,
        )
        AppendField(fieldName, fieldSamples)

    # HE-STF and EHT-STF are 4 us fields. A legacy-rate OFDM construction
    # preserves that duration while providing a repeatable wideband stimulus.
    AppendField(
        shortTrainingFieldName,
        _TrainingField(
            1,
            legacyFftLength,
            config.oversampling,
            randomGenerator,
            len(preambleSpecification),
        ),
    )

    # A 1.6 us GI is paired with 2x HE/EHT-LTF compression. The 0.8 us and
    # 3.2 us GIs use a 4x LTF. For 2x compression every second tone is sent.
    if config.guardIntervalUs == 1.6:
        longTrainingFftLength = fftLength // 2
        evenToneMask = np.mod(activeSubcarriers, 2) == 0
        longTrainingSubcarriers = activeSubcarriers[evenToneMask] // 2
    else:
        longTrainingFftLength = fftLength
        longTrainingSubcarriers = activeSubcarriers
    longTrainingCpLength = int(
        round(config.guardIntervalUs * 1e-6 * sampleRateHz)
    )
    longTrainingValues = _PilotSequence(
        longTrainingSubcarriers.size, randomGenerator
    )
    AppendField(
        longTrainingFieldName,
        _OfdmSymbol(
            longTrainingFftLength,
            longTrainingCpLength,
            longTrainingSubcarriers,
            longTrainingValues,
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
        pilotValues = _PilotSequence(pilotSubcarriers.size, randomGenerator)
        subcarrierValues = np.r_[qamSymbols[symbolIndex], pilotValues]
        mappedSubcarriers = np.r_[dataSubcarriers, pilotSubcarriers]
        dataSymbolStarts.append(dataStart + symbolIndex * symbolLength)
        dataTimeSymbols.append(
            _OfdmSymbol(
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
