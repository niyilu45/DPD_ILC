"""Recover analysis metadata from a received project Wi-Fi waveform.

The simulated VHT, HE, and EHT generator writes a compact, LDPC-protected PHY
descriptor into the format-specific signaling field. ``ParseWifi`` locates
that descriptor, restores the transmitted configuration, regenerates the ideal
reference packet, and returns every object required by ``Analysis``. Legacy
CRC-protected descriptors remain decodable for saved waveform compatibility.

The descriptor is a project receiver aid, not a bit-exact IEEE signaling-field
codec. It is necessary because the project intentionally generates randomized
post-FEC payload symbols instead of a complete MAC/FEC/PHY protocol stack.
"""

from collections import ChainMap
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np

from .Fec import DecodeDescriptorLdpc, EncodeDescriptorLdpc

# Support both the canonical ``inc.lib`` package and the compatibility
# ``lib`` package used when callers place the ``inc`` directory on sys.path.
# Selecting by package depth avoids catching unrelated import failures.
if __package__ and "." in __package__:
    from ..utils.ConfigUtils import (
        FilterRecognizedParameters,
        RecognizedParameterView,
    )
    from ..utils.FixedPoint import FixedPoint
    from ..utils.SigProc import SigProc
    from ..utils.WifiMetadata import WifiWaveform
else:
    from utils.ConfigUtils import (
        FilterRecognizedParameters,
        RecognizedParameterView,
    )
    from utils.FixedPoint import FixedPoint
    from utils.SigProc import SigProc
    from utils.WifiMetadata import WifiWaveform


@dataclass(frozen=True)
class ParsedWifiFrame:
    """Store a parsed receive frame and its reconstructed analysis context.

    Attributes:
        receivedSignal: Packet-aligned received samples with leading and
            trailing capture samples removed.
        referenceSignal: Regenerated ideal packet samples.
        waveform: Regenerated ``WifiWaveform`` metadata used by ``Analysis``.
        packetStartSample: Detected packet start in the original capture.
        parseConfidence: Normalized signaling-field pilot or magic correlation.
        detectedParameters: Descriptor values recovered from the frame.
    """

    receivedSignal: np.ndarray
    referenceSignal: np.ndarray
    waveform: WifiWaveform
    packetStartSample: int
    parseConfidence: float
    detectedParameters: Mapping[str, object]


def IntegerToBits(integerValue: int, bitWidth: int) -> np.ndarray:
    """Convert a nonnegative integer to a fixed-width MSB-first bit vector.

    Processing details:
        Algorithm: Validate the integer range, shift from the most-significant
        position to the least-significant position, and retain one binary value
        per output element.

    Args:
        integerValue: Nonnegative value represented by the output bits.
        bitWidth: Positive number of output bits.

    Returns:
        result: Unsigned byte vector containing zeros and ones.
    """

    if (
        not isinstance(bitWidth, int)
        or isinstance(bitWidth, bool)
        or bitWidth < 1
    ):
        raise ValueError("bitWidth must be a positive integer")
    if (
        not isinstance(integerValue, int)
        or isinstance(integerValue, bool)
        or integerValue < 0
        or integerValue >= (1 << bitWidth)
    ):
        raise ValueError("integerValue does not fit the requested bitWidth")
    bitShifts = np.arange(bitWidth - 1, -1, -1, dtype=np.int64)
    return ((integerValue >> bitShifts) & 1).astype(np.uint8)


def BitsToInteger(inputBits: np.ndarray) -> int:
    """Convert an MSB-first binary vector to one Python integer.

    Processing details:
        Algorithm: Validate that every element is zero or one, then accumulate
        the value with one left shift and one logical OR per input bit.

    Args:
        inputBits: Nonempty one-dimensional binary sequence.

    Returns:
        result: Integer encoded by ``inputBits``.
    """

    bitArray = np.asarray(inputBits, dtype=np.uint8).reshape(-1)
    if bitArray.size == 0 or np.any(bitArray > 1):
        raise ValueError("inputBits must contain one or more binary values")
    integerValue = 0
    for bitValue in bitArray:
        integerValue = (integerValue << 1) | int(bitValue)
    return integerValue


def CalculateDescriptorCrc(payloadBits: np.ndarray) -> int:
    """Calculate the CRC-16-CCITT value of a descriptor bit sequence.

    Processing details:
        Algorithm: Start from hexadecimal ``FFFF`` and clock each MSB-first
        payload bit through the ``x^16+x^12+x^5+1`` polynomial. The same
        operation is used by the writer and parser, so a valid descriptor
        rejects wrong timing, sample-rate, and random waveform candidates.

    Args:
        payloadBits: Descriptor payload before the CRC field.

    Returns:
        result: Unsigned 16-bit CRC integer.
    """

    bitArray = np.asarray(payloadBits, dtype=np.uint8).reshape(-1)
    if bitArray.size == 0 or np.any(bitArray > 1):
        raise ValueError("payloadBits must contain one or more binary values")
    crcValue = 0xFFFF
    polynomial = 0x1021
    for bitValue in bitArray:
        feedbackBit = ((crcValue >> 15) & 1) ^ int(bitValue)
        crcValue = (crcValue << 1) & 0xFFFF
        if feedbackBit:
            crcValue ^= polynomial
    return crcValue


def DescriptorLdpcPhysicalLayout(
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return pilot and code-bit positions for two descriptor symbols.

    Processing details:
        Algorithm: Place seven known BPSK pilots across each 52-tone symbol,
        retain the remaining 90 positions for the LDPC codeword, and map even
        and odd codeword indices into different OFDM symbols. The distribution
        gives each symbol an independent complex-gain estimate and prevents
        one symbol-local PA error burst from corrupting a contiguous field.

    Returns:
        result: Pilot positions, pilot bits, nonpilot physical positions, and
            the codeword-index order stored at those nonpilot positions.
    """

    pilotPositionsPerSymbol = np.array(
        [0, 8, 17, 26, 34, 43, 51],
        dtype=np.int64,
    )
    pilotPositions = np.r_[
        pilotPositionsPerSymbol,
        52 + pilotPositionsPerSymbol,
    ]
    firstPilotBits = np.array(
        [0, 0, 1, 0, 1, 1, 0],
        dtype=np.uint8,
    )
    pilotBits = np.r_[firstPilotBits, 1 - firstPilotBits].astype(
        np.uint8
    )
    codePhysicalPositions = np.setdiff1d(
        np.arange(104, dtype=np.int64),
        pilotPositions,
        assume_unique=True,
    )
    codewordOrder = np.r_[
        np.arange(0, 90, 2, dtype=np.int64),
        np.arange(1, 90, 2, dtype=np.int64),
    ]
    return (
        pilotPositions,
        pilotBits,
        codePhysicalPositions,
        codewordOrder,
    )


def DecodeWifiDescriptorPayload(
    payloadBits: np.ndarray,
) -> Dict[str, object]:
    """Decode and validate the 55-bit version-two descriptor payload.

    Processing details:
        Algorithm: Slice the fixed-width PHY fields, require the version-two
        magic and version values, and apply the same format, MCS, guard
        interval, spatial, and stream-count semantic checks used by the legacy
        receiver. The random seed occupies ten bits.

    Args:
        payloadBits: Corrected systematic LDPC payload of exactly 55 bits.

    Returns:
        result: Generator-compatible descriptor parameter dictionary.
    """

    bitArray = np.asarray(payloadBits, dtype=np.uint8).reshape(-1)
    if bitArray.size != 55 or np.any(bitArray > 1):
        raise ValueError("payloadBits must contain exactly 55 binary values")
    fieldWidths = (12, 2, 2, 2, 4, 2, 12, 3, 3, 2, 1, 10)
    fieldValues = []
    bitCursor = 0
    for bitWidth in fieldWidths:
        fieldValues.append(
            BitsToInteger(bitArray[bitCursor:bitCursor + bitWidth])
        )
        bitCursor += bitWidth
    if fieldValues[0] != 0xD5B:
        raise ValueError("Wi-Fi descriptor magic word is invalid")
    if fieldValues[1] != 2:
        raise ValueError("Wi-Fi descriptor version is unsupported")
    return BuildDecodedDescriptorParameters(fieldValues)


def BuildDecodedDescriptorParameters(
    fieldValues: Sequence[int],
) -> Dict[str, object]:
    """Validate decoded descriptor fields and build generator parameters.

    Processing details:
        Algorithm: Map compact enum fields to public names and reject illegal
        MCS, guard interval, antenna, stream, or spatial-mapping combinations.
        This shared semantic layer keeps version-one CRC and version-two LDPC
        decoding behavior identical after error correction.

    Args:
        fieldValues: Twelve decoded integer fields in transmitter order.

    Returns:
        result: Valid generator-compatible parameter dictionary.
    """

    if len(fieldValues) != 12:
        raise ValueError("fieldValues must contain exactly twelve values")
    formats: Mapping[int, str] = MappingProxyType(
        {0: "VHT", 1: "HE", 2: "EHT"}
    )
    bandwidths: Mapping[int, int] = MappingProxyType(
        {0: 20, 1: 40, 2: 80, 3: 160}
    )
    guardIntervals: Mapping[int, float] = MappingProxyType(
        {0: 0.4, 1: 0.8, 2: 1.6, 3: 3.2}
    )
    mappings: Mapping[int, str] = MappingProxyType(
        {0: "direct", 1: "dft", 2: "custom"}
    )
    if fieldValues[2] not in formats or fieldValues[9] not in mappings:
        raise ValueError("Wi-Fi descriptor enum code is invalid")
    if fieldValues[3] not in bandwidths:
        raise ValueError("Wi-Fi descriptor bandwidth code is invalid")
    if fieldValues[5] not in guardIntervals:
        raise ValueError("Wi-Fi descriptor guard interval code is invalid")
    frameFormat = formats[fieldValues[2]]
    guardIntervalUs = guardIntervals[fieldValues[5]]
    mcs = fieldValues[4]
    numDataSymbols = fieldValues[6]
    numTransmitAntennas = fieldValues[7] + 1
    numSpatialStreams = fieldValues[8] + 1
    maximumMcsByFormat: Mapping[str, int] = MappingProxyType(
        {"VHT": 9, "HE": 11, "EHT": 13}
    )
    supportedGuardIntervals: Mapping[str, Tuple[float, ...]] = (
        MappingProxyType(
            {
                "VHT": (0.4, 0.8),
                "HE": (0.8, 1.6, 3.2),
                "EHT": (0.8, 1.6, 3.2),
            }
        )
    )
    if mcs > maximumMcsByFormat[frameFormat]:
        raise ValueError("Wi-Fi descriptor MCS is invalid for its format")
    if guardIntervalUs not in supportedGuardIntervals[frameFormat]:
        raise ValueError(
            "Wi-Fi descriptor guard interval is invalid for its format"
        )
    if numDataSymbols < 1:
        raise ValueError(
            "Wi-Fi descriptor data-symbol count must be positive"
        )
    if numSpatialStreams > numTransmitAntennas:
        raise ValueError(
            "Wi-Fi descriptor spatial streams exceed transmit antennas"
        )
    return {
        "frameFormat": frameFormat,
        "bandwidthMhz": bandwidths[fieldValues[3]],
        "mcs": mcs,
        "guardIntervalUs": guardIntervalUs,
        "numDataSymbols": numDataSymbols,
        "numTransmitAntennas": numTransmitAntennas,
        "numSpatialStreams": numSpatialStreams,
        "spatialMapping": mappings[fieldValues[9]],
        "cyclicShiftEnabled": bool(fieldValues[10]),
        "seed": fieldValues[11],
    }


def BuildWifiDescriptorBits(
    frameFormat: str,
    bandwidthMhz: int,
    mcs: int,
    numDataSymbols: int,
    guardIntervalUs: float,
    seed: int,
    numTransmitAntennas: int,
    numSpatialStreams: int,
    spatialMapping: str,
    cyclicShiftEnabled: bool,
) -> np.ndarray:
    """Pack one Wi-Fi simulation configuration into 104 protected bits.

    Processing details:
        Algorithm: Encode a version-two 55-bit payload with a ten-bit seed,
        apply the systematic rate-55/90 LDPC code, interleave even and odd
        codeword positions across two symbols, and insert fourteen distributed
        BPSK pilots for independent per-symbol complex-gain estimation.

    Args:
        frameFormat: Canonical VHT, HE, or EHT PHY name.
        bandwidthMhz: Nominal channel bandwidth in megahertz.
        mcs: Modulation-and-coding-scheme index.
        numDataSymbols: Number of payload OFDM symbols.
        guardIntervalUs: Payload guard interval in microseconds.
        seed: Unsigned 10-bit waveform random seed.
        numTransmitAntennas: Physical transmit-chain count from one through
            eight.
        numSpatialStreams: Spatial-stream count from one through eight.
        spatialMapping: Direct, DFT, or custom spatial mapping.
        cyclicShiftEnabled: Whether cyclic-shift diversity is enabled.

    Returns:
        result: Binary vector of length 104.
    """

    formatCodes: Mapping[str, int] = MappingProxyType(
        {"VHT": 0, "HE": 1, "EHT": 2}
    )
    bandwidthCodes: Mapping[int, int] = MappingProxyType(
        {20: 0, 40: 1, 80: 2, 160: 3}
    )
    guardIntervalCodes: Mapping[float, int] = MappingProxyType(
        {0.4: 0, 0.8: 1, 1.6: 2, 3.2: 3}
    )
    mappingCodes: Mapping[str, int] = MappingProxyType(
        {"direct": 0, "dft": 1, "custom": 2}
    )
    normalizedFormat = str(frameFormat).strip().upper()
    normalizedMapping = str(spatialMapping).strip().lower()
    if normalizedFormat not in formatCodes:
        raise ValueError("frameFormat must be VHT, HE, or EHT")
    if bandwidthMhz not in bandwidthCodes:
        raise ValueError("bandwidthMhz must be 20, 40, 80, or 160")
    if guardIntervalUs not in guardIntervalCodes:
        raise ValueError("unsupported descriptor guardIntervalUs")
    if normalizedMapping not in mappingCodes:
        raise ValueError("unsupported descriptor spatialMapping")
    if not isinstance(cyclicShiftEnabled, bool):
        raise TypeError("cyclicShiftEnabled must be boolean")

    fieldValues: Tuple[Tuple[int, int], ...] = (
        (0xD5B, 12),
        (2, 2),
        (formatCodes[normalizedFormat], 2),
        (bandwidthCodes[bandwidthMhz], 2),
        (int(mcs), 4),
        (int(round(guardIntervalCodes[guardIntervalUs])), 2),
        (int(numDataSymbols), 12),
        (int(numTransmitAntennas) - 1, 3),
        (int(numSpatialStreams) - 1, 3),
        (mappingCodes[normalizedMapping], 2),
        (int(cyclicShiftEnabled), 1),
        (int(seed), 10),
    )
    messageBits = np.concatenate(
        [
            IntegerToBits(integerValue, bitWidth)
            for integerValue, bitWidth in fieldValues
        ]
    )
    codeword = EncodeDescriptorLdpc(messageBits)
    (
        pilotPositions,
        pilotBits,
        codePhysicalPositions,
        codewordOrder,
    ) = DescriptorLdpcPhysicalLayout()
    descriptorBits = np.zeros(104, dtype=np.uint8)
    descriptorBits[pilotPositions] = pilotBits
    descriptorBits[codePhysicalPositions] = codeword[codewordOrder]
    return descriptorBits


def DecodeLegacyWifiDescriptorBits(
    descriptorBits: np.ndarray,
) -> Dict[str, object]:
    """Decode and validate one legacy version-one CRC descriptor.

    Processing details:
        Algorithm: Slice the original 32-bit-seed layout, validate its magic,
        version, CRC-16, reserved bits, and field semantics, and return the
        same public parameter dictionary as the version-two LDPC path.

    Args:
        descriptorBits: Two-symbol binary descriptor candidate.

    Returns:
        result: Decoded generator parameters.
    """

    bitArray = np.asarray(descriptorBits, dtype=np.uint8).reshape(-1)
    if bitArray.size != 104 or np.any(bitArray > 1):
        raise ValueError("descriptorBits must contain exactly 104 bits")
    fieldWidths = (12, 2, 2, 2, 4, 2, 12, 3, 3, 2, 1, 32)
    fieldValues = []
    bitCursor = 0
    for bitWidth in fieldWidths:
        fieldValues.append(
            BitsToInteger(bitArray[bitCursor : bitCursor + bitWidth])
        )
        bitCursor += bitWidth
    payloadStop = bitCursor
    receivedCrc = BitsToInteger(bitArray[payloadStop : payloadStop + 16])
    calculatedCrc = CalculateDescriptorCrc(bitArray[:payloadStop])
    reservedBits = bitArray[payloadStop + 16 :]
    if fieldValues[0] != 0xD5B:
        raise ValueError("Wi-Fi descriptor magic word is invalid")
    if fieldValues[1] != 1:
        raise ValueError("Wi-Fi descriptor version is unsupported")
    if receivedCrc != calculatedCrc:
        raise ValueError("Wi-Fi descriptor CRC is invalid")
    if np.any(reservedBits):
        raise ValueError("Wi-Fi descriptor reserved bits are nonzero")

    return BuildDecodedDescriptorParameters(fieldValues)


def DecodeWifiDescriptorLdpcValues(
    normalizedDescriptorValues: np.ndarray,
) -> Dict[str, object]:
    """Decode normalized version-two descriptor symbols with LDPC.

    Processing details:
        Algorithm: Remove the fourteen known pilot positions, invert the
        even/odd cross-symbol interleaver, run soft normalized-min-sum LDPC
        decoding, and validate the recovered 55-bit semantic payload.

    Args:
        normalizedDescriptorValues: Complex or real soft BPSK values in the
            104 physical descriptor-tone order.

    Returns:
        result: Generator-compatible descriptor parameters.
    """

    physicalValues = np.asarray(
        normalizedDescriptorValues,
        dtype=np.complex128,
    ).reshape(-1)
    if physicalValues.size != 104 or not np.all(
        np.isfinite(physicalValues)
    ):
        raise ValueError(
            "normalizedDescriptorValues must contain 104 finite values"
        )
    (
        _,
        _,
        codePhysicalPositions,
        codewordOrder,
    ) = DescriptorLdpcPhysicalLayout()
    softCodeword = np.zeros(90, dtype=float)
    softCodeword[codewordOrder] = physicalValues[
        codePhysicalPositions
    ].real
    messageBits = DecodeDescriptorLdpc(softCodeword)
    return DecodeWifiDescriptorPayload(messageBits)


def DecodeWifiDescriptorBits(descriptorBits: np.ndarray) -> Dict[str, object]:
    """Decode version-two LDPC or legacy version-one descriptor bits.

    Processing details:
        Algorithm: Interpret hard values first as the new pilot-interleaved
        LDPC format and fall back to the original sequential CRC format. The
        fallback preserves receive compatibility with waveforms generated
        before the ten-bit seed transition.

    Args:
        descriptorBits: Two-symbol binary descriptor candidate.

    Returns:
        result: Decoded generator parameters.
    """

    bitArray = np.asarray(descriptorBits, dtype=np.uint8).reshape(-1)
    if bitArray.size != 104 or np.any(bitArray > 1):
        raise ValueError("descriptorBits must contain exactly 104 bits")
    hardValues = 1.0 - 2.0 * bitArray.astype(float)
    try:
        return DecodeWifiDescriptorLdpcValues(hardValues)
    except ValueError:
        return DecodeLegacyWifiDescriptorBits(bitArray)


def DecodeWifiDescriptorBitsWithCorrection(
    descriptorBits: np.ndarray,
    bitReliabilities: np.ndarray,
    maximumCorrectedBits: int = 8,
    candidateBitCount: int = 24,
    candidateEvaluator: Optional[
        Callable[[Mapping[str, object]], float]
    ] = None,
    maximumEvaluatedCandidates: int = 32,
) -> Dict[str, object]:
    """Decode a high-confidence descriptor with bounded CRC-aided correction.

    Processing details:
        Algorithm: Restore the known magic, version, and reserved fields,
        attempt a normal CRC-valid decode, then use a meet-in-the-middle CRC
        syndrome search across the least reliable unknown payload or CRC
        decisions. Every accepted candidate must still pass the complete
        descriptor CRC and semantic validation. The bounded search corrects
        occasional PA-induced BPSK decision errors without turning CRC failure
        into an unrestricted metadata guess.

    Args:
        descriptorBits: Hard BPSK decisions for all 104 descriptor bits.
        bitReliabilities: Absolute in-phase decision distances after common
            complex-gain compensation; smaller values are less reliable.
        maximumCorrectedBits: Maximum number of uncertain decisions to flip.
        candidateBitCount: Number of least reliable unknown positions admitted
            to the bounded combination search.
        candidateEvaluator: Optional full-waveform consistency score used to
            disambiguate multiple CRC-valid correction candidates.
        maximumEvaluatedCandidates: Maximum number of lowest-cost semantic
            candidates passed to ``candidateEvaluator``.

    Returns:
        result: Generator-compatible descriptor parameters that pass all
            magic, version, reserved-bit, enum, and CRC checks.
    """

    hardBits = np.asarray(descriptorBits, dtype=np.uint8).reshape(-1)
    reliabilityValues = np.asarray(
        bitReliabilities, dtype=float
    ).reshape(-1)
    if (
        hardBits.size != 104
        or reliabilityValues.size != 104
        or np.any(hardBits > 1)
    ):
        raise ValueError(
            "descriptorBits and bitReliabilities must contain 104 values"
        )
    if (
        not np.all(np.isfinite(reliabilityValues))
        or np.any(reliabilityValues < 0.0)
    ):
        raise ValueError(
            "bitReliabilities must be finite and nonnegative"
        )
    if (
        not isinstance(maximumCorrectedBits, int)
        or isinstance(maximumCorrectedBits, bool)
        or maximumCorrectedBits < 0
    ):
        raise ValueError(
            "maximumCorrectedBits must be a nonnegative integer"
        )
    if (
        not isinstance(candidateBitCount, int)
        or isinstance(candidateBitCount, bool)
        or candidateBitCount < 1
    ):
        raise ValueError("candidateBitCount must be a positive integer")
    if candidateEvaluator is not None and not callable(candidateEvaluator):
        raise TypeError("candidateEvaluator must be callable or None")
    if (
        not isinstance(maximumEvaluatedCandidates, int)
        or isinstance(maximumEvaluatedCandidates, bool)
        or maximumEvaluatedCandidates < 1
    ):
        raise ValueError(
            "maximumEvaluatedCandidates must be a positive integer"
        )

    correctedBits = hardBits.copy()
    correctedBits[:12] = IntegerToBits(0xD5B, 12)
    correctedBits[12:14] = IntegerToBits(1, 2)
    correctedBits[93:] = 0
    try:
        return DecodeLegacyWifiDescriptorBits(correctedBits)
    except ValueError:
        pass

    searchableIndices = np.arange(14, 93, dtype=int)
    rankedIndices = searchableIndices[
        np.argsort(
            reliabilityValues[searchableIndices],
            kind="stable",
        )
    ]
    selectedIndices = rankedIndices[
        : min(candidateBitCount, rankedIndices.size)
    ]
    maximumFlipCount = min(maximumCorrectedBits, selectedIndices.size)
    payloadStop = 77
    crcStop = 93
    baseSyndrome = (
        CalculateDescriptorCrc(correctedBits[:payloadStop])
        ^ BitsToInteger(correctedBits[payloadStop:crcStop])
    )
    syndromeContributions = []
    for selectedIndex in selectedIndices:
        trialBits = correctedBits.copy()
        trialBits[selectedIndex] ^= 1
        trialSyndrome = (
            CalculateDescriptorCrc(trialBits[:payloadStop])
            ^ BitsToInteger(trialBits[payloadStop:crcStop])
        )
        syndromeContributions.append(
            int(baseSyndrome ^ trialSyndrome)
        )

    splitIndex = selectedIndices.size // 2
    indexHalves = (
        selectedIndices[:splitIndex],
        selectedIndices[splitIndex:],
    )
    syndromeHalves = (
        syndromeContributions[:splitIndex],
        syndromeContributions[splitIndex:],
    )
    subsetRecords = []
    for halfIndices, halfContributions in zip(
        indexHalves,
        syndromeHalves,
    ):
        halfRecords = []
        for subsetMask in range(1 << halfIndices.size):
            flipCount = bin(subsetMask).count("1")
            if flipCount > maximumFlipCount:
                continue
            subsetSyndrome = 0
            correctionCost = 0.0
            flippedIndices = []
            for localIndex, descriptorIndex in enumerate(halfIndices):
                if not (subsetMask & (1 << localIndex)):
                    continue
                subsetSyndrome ^= halfContributions[localIndex]
                correctionCost += float(
                    reliabilityValues[descriptorIndex]
                )
                flippedIndices.append(int(descriptorIndex))
            halfRecords.append(
                (
                    int(subsetSyndrome),
                    flipCount,
                    correctionCost,
                    tuple(flippedIndices),
                )
            )
        subsetRecords.append(halfRecords)

    rightCandidates = {}
    for (
        subsetSyndrome,
        flipCount,
        correctionCost,
        flippedIndices,
    ) in subsetRecords[1]:
        candidateKey = (subsetSyndrome, flipCount)
        previousCandidate = rightCandidates.get(candidateKey)
        if (
            previousCandidate is None
            or correctionCost < previousCandidate[0]
        ):
            rightCandidates[candidateKey] = (
                correctionCost,
                flippedIndices,
            )

    correctionCandidates = []
    for (
        leftSyndrome,
        leftFlipCount,
        leftCost,
        leftIndices,
    ) in subsetRecords[0]:
        requiredRightSyndrome = int(baseSyndrome ^ leftSyndrome)
        for rightFlipCount in range(
            0,
            maximumFlipCount - leftFlipCount + 1,
        ):
            rightCandidate = rightCandidates.get(
                (requiredRightSyndrome, rightFlipCount)
            )
            if rightCandidate is None:
                continue
            rightCost, rightIndices = rightCandidate
            allFlippedIndices = leftIndices + rightIndices
            if not allFlippedIndices:
                continue
            correctionCandidates.append(
                (
                    leftCost + rightCost,
                    len(allFlippedIndices),
                    allFlippedIndices,
                )
            )

    decodedCandidates = []
    for correctionCost, _, flippedIndices in sorted(correctionCandidates):
        candidateBits = correctedBits.copy()
        candidateBits[list(flippedIndices)] ^= 1
        try:
            decodedParameters = DecodeLegacyWifiDescriptorBits(
                candidateBits
            )
        except ValueError:
            continue
        if candidateEvaluator is None:
            return decodedParameters
        decodedCandidates.append(
            (correctionCost, decodedParameters)
        )
        if len(decodedCandidates) >= maximumEvaluatedCandidates:
            break
    if decodedCandidates and candidateEvaluator is not None:
        evaluatedCandidates = []
        for correctionCost, decodedParameters in decodedCandidates:
            consistencyScore = float(
                candidateEvaluator(decodedParameters)
            )
            if not np.isfinite(consistencyScore):
                continue
            evaluatedCandidates.append(
                (
                    consistencyScore,
                    -correctionCost,
                    decodedParameters,
                )
            )
        if evaluatedCandidates:
            return max(
                evaluatedCandidates,
                key=lambda candidate: (
                    candidate[0],
                    candidate[1],
                ),
            )[2]
    raise ValueError(
        "no legacy CRC-valid Wi-Fi descriptor was found within the "
        "correction limit"
    )


def BuildWifiDescriptorField(
    frameFormat: str,
    bandwidthMhz: int,
    mcs: int,
    numDataSymbols: int,
    guardIntervalUs: float,
    seed: int,
    numTransmitAntennas: int,
    numSpatialStreams: int,
    spatialMapping: str,
    cyclicShiftEnabled: bool,
    legacyFftLength: int,
    subchannelCount: int,
) -> np.ndarray:
    """Create the two-symbol BPSK field carrying the project descriptor.

    Processing details:
        Algorithm: Repeat the same 52 bits on every bonded 20 MHz legacy
        subchannel, place the first and second halves in consecutive OFDM
        symbols, perform an energy-normalized IFFT, and prepend a 0.8 us
        legacy cyclic prefix.

    Args:
        frameFormat: Canonical VHT, HE, or EHT PHY name.
        bandwidthMhz: Nominal channel bandwidth in megahertz.
        mcs: Modulation-and-coding-scheme index.
        numDataSymbols: Number of payload OFDM symbols.
        guardIntervalUs: Payload guard interval in microseconds.
        seed: Unsigned 10-bit waveform random seed.
        numTransmitAntennas: Physical transmit-chain count.
        numSpatialStreams: Spatial-stream count.
        spatialMapping: Direct, DFT, or custom mapping name.
        cyclicShiftEnabled: Whether cyclic-shift diversity is enabled.
        legacyFftLength: FFT length representing 3.2 microseconds.
        subchannelCount: Number of bonded 20 MHz legacy subchannels.

    Returns:
        result: Complex time-domain vector containing two 4 us symbols.
    """

    if (
        not isinstance(legacyFftLength, int)
        or isinstance(legacyFftLength, bool)
        or legacyFftLength < 64
        or legacyFftLength % 4
    ):
        raise ValueError("legacyFftLength must be a valid legacy FFT length")
    if (
        not isinstance(subchannelCount, int)
        or isinstance(subchannelCount, bool)
        or subchannelCount not in (1, 2, 4, 8)
    ):
        raise ValueError("subchannelCount must be 1, 2, 4, or 8")
    descriptorBits = BuildWifiDescriptorBits(
        frameFormat,
        bandwidthMhz,
        mcs,
        numDataSymbols,
        guardIntervalUs,
        seed,
        numTransmitAntennas,
        numSpatialStreams,
        spatialMapping,
        cyclicShiftEnabled,
    )
    subchannelCenters = (
        np.arange(subchannelCount) - (subchannelCount - 1) / 2.0
    ) * 64
    localTones = np.r_[np.arange(-26, 0), np.arange(1, 27)]
    legacyCpLength = legacyFftLength // 4
    timeSymbols = []
    for symbolIndex in range(2):
        frequencyGrid = np.zeros(legacyFftLength, dtype=np.complex128)
        symbolBits = descriptorBits[
            symbolIndex * 52 : (symbolIndex + 1) * 52
        ]
        symbolValues = 1.0 - 2.0 * symbolBits.astype(float)
        for subchannelCenter in subchannelCenters.astype(int):
            toneIndices = subchannelCenter + localTones
            frequencyGrid[np.mod(toneIndices, legacyFftLength)] = symbolValues
        usefulSamples = np.fft.ifft(frequencyGrid) * np.sqrt(legacyFftLength)
        timeSymbols.append(
            np.r_[usefulSamples[-legacyCpLength:], usefulSamples]
        )
    return np.concatenate(timeSymbols)


class ParseWifi:
    """Parse project VHT/HE/EHT receive frames without a supplied reference.

    The receiver clock is normally known by hardware. When ``sampleRateHz`` is
    left as ``None``, the parser tries common complex-baseband rates and accepts
    only a candidate whose protected descriptor passes every validation.
    """

    def __init__(
        self,
        parameters: Optional[Mapping[str, object]] = None,
        **parameterOverrides: object,
    ) -> None:
        """Initialize parser parameter layers with constructor-local defaults.

        Processing details:
            Algorithm: Keep explicit keyword overrides first, a caller-owned
            mapping second, and immutable defaults last in one ``ChainMap``.

        Args:
            parameters: Optional live external parameter mapping.
            parameterOverrides: Highest-priority parser values.

        Returns:
            result: None. The parser is validated and ready for ``Parse``.
        """

        self.defaultParameters: Mapping[str, object] = MappingProxyType(
            {
                "sampleRateHz": None,
                "sampleRateCandidatesHz": (
                    20.0e6,
                    40.0e6,
                    80.0e6,
                    160.0e6,
                    320.0e6,
                    640.0e6,
                ),
                "maximumPacketOffsetSamples": 2000,
                "minimumParseConfidence": 0.80,
                "referenceSearchSamples": 4096,
                "spatialMappingMatrix": None,
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
                "ParseWifi",
            )
        )
        recognizedOverrides = FilterRecognizedParameters(
            parameterOverrides,
            self.defaultParameters,
            "ParseWifi",
        )
        self.parameters: ChainMap[str, object] = ChainMap(
            recognizedOverrides,
            externalParameters,
            self.defaultParameters,
        )
        self.ValidateParameters()

    def GetParameters(self) -> Dict[str, object]:
        """Return a flattened snapshot of resolved parser parameters.

        Processing details:
            Algorithm: Resolve the current ``ChainMap`` without mutating
            caller-owned data or internal defaults.

        Returns:
            result: Independent parser-parameter dictionary.
        """

        return dict(self.parameters)

    def UpdateParameters(self, **parameterOverrides: object) -> None:
        """Apply validated highest-priority parser overrides transactionally.

        Processing details:
            Algorithm: Save the current local layer, apply new values, validate
            the complete map, and restore the old layer if validation fails.

        Args:
            parameterOverrides: Parser values to add or replace.

        Returns:
            result: None. Valid overrides remain active.
        """

        recognizedOverrides = FilterRecognizedParameters(
            parameterOverrides,
            self.defaultParameters,
            "ParseWifi.UpdateParameters",
        )
        previousOverrides = dict(self.parameters.maps[0])
        self.parameters.maps[0].update(recognizedOverrides)
        try:
            self.ValidateParameters()
        except (TypeError, ValueError):
            self.parameters.maps[0].clear()
            self.parameters.maps[0].update(previousOverrides)
            raise

    def ValidateParameters(self) -> None:
        """Validate parser options before searching a receive capture.

        Processing details:
            Algorithm: Check clock candidates, packet-offset ranges, confidence
            values, and optional custom spatial matrices after unknown keys
            have been warned about and filtered at the configuration boundary.

        Returns:
            result: None. Invalid parameters raise descriptive exceptions.
        """

        sampleRateHz = self.parameters["sampleRateHz"]
        if sampleRateHz is not None and (
            not isinstance(sampleRateHz, (int, float))
            or isinstance(sampleRateHz, bool)
            or not np.isfinite(sampleRateHz)
            or float(sampleRateHz) <= 0.0
        ):
            raise ValueError("sampleRateHz must be finite and positive or None")
        candidateRates = self.parameters["sampleRateCandidatesHz"]
        if not isinstance(candidateRates, Sequence) or isinstance(
            candidateRates, (str, bytes)
        ):
            raise TypeError("sampleRateCandidatesHz must be a sequence")
        numericCandidates = np.asarray(candidateRates, dtype=float).reshape(-1)
        if (
            numericCandidates.size == 0
            or not np.all(np.isfinite(numericCandidates))
            or np.any(numericCandidates <= 0.0)
        ):
            raise ValueError(
                "sampleRateCandidatesHz must contain positive finite rates"
            )
        maximumOffset = self.parameters["maximumPacketOffsetSamples"]
        if (
            not isinstance(maximumOffset, int)
            or isinstance(maximumOffset, bool)
            or maximumOffset < 0
        ):
            raise ValueError(
                "maximumPacketOffsetSamples must be a nonnegative integer"
            )
        minimumConfidence = self.parameters["minimumParseConfidence"]
        if (
            not isinstance(minimumConfidence, (int, float))
            or isinstance(minimumConfidence, bool)
            or not np.isfinite(minimumConfidence)
            or not 0.0 <= float(minimumConfidence) <= 1.0
        ):
            raise ValueError(
                "minimumParseConfidence must be between zero and one"
            )
        referenceSearchSamples = self.parameters["referenceSearchSamples"]
        if (
            not isinstance(referenceSearchSamples, int)
            or isinstance(referenceSearchSamples, bool)
            or referenceSearchSamples < 64
        ):
            raise ValueError(
                "referenceSearchSamples must be an integer of at least 64"
            )
        customMatrix = self.parameters["spatialMappingMatrix"]
        if customMatrix is not None:
            matrix = np.asarray(customMatrix, dtype=np.complex128)
            if matrix.ndim != 2 or matrix.size == 0:
                raise ValueError(
                    "spatialMappingMatrix must be a nonempty matrix or None"
                )
            if not np.all(np.isfinite(matrix)):
                raise ValueError(
                    "spatialMappingMatrix must contain finite values"
                )
        FixedPoint(cast(int, self.parameters["width"]))

    def ResolveSampleRates(self) -> Tuple[float, ...]:
        """Return explicit or automatically searched receiver sample rates.

        Processing details:
            Algorithm: Prefer a caller-supplied receiver clock; otherwise
            remove duplicate values from the ordered candidate list.

        Returns:
            result: Tuple of sample rates in hertz.
        """

        explicitRate = self.parameters["sampleRateHz"]
        if explicitRate is not None:
            return (float(cast(float, explicitRate)),)
        resolvedRates = []
        for sampleRateHz in cast(
            Sequence[float], self.parameters["sampleRateCandidatesHz"]
        ):
            numericRate = float(sampleRateHz)
            if numericRate not in resolvedRates:
                resolvedRates.append(numericRate)
        return tuple(resolvedRates)

    def ValidateReceivedSignal(
        self,
        receivedSignal: Union[np.ndarray, WifiWaveform],
    ) -> np.ndarray:
        """Extract and validate an array or WifiWaveform receive input.

        Processing details:
            Algorithm: Select ``WifiWaveform.samples`` when an object is
            supplied, otherwise use the array directly; convert to complex
            double precision, require finite samples, and preserve shape.

        Args:
            receivedSignal: Captured NumPy waveform or ``WifiWaveform`` object.

        Returns:
            result: Valid complex receive array.
        """

        rawSamples = (
            receivedSignal.samples
            if isinstance(receivedSignal, WifiWaveform)
            else receivedSignal
        )
        # The parser receives the same external representation as Analysis.
        # Quantized values are immediately dequantized to complex128 so every
        # synchronization and descriptor operation below remains floating
        # point and the public array type is identical in both modes.
        complexReceived = FixedPoint(
            cast(int, self.parameters["width"])
        ).QuantizeComplex(rawSamples)
        if complexReceived.ndim not in (1, 2):
            raise ValueError(
                "receivedSignal must be a vector or samples-by-chain matrix"
            )
        if (
            complexReceived.size == 0
            or complexReceived.shape[0] == 0
            or not np.all(np.isfinite(complexReceived))
        ):
            raise ValueError(
                "receivedSignal must contain finite complex samples"
            )
        if complexReceived.ndim == 2 and complexReceived.shape[1] < 1:
            raise ValueError("receivedSignal must contain at least one chain")
        return complexReceived

    def ScoreDescriptorCandidate(
        self,
        decodedParameters: Mapping[str, object],
        receivedSignal: np.ndarray,
        packetStartSample: int,
        sampleRateHz: float,
    ) -> float:
        """Score one protected descriptor against the complete receive packet.

        Processing details:
            Algorithm: Regenerate the candidate deterministic Wi-Fi waveform,
            align it at the proposed packet boundary, calculate normalized
            magnitude correlation independently per physical chain, average
            the chain scores, and mildly penalize candidates that explain only
            a short prefix of the available capture. The full packet score
            rejects a semantically valid but incorrect LDPC or legacy CRC
            candidate that cannot be distinguished from the descriptor alone.

        Args:
            decodedParameters: Semantically valid descriptor parameter map.
            receivedSignal: Validated receive vector or matrix.
            packetStartSample: Proposed first packet sample.
            sampleRateHz: Proposed receiver sample rate in hertz.

        Returns:
            result: Finite normalized consistency score, or negative infinity
                when the candidate cannot describe the available capture.
        """

        candidateParameters = dict(decodedParameters)
        candidateParameters["sampleRateHz"] = float(sampleRateHz)
        candidateParameters["width"] = cast(
            int, self.parameters["width"]
        )
        if candidateParameters.get("spatialMapping") == "custom":
            customMatrix = self.parameters["spatialMappingMatrix"]
            if customMatrix is None:
                return float("-inf")
            candidateParameters["spatialMappingMatrix"] = np.asarray(
                customMatrix,
                dtype=np.complex128,
            ).copy()

        # Import locally because WaveGenWifi reuses the descriptor writer from
        # this module during normal packet construction.
        from .WaveGenWifi import WaveGenWifi

        try:
            candidateWaveform = WaveGenWifi(
                parameters=candidateParameters
            ).Generate()
        except (TypeError, ValueError):
            return float("-inf")
        referenceArray = np.asarray(
            candidateWaveform.samples,
            dtype=np.complex128,
        )
        if packetStartSample < 0:
            return float("-inf")
        availableSampleCount = (
            receivedSignal.shape[0] - packetStartSample
        )
        scoredSampleCount = min(
            referenceArray.shape[0],
            availableSampleCount,
        )
        if scoredSampleCount <= 0:
            return float("-inf")
        packetStopSample = packetStartSample + scoredSampleCount
        receiveSegment = receivedSignal[
            packetStartSample:packetStopSample
        ]
        referenceSegment = referenceArray[:scoredSampleCount]
        referenceMatrix = (
            referenceSegment.reshape(-1, 1)
            if referenceSegment.ndim == 1
            else referenceSegment
        )
        receiveMatrix = (
            receiveSegment.reshape(-1, 1)
            if receiveSegment.ndim == 1
            else receiveSegment
        )
        if referenceMatrix.shape != receiveMatrix.shape:
            return float("-inf")
        chainScores = []
        for chainIndex in range(referenceMatrix.shape[1]):
            referenceColumn = referenceMatrix[:, chainIndex]
            receiveColumn = receiveMatrix[:, chainIndex]
            denominator = np.sqrt(
                np.vdot(referenceColumn, referenceColumn).real
                * np.vdot(receiveColumn, receiveColumn).real
            )
            if denominator <= np.finfo(float).tiny:
                return float("-inf")
            chainScores.append(
                float(
                    np.abs(
                        np.vdot(referenceColumn, receiveColumn)
                    )
                    / denominator
                )
            )
        completenessWeight = np.sqrt(
            scoredSampleCount / referenceArray.shape[0]
        )
        return float(np.mean(chainScores) * completenessWeight)

    def DecodeDescriptorAt(
        self,
        receivedSignal: np.ndarray,
        packetStartSample: int,
        sampleRateHz: float,
        descriptorOffsetSymbols: int,
        evaluateCorrectionCandidates: bool = True,
    ) -> Tuple[Dict[str, object], float]:
        """Decode one descriptor at a proposed packet start and sample rate.

        Processing details:
            Algorithm: Remove each legacy cyclic prefix, FFT two signaling
            symbols from the first receive chain, recover one repeated copy per
            20 MHz subchannel, and first try the version-two distributed pilots
            with independent per-symbol gain estimates and soft LDPC decoding.
            Fall back to version-one magic-word gain estimation and bounded
            CRC correction for previously generated waveforms.

        Args:
            receivedSignal: Validated receive vector or matrix.
            packetStartSample: Proposed first packet sample.
            sampleRateHz: Proposed receiver sample rate in hertz.
            descriptorOffsetSymbols: Five for VHT or six for HE/EHT.
            evaluateCorrectionCandidates: Whether valid protected candidates
                are checked with full-packet correlation.

        Returns:
            result: Pair containing decoded parameters and correlation score.
        """

        legacyFftLengthFloat = 3.2e-6 * sampleRateHz
        if not np.isclose(
            legacyFftLengthFloat,
            round(legacyFftLengthFloat),
            rtol=0.0,
            atol=1.0e-8,
        ):
            raise ValueError("sampleRateHz does not produce a legacy FFT grid")
        legacyFftLength = int(round(legacyFftLengthFloat))
        legacyCpLength = legacyFftLength // 4
        legacySymbolLength = legacyFftLength + legacyCpLength
        descriptorStart = (
            int(packetStartSample)
            + descriptorOffsetSymbols * legacySymbolLength
        )
        descriptorStop = descriptorStart + 2 * legacySymbolLength
        if descriptorStart < 0 or descriptorStop > receivedSignal.shape[0]:
            raise ValueError("descriptor candidate exceeds receive capture")
        firstChain = (
            receivedSignal
            if receivedSignal.ndim == 1
            else receivedSignal[:, 0]
        )
        symbolSpectra = []
        for symbolIndex in range(2):
            usefulStart = (
                descriptorStart
                + symbolIndex * legacySymbolLength
                + legacyCpLength
            )
            usefulStop = usefulStart + legacyFftLength
            usefulSamples = firstChain[usefulStart:usefulStop]
            symbolSpectra.append(
                np.fft.fft(usefulSamples) / np.sqrt(legacyFftLength)
            )

        descriptorCopies = []
        correlationScores = []
        expectedMagicBits = IntegerToBits(0xD5B, 12)
        expectedMagicSymbols = (
            1.0 - 2.0 * expectedMagicBits.astype(float)
        ).astype(np.complex128)
        localTones = np.r_[np.arange(-26, 0), np.arange(1, 27)]
        maximumSubchannels = max(1, legacyFftLength // 64)
        for subchannelCount in (1, 2, 4, 8):
            if subchannelCount > maximumSubchannels:
                continue
            subchannelCenters = (
                np.arange(subchannelCount)
                - (subchannelCount - 1) / 2.0
            ) * 64
            for subchannelCenter in subchannelCenters.astype(int):
                toneIndices = subchannelCenter + localTones
                if np.any(np.abs(toneIndices) > legacyFftLength // 2):
                    continue
                receivedValues = np.concatenate(
                    [
                        spectrum[np.mod(toneIndices, legacyFftLength)]
                        for spectrum in symbolSpectra
                    ]
                )

                # Version two distributes seven known pilots over each OFDM
                # symbol. Separate gain estimates prevent PA memory or burst
                # distortion in one symbol from rotating the other symbol.
                (
                    pilotPositions,
                    pilotBits,
                    _,
                    _,
                ) = DescriptorLdpcPhysicalLayout()
                ldpcNormalizedValues = np.zeros(
                    104,
                    dtype=np.complex128,
                )
                pilotCorrelations = []
                ldpcGainIsValid = True
                for symbolIndex in range(2):
                    symbolStart = 52 * symbolIndex
                    symbolStop = symbolStart + 52
                    symbolPilotMask = (
                        (pilotPositions >= symbolStart)
                        & (pilotPositions < symbolStop)
                    )
                    symbolPilotPositions = pilotPositions[
                        symbolPilotMask
                    ]
                    expectedPilotSymbols = (
                        1.0
                        - 2.0
                        * pilotBits[symbolPilotMask].astype(float)
                    ).astype(np.complex128)
                    receivedPilots = receivedValues[
                        symbolPilotPositions
                    ]
                    pilotEnergy = float(
                        np.vdot(receivedPilots, receivedPilots).real
                    )
                    expectedPilotEnergy = float(
                        np.vdot(
                            expectedPilotSymbols,
                            expectedPilotSymbols,
                        ).real
                    )
                    if (
                        pilotEnergy <= np.finfo(float).tiny
                        or expectedPilotEnergy <= np.finfo(float).tiny
                    ):
                        ldpcGainIsValid = False
                        break
                    symbolGain = np.vdot(
                        expectedPilotSymbols,
                        receivedPilots,
                    ) / expectedPilotEnergy
                    if np.abs(symbolGain) <= np.finfo(float).tiny:
                        ldpcGainIsValid = False
                        break
                    ldpcNormalizedValues[
                        symbolStart:symbolStop
                    ] = (
                        receivedValues[symbolStart:symbolStop]
                        / symbolGain
                    )
                    pilotCorrelations.append(
                        float(
                            np.abs(
                                np.vdot(
                                    expectedPilotSymbols,
                                    receivedPilots,
                                )
                            )
                            / np.sqrt(
                                expectedPilotEnergy * pilotEnergy
                            )
                        )
                    )
                if ldpcGainIsValid and pilotCorrelations:
                    ldpcCorrelation = float(
                        np.mean(pilotCorrelations)
                    )
                    if ldpcCorrelation >= float(
                        self.parameters["minimumParseConfidence"]
                    ):
                        try:
                            ldpcParameters = (
                                DecodeWifiDescriptorLdpcValues(
                                    ldpcNormalizedValues
                                )
                            )
                        except ValueError:
                            ldpcParameters = None
                        if ldpcParameters is not None:
                            ldpcCandidateScore = ldpcCorrelation
                            if evaluateCorrectionCandidates:
                                fullPacketScore = (
                                    self.ScoreDescriptorCandidate(
                                        ldpcParameters,
                                        receivedSignal,
                                        packetStartSample,
                                        sampleRateHz,
                                    )
                                )
                                if np.isfinite(fullPacketScore):
                                    ldpcCandidateScore = min(
                                        ldpcCandidateScore,
                                        max(fullPacketScore, 0.0),
                                    )
                                else:
                                    ldpcParameters = None
                            if ldpcParameters is not None:
                                descriptorCopies.append(ldpcParameters)
                                correlationScores.append(
                                    ldpcCandidateScore
                                )

                # Version-one fallback uses the original sequential magic,
                # CRC, and 32-bit seed layout.
                receivedMagic = receivedValues[:12]
                magicEnergy = float(np.vdot(receivedMagic, receivedMagic).real)
                if magicEnergy <= np.finfo(float).tiny:
                    continue
                complexGain = np.vdot(
                    expectedMagicSymbols, receivedMagic
                ) / np.vdot(expectedMagicSymbols, expectedMagicSymbols)
                if np.abs(complexGain) <= np.finfo(float).tiny:
                    continue
                normalizedValues = receivedValues / complexGain
                decidedBits = (normalizedValues.real < 0.0).astype(np.uint8)
                correlation = float(
                    np.abs(
                        np.vdot(expectedMagicSymbols, receivedMagic)
                    )
                    / np.sqrt(
                        np.vdot(
                            expectedMagicSymbols, expectedMagicSymbols
                        ).real
                        * magicEnergy
                    )
                )
                if correlation < float(
                    self.parameters["minimumParseConfidence"]
                ):
                    continue
                try:
                    descriptorParameters = (
                        DecodeWifiDescriptorBitsWithCorrection(
                            decidedBits,
                            np.abs(normalizedValues.real),
                            candidateEvaluator=(
                                None
                                if not evaluateCorrectionCandidates
                                else (
                                    lambda candidateParameters: (
                                        self.ScoreDescriptorCandidate(
                                            candidateParameters,
                                            receivedSignal,
                                            packetStartSample,
                                            sampleRateHz,
                                        )
                                    )
                                )
                            ),
                        )
                    )
                except ValueError:
                    continue
                descriptorCopies.append(descriptorParameters)
                correlationScores.append(correlation)
        if not descriptorCopies:
            raise ValueError(
                "no LDPC-valid or legacy CRC-valid Wi-Fi descriptor was found"
            )
        bestIndex = int(np.argmax(correlationScores))
        return descriptorCopies[bestIndex], correlationScores[bestIndex]

    def FindDescriptor(
        self,
        receivedSignal: np.ndarray,
        preferredSampleRateHz: Optional[float] = None,
    ) -> Tuple[Dict[str, object], int, float, float]:
        """Search packet start, PHY family offset, and receiver sample rate.

        Processing details:
            Algorithm: Test exact capture start first, then scan the configured
            leading-offset range. For each receiver-clock candidate, try the
            VHT and HE/EHT signaling positions and accept only protected
            descriptors whose format-dependent offset, antenna count, and
            confidence agree.

        Args:
            receivedSignal: Validated receive waveform.
            preferredSampleRateHz: Optional metadata-derived receiver clock
                tested before explicit or default parser candidates.

        Returns:
            result: Decoded parameters, packet start, sample rate, confidence.
        """

        maximumOffset = min(
            int(self.parameters["maximumPacketOffsetSamples"]),
            max(receivedSignal.shape[0] - 1, 0),
        )
        packetStartCandidates = range(maximumOffset + 1)
        receiveChainCount = (
            1 if receivedSignal.ndim == 1 else receivedSignal.shape[1]
        )
        searchRates = list(self.ResolveSampleRates())
        if preferredSampleRateHz is not None:
            if (
                not isinstance(preferredSampleRateHz, (int, float))
                or isinstance(preferredSampleRateHz, bool)
                or not np.isfinite(preferredSampleRateHz)
                or float(preferredSampleRateHz) <= 0.0
            ):
                raise ValueError(
                    "preferredSampleRateHz must be finite and positive or None"
                )
            preferredRate = float(preferredSampleRateHz)
            searchRates = [
                preferredRate,
                *[
                    sampleRateHz
                    for sampleRateHz in searchRates
                    if sampleRateHz != preferredRate
                ],
            ]
        for packetStartSample in packetStartCandidates:
            for sampleRateHz in searchRates:
                legacySymbolLength = int(round(4.0e-6 * sampleRateHz))
                if legacySymbolLength < 80:
                    continue
                for descriptorOffsetSymbols in (5, 6):
                    try:
                        decodedParameters, confidence = (
                            self.DecodeDescriptorAt(
                                receivedSignal,
                                packetStartSample,
                                sampleRateHz,
                                descriptorOffsetSymbols,
                                False,
                            )
                        )
                    except ValueError:
                        continue
                    expectedOffset = (
                        5
                        if decodedParameters["frameFormat"] == "VHT"
                        else 6
                    )
                    if descriptorOffsetSymbols != expectedOffset:
                        continue
                    if (
                        int(decodedParameters["numTransmitAntennas"])
                        != receiveChainCount
                    ):
                        continue
                    if sampleRateHz < (
                        float(decodedParameters["bandwidthMhz"]) * 1.0e6
                    ):
                        continue
                    if confidence < float(
                        self.parameters["minimumParseConfidence"]
                    ):
                        continue
                    # A cyclic prefix can make a descriptor decodable a few
                    # samples before or after the true packet boundary. Refine
                    # around the first protected-descriptor point and choose
                    # the maximum pilot or magic correlation so the returned
                    # packet crop is aligned to the original transmit grid.
                    legacyCpLength = int(
                        round(0.8e-6 * sampleRateHz)
                    )
                    refinementStart = max(
                        0, packetStartSample - legacyCpLength
                    )
                    refinementStop = min(
                        maximumOffset,
                        packetStartSample + legacyCpLength,
                    )
                    refinedCandidates = []
                    for refinedStart in range(
                        refinementStart, refinementStop + 1
                    ):
                        try:
                            refinedParameters, refinedConfidence = (
                                self.DecodeDescriptorAt(
                                    receivedSignal,
                                    refinedStart,
                                    sampleRateHz,
                                    descriptorOffsetSymbols,
                                    False,
                                )
                            )
                        except ValueError:
                            continue
                        refinedExpectedOffset = (
                            5
                            if refinedParameters["frameFormat"] == "VHT"
                            else 6
                        )
                        if descriptorOffsetSymbols != refinedExpectedOffset:
                            continue
                        if (
                            int(
                                refinedParameters[
                                    "numTransmitAntennas"
                                ]
                            )
                            != receiveChainCount
                        ):
                            continue
                        if sampleRateHz < (
                            float(refinedParameters["bandwidthMhz"])
                            * 1.0e6
                        ):
                            continue
                        if refinedConfidence < float(
                            self.parameters["minimumParseConfidence"]
                        ):
                            continue
                        refinedCandidates.append(
                            (
                                refinedConfidence,
                                refinedStart,
                                refinedParameters,
                            )
                        )
                    if not refinedCandidates:
                        continue
                    (
                        bestConfidence,
                        bestPacketStart,
                        _,
                    ) = max(
                        refinedCandidates,
                        key=lambda candidate: candidate[0],
                    )
                    try:
                        bestParameters, bestConfidence = (
                            self.DecodeDescriptorAt(
                                receivedSignal,
                                bestPacketStart,
                                sampleRateHz,
                                descriptorOffsetSymbols,
                                True,
                            )
                        )
                    except ValueError:
                        continue
                    bestExpectedOffset = (
                        5
                        if bestParameters["frameFormat"] == "VHT"
                        else 6
                    )
                    if descriptorOffsetSymbols != bestExpectedOffset:
                        continue
                    if (
                        int(bestParameters["numTransmitAntennas"])
                        != receiveChainCount
                    ):
                        continue
                    if sampleRateHz < (
                        float(bestParameters["bandwidthMhz"]) * 1.0e6
                    ):
                        continue
                    return (
                        bestParameters,
                        bestPacketStart,
                        sampleRateHz,
                        bestConfidence,
                    )
        raise ValueError(
            "unable to parse the Wi-Fi frame descriptor; verify that the "
            "capture was generated by this project, includes the signaling "
            "field, and uses one of the configured sample rates; if severe "
            "PA or channel distortion corrupts that field, pass the original "
            "NumPy transmit waveform or WifiWaveform as transmittedSignal"
        )

    def EstimatePacketStartFromReference(
        self,
        receivedSignal: np.ndarray,
        transmittedSignal: np.ndarray,
    ) -> Tuple[int, float]:
        """Estimate packet start using an optional known transmit waveform.

        Processing details:
            Algorithm: Delegate to ``EstimateSignalOverlap`` so the public
            compatibility interface still returns only the receive-side packet
            start and confidence while accepting unequal waveform lengths.

        Args:
            receivedSignal: Validated receive vector or matrix.
            transmittedSignal: Known transmitted NumPy waveform without any
                required metadata or configuration object.

        Returns:
            result: Detected packet-start sample and normalized confidence.
        """

        (
            receiveStartSample,
            _,
            _,
            confidence,
        ) = self.EstimateSignalOverlap(
            receivedSignal,
            transmittedSignal,
        )
        return receiveStartSample, confidence

    def EstimateSignalOverlap(
        self,
        receivedSignal: np.ndarray,
        transmittedSignal: np.ndarray,
    ) -> Tuple[int, int, int, float]:
        """Estimate the best valid overlap between transmit and receive data.

        Processing details:
            Algorithm: Remove only negligible outer zero padding from the
            transmit reference, enumerate signed lags that retain a useful
            overlap, and calculate energy-normalized correlation independently
            on every physical chain. A negative lag represents a receive
            waveform that starts inside a longer or cropped transmit waveform.
            The comparison therefore never assumes that receive length is at
            least transmit length. The configurable reference-search length
            bounds the work per candidate, while the packet-offset limit
            constrains only leading samples on the receive side.

        Args:
            receivedSignal: Validated receive vector or samples-by-chains
                matrix. It may be shorter or longer than ``transmittedSignal``.
            transmittedSignal: Known transmit vector or matrix. Leading and
                trailing zero padding and transmit-side cropping are allowed.

        Returns:
            result: Receive start, transmit start, full available overlap
                length, and normalized multi-chain correlation confidence.
        """

        overlapResult = SigProc.EstimateSignalOverlap(
            receivedSignal,
            transmittedSignal,
            int(self.parameters["maximumPacketOffsetSamples"]),
            int(self.parameters["referenceSearchSamples"]),
            float(self.parameters["minimumParseConfidence"]),
        )
        return (
            overlapResult.receivedStartSample,
            overlapResult.referenceStartSample,
            overlapResult.overlapLength,
            overlapResult.confidence,
        )

    def BuildDetectedParameters(
        self, transmittedWaveform: WifiWaveform
    ) -> Dict[str, object]:
        """Convert a supplied WifiWaveform object to parser result fields.

        Processing details:
            Algorithm: Read neutral metadata directly when the optional transmit
            input is a ``WifiWaveform``. This bypasses signaling-field decoding
            without exposing any type-selection step to the caller.

        Args:
            transmittedWaveform: Known transmit waveform object.

        Returns:
            result: Analysis-relevant detected parameter dictionary.
        """

        if not isinstance(transmittedWaveform, WifiWaveform):
            raise TypeError("transmittedWaveform must be a WifiWaveform")
        return {
            "frameFormat": transmittedWaveform.frameFormat,
            "bandwidthMhz": int(
                round(transmittedWaveform.bandwidthHz / 1.0e6)
            ),
            "mcs": transmittedWaveform.mcsInfo.index,
            "guardIntervalUs": float(
                transmittedWaveform.cpLength
                / transmittedWaveform.sampleRateHz
                * 1.0e6
            ),
            "numDataSymbols": int(
                np.asarray(
                    transmittedWaveform.dataSymbolStarts
                ).size
            ),
            "numTransmitAntennas": (
                transmittedWaveform.numTransmitAntennas
            ),
            "numSpatialStreams": transmittedWaveform.numSpatialStreams,
            "spatialMapping": transmittedWaveform.spatialMapping,
            "cyclicShiftEnabled": transmittedWaveform.cyclicShiftEnabled,
            "seed": transmittedWaveform.seed,
            "sampleRateHz": transmittedWaveform.sampleRateHz,
        }

    def Parse(
        self,
        receivedSignal: Union[np.ndarray, WifiWaveform],
        transmittedSignal: Optional[
            Union[np.ndarray, WifiWaveform]
        ] = None,
    ) -> ParsedWifiFrame:
        """Parse one receive capture and reconstruct the Analysis reference.

        Processing details:
            Algorithm: Decode metadata from the optional transmit waveform when
            available because it normally has higher SNR, otherwise decode the
            receive waveform. Regenerate neutral metadata from that descriptor,
            then use normalized multi-chain transmit/receive correlation for
            the most accurate receive boundary and retain the supplied NumPy
            transmit samples as the exact reference. Without a transmit input,
            regenerate the deterministic reference from descriptor parameters.

        Args:
            receivedSignal: SISO or MIMO receive input supplied as a NumPy
                array or ``WifiWaveform``. Type selection is internal.
            transmittedSignal: Optional known transmit input. A ``WifiWaveform``
                supplies samples and metadata directly; a NumPy array supplies
                samples only and is parsed automatically.

        Returns:
            result: Parsed packet, ideal reference, metadata, and diagnostics.
        """

        receivedWaveform = (
            receivedSignal
            if isinstance(receivedSignal, WifiWaveform)
            else None
        )
        complexReceived = self.ValidateReceivedSignal(receivedSignal)
        if isinstance(transmittedSignal, WifiWaveform):
            referenceSignal = self.ValidateReceivedSignal(
                transmittedSignal.samples
            )
            (
                packetStartSample,
                _,
                overlapLength,
                confidence,
            ) = self.EstimateSignalOverlap(
                complexReceived,
                referenceSignal,
            )
            packetStopSample = packetStartSample + overlapLength
            alignedReceived = complexReceived[
                packetStartSample:packetStopSample
            ].copy()
            return ParsedWifiFrame(
                receivedSignal=alignedReceived,
                referenceSignal=referenceSignal.copy(),
                waveform=transmittedSignal,
                packetStartSample=packetStartSample,
                parseConfidence=confidence,
                detectedParameters=MappingProxyType(
                    self.BuildDetectedParameters(transmittedSignal)
                ),
            )
        validatedTransmit = (
            None
            if transmittedSignal is None
            else self.ValidateReceivedSignal(
                cast(np.ndarray, transmittedSignal)
            )
        )
        descriptorSource = (
            complexReceived
            if validatedTransmit is None
            else validatedTransmit
        )
        (
            detectedParameters,
            descriptorPacketStart,
            sampleRateHz,
            descriptorConfidence,
        ) = self.FindDescriptor(
            descriptorSource,
            preferredSampleRateHz=(
                None
                if receivedWaveform is None
                else receivedWaveform.sampleRateHz
            ),
        )
        generatorParameters = dict(detectedParameters)
        generatorParameters["sampleRateHz"] = sampleRateHz
        generatorParameters["width"] = cast(
            int, self.parameters["width"]
        )
        if generatorParameters["spatialMapping"] == "custom":
            customMatrix = (
                receivedWaveform.spatialMappingMatrix
                if receivedWaveform is not None
                else self.parameters["spatialMappingMatrix"]
            )
            if customMatrix is None:
                raise ValueError(
                    "a custom-mapped frame requires spatialMappingMatrix in "
                    "ParseWifi parameters"
                )
            generatorParameters["spatialMappingMatrix"] = np.asarray(
                customMatrix, dtype=np.complex128
            ).copy()

        # Import locally to keep the receive-parser module free of a module-load
        # cycle while allowing the transmitter to reuse the descriptor writer.
        from .WaveGenWifi import WaveGenWifi

        referenceWaveform = WaveGenWifi(
            parameters=generatorParameters
        ).Generate()
        regeneratedReference = np.asarray(
            referenceWaveform.samples, dtype=np.complex128
        )
        if validatedTransmit is None:
            referenceSignal = regeneratedReference.copy()
            packetStartSample = descriptorPacketStart
            confidence = descriptorConfidence
        else:
            transmitPacketStop = (
                descriptorPacketStart + regeneratedReference.shape[0]
            )
            availableTransmitStop = min(
                transmitPacketStop,
                validatedTransmit.shape[0],
            )
            availableTransmit = validatedTransmit[
                descriptorPacketStart:availableTransmitStop
            ]
            referenceSignal = regeneratedReference.copy()
            referenceSignal[
                :availableTransmit.shape[0]
            ] = availableTransmit
            (
                packetStartSample,
                _,
                _,
                correlationConfidence,
            ) = self.EstimateSignalOverlap(
                complexReceived,
                referenceSignal,
            )
            confidence = min(
                descriptorConfidence, correlationConfidence
            )
        availableReceiveLength = (
            complexReceived.shape[0] - packetStartSample
        )
        alignedSampleCount = min(
            referenceSignal.shape[0],
            availableReceiveLength,
        )
        if alignedSampleCount <= 0:
            raise ValueError(
                "decoded Wi-Fi packet has no samples inside receivedSignal"
            )
        packetStopSample = packetStartSample + alignedSampleCount
        alignedReceived = complexReceived[
            packetStartSample:packetStopSample
        ].copy()
        detectedOutput = dict(detectedParameters)
        detectedOutput["sampleRateHz"] = sampleRateHz
        return ParsedWifiFrame(
            receivedSignal=alignedReceived,
            referenceSignal=referenceSignal,
            waveform=referenceWaveform,
            packetStartSample=packetStartSample,
            parseConfidence=confidence,
            detectedParameters=MappingProxyType(detectedOutput),
        )
