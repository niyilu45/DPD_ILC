"""Recover analysis metadata from a received project Wi-Fi waveform.

The simulated VHT, HE, and EHT generator writes a compact, CRC-protected PHY
descriptor into the format-specific signaling field. ``ParseWifi`` locates
that descriptor, restores the transmitted configuration, regenerates the ideal
reference packet, and returns every object required by ``Analysis``.

The descriptor is a project receiver aid, not a bit-exact IEEE signaling-field
codec. It is necessary because the project intentionally generates randomized
post-FEC payload symbols instead of a complete MAC/FEC/PHY protocol stack.
"""

from collections import ChainMap
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union, cast

import numpy as np

from .ConfigUtils import (
    FilterRecognizedParameters,
    RecognizedParameterView,
)
from .WifiMetadata import WifiWaveform


@dataclass(frozen=True)
class ParsedWifiFrame:
    """Store a parsed receive frame and its reconstructed analysis context.

    Attributes:
        receivedSignal: Packet-aligned received samples with leading and
            trailing capture samples removed.
        referenceSignal: Regenerated ideal packet samples.
        waveform: Regenerated ``WifiWaveform`` metadata used by ``Analysis``.
        packetStartSample: Detected packet start in the original capture.
        parseConfidence: Normalized signaling-field magic-word correlation.
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
        Algorithm: Encode a magic word, descriptor version, PHY parameters,
        spatial configuration, and random seed into fixed-width fields; append
        CRC-16; then zero-pad the result to two 52-tone legacy OFDM symbols.

    Args:
        frameFormat: Canonical VHT, HE, or EHT PHY name.
        bandwidthMhz: Nominal channel bandwidth in megahertz.
        mcs: Modulation-and-coding-scheme index.
        numDataSymbols: Number of payload OFDM symbols.
        guardIntervalUs: Payload guard interval in microseconds.
        seed: Unsigned 32-bit waveform random seed.
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
        (1, 2),
        (formatCodes[normalizedFormat], 2),
        (bandwidthCodes[bandwidthMhz], 2),
        (int(mcs), 4),
        (int(round(guardIntervalCodes[guardIntervalUs])), 2),
        (int(numDataSymbols), 12),
        (int(numTransmitAntennas) - 1, 3),
        (int(numSpatialStreams) - 1, 3),
        (mappingCodes[normalizedMapping], 2),
        (int(cyclicShiftEnabled), 1),
        (int(seed), 32),
    )
    payloadBits = np.concatenate(
        [
            IntegerToBits(integerValue, bitWidth)
            for integerValue, bitWidth in fieldValues
        ]
    )
    crcBits = IntegerToBits(CalculateDescriptorCrc(payloadBits), 16)
    descriptorBits = np.r_[payloadBits, crcBits]
    if descriptorBits.size > 104:
        raise RuntimeError("internal Wi-Fi descriptor width overflow")
    return np.pad(
        descriptorBits,
        (0, 104 - descriptorBits.size),
        mode="constant",
    ).astype(np.uint8)


def DecodeWifiDescriptorBits(descriptorBits: np.ndarray) -> Dict[str, object]:
    """Decode and validate one 104-bit project Wi-Fi descriptor.

    Processing details:
        Algorithm: Slice fields in the same order used by the transmitter,
        validate the magic word, version, CRC, reserved bits, and enum codes,
        then return generator-compatible parameter names.

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
    return {
        "frameFormat": formats[fieldValues[2]],
        "bandwidthMhz": bandwidths[fieldValues[3]],
        "mcs": fieldValues[4],
        "guardIntervalUs": guardIntervals[fieldValues[5]],
        "numDataSymbols": fieldValues[6],
        "numTransmitAntennas": fieldValues[7] + 1,
        "numSpatialStreams": fieldValues[8] + 1,
        "spatialMapping": mappings[fieldValues[9]],
        "cyclicShiftEnabled": bool(fieldValues[10]),
        "seed": fieldValues[11],
    }


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
        seed: Unsigned 32-bit waveform random seed.
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
                "maximumPacketOffsetSamples": 4096,
                "minimumParseConfidence": 0.80,
                "referenceSearchSamples": 4096,
                "spatialMappingMatrix": None,
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
        complexReceived = np.asarray(rawSamples, dtype=np.complex128)
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

    def DecodeDescriptorAt(
        self,
        receivedSignal: np.ndarray,
        packetStartSample: int,
        sampleRateHz: float,
        descriptorOffsetSymbols: int,
    ) -> Tuple[Dict[str, object], float]:
        """Decode one descriptor at a proposed packet start and sample rate.

        Processing details:
            Algorithm: Remove each legacy cyclic prefix, FFT two signaling
            symbols from the first receive chain, recover one repeated copy per
            20 MHz subchannel, use the known magic word to remove common phase,
            hard-decision the BPSK values, and accept only CRC-valid metadata.

        Args:
            receivedSignal: Validated receive vector or matrix.
            packetStartSample: Proposed first packet sample.
            sampleRateHz: Proposed receiver sample rate in hertz.
            descriptorOffsetSymbols: Five for VHT or six for HE/EHT.

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
                try:
                    descriptorParameters = DecodeWifiDescriptorBits(
                        decidedBits
                    )
                except ValueError:
                    continue
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
                descriptorCopies.append(descriptorParameters)
                correlationScores.append(correlation)
        if not descriptorCopies:
            raise ValueError("no CRC-valid Wi-Fi descriptor was found")
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
            VHT and HE/EHT signaling positions and accept only descriptors whose
            CRC, format-dependent offset, antenna count, and confidence agree.

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
                    if confidence < float(
                        self.parameters["minimumParseConfidence"]
                    ):
                        continue
                    # A cyclic prefix can make a descriptor decodable a few
                    # samples before or after the true packet boundary. Refine
                    # around the first CRC-valid point and choose the maximum
                    # magic-word correlation so the returned packet crop is
                    # aligned to the original transmit sample grid.
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
                                )
                            )
                        except ValueError:
                            continue
                        if refinedParameters != decodedParameters:
                            continue
                        refinedCandidates.append(
                            (
                                refinedConfidence,
                                refinedStart,
                            )
                        )
                    if not refinedCandidates:
                        continue
                    (
                        bestConfidence,
                        bestPacketStart,
                    ) = max(refinedCandidates)
                    return (
                        decodedParameters,
                        bestPacketStart,
                        sampleRateHz,
                        bestConfidence,
                    )
        raise ValueError(
            "unable to parse the Wi-Fi frame descriptor; verify that the "
            "capture was generated by this project, includes the signaling "
            "field, and uses one of the configured sample rates"
        )

    def EstimatePacketStartFromReference(
        self,
        receivedSignal: np.ndarray,
        transmittedSignal: np.ndarray,
    ) -> Tuple[int, float]:
        """Estimate packet start using an optional known transmit waveform.

        Processing details:
            Algorithm: Correlate a configurable leading reference segment with
            every allowed capture offset, normalize each physical chain by its
            own reference and receive energy, combine chain scores without
            phase cancellation, and select the highest normalized correlation.

        Args:
            receivedSignal: Validated receive vector or matrix.
            transmittedSignal: Known transmitted NumPy waveform without any
                required metadata or configuration object.

        Returns:
            result: Detected packet-start sample and normalized confidence.
        """

        referenceSignal = self.ValidateReceivedSignal(transmittedSignal)
        receivedMatrix = (
            receivedSignal.reshape(-1, 1)
            if receivedSignal.ndim == 1
            else receivedSignal
        )
        referenceMatrix = (
            referenceSignal.reshape(-1, 1)
            if referenceSignal.ndim == 1
            else referenceSignal
        )
        if receivedMatrix.shape[1] != referenceMatrix.shape[1]:
            raise ValueError(
                "receivedSignal and transmittedSignal must have the same "
                "number of physical chains"
            )
        maximumValidOffset = receivedMatrix.shape[0] - referenceMatrix.shape[0]
        if maximumValidOffset < 0:
            raise ValueError(
                "receivedSignal is shorter than transmittedSignal"
            )
        searchStop = min(
            int(self.parameters["maximumPacketOffsetSamples"]),
            maximumValidOffset,
        )
        probeLength = min(
            referenceMatrix.shape[0],
            int(self.parameters["referenceSearchSamples"]),
        )
        candidateScores = np.zeros(searchStop + 1, dtype=float)
        numericFloor = np.finfo(float).tiny
        for chainIndex in range(referenceMatrix.shape[1]):
            referenceProbe = referenceMatrix[:probeLength, chainIndex]
            captureProbe = receivedMatrix[
                : searchStop + probeLength, chainIndex
            ]
            correlation = np.correlate(
                captureProbe, referenceProbe, mode="valid"
            )
            referenceEnergy = max(
                float(np.vdot(referenceProbe, referenceProbe).real),
                numericFloor,
            )
            samplePower = np.abs(captureProbe) ** 2
            cumulativeEnergy = np.r_[
                0.0, np.cumsum(samplePower, dtype=float)
            ]
            windowEnergy = (
                cumulativeEnergy[probeLength:]
                - cumulativeEnergy[:-probeLength]
            )
            normalizedPower = (
                np.abs(correlation) ** 2
                / np.maximum(
                    referenceEnergy * windowEnergy,
                    numericFloor,
                )
            )
            candidateScores += normalizedPower
        candidateScores /= float(referenceMatrix.shape[1])
        bestOffset = int(np.argmax(candidateScores))
        bestConfidence = float(np.sqrt(candidateScores[bestOffset]))
        if bestConfidence < float(
            self.parameters["minimumParseConfidence"]
        ):
            raise ValueError(
                "transmit/receive waveform correlation is below "
                "minimumParseConfidence"
            )
        return bestOffset, bestConfidence

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
            packetStartSample, confidence = (
                self.EstimatePacketStartFromReference(
                    complexReceived, referenceSignal
                )
            )
            packetStopSample = (
                packetStartSample + referenceSignal.shape[0]
            )
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
        from ..lib.WaveGenWifi import WaveGenWifi

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
            if transmitPacketStop > validatedTransmit.shape[0]:
                raise ValueError(
                    "transmittedSignal ends before its decoded Wi-Fi packet"
                )
            referenceSignal = validatedTransmit[
                descriptorPacketStart:transmitPacketStop
            ].copy()
            packetStartSample, correlationConfidence = (
                self.EstimatePacketStartFromReference(
                    complexReceived, referenceSignal
                )
            )
            confidence = min(
                descriptorConfidence, correlationConfidence
            )
        packetStopSample = (
            packetStartSample + referenceSignal.shape[0]
        )
        if packetStopSample > complexReceived.shape[0]:
            raise ValueError(
                "receivedSignal ends before the decoded Wi-Fi packet"
            )
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
