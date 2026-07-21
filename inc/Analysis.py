"""SNR, EVM, and ACLR analysis for EHT DPD simulations."""

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np

from .waveGen import EHTWaveform


@dataclass(frozen=True)
class SignalMetrics:
    """Collect the requested signal-quality and spectral-quality metrics."""

    snrDb: float
    evmDb: float
    evmPercent: float
    aclrLowerDb: float
    aclrUpperDb: float
    aclrWorstDb: float

    def ToDict(self) -> Dict[str, float]:
        """Convert metrics to a JSON/CSV-ready dictionary."""

        return {key: float(value) for key, value in asdict(self).items()}


def _BestComplexGain(referenceSignal: np.ndarray, measuredSignal: np.ndarray) -> complex:
    """Find the least-squares complex gain mapping reference onto measurement."""

    denominator = max(
        np.vdot(referenceSignal, referenceSignal).real,
        np.finfo(float).tiny,
    )
    return np.vdot(referenceSignal, measuredSignal) / denominator


def CalculateSnr(referenceSignal: np.ndarray, measuredSignal: np.ndarray) -> float:
    """Calculate reconstruction SNR after removing one complex gain and phase."""

    complexReference = np.asarray(referenceSignal, dtype=np.complex128).reshape(-1)
    complexMeasured = np.asarray(measuredSignal, dtype=np.complex128).reshape(-1)
    if complexReference.size != complexMeasured.size:
        raise ValueError("referenceSignal and measuredSignal must have equal length")
    complexGain = _BestComplexGain(complexReference, complexMeasured)
    fittedReference = complexGain * complexReference
    errorSignal = complexMeasured - fittedReference
    signalPower = np.mean(np.abs(fittedReference) ** 2)
    errorPower = np.mean(np.abs(errorSignal) ** 2)
    return float(
        10.0
        * np.log10(
            max(signalPower, np.finfo(float).tiny)
            / max(errorPower, np.finfo(float).tiny)
        )
    )


def DemodulateEhtData(
    measuredSignal: np.ndarray, waveform: EHTWaveform
) -> np.ndarray:
    """Remove each cyclic prefix and FFT-demodulate EHT data subcarriers."""

    complexMeasured = np.asarray(measuredSignal, dtype=np.complex128).reshape(-1)
    demodulatedSymbols = []
    for symbolStart in waveform.dataSymbolStarts:
        usefulStart = int(symbolStart) + waveform.cpLength
        usefulStop = usefulStart + waveform.fftLength
        if usefulStop > complexMeasured.size:
            raise ValueError("measuredSignal is shorter than the EHT data field")
        usefulSamples = complexMeasured[usefulStart:usefulStop]
        frequencyGrid = np.fft.fft(usefulSamples) / np.sqrt(waveform.fftLength)
        demodulatedSymbols.append(
            frequencyGrid[
                np.mod(waveform.dataSubcarriers, waveform.fftLength)
            ]
        )
    return np.asarray(demodulatedSymbols)


def CalculateEvm(
    measuredSignal: np.ndarray, waveform: EHTWaveform
) -> Tuple[float, float]:
    """Calculate RMS EVM in dB and percent on EHT data subcarriers."""

    measuredSymbols = DemodulateEhtData(measuredSignal, waveform)
    referenceSymbols = waveform.referenceDataSymbols
    flattenedReference = referenceSymbols.reshape(-1)
    flattenedMeasured = measuredSymbols.reshape(-1)
    complexGain = _BestComplexGain(flattenedReference, flattenedMeasured)
    fittedReference = complexGain * flattenedReference
    symbolError = flattenedMeasured - fittedReference
    evmRatio = np.sqrt(
        np.sum(np.abs(symbolError) ** 2)
        / max(np.sum(np.abs(fittedReference) ** 2), np.finfo(float).tiny)
    )
    evmPercent = 100.0 * evmRatio
    evmDb = 20.0 * np.log10(max(evmRatio, np.finfo(float).tiny))
    return float(evmDb), float(evmPercent)


def _AveragePeriodogram(
    inputSignal: np.ndarray,
    sampleRateHz: float,
    maxSegmentLength: int = 16384,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate a low-variance PSD using overlapping Hann-windowed segments."""

    complexInput = np.asarray(inputSignal, dtype=np.complex128).reshape(-1)
    if complexInput.size < 16:
        raise ValueError("inputSignal is too short for spectral analysis")

    segmentLength = min(maxSegmentLength, complexInput.size)
    segmentLength = 1 << int(np.floor(np.log2(segmentLength)))
    segmentStep = max(segmentLength // 2, 1)
    analysisWindow = np.hanning(segmentLength)
    windowPower = max(np.sum(analysisWindow**2), np.finfo(float).tiny)
    accumulatedPsd = np.zeros(segmentLength, dtype=float)
    segmentCount = 0

    for startIndex in range(0, complexInput.size - segmentLength + 1, segmentStep):
        signalSegment = complexInput[startIndex : startIndex + segmentLength]
        signalSpectrum = np.fft.fftshift(
            np.fft.fft(signalSegment * analysisWindow)
        )
        accumulatedPsd += np.abs(signalSpectrum) ** 2 / windowPower
        segmentCount += 1

    if segmentCount == 0:
        raise RuntimeError("unable to create a PSD segment")
    averagePsd = accumulatedPsd / segmentCount
    frequencyBins = np.fft.fftshift(
        np.fft.fftfreq(segmentLength, d=1.0 / sampleRateHz)
    )
    return frequencyBins, averagePsd


def CalculateAclr(
    inputSignal: np.ndarray,
    sampleRateHz: float,
    channelBandwidthHz: float,
) -> Tuple[float, float, float]:
    """Calculate lower, upper, and worst-case adjacent-channel leakage ratios.

    The wanted channel spans ``[-BW/2, BW/2]``. Equal-bandwidth adjacent
    channels occupy ``[-3BW/2, -BW/2]`` and ``[BW/2, 3BW/2]``. At least 4x
    sampling is required so both adjacent channels lie inside Nyquist.
    """

    if sampleRateHz < 3.0 * channelBandwidthHz:
        raise ValueError("ACLR analysis requires at least 3x oversampling")
    frequencyBins, powerSpectrum = _AveragePeriodogram(
        inputSignal, sampleRateHz
    )
    halfBandwidth = channelBandwidthHz / 2.0
    mainMask = np.abs(frequencyBins) < halfBandwidth
    lowerMask = (frequencyBins >= -3.0 * halfBandwidth) & (
        frequencyBins < -halfBandwidth
    )
    upperMask = (frequencyBins > halfBandwidth) & (
        frequencyBins <= 3.0 * halfBandwidth
    )

    mainPower = max(np.sum(powerSpectrum[mainMask]), np.finfo(float).tiny)
    lowerPower = max(np.sum(powerSpectrum[lowerMask]), np.finfo(float).tiny)
    upperPower = max(np.sum(powerSpectrum[upperMask]), np.finfo(float).tiny)
    lowerAclrDb = 10.0 * np.log10(mainPower / lowerPower)
    upperAclrDb = 10.0 * np.log10(mainPower / upperPower)
    worstAclrDb = min(lowerAclrDb, upperAclrDb)
    return float(lowerAclrDb), float(upperAclrDb), float(worstAclrDb)


def AnalyzeSignal(
    referenceSignal: np.ndarray,
    measuredSignal: np.ndarray,
    waveform: EHTWaveform,
) -> SignalMetrics:
    """Calculate SNR, subcarrier EVM, and ACLR for one PA output waveform."""

    complexReference = np.asarray(referenceSignal, dtype=np.complex128).reshape(-1)
    complexMeasured = np.asarray(measuredSignal, dtype=np.complex128).reshape(-1)
    if complexReference.size != complexMeasured.size:
        raise ValueError("referenceSignal and measuredSignal must have equal length")

    # SNR and ACLR use only the EHT data field. This removes deterministic
    # preamble format transitions from the nonlinear data-quality statistics.
    dataSlice = waveform.fieldSlices["EHT-Data"]
    referenceData = complexReference[dataSlice]
    measuredData = complexMeasured[dataSlice]
    snrDb = CalculateSnr(referenceData, measuredData)
    evmDb, evmPercent = CalculateEvm(complexMeasured, waveform)
    aclrLowerDb, aclrUpperDb, aclrWorstDb = CalculateAclr(
        measuredData, waveform.sampleRateHz, waveform.bandwidthHz
    )
    return SignalMetrics(
        snrDb=snrDb,
        evmDb=evmDb,
        evmPercent=evmPercent,
        aclrLowerDb=aclrLowerDb,
        aclrUpperDb=aclrUpperDb,
        aclrWorstDb=aclrWorstDb,
    )


def PrintMetrics(stageMetrics: Mapping[str, SignalMetrics]) -> None:
    """Print an aligned console table for baseline, ILC, and fitted DPD stages."""

    header = (
        f"{'Stage':<16} {'SNR(dB)':>10} {'EVM(dB)':>10} "
        f"{'EVM(%)':>10} {'ACLR-L':>10} {'ACLR-U':>10} {'ACLR-W':>10}"
    )
    print(header)
    print("-" * len(header))
    for stageName, metrics in stageMetrics.items():
        print(
            f"{stageName:<16} {metrics.snrDb:>10.2f} {metrics.evmDb:>10.2f} "
            f"{metrics.evmPercent:>10.3f} {metrics.aclrLowerDb:>10.2f} "
            f"{metrics.aclrUpperDb:>10.2f} {metrics.aclrWorstDb:>10.2f}"
        )


def SaveMetrics(
    stageMetrics: Mapping[str, SignalMetrics],
    outputDirectory: Path,
    runMetadata: Mapping[str, object],
) -> Tuple[Path, Path]:
    """Save metrics as both JSON and CSV for automation and spreadsheets."""

    outputPath = Path(outputDirectory)
    outputPath.mkdir(parents=True, exist_ok=True)
    jsonPath = outputPath / "metrics.json"
    csvPath = outputPath / "metrics.csv"

    serializableMetrics = {
        stageName: metrics.ToDict()
        for stageName, metrics in stageMetrics.items()
    }
    jsonPayload = {
        "metadata": dict(runMetadata),
        "metrics": serializableMetrics,
    }
    with jsonPath.open("w", encoding="utf-8") as jsonFile:
        json.dump(jsonPayload, jsonFile, indent=2, ensure_ascii=False)

    fieldNames = ["stage"] + list(SignalMetrics.__dataclass_fields__.keys())
    with csvPath.open("w", newline="", encoding="utf-8-sig") as csvFile:
        csvWriter = csv.DictWriter(csvFile, fieldnames=fieldNames)
        csvWriter.writeheader()
        for stageName, metrics in stageMetrics.items():
            rowData = {"stage": stageName}
            rowData.update(metrics.ToDict())
            csvWriter.writerow(rowData)
    return jsonPath, csvPath


def SaveConvergence(ilcHistory, outputDirectory: Path) -> Path:
    """Save the per-iteration ILC convergence history as a CSV file."""

    outputPath = Path(outputDirectory)
    outputPath.mkdir(parents=True, exist_ok=True)
    convergencePath = outputPath / "ilc_convergence.csv"
    fieldNames = ["iteration", "errorRms", "nmseDb", "inputPeak"]
    with convergencePath.open("w", newline="", encoding="utf-8-sig") as csvFile:
        csvWriter = csv.DictWriter(csvFile, fieldnames=fieldNames)
        csvWriter.writeheader()
        for iterationRecord in ilcHistory:
            csvWriter.writerow(
                {
                    "iteration": iterationRecord.iteration,
                    "errorRms": iterationRecord.errorRms,
                    "nmseDb": iterationRecord.nmseDb,
                    "inputPeak": iterationRecord.inputPeak,
                }
            )
    return convergencePath
