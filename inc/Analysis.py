"""Object-oriented SNR, EVM, and ACLR analysis for HE/EHT simulations."""

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from .waveGen import WifiWaveform


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


def _BestComplexGain(
    referenceSignal: np.ndarray, measuredSignal: np.ndarray
) -> complex:
    """Find the least-squares complex gain mapping reference onto measurement."""

    denominator = max(
        np.vdot(referenceSignal, referenceSignal).real,
        np.finfo(float).tiny,
    )
    return np.vdot(referenceSignal, measuredSignal) / denominator


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

    for startIndex in range(
        0, complexInput.size - segmentLength + 1, segmentStep
    ):
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


class Analysis:
    """Analyze and save PA/DPD results for one reference Wi-Fi waveform.

    Callers construct one instance from the transmitted reference signal and
    its ``WifiWaveform`` metadata. Every PA, ILC, or deployed-DPD output is
    then evaluated through the same stored analysis context.

    Example:
        ``resultAnalysis = Analysis(referenceSignal, waveform)``
        ``metrics = resultAnalysis.Analyze(paOutput)``
    """

    def __init__(
        self, referenceSignal: np.ndarray, waveform: WifiWaveform
    ) -> None:
        complexReference = np.asarray(
            referenceSignal, dtype=np.complex128
        ).reshape(-1)
        if complexReference.size == 0:
            raise ValueError("referenceSignal cannot be empty")
        if complexReference.size != waveform.samples.size:
            raise ValueError(
                "referenceSignal length must match the Wi-Fi waveform"
            )
        if not np.all(np.isfinite(complexReference)):
            raise ValueError("referenceSignal contains NaN or infinite values")
        self.referenceSignal = complexReference
        self.waveform = waveform
        self.stageMetrics: Dict[str, SignalMetrics] = {}

    def _PrepareMeasuredSignal(self, measuredSignal: np.ndarray) -> np.ndarray:
        """Validate and normalize one measured signal for metric processing."""

        complexMeasured = np.asarray(
            measuredSignal, dtype=np.complex128
        ).reshape(-1)
        if complexMeasured.size != self.referenceSignal.size:
            raise ValueError(
                "measuredSignal and referenceSignal must have equal length"
            )
        if not np.all(np.isfinite(complexMeasured)):
            raise ValueError("measuredSignal contains NaN or infinite values")
        return complexMeasured

    def CalculateSnr(self, measuredSignal: np.ndarray) -> float:
        """Calculate data-field SNR after removing one complex gain and phase."""

        complexMeasured = self._PrepareMeasuredSignal(measuredSignal)
        dataSlice = self.waveform.fieldSlices[self.waveform.dataFieldName]
        referenceData = self.referenceSignal[dataSlice]
        measuredData = complexMeasured[dataSlice]
        complexGain = _BestComplexGain(referenceData, measuredData)
        fittedReference = complexGain * referenceData
        errorSignal = measuredData - fittedReference
        signalPower = np.mean(np.abs(fittedReference) ** 2)
        errorPower = np.mean(np.abs(errorSignal) ** 2)
        return float(
            10.0
            * np.log10(
                max(signalPower, np.finfo(float).tiny)
                / max(errorPower, np.finfo(float).tiny)
            )
        )

    def DemodulateWifiData(self, measuredSignal: np.ndarray) -> np.ndarray:
        """Remove cyclic prefixes and FFT-demodulate Wi-Fi data subcarriers."""

        complexMeasured = self._PrepareMeasuredSignal(measuredSignal)
        demodulatedSymbols = []
        for symbolStart in self.waveform.dataSymbolStarts:
            usefulStart = int(symbolStart) + self.waveform.cpLength
            usefulStop = usefulStart + self.waveform.fftLength
            if usefulStop > complexMeasured.size:
                raise ValueError(
                    "measuredSignal is shorter than the Wi-Fi data field"
                )
            usefulSamples = complexMeasured[usefulStart:usefulStop]
            frequencyGrid = np.fft.fft(usefulSamples) / np.sqrt(
                self.waveform.fftLength
            )
            demodulatedSymbols.append(
                frequencyGrid[
                    np.mod(
                        self.waveform.dataSubcarriers,
                        self.waveform.fftLength,
                    )
                ]
            )
        return np.asarray(demodulatedSymbols)

    def CalculateEvm(self, measuredSignal: np.ndarray) -> Tuple[float, float]:
        """Calculate RMS EVM in dB and percent on Wi-Fi data subcarriers."""

        measuredSymbols = self.DemodulateWifiData(measuredSignal)
        flattenedReference = self.waveform.referenceDataSymbols.reshape(-1)
        flattenedMeasured = measuredSymbols.reshape(-1)
        complexGain = _BestComplexGain(
            flattenedReference, flattenedMeasured
        )
        fittedReference = complexGain * flattenedReference
        symbolError = flattenedMeasured - fittedReference
        evmRatio = np.sqrt(
            np.sum(np.abs(symbolError) ** 2)
            / max(
                np.sum(np.abs(fittedReference) ** 2),
                np.finfo(float).tiny,
            )
        )
        evmPercent = 100.0 * evmRatio
        evmDb = 20.0 * np.log10(max(evmRatio, np.finfo(float).tiny))
        return float(evmDb), float(evmPercent)

    def CalculateAclr(
        self, measuredSignal: np.ndarray
    ) -> Tuple[float, float, float]:
        """Calculate lower, upper, and worst adjacent-channel leakage ratios."""

        complexMeasured = self._PrepareMeasuredSignal(measuredSignal)
        dataSlice = self.waveform.fieldSlices[self.waveform.dataFieldName]
        measuredData = complexMeasured[dataSlice]
        sampleRateHz = self.waveform.sampleRateHz
        channelBandwidthHz = self.waveform.bandwidthHz
        if sampleRateHz < 3.0 * channelBandwidthHz:
            raise ValueError("ACLR analysis requires at least 3x oversampling")
        frequencyBins, powerSpectrum = _AveragePeriodogram(
            measuredData, sampleRateHz
        )
        halfBandwidth = channelBandwidthHz / 2.0
        mainMask = np.abs(frequencyBins) < halfBandwidth
        lowerMask = (frequencyBins >= -3.0 * halfBandwidth) & (
            frequencyBins < -halfBandwidth
        )
        upperMask = (frequencyBins > halfBandwidth) & (
            frequencyBins <= 3.0 * halfBandwidth
        )

        mainPower = max(
            np.sum(powerSpectrum[mainMask]), np.finfo(float).tiny
        )
        lowerPower = max(
            np.sum(powerSpectrum[lowerMask]), np.finfo(float).tiny
        )
        upperPower = max(
            np.sum(powerSpectrum[upperMask]), np.finfo(float).tiny
        )
        lowerAclrDb = 10.0 * np.log10(mainPower / lowerPower)
        upperAclrDb = 10.0 * np.log10(mainPower / upperPower)
        worstAclrDb = min(lowerAclrDb, upperAclrDb)
        return float(lowerAclrDb), float(upperAclrDb), float(worstAclrDb)

    def Analyze(self, measuredSignal: np.ndarray) -> SignalMetrics:
        """Calculate SNR, EVM, and ACLR for one PA or DPD output waveform."""

        complexMeasured = self._PrepareMeasuredSignal(measuredSignal)
        snrDb = self.CalculateSnr(complexMeasured)
        evmDb, evmPercent = self.CalculateEvm(complexMeasured)
        aclrLowerDb, aclrUpperDb, aclrWorstDb = self.CalculateAclr(
            complexMeasured
        )
        return SignalMetrics(
            snrDb=snrDb,
            evmDb=evmDb,
            evmPercent=evmPercent,
            aclrLowerDb=aclrLowerDb,
            aclrUpperDb=aclrUpperDb,
            aclrWorstDb=aclrWorstDb,
        )

    def AnalyzeStages(
        self, stageSignals: Mapping[str, np.ndarray]
    ) -> Dict[str, SignalMetrics]:
        """Analyze multiple named stages and retain their result table."""

        self.stageMetrics = {
            stageName: self.Analyze(stageSignal)
            for stageName, stageSignal in stageSignals.items()
        }
        return dict(self.stageMetrics)

    def Print(
        self,
        stageMetrics: Optional[Mapping[str, SignalMetrics]] = None,
    ) -> None:
        """Print an aligned table for all selected result stages."""

        selectedMetrics = (
            self.stageMetrics if stageMetrics is None else stageMetrics
        )
        if not selectedMetrics:
            raise ValueError("no stage metrics are available to print")
        header = (
            f"{'Stage':<16} {'SNR(dB)':>10} {'EVM(dB)':>10} "
            f"{'EVM(%)':>10} {'ACLR-L':>10} {'ACLR-U':>10} {'ACLR-W':>10}"
        )
        print(header)
        print("-" * len(header))
        for stageName, metrics in selectedMetrics.items():
            print(
                f"{stageName:<16} {metrics.snrDb:>10.2f} "
                f"{metrics.evmDb:>10.2f} {metrics.evmPercent:>10.3f} "
                f"{metrics.aclrLowerDb:>10.2f} "
                f"{metrics.aclrUpperDb:>10.2f} "
                f"{metrics.aclrWorstDb:>10.2f}"
            )

    def Save(
        self,
        outputDirectory: Path,
        runMetadata: Mapping[str, object],
        stageMetrics: Optional[Mapping[str, SignalMetrics]] = None,
    ) -> Tuple[Path, Path]:
        """Save selected metrics as JSON and CSV result files."""

        selectedMetrics = (
            self.stageMetrics if stageMetrics is None else stageMetrics
        )
        if not selectedMetrics:
            raise ValueError("no stage metrics are available to save")
        outputPath = Path(outputDirectory)
        outputPath.mkdir(parents=True, exist_ok=True)
        jsonPath = outputPath / "metrics.json"
        csvPath = outputPath / "metrics.csv"
        serializableMetrics = {
            stageName: metrics.ToDict()
            for stageName, metrics in selectedMetrics.items()
        }
        jsonPayload = {
            "metadata": dict(runMetadata),
            "metrics": serializableMetrics,
        }
        with jsonPath.open("w", encoding="utf-8") as jsonFile:
            json.dump(jsonPayload, jsonFile, indent=2, ensure_ascii=False)

        fieldNames = ["stage"] + list(
            SignalMetrics.__dataclass_fields__.keys()
        )
        with csvPath.open("w", newline="", encoding="utf-8-sig") as csvFile:
            csvWriter = csv.DictWriter(csvFile, fieldnames=fieldNames)
            csvWriter.writeheader()
            for stageName, metrics in selectedMetrics.items():
                rowData = {"stage": stageName}
                rowData.update(metrics.ToDict())
                csvWriter.writerow(rowData)
        return jsonPath, csvPath

    def SaveConvergence(self, ilcHistory, outputDirectory: Path) -> Path:
        """Save the per-iteration ILC convergence history as a CSV file."""

        outputPath = Path(outputDirectory)
        outputPath.mkdir(parents=True, exist_ok=True)
        convergencePath = outputPath / "ilc_convergence.csv"
        fieldNames = ["iteration", "errorRms", "nmseDb", "inputPeak"]
        with convergencePath.open(
            "w", newline="", encoding="utf-8-sig"
        ) as csvFile:
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
