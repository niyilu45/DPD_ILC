"""Object-oriented SNR, EVM, and ACLR analysis for HE/EHT simulations."""

import csv
import json
from collections import ChainMap
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .waveGen import WifiWaveform


analysisDefaultParameters: Mapping[str, object] = MappingProxyType(
    {
        "maxSegmentLength": 16384,
        "minimumAclrOversampling": 3.0,
        "powerEvmFileStem": "power_evm_curve",
    }
)


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


@dataclass
class PowerEvmCurve:
    """Store a multi-method EVM sweep over PA input RMS drive levels."""

    driveRmsValues: np.ndarray
    inputPowerDb: np.ndarray
    evmDbByMethod: Dict[str, np.ndarray]
    evmPercentByMethod: Dict[str, np.ndarray]

    def ToDict(self) -> Dict[str, object]:
        """Convert all curve samples to a JSON-ready dictionary."""

        return {
            "driveRmsValues": self.driveRmsValues.astype(float).tolist(),
            "inputPowerDb": self.inputPowerDb.astype(float).tolist(),
            "methods": {
                methodName: {
                    "evmDb": self.evmDbByMethod[methodName]
                    .astype(float)
                    .tolist(),
                    "evmPercent": self.evmPercentByMethod[methodName]
                    .astype(float)
                    .tolist(),
                }
                for methodName in self.evmDbByMethod
            },
        }


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
        self,
        referenceSignal: np.ndarray,
        waveform: WifiWaveform,
        parameters: Optional[Mapping[str, object]] = None,
        **parameterOverrides: object,
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
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping or None")
        externalParameters = {} if parameters is None else parameters
        self.parameters: ChainMap[str, object] = ChainMap(
            dict(parameterOverrides),
            externalParameters,
            analysisDefaultParameters,
        )
        self._ValidateParameters()
        self.stageMetrics: Dict[str, SignalMetrics] = {}
        self.powerEvmCurve: Optional[PowerEvmCurve] = None

    def GetParameters(self) -> Dict[str, object]:
        """Return a flattened snapshot of all resolved analysis parameters."""

        return dict(self.parameters)

    def UpdateParameters(self, **parameterOverrides: object) -> None:
        """Apply validated high-priority analysis parameter overrides."""

        previousOverrides = dict(self.parameters.maps[0])
        self.parameters.maps[0].update(parameterOverrides)
        try:
            self._ValidateParameters()
        except (TypeError, ValueError):
            self.parameters.maps[0].clear()
            self.parameters.maps[0].update(previousOverrides)
            raise

    def _ValidateParameters(self) -> None:
        """Validate the currently resolved ChainMap analysis settings."""

        unknownParameters = set(self.parameters).difference(
            analysisDefaultParameters
        )
        if unknownParameters:
            unknownNames = ", ".join(
                sorted(str(parameterName) for parameterName in unknownParameters)
            )
            raise TypeError(f"unknown Analysis parameters: {unknownNames}")
        maxSegmentLength = self.parameters["maxSegmentLength"]
        if (
            not isinstance(maxSegmentLength, int)
            or isinstance(maxSegmentLength, bool)
            or maxSegmentLength < 16
        ):
            raise ValueError(
                "maxSegmentLength must be an integer of at least 16"
            )
        minimumAclrOversampling = self.parameters["minimumAclrOversampling"]
        if not isinstance(minimumAclrOversampling, (int, float)) or isinstance(
            minimumAclrOversampling, bool
        ):
            raise TypeError("minimumAclrOversampling must be numeric")
        if minimumAclrOversampling < 3.0:
            raise ValueError("minimumAclrOversampling cannot be less than 3.0")
        powerEvmFileStem = self.parameters["powerEvmFileStem"]
        if not isinstance(powerEvmFileStem, str):
            raise TypeError("powerEvmFileStem must be a string")
        if not powerEvmFileStem or any(
            character in powerEvmFileStem for character in '<>:"/\\|?*'
        ):
            raise ValueError("powerEvmFileStem must be a valid simple file name")

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

        self._ValidateParameters()
        complexMeasured = self._PrepareMeasuredSignal(measuredSignal)
        dataSlice = self.waveform.fieldSlices[self.waveform.dataFieldName]
        measuredData = complexMeasured[dataSlice]
        sampleRateHz = self.waveform.sampleRateHz
        channelBandwidthHz = self.waveform.bandwidthHz
        minimumAclrOversampling = float(
            self.parameters["minimumAclrOversampling"]
        )
        if sampleRateHz < minimumAclrOversampling * channelBandwidthHz:
            raise ValueError(
                "ACLR analysis requires at least "
                f"{minimumAclrOversampling:g}x oversampling"
            )
        frequencyBins, powerSpectrum = _AveragePeriodogram(
            measuredData,
            sampleRateHz,
            int(self.parameters["maxSegmentLength"]),
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

    def AnalyzePowerEvmCurve(
        self,
        driveRmsValues: Sequence[float],
        methodEvaluators: Mapping[
            str, Callable[[np.ndarray, float], np.ndarray]
        ],
    ) -> PowerEvmCurve:
        """Evaluate multiple methods over one common RMS input-power sweep.

        Every evaluator receives the reference waveform scaled to the current
        RMS drive and the numeric drive value. An ILC evaluator may relearn at
        every point, while a deployed DPD evaluator may reuse fixed nominal-
        power coefficients. All methods are scored with identical references.
        """

        driveArray = np.asarray(driveRmsValues, dtype=float).reshape(-1)
        if driveArray.size < 2:
            raise ValueError("driveRmsValues must contain at least two points")
        if not np.all(np.isfinite(driveArray)) or np.any(driveArray <= 0.0):
            raise ValueError("driveRmsValues must contain finite positive values")
        if np.any(np.diff(driveArray) <= 0.0):
            raise ValueError("driveRmsValues must be strictly increasing")
        if not methodEvaluators:
            raise ValueError("methodEvaluators cannot be empty")

        evmDbByMethod: Dict[str, np.ndarray] = {}
        evmPercentByMethod: Dict[str, np.ndarray] = {}
        for methodName, methodEvaluator in methodEvaluators.items():
            methodEvmDb = []
            methodEvmPercent = []
            for driveRms in driveArray:
                pointReference = float(driveRms) * self.waveform.samples
                measuredSignal = methodEvaluator(
                    pointReference, float(driveRms)
                )
                pointAnalysis = Analysis(
                    pointReference,
                    self.waveform,
                    parameters=self.parameters,
                )
                pointMetrics = pointAnalysis.Analyze(measuredSignal)
                methodEvmDb.append(pointMetrics.evmDb)
                methodEvmPercent.append(pointMetrics.evmPercent)
            evmDbByMethod[methodName] = np.asarray(methodEvmDb, dtype=float)
            evmPercentByMethod[methodName] = np.asarray(
                methodEvmPercent, dtype=float
            )

        self.powerEvmCurve = PowerEvmCurve(
            driveRmsValues=driveArray,
            inputPowerDb=20.0 * np.log10(driveArray),
            evmDbByMethod=evmDbByMethod,
            evmPercentByMethod=evmPercentByMethod,
        )
        return self.powerEvmCurve

    def SavePowerEvmCurve(
        self,
        outputDirectory: Path,
        powerEvmCurve: Optional[PowerEvmCurve] = None,
        fileStem: Optional[str] = None,
    ) -> Tuple[Path, Path, Path]:
        """Save a multi-method power-EVM curve as CSV, JSON, and PNG."""

        self._ValidateParameters()
        selectedCurve = (
            self.powerEvmCurve if powerEvmCurve is None else powerEvmCurve
        )
        if selectedCurve is None:
            raise ValueError("no power-EVM curve is available to save")
        selectedFileStem = (
            str(self.parameters["powerEvmFileStem"])
            if fileStem is None
            else fileStem
        )
        if not selectedFileStem or any(
            character in selectedFileStem for character in '<>:"/\\|?*'
        ):
            raise ValueError("fileStem must be a valid simple file name")

        outputPath = Path(outputDirectory)
        outputPath.mkdir(parents=True, exist_ok=True)
        csvPath = outputPath / f"{selectedFileStem}.csv"
        jsonPath = outputPath / f"{selectedFileStem}.json"
        figurePath = outputPath / f"{selectedFileStem}.png"
        methodNames = list(selectedCurve.evmDbByMethod)

        fieldNames = ["driveRms", "inputPowerDb"]
        for methodName in methodNames:
            fieldNames.extend(
                [f"{methodName} evmDb", f"{methodName} evmPercent"]
            )
        with csvPath.open("w", newline="", encoding="utf-8-sig") as csvFile:
            csvWriter = csv.DictWriter(csvFile, fieldnames=fieldNames)
            csvWriter.writeheader()
            for pointIndex, driveRms in enumerate(
                selectedCurve.driveRmsValues
            ):
                rowData = {
                    "driveRms": float(driveRms),
                    "inputPowerDb": float(
                        selectedCurve.inputPowerDb[pointIndex]
                    ),
                }
                for methodName in methodNames:
                    rowData[f"{methodName} evmDb"] = float(
                        selectedCurve.evmDbByMethod[methodName][pointIndex]
                    )
                    rowData[f"{methodName} evmPercent"] = float(
                        selectedCurve.evmPercentByMethod[methodName][pointIndex]
                    )
                csvWriter.writerow(rowData)

        with jsonPath.open("w", encoding="utf-8") as jsonFile:
            json.dump(
                selectedCurve.ToDict(),
                jsonFile,
                indent=2,
                ensure_ascii=False,
            )

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise RuntimeError(
                "matplotlib is required to save the power-EVM figure"
            ) from error

        figure, axes = plt.subplots(figsize=(10.5, 6.2))
        markerStyles = ("o", "s", "^", "D", "v", "P", "X", "<", ">")
        lineStyles = ("-", "--", "-.", ":")
        for methodIndex, methodName in enumerate(methodNames):
            axes.plot(
                selectedCurve.inputPowerDb,
                selectedCurve.evmDbByMethod[methodName],
                label=methodName,
                marker=markerStyles[methodIndex % len(markerStyles)],
                linestyle=lineStyles[
                    (methodIndex // len(markerStyles)) % len(lineStyles)
                ],
                linewidth=1.8,
                markersize=5.0,
            )
        axes.set_xlabel("Input RMS power relative to unit saturation (dB)")
        axes.set_ylabel("RMS EVM (dB, lower is better)")
        axes.set_title("Power-EVM comparison")
        axes.grid(True, which="both", linestyle=":", linewidth=0.7)
        if len(methodNames) <= 6:
            axes.legend(loc="best")
        else:
            axes.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
            )
        figure.tight_layout()
        figure.savefig(figurePath, dpi=180, bbox_inches="tight")
        plt.close(figure)
        return csvPath, jsonPath, figurePath

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
