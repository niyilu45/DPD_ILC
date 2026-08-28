"""Object-oriented IM3, IM5, and IM7 analysis for two-tone PA tests."""

import csv
import json
from collections import ChainMap
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, TypedDict, cast

import numpy as np

from .WaveGenTwoTone import TwoToneWaveform

# Cross-package imports support both repository-root and ``inc``-root imports.
if __package__ and "." in __package__:
    from ..utils.ConfigUtils import (
        FilterRecognizedParameters,
        RecognizedParameterView,
    )
    from ..utils.FixedPoint import FixedPoint
    from ..utils.SigProc import PowerCalibration
else:
    from utils.ConfigUtils import (
        FilterRecognizedParameters,
        RecognizedParameterView,
    )
    from utils.FixedPoint import FixedPoint
    from utils.SigProc import PowerCalibration


class TwoToneMetrics(TypedDict):
    """Define the ordinary dictionary returned by two-tone analysis."""

    fundamentalLowerDbfs: float
    fundamentalUpperDbfs: float
    fundamentalAverageDbfs: float
    im3LowerDbc: float
    im3UpperDbc: float
    im3WorstDbc: float
    im5LowerDbc: float
    im5UpperDbc: float
    im5WorstDbc: float
    im7LowerDbc: float
    im7UpperDbc: float
    im7WorstDbc: float
    worstIntermodulationDbc: float
    outputPowerDbm: float


@dataclass(frozen=True)
class TwoToneILCIteration:
    """Combine one native ILC iteration with independently measured IM levels."""

    iteration: int
    nmseDb: float
    linearCompensatedNmseDb: float
    metrics: TwoToneMetrics

    def ToDict(self) -> Dict[str, float]:
        """Flatten one iteration into serialization-ready scalar fields.

        Processing details:
            Algorithm: Copy the native iteration indices and NMSE values, then
            merge every independently calculated two-tone metric without
            changing signs, units, or the caller-owned source dictionary.

        Returns:
            result: Flat dictionary suitable for CSV or JSON output.
        """

        rowData = {
            "iteration": float(self.iteration),
            "nmseDb": float(self.nmseDb),
            "linearCompensatedNmseDb": float(
                self.linearCompensatedNmseDb
            ),
        }
        rowData.update(
            {
                metricName: float(metricValue)
                for metricName, metricValue in self.metrics.items()
            }
        )
        return rowData


@dataclass(frozen=True)
class TwoToneILCAnalysisResult:
    """Store analyzed history and the IM-best measured ILC candidate."""

    history: Tuple[TwoToneILCIteration, ...]
    bestIteration: int
    bestInputSignal: np.ndarray
    bestOutputSignal: np.ndarray
    bestMetrics: TwoToneMetrics


class TwoToneAnalysis:
    """Measure fundamentals and odd intermodulation products in two-tone data."""

    def __init__(
        self,
        waveform: TwoToneWaveform,
        parameters: Optional[Mapping[str, object]] = None,
        width: Optional[int] = None,
        outputFullScaleAmplitude: Optional[float] = None,
        **parameterOverrides: object,
    ) -> None:
        """Initialize a metadata-aware two-tone analysis context.

        Processing details:
            Algorithm: Retain immutable tone metadata, build an internal
            ChainMap of spectral-window, edge-discard, power-reference, and
            interface-width defaults, then validate every resolved value.

        Args:
            waveform: Generated two-tone metadata defining exact frequencies.
            parameters: Optional caller-owned mapping layered ahead of defaults.
            width: Optional public I/Q width; None inherits waveform metadata.
            outputFullScaleAmplitude: Optional physical component magnitude
                represented by measured PA-output code rails. The default is
                1.0 for compatibility and is independent of the transmit
                waveform's DAC scale.
            parameterOverrides: Highest-priority recognized analysis values.

        Returns:
            result: None. A reusable analyzer is ready for PA and ILC outputs.
        """

        if not isinstance(waveform, TwoToneWaveform):
            raise TypeError("waveform must be a TwoToneWaveform")
        self.waveform = waveform
        self.defaultParameters: Mapping[str, object] = MappingProxyType(
            {
                "windowName": "hann",
                "settlingSamples": 256,
                "minimumSpectralPower": 1.0e-30,
                "maximumOutputPowerDbm": 25.0,
                "loadResistanceOhm": 50.0,
                "activePowerThresholdDb": -60.0,
                "activeGapToleranceSamples": 16,
                "width": waveform.width,
                "outputFullScaleAmplitude": 1.0,
            }
        )
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping or None")
        directOverrides = dict(parameterOverrides)
        if width is not None:
            directOverrides["width"] = width
        if outputFullScaleAmplitude is not None:
            directOverrides["outputFullScaleAmplitude"] = (
                outputFullScaleAmplitude
            )
        externalParameters: Mapping[str, object] = (
            {}
            if parameters is None
            else RecognizedParameterView(
                parameters,
                self.defaultParameters,
                "TwoToneAnalysis",
            )
        )
        recognizedOverrides = FilterRecognizedParameters(
            directOverrides,
            self.defaultParameters,
            "TwoToneAnalysis",
        )
        self.parameters: ChainMap[str, object] = ChainMap(
            recognizedOverrides,
            externalParameters,
            self.defaultParameters,
        )
        self.lastMetrics: Optional[TwoToneMetrics] = None
        self.ValidateParameters()

    @property
    def Width(self) -> int:
        """Return the public I/Q component width used at the analysis boundary.

        Processing details:
            Algorithm: Read the validated ChainMap value without changing the
            waveform metadata or any live external parameter mapping.

        Returns:
            result: Zero for floating mode or a positive signed-code width.
        """

        return cast(int, self.parameters["width"])

    width = Width

    @property
    def OutputFullScaleAmplitude(self) -> float:
        """Return the physical magnitude represented by measured PA codes.

        Processing details:
            Algorithm: Read the validated measured-output scale separately
            from the source waveform's DAC convention.

        Returns:
            result: Positive physical I/Q component full-scale amplitude.
        """

        return float(
            cast(float, self.parameters["outputFullScaleAmplitude"])
        )

    outputFullScaleAmplitude = OutputFullScaleAmplitude

    def GetParameters(self) -> Dict[str, object]:
        """Return a flattened snapshot of all resolved analysis parameters.

        Processing details:
            Algorithm: Resolve every ChainMap layer into an independent normal
            dictionary so callers cannot mutate internal state through it.

        Returns:
            result: Dictionary containing every supported analysis setting.
        """

        return dict(self.parameters)

    def UpdateParameters(self, **parameterOverrides: object) -> None:
        """Apply recognized analysis parameter changes transactionally.

        Processing details:
            Algorithm: Warn and discard unknown names, update only the local
            ChainMap layer, validate the complete result, and restore the
            previous layer if any recognized value is invalid.

        Args:
            parameterOverrides: Highest-priority values for later analyses.

        Returns:
            result: None. Valid changes remain active.
        """

        recognizedOverrides = FilterRecognizedParameters(
            parameterOverrides,
            self.defaultParameters,
            "TwoToneAnalysis.UpdateParameters",
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
        """Validate window, record length, numerical floor, power, and width.

        Processing details:
            Algorithm: Restrict the window to supported deterministic choices,
            ensure symmetric edge removal leaves at least 64 samples, require
            positive finite spectral and RF references, and validate the
            measured-signal fixed-point boundary independently of transmit
            waveform metadata.

        Returns:
            result: None. Invalid settings raise descriptive exceptions.
        """

        windowName = self.parameters["windowName"]
        if not isinstance(windowName, str):
            raise TypeError("windowName must be a string")
        if windowName.lower() not in ("hann", "rectangular"):
            raise ValueError("windowName must be 'hann' or 'rectangular'")
        settlingSamples = self.parameters["settlingSamples"]
        if (
            not isinstance(settlingSamples, int)
            or isinstance(settlingSamples, bool)
            or settlingSamples < 0
            or 2 * settlingSamples + 64 > self.waveform.numSamples
        ):
            raise ValueError(
                "settlingSamples must leave at least 64 analyzed samples"
            )
        minimumSpectralPower = self.parameters["minimumSpectralPower"]
        if (
            not isinstance(minimumSpectralPower, (int, float))
            or isinstance(minimumSpectralPower, bool)
            or not np.isfinite(minimumSpectralPower)
            or float(minimumSpectralPower) <= 0.0
        ):
            raise ValueError(
                "minimumSpectralPower must be finite and positive"
            )
        for parameterName in (
            "maximumOutputPowerDbm",
            "loadResistanceOhm",
        ):
            parameterValue = self.parameters[parameterName]
            if (
                not isinstance(parameterValue, (int, float))
                or isinstance(parameterValue, bool)
                or not np.isfinite(parameterValue)
            ):
                raise ValueError(f"{parameterName} must be finite")
        if float(self.parameters["loadResistanceOhm"]) <= 0.0:
            raise ValueError("loadResistanceOhm must be positive")
        activePowerThresholdDb = self.parameters[
            "activePowerThresholdDb"
        ]
        if (
            not isinstance(activePowerThresholdDb, (int, float))
            or isinstance(activePowerThresholdDb, bool)
            or not np.isfinite(activePowerThresholdDb)
            or float(activePowerThresholdDb) >= 0.0
        ):
            raise ValueError(
                "activePowerThresholdDb must be finite and negative"
            )
        activeGapToleranceSamples = self.parameters[
            "activeGapToleranceSamples"
        ]
        if (
            not isinstance(activeGapToleranceSamples, int)
            or isinstance(activeGapToleranceSamples, bool)
            or activeGapToleranceSamples < 0
        ):
            raise ValueError(
                "activeGapToleranceSamples must be a nonnegative integer"
            )
        FixedPoint(
            self.width,
            self.parameters["outputFullScaleAmplitude"],
        )

    def BuildAnalysisWindow(self, sampleCount: int) -> np.ndarray:
        """Construct the configured deterministic spectral analysis window.

        Processing details:
            Algorithm: Return a periodic-record rectangular window or a Hann
            taper of the exact requested length, and reject records too short
            for stable discrete-tone projection.

        Args:
            sampleCount: Number of retained steady-state complex samples.

        Returns:
            result: Real window vector with positive coherent gain.
        """

        if (
            not isinstance(sampleCount, int)
            or isinstance(sampleCount, bool)
            or sampleCount < 64
        ):
            raise ValueError("sampleCount must be an integer no smaller than 64")
        if str(self.parameters["windowName"]).lower() == "hann":
            return np.hanning(sampleCount)
        return np.ones(sampleCount, dtype=float)

    def CalculateToneCoefficient(
        self,
        inputSignal: np.ndarray,
        frequencyHz: float,
        analysisWindow: np.ndarray,
    ) -> complex:
        """Project a finite record onto one exact physical tone frequency.

        Processing details:
            Algorithm: Multiply the decoded steady-state record by the selected
            window and a negative-frequency complex exponential, then divide by
            coherent window gain. Exact-frequency projection avoids assigning
            the product to the nearest FFT bin.

        Args:
            inputSignal: One-dimensional decoded steady-state complex samples.
            frequencyHz: Physical projection frequency in hertz.
            analysisWindow: Real taper with the same number of samples.

        Returns:
            result: Complex sinusoidal amplitude at the requested frequency.
        """

        complexInput = np.asarray(inputSignal, dtype=np.complex128).reshape(-1)
        realWindow = np.asarray(analysisWindow, dtype=float).reshape(-1)
        if complexInput.size != realWindow.size or complexInput.size == 0:
            raise ValueError(
                "inputSignal and analysisWindow must have equal nonzero length"
            )
        if (
            not isinstance(frequencyHz, (int, float))
            or isinstance(frequencyHz, bool)
            or not np.isfinite(frequencyHz)
        ):
            raise ValueError("frequencyHz must be finite")
        coherentGain = float(np.sum(realWindow))
        if coherentGain <= np.finfo(float).tiny:
            raise ValueError("analysisWindow must have positive coherent gain")
        sampleIndices = np.arange(complexInput.size, dtype=float)
        projection = np.exp(
            -1j
            * 2.0
            * np.pi
            * float(frequencyHz)
            * sampleIndices
            / self.waveform.sampleRateHz
        )
        return complex(
            np.sum(realWindow * complexInput * projection) / coherentGain
        )

    def CalculateOutputPowerDbm(self, inputSignal: np.ndarray) -> float:
        """Calculate active continuous-record PA output power in dBm.

        Processing details:
            Algorithm: Decode the public samples before any arithmetic, remove
            the same settling edges used for spectral measurement, detect the
            active two-tone interval while excluding long padding or idle
            runs, calculate its complex-envelope RMS, and map normalized RMS
            one to the configured maximum PA power.

        Args:
            inputSignal: Floating or integer-code PA output vector.

        Returns:
            result: Average steady-state PA output power in dBm.
        """

        publicSignal = np.asarray(
            inputSignal, dtype=np.complex128
        ).reshape(-1)
        if publicSignal.size == 0 or not np.all(np.isfinite(publicSignal)):
            raise ValueError(
                "inputSignal must contain finite nonempty samples"
            )
        resolvedWidth = self.width
        widthWasExplicitlyConfigured = any(
            "width" in parameterLayer
            for parameterLayer in self.parameters.maps[:-1]
        )
        if not widthWasExplicitlyConfigured and self.waveform.width == 0:
            componentValues = np.concatenate(
                (publicSignal.real, publicSignal.imag)
            )
            integerCodeShape = bool(
                np.all(
                    np.isclose(
                        componentValues,
                        np.rint(componentValues),
                        rtol=0.0,
                        atol=1.0e-9,
                    )
                )
            )
            exceedsNormalizedRange = bool(
                np.max(np.abs(componentValues)) > 1.0 + 1.0e-9
            )
            if integerCodeShape and exceedsNormalizedRange:
                # A floating transmit reference may accompany the project's
                # default 16-bit receiver capture. Resolve that common case at
                # the measurement boundary instead of treating DAC codes as
                # normalized amplitudes and adding about 90.31 dB to power.
                resolvedWidth = 16
        decodedSignal = FixedPoint(
            resolvedWidth, self.outputFullScaleAmplitude
        ).DecodeComplex(
            publicSignal
        ).reshape(-1)
        settlingSamples = cast(int, self.parameters["settlingSamples"])
        steadySignal = (
            decodedSignal
            if settlingSamples == 0
            else decodedSignal[settlingSamples:-settlingSamples]
        )
        powerCalibration = PowerCalibration(
            parameters={
                "loadResistanceOhm": self.parameters["loadResistanceOhm"],
                "maximumOutputPowerDbm": self.parameters[
                    "maximumOutputPowerDbm"
                ],
                "activePowerThresholdDb": self.parameters[
                    "activePowerThresholdDb"
                ],
                "activeGapToleranceSamples": self.parameters[
                    "activeGapToleranceSamples"
                ],
                "width": 0,
            }
        )
        signalRms = powerCalibration.CalculateActiveRmsPerChain(
            steadySignal
        )[0]
        return powerCalibration.NormalizedRmsToOutputPowerDbm(signalRms)

    def Analyze(self, measuredSignal: np.ndarray) -> TwoToneMetrics:
        """Measure fundamentals and paired IM3, IM5, and IM7 levels.

        Processing details:
            Algorithm: Decode and edge-trim the PA record, project it at the
            exact two fundamentals and six odd-order product frequencies,
            convert squared amplitudes to dBc relative to the adjacent carrier,
            report the worse side of each order and the worst order overall,
            and independently calculate average output power.

        Args:
            measuredSignal: Floating or fixed-point PA output waveform.

        Returns:
            result: Ordinary dictionary of fundamental, IM, and power metrics.
        """

        self.ValidateParameters()
        publicSignal = np.asarray(
            measuredSignal, dtype=np.complex128
        ).reshape(-1)
        if publicSignal.size == 0 or not np.all(np.isfinite(publicSignal)):
            raise ValueError(
                "measuredSignal must contain finite nonempty samples"
            )
        resolvedWidth = self.width
        widthWasExplicitlyConfigured = any(
            "width" in parameterLayer
            for parameterLayer in self.parameters.maps[:-1]
        )
        if not widthWasExplicitlyConfigured and self.waveform.width == 0:
            componentValues = np.concatenate(
                (publicSignal.real, publicSignal.imag)
            )
            integerCodeShape = bool(
                np.all(
                    np.isclose(
                        componentValues,
                        np.rint(componentValues),
                        rtol=0.0,
                        atol=1.0e-9,
                    )
                )
            )
            exceedsNormalizedRange = bool(
                np.max(np.abs(componentValues)) > 1.0 + 1.0e-9
            )
            if integerCodeShape and exceedsNormalizedRange:
                resolvedWidth = 16
        decodedSignal = FixedPoint(
            resolvedWidth, self.outputFullScaleAmplitude
        ).DecodeComplex(
            publicSignal
        ).reshape(-1)
        decodedSignal = decodedSignal / self.outputFullScaleAmplitude
        if (
            decodedSignal.size != self.waveform.numSamples
            or not np.all(np.isfinite(decodedSignal))
        ):
            raise ValueError(
                "measuredSignal must be finite and match waveform.numSamples"
            )
        settlingSamples = cast(int, self.parameters["settlingSamples"])
        steadySignal = (
            decodedSignal
            if settlingSamples == 0
            else decodedSignal[settlingSamples:-settlingSamples]
        )
        analysisWindow = self.BuildAnalysisWindow(steadySignal.size)
        lowerToneHz, upperToneHz = self.waveform.toneFrequenciesHz
        lowerFundamental = self.CalculateToneCoefficient(
            steadySignal, lowerToneHz, analysisWindow
        )
        upperFundamental = self.CalculateToneCoefficient(
            steadySignal, upperToneHz, analysisWindow
        )
        minimumPower = float(self.parameters["minimumSpectralPower"])
        lowerFundamentalPower = max(
            abs(lowerFundamental) ** 2, minimumPower
        )
        upperFundamentalPower = max(
            abs(upperFundamental) ** 2, minimumPower
        )
        averageFundamentalPower = 0.5 * (
            lowerFundamentalPower + upperFundamentalPower
        )
        metrics: Dict[str, float] = {
            "fundamentalLowerDbfs": float(
                10.0 * np.log10(lowerFundamentalPower)
            ),
            "fundamentalUpperDbfs": float(
                10.0 * np.log10(upperFundamentalPower)
            ),
            "fundamentalAverageDbfs": float(
                10.0 * np.log10(averageFundamentalPower)
            ),
        }
        worstValues = []
        for nonlinearOrder in (3, 5, 7):
            lowerProductHz, upperProductHz = (
                self.waveform.IntermodulationFrequencies(nonlinearOrder)
            )
            lowerProductPower = max(
                abs(
                    self.CalculateToneCoefficient(
                        steadySignal,
                        lowerProductHz,
                        analysisWindow,
                    )
                )
                ** 2,
                minimumPower,
            )
            upperProductPower = max(
                abs(
                    self.CalculateToneCoefficient(
                        steadySignal,
                        upperProductHz,
                        analysisWindow,
                    )
                )
                ** 2,
                minimumPower,
            )
            lowerDbc = float(
                10.0
                * np.log10(lowerProductPower / lowerFundamentalPower)
            )
            upperDbc = float(
                10.0
                * np.log10(upperProductPower / upperFundamentalPower)
            )
            worstDbc = max(lowerDbc, upperDbc)
            metrics[f"im{nonlinearOrder}LowerDbc"] = lowerDbc
            metrics[f"im{nonlinearOrder}UpperDbc"] = upperDbc
            metrics[f"im{nonlinearOrder}WorstDbc"] = worstDbc
            worstValues.append(worstDbc)
        metrics["worstIntermodulationDbc"] = max(worstValues)
        metrics["outputPowerDbm"] = self.CalculateOutputPowerDbm(
            measuredSignal
        )
        self.lastMetrics = cast(TwoToneMetrics, metrics)
        return cast(TwoToneMetrics, dict(metrics))

    def AnalyzeIlcHistory(
        self, ilcHistory: Sequence[Any]
    ) -> TwoToneILCAnalysisResult:
        """Analyze every native ILC output and select the best IM candidate.

        Processing details:
            Algorithm: Preserve native iteration order, independently calculate
            IM3, IM5, and IM7 on each stored PA output, minimize the maximum of
            the three worse-side products, and copy the corresponding input and
            output signals for later equal-power deployment comparison.

        Args:
            ilcHistory: Ordered objects exposing native ILC iteration fields.

        Returns:
            result: Complete analyzed history and the IM-best measured record.
        """

        historyRecords = tuple(ilcHistory)
        if not historyRecords:
            raise ValueError("ilcHistory cannot be empty")
        analyzedHistory = []
        bestIndex = 0
        bestScoreDbc = np.inf
        previousIteration = 0
        for recordIndex, iterationRecord in enumerate(historyRecords):
            iteration = int(iterationRecord.iteration)
            if iteration <= previousIteration:
                raise ValueError(
                    "ILC iterations must be strictly increasing"
                )
            previousIteration = iteration
            metrics = self.Analyze(iterationRecord.outputSignal)
            analyzedHistory.append(
                TwoToneILCIteration(
                    iteration=iteration,
                    nmseDb=float(iterationRecord.nmseDb),
                    linearCompensatedNmseDb=float(
                        iterationRecord.linearCompensatedNmseDb
                    ),
                    metrics=metrics,
                )
            )
            scoreDbc = metrics["worstIntermodulationDbc"]
            if scoreDbc < bestScoreDbc:
                bestScoreDbc = scoreDbc
                bestIndex = recordIndex
        bestRecord = historyRecords[bestIndex]
        return TwoToneILCAnalysisResult(
            history=tuple(analyzedHistory),
            bestIteration=int(bestRecord.iteration),
            bestInputSignal=np.asarray(
                bestRecord.inputSignal, dtype=np.complex128
            ).copy(),
            bestOutputSignal=np.asarray(
                bestRecord.outputSignal, dtype=np.complex128
            ).copy(),
            bestMetrics=analyzedHistory[bestIndex].metrics,
        )

    def Print(self, metrics: Optional[TwoToneMetrics] = None) -> None:
        """Print one compact IM3, IM5, IM7, and output-power summary.

        Processing details:
            Algorithm: Use caller metrics or the most recent analysis and print
            worse-side dBc values with the convention that more-negative
            intermodulation is better.

        Args:
            metrics: Optional result dictionary; None uses ``lastMetrics``.

        Returns:
            result: None. Human-readable metrics are written to standard output.
        """

        selectedMetrics = self.lastMetrics if metrics is None else metrics
        if selectedMetrics is None:
            raise RuntimeError("Analyze must run before Print without metrics")
        print(
            "Two-tone metrics: "
            f"IM3={selectedMetrics['im3WorstDbc']:.2f} dBc, "
            f"IM5={selectedMetrics['im5WorstDbc']:.2f} dBc, "
            f"IM7={selectedMetrics['im7WorstDbc']:.2f} dBc, "
            f"Pout={selectedMetrics['outputPowerDbm']:.2f} dBm"
        )

    def SaveIlcHistory(
        self,
        analysisResult: TwoToneILCAnalysisResult,
        outputDirectory: Path,
        fileStem: str,
    ) -> Tuple[Path, Path]:
        """Save per-iteration IM convergence to matching CSV and JSON files.

        Processing details:
            Algorithm: Validate a simple file stem, flatten each analyzed
            iteration once, create the output directory, and serialize identical
            numerical records plus the selected best iteration in both formats.

        Args:
            analysisResult: Independently analyzed two-tone ILC history.
            outputDirectory: Destination directory for result files.
            fileStem: Simple base name used by both serialized artifacts.

        Returns:
            result: CSV path followed by JSON path.
        """

        if not isinstance(analysisResult, TwoToneILCAnalysisResult):
            raise TypeError(
                "analysisResult must be a TwoToneILCAnalysisResult"
            )
        if (
            not isinstance(fileStem, str)
            or not fileStem
            or any(character in fileStem for character in '<>:"/\\|?*')
        ):
            raise ValueError("fileStem must be a valid simple file name")
        outputPath = Path(outputDirectory)
        outputPath.mkdir(parents=True, exist_ok=True)
        csvPath = outputPath / f"{fileStem}.csv"
        jsonPath = outputPath / f"{fileStem}.json"
        flatRows = [historyRecord.ToDict() for historyRecord in analysisResult.history]
        with csvPath.open("w", newline="", encoding="utf-8-sig") as csvFile:
            csvWriter = csv.DictWriter(
                csvFile, fieldnames=list(flatRows[0].keys())
            )
            csvWriter.writeheader()
            csvWriter.writerows(flatRows)
        with jsonPath.open("w", encoding="utf-8") as jsonFile:
            json.dump(
                {
                    "bestIteration": analysisResult.bestIteration,
                    "bestMetrics": dict(analysisResult.bestMetrics),
                    "history": flatRows,
                },
                jsonFile,
                ensure_ascii=False,
                indent=2,
            )
        return csvPath, jsonPath
