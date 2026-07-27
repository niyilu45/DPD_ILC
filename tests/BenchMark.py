"""Scenario-based performance benchmark for every supported ILC method.

This test module owns waveform construction, impairment scenarios, metric
comparison, convergence reporting, held-out deployment validation, and the
multi-method power-EVM curve. Production ILC algorithms remain in
``inc.lib.DpdIlc`` and do not depend on this benchmark harness.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np


def GetProjectRoot() -> Path:
    """Return the repository root without retaining module-level state.

    Processing details:
        Algorithm: Resolve this benchmark file and select its parent
        repository directory whenever imports or output paths need it.

    Returns:
        result: Absolute repository directory containing ``inc`` and
        ``tests``.
    """

    return Path(__file__).resolve().parents[1]


if str(GetProjectRoot()) not in sys.path:
    sys.path.insert(0, str(GetProjectRoot()))

from inc.lib.Analysis import Analysis, ILCAnalysisResult, SignalMetrics
from inc.utils.Draw import Draw
from inc.lib.DpdIlc import (
    FitGmpPredistorter,
    FitLutPredistorter,
    FitNeuralPredistorter,
    FitVolterraPredistorter,
    ILCConfig,
    ILCResult,
    LimitAmplitude,
    RunAugmentedIqIlc,
    RunComplexGainIlc,
    RunDirectionalGaussNewtonIlc,
    RunFirIlc,
    RunFrequencyDomainIlc,
    RunParameterDomainIlc,
    RunScalarPIlc,
)
from inc.lib.PaModel import IQImbalancePA, PaModel
from inc.utils.FixedPoint import FixedPoint
from inc.utils.SigProc import PowerCalibration
from inc.lib.WaveGenWifi import WaveGenWifi
from inc.utils.WifiMetadata import WifiWaveform


# =============================================================================
# All-method ILC benchmark and reporting
# =============================================================================

@dataclass(frozen=True)
class BenchmarkConfig:
    """Configure a deterministic, representative all-method comparison."""

    frameFormat: str = "EHT"
    bandwidthMhz: int = 20
    mcs: int = 7
    numDataSymbols: int = 10
    sampleRateHz: Optional[float] = None
    oversampling: int = 4
    width: int = 16
    guardIntervalUs: float = 0.8
    outputPowerDbm: float = 20.0
    maximumOutputPowerDbm: float = 25.0
    loadResistanceOhm: float = 50.0
    numIterations: int = 10
    paModelName: str = "wiener"
    seed: int = 101
    powerStartDbm: float = 10.0
    powerStopDbm: float = 25.0
    powerPointCount: int = 5
    generatePowerEvmCurve: bool = True
    outputDirectory: Path = Path("results/all_ilc_benchmark")

    def Validate(self) -> None:
        """Validate waveform, iteration, sweep, and output parameters.

        Processing details:
            Algorithm: Reject nonphysical waveform or power settings before
            any long-running scenario starts. PHY-specific format, bandwidth,
            MCS, and guard-interval combinations are validated later by
            ``WaveGenWifi`` using the same configuration.

        Returns:
            result: None. Invalid settings raise ``ValueError`` with the
            parameter responsible for the failure.
        """

        if self.numDataSymbols < 1:
            raise ValueError("numDataSymbols must be positive")
        if (
            not isinstance(self.oversampling, int)
            or isinstance(self.oversampling, bool)
            or self.oversampling < 1
        ):
            raise ValueError("oversampling must be a positive integer")
        if self.sampleRateHz is not None and (
            not isinstance(self.sampleRateHz, (int, float))
            or isinstance(self.sampleRateHz, bool)
            or not np.isfinite(self.sampleRateHz)
            or self.sampleRateHz <= 0.0
        ):
            raise ValueError(
                "sampleRateHz must be finite and positive or None"
            )
        if (
            not isinstance(self.width, int)
            or isinstance(self.width, bool)
            or self.width < 0
            or self.width > 53
        ):
            raise ValueError("width must be an integer from zero through 53")
        bandwidthHz = float(self.bandwidthMhz) * 1.0e6
        effectiveSampleRateHz = (
            bandwidthHz * float(self.oversampling)
            if self.sampleRateHz is None
            else float(self.sampleRateHz)
        )
        if effectiveSampleRateHz < 3.0 * bandwidthHz:
            raise ValueError(
                "sampleRateHz must be at least three times the channel "
                "bandwidth for ACLR analysis"
            )
        powerCalibration = PowerCalibration(
            loadResistanceOhm=self.loadResistanceOhm,
            maximumOutputPowerDbm=self.maximumOutputPowerDbm,
        )
        powerCalibration.OutputPowerToDriveScale(
            self.outputPowerDbm
        )
        if self.numIterations < 1:
            raise ValueError("numIterations must be positive")
        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or self.seed < 0
            or self.seed + 97 >= 2**10
        ):
            raise ValueError(
                "seed must be from zero through 926 so the independent "
                "validation waveform remains inside the 10-bit range"
            )
        powerCalibration.OutputPowerToDriveScale(
            self.powerStartDbm
        )
        powerCalibration.OutputPowerToDriveScale(
            self.powerStopDbm
        )
        if self.powerStopDbm <= self.powerStartDbm:
            raise ValueError("powerStopDbm must exceed powerStartDbm")
        if self.powerPointCount < 2:
            raise ValueError("powerPointCount must be at least two")


@dataclass(frozen=True)
class BenchmarkRow:
    """Store one method result and its improvement over scenario baseline."""

    methodName: str
    category: str
    scenario: str
    metrics: SignalMetrics
    snrImprovementDb: float
    evmImprovementDb: float
    aclrImprovementDb: float

    def ToDict(self) -> Dict[str, object]:
        """Convert a row to flat JSON/CSV-compatible values.

        Processing details:
            Algorithm: Copy scenario identity and signed improvement fields,
            merge the ordinary ``SignalMetrics`` dictionary without changing
            stored numerical values.

        Returns:
            result: One flat mapping suitable for a CSV row or JSON object.
        """

        rowData: Dict[str, object] = {
            "methodName": self.methodName,
            "category": self.category,
            "scenario": self.scenario,
            "snrImprovementDb": self.snrImprovementDb,
            "evmImprovementDb": self.evmImprovementDb,
            "aclrImprovementDb": self.aclrImprovementDb,
        }
        rowData.update(self.metrics)
        return rowData

def AddRow(
    rows: List[BenchmarkRow],
    methodName: str,
    category: str,
    scenario: str,
    metrics: SignalMetrics,
    baselineMetrics: SignalMetrics,
) -> None:
    """Append metrics and consistently signed improvements to the result table.

    Processing details:
        Algorithm: Subtract baseline from method for SNR and ACLR, subtract
        method EVM dB from baseline EVM dB because more-negative EVM is
        better, then append one immutable row carrying absolute and relative
        values.

    Args:
        rows: Benchmark rows accumulated or emitted by the reporting operation.
        methodName: Human-readable algorithm or deployment-model label.
        category: Group used to separate baselines, update laws, and deployed
            label models.
        scenario: Description of the impairment or validation scenario.
        metrics: Signal-quality metrics calculated for the selected output.
        baselineMetrics: Reference metrics used to calculate improvements.

    Returns:
        result: None. The caller-owned row list receives one new record.
    """

    rows.append(
        BenchmarkRow(
            methodName=methodName,
            category=category,
            scenario=scenario,
            metrics=metrics,
            snrImprovementDb=(
                metrics["snrDb"] - baselineMetrics["snrDb"]
            ),
            # More-negative EVM dB is better, so baseline minus result is a
            # positive improvement.
            evmImprovementDb=(
                baselineMetrics["evmDb"] - metrics["evmDb"]
            ),
            aclrImprovementDb=(
                metrics["aclrWorstDb"]
                - baselineMetrics["aclrWorstDb"]
            ),
        )
    )

def SaveHistory(
    methodName: str,
    analysisResult: ILCAnalysisResult,
    outputDirectory: Path,
) -> None:
    """Save a separate convergence CSV without overwriting other methods.

    Processing details:
        Algorithm: Sanitize the method label into a stable file stem, write
        every post-ILC performance record without recalculation, then ask
        ``Draw`` to render the same independently analyzed history.

    Args:
        methodName: Human-readable algorithm or deployment-model label.
        analysisResult: Post-ILC RF performance history and best candidate.
        outputDirectory: Directory in which result artifacts are written.

    Returns:
        result: None. One CSV and one PNG are written to the output directory.
    """

    safeName = "".join(
        character.lower() if character.isalnum() else "_"
        for character in methodName
    ).strip("_")
    historyPath = outputDirectory / f"convergence_{safeName}.csv"
    with historyPath.open("w", newline="", encoding="utf-8-sig") as csvFile:
        csvWriter = csv.DictWriter(
            csvFile,
            fieldnames=(
                "iteration",
                "mse",
                "errorRms",
                "nmseDb",
                "linearCompensatedMse",
                "linearCompensatedNmseDb",
                "outputPowerDbm",
                "snrDb",
                "evmAlignedMse",
                "evmDb",
                "evmPercent",
                "aclrLowerDb",
                "aclrUpperDb",
                "aclrWorstDb",
                "complexGainMagnitudeDb",
                "complexGainPhaseDegrees",
                "feedbackIntegerDelaySamples",
                "feedbackFractionalDelaySamples",
                "feedbackCarrierFrequencyOffsetHz",
                "feedbackSamplingFrequencyOffsetPpm",
                "feedbackComplexGainMagnitudeDb",
                "feedbackComplexGainPhaseDegrees",
                "inputPeak",
            ),
        )
        csvWriter.writeheader()
        for iterationRecord in analysisResult.history:
            csvWriter.writerow(iterationRecord.ToDict())
    Draw(convergenceFileStem=f"convergence_{safeName}").SaveConvergenceCurve(
        analysisResult.history, outputDirectory
    )

def ReportHistory(
    methodName: str,
    ilcResult: ILCResult,
    resultAnalysis: Analysis,
    outputDirectory: Path,
) -> ILCAnalysisResult:
    """Print and save one method's complete per-iteration MSE history.

    Processing details:
        Algorithm: Feed every PA output stored by ILC into ``Analysis``, select
        the EVM-best measured candidate outside the algorithm, then print,
        serialize, and render the same immutable performance records.

    Args:
        methodName: Human-readable ILC method label.
        ilcResult: Completed ILC result containing ordered history records.
        resultAnalysis: Analysis instance used for consistent presentation.
        outputDirectory: Destination for CSV and PNG result artifacts.

    Returns:
        result: Post-ILC history and the EVM-best measured candidate.
    """

    analysisResult = resultAnalysis.AnalyzeIlcHistory(ilcResult.history)
    resultAnalysis.PrintConvergence(
        analysisResult.history, f"{methodName} iteration metrics"
    )
    SaveHistory(methodName, analysisResult, outputDirectory)
    return analysisResult

def EvaluateDeployment(
    predistorter: Any,
    validationSignal: np.ndarray,
    paModel: Any,
    resultAnalysis: Analysis,
    maxAmplitude: float,
) -> SignalMetrics:
    """Evaluate one fitted DPD on a held-out Wi-Fi packet.

    Processing details:
        Algorithm: Run the fitted predistorter on a packet generated with an
        independent seed, project the complex envelope onto the configured
        amplitude disk, pass it through the unchanged PA, and analyze the
        resulting signal with validation-frame metadata.

    Args:
        predistorter: Fitted model exposing a ``Process`` method.
        validationSignal: Independent complex waveform used to evaluate generalization.
        paModel: PA object exposing Process and SmallSignalGain operations.
        resultAnalysis: Analyzer bound to the independent validation packet.
        maxAmplitude: Maximum allowed complex-envelope magnitude.

    Returns:
        result: SNR, EVM, and ACLR of the held-out PA output.
    """

    interfaceFormat = FixedPoint(int(getattr(paModel, "width", 0)))
    floatingValidationSignal = interfaceFormat.DecodeComplex(
        validationSignal
    )
    floatingPredistortedInput = LimitAmplitude(
        predistorter.Process(floatingValidationSignal), maxAmplitude
    )
    predistortedInput = interfaceFormat.EncodeComplex(
        floatingPredistortedInput
    )
    paOutput = paModel.Process(predistortedInput)
    return resultAnalysis.Analyze(paOutput)

def RunIlcCurvePoint(
    referenceSignal: np.ndarray,
    outputPowerDbm: float,
    paModel: Any,
    waveform: WifiWaveform,
    width: int,
    methodName: str,
    methodFunction: Optional[Callable[..., ILCResult]],
    methodConfig: ILCConfig,
) -> np.ndarray:
    """Run one selected ILC method at one power-EVM sweep point.

    Processing details:
        Algorithm: Run ILC without any RF metric callback, construct a fresh
        Analysis context after power scaling, analyze every stored PA output,
        and return the externally selected minimum-EVM measured candidate.

    Args:
        referenceSignal: Ideal complex baseband samples used as the target or regression input.
        outputPowerDbm: Current absolute PA output power in dBm.
        paModel: PA object exposing Process and SmallSignalGain operations.
        waveform: Wi-Fi metadata defining field locations, FFT sizes, and subcarriers.
        width: External I/Q component width shared by all benchmark modules.
        methodName: Human-readable algorithm or deployment-model label.
        methodFunction: Selected ILC update-law callable. ``None`` selects
            the dedicated frequency-domain call path, which also needs the
            waveform sample rate and occupied bandwidth.
        methodConfig: Validated ILC configuration for the selected update law.

    Returns:
        result: Complex PA output learned specifically for this power point.
    """

    del outputPowerDbm
    pointAnalysis = Analysis(
        referenceSignal,
        waveform,
        parameters={"width": width},
    )
    if methodName == "Frequency-domain ILC":
        methodResult = RunFrequencyDomainIlc(
            referenceSignal,
            paModel,
            waveform.sampleRateHz,
            waveform.bandwidthHz,
            methodConfig,
        )
    else:
        if methodFunction is None:
            raise ValueError(
                "methodFunction is required for non-frequency ILC"
            )
        methodResult = methodFunction(
            referenceSignal,
            paModel,
            methodConfig,
        )
    analysisResult = pointAnalysis.AnalyzeIlcHistory(
        methodResult.history
    )
    return paModel.Process(analysisResult.bestInputSignal)

def RunAllIlcBenchmark(
    config: Optional[BenchmarkConfig] = None,
) -> List[BenchmarkRow]:
    """Run every update law and every ILC-label deployment model.

    Processing details:
        Algorithm: Construct a nominal repeated-packet scenario, a
        peak-constrained scenario, a noisy-feedback scenario, an IQ-image
        scenario, and an independent-packet deployment scenario. Each
        specialized plant includes a matching baseline and at least one
        structurally simpler comparison method. Run all applicable ILC
        methods under controlled settings, calculate SNR, EVM, and ACLR,
        save convergence histories, and optionally generate the common
        power-EVM comparison.

    Args:
        config: Optional caller overrides. ``None`` creates defaults inside
            this function so call sites never reconstruct a default layer.

    Returns:
        result: Ordered benchmark rows containing absolute metrics and
        improvements relative to each scenario's matching baseline.
    """

    if config is None:
        config = BenchmarkConfig()
    config.Validate()
    outputDirectory = Path(config.outputDirectory)
    outputDirectory.mkdir(parents=True, exist_ok=True)
    sharedWifiParameters = {
        "frameFormat": config.frameFormat,
        "bandwidthMhz": config.bandwidthMhz,
        "mcs": config.mcs,
        "numDataSymbols": config.numDataSymbols,
        "guardIntervalUs": config.guardIntervalUs,
        "width": config.width,
    }
    if config.sampleRateHz is None:
        sharedWifiParameters["oversampling"] = config.oversampling
    else:
        sharedWifiParameters["sampleRateHz"] = config.sampleRateHz
    trainingParameters = dict(sharedWifiParameters)
    trainingParameters["seed"] = config.seed
    validationParameters = dict(sharedWifiParameters)
    validationParameters["seed"] = config.seed + 97
    trainingGenerator = WaveGenWifi(parameters=trainingParameters)
    validationGenerator = WaveGenWifi(parameters=validationParameters)
    trainingWaveform = trainingGenerator.Generate()
    validationWaveform = validationGenerator.Generate()
    powerCalibration = PowerCalibration(
        loadResistanceOhm=config.loadResistanceOhm,
        maximumOutputPowerDbm=config.maximumOutputPowerDbm,
    )
    driveScale = powerCalibration.OutputPowerToDriveScale(
        config.outputPowerDbm
    )
    trainingSignal = driveScale * trainingWaveform.samples
    validationSignal = driveScale * validationWaveform.samples
    interfaceFormat = FixedPoint(config.width)
    floatingTrainingSignal = interfaceFormat.DecodeComplex(trainingSignal)
    paParameters = {
        "modelName": config.paModelName,
        "width": config.width,
    }
    paModel = PaModel(parameters=paParameters)
    trainingAnalysis = Analysis(
        trainingSignal,
        trainingWaveform,
        parameters={
            "loadResistanceOhm": config.loadResistanceOhm,
            "width": config.width,
        },
    )
    validationAnalysis = Analysis(
        validationSignal,
        validationWaveform,
        parameters={
            "loadResistanceOhm": config.loadResistanceOhm,
            "width": config.width,
        },
    )
    maxAmplitude = max(
        2.0, 1.6 * np.max(np.abs(floatingTrainingSignal))
    )

    baselineOutput = paModel.Process(trainingSignal)
    baselineMetrics = trainingAnalysis.Analyze(baselineOutput)
    powerEvaluators = {
        "PA baseline": lambda pointReference, _: paModel.Process(
            pointReference
        )
    }
    rows: List[BenchmarkRow] = []
    AddRow(
        rows,
        "PA baseline",
        "baseline",
        "nominal repeated waveform",
        baselineMetrics,
        baselineMetrics,
    )

    # Each algorithm receives tuned but conservative learning parameters. The
    # waveform, PA, iteration budget, and metrics remain identical.
    scalarConfig = ILCConfig(
        numIterations=config.numIterations,
        learningRate=0.10,
        maxAmplitude=maxAmplitude,
        randomSeed=config.seed + 1,
    )
    complexConfig = ILCConfig(
        numIterations=config.numIterations,
        learningRate=0.15,
        maxAmplitude=maxAmplitude,
        randomSeed=config.seed + 2,
    )
    firConfig = ILCConfig(
        numIterations=config.numIterations,
        learningRate=0.15,
        maxAmplitude=maxAmplitude,
        randomSeed=config.seed + 3,
    )
    frequencyConfig = ILCConfig(
        numIterations=config.numIterations,
        learningRate=0.15,
        maxAmplitude=maxAmplitude,
        randomSeed=config.seed + 4,
    )
    gaussNewtonConfig = ILCConfig(
        numIterations=config.numIterations,
        learningRate=0.65,
        maxAmplitude=maxAmplitude,
        randomSeed=config.seed + 5,
    )
    parameterConfig = ILCConfig(
        numIterations=config.numIterations,
        learningRate=0.20,
        maxAmplitude=maxAmplitude,
        randomSeed=config.seed + 6,
    )

    methodRuns = (
        ("Scalar P ILC", RunScalarPIlc, scalarConfig),
        ("Complex-gain ILC", RunComplexGainIlc, complexConfig),
        ("FIR ILC", RunFirIlc, firConfig),
        ("Frequency-domain ILC", None, frequencyConfig),
        (
            "Directional Gauss-Newton ILC",
            RunDirectionalGaussNewtonIlc,
            gaussNewtonConfig,
        ),
        ("Parameter-domain MP ILC", RunParameterDomainIlc, parameterConfig),
    )
    frequencyResult = None
    frequencyAnalysisResult = None
    for methodName, methodFunction, methodConfig in methodRuns:
        if methodName == "Frequency-domain ILC":
            methodResult = RunFrequencyDomainIlc(
                trainingSignal,
                paModel,
                trainingWaveform.sampleRateHz,
                trainingWaveform.bandwidthHz,
                methodConfig,
            )
            frequencyResult = methodResult
        else:
            methodResult = methodFunction(
                trainingSignal,
                paModel,
                methodConfig,
            )
        methodAnalysisResult = ReportHistory(
            methodName,
            methodResult,
            trainingAnalysis,
            outputDirectory,
        )
        methodMetrics = trainingAnalysis.Analyze(
            paModel.Process(methodAnalysisResult.bestInputSignal)
        )
        if methodName == "Frequency-domain ILC":
            frequencyAnalysisResult = methodAnalysisResult
        AddRow(
            rows,
            methodName,
            "ILC update law",
            "nominal repeated waveform",
            methodMetrics,
            baselineMetrics,
        )
        powerEvaluators[methodName] = (
            lambda pointReference,
            pointDrive,
            selectedName=methodName,
            selectedFunction=methodFunction,
            selectedConfig=methodConfig: RunIlcCurvePoint(
                pointReference,
                pointDrive,
                paModel,
                trainingWaveform,
                config.width,
                selectedName,
                selectedFunction,
                selectedConfig,
            )
        )

    if frequencyResult is None or frequencyAnalysisResult is None:
        raise RuntimeError("frequency-domain ILC result was not generated")

    # Repeat the physically identical baseline and unconstrained result under
    # the peak scenario label so scenario-filtered reports contain the full
    # baseline-versus-performance-versus-feasibility comparison.
    frequencyMetrics = trainingAnalysis.Analyze(
        paModel.Process(frequencyAnalysisResult.bestInputSignal)
    )
    AddRow(
        rows,
        "Peak-constrained baseline",
        "baseline",
        "peak-constrained waveform",
        baselineMetrics,
        baselineMetrics,
    )
    AddRow(
        rows,
        "Unconstrained frequency-domain ILC",
        "ILC update law",
        "peak-constrained waveform",
        frequencyMetrics,
        baselineMetrics,
    )

    # Constrained ILC uses a peak only 5 percent above the original waveform.
    constrainedPeak = 1.05 * np.max(
        np.abs(floatingTrainingSignal)
    )
    constrainedResult = RunFrequencyDomainIlc(
        trainingSignal,
        paModel,
        trainingWaveform.sampleRateHz,
        trainingWaveform.bandwidthHz,
        ILCConfig(
            numIterations=config.numIterations,
            learningRate=0.12,
            maxAmplitude=constrainedPeak,
            randomSeed=config.seed + 7,
        ),
    )
    constrainedAnalysisResult = ReportHistory(
        "Constrained CFR-ILC",
        constrainedResult,
        trainingAnalysis,
        outputDirectory,
    )
    constrainedMetrics = trainingAnalysis.Analyze(
        paModel.Process(constrainedAnalysisResult.bestInputSignal)
    )
    AddRow(
        rows,
        "Constrained CFR-ILC",
        "ILC update law",
        "peak-constrained waveform",
        constrainedMetrics,
        baselineMetrics,
    )
    powerEvaluators["Constrained CFR-ILC"] = (
        lambda pointReference, pointDrive: RunIlcCurvePoint(
            pointReference,
            pointDrive,
            paModel,
            trainingWaveform,
            config.width,
            "Frequency-domain ILC",
            None,
            ILCConfig(
                numIterations=config.numIterations,
                learningRate=0.12,
                maxAmplitude=1.05
                * np.max(
                    np.abs(
                        interfaceFormat.DecodeComplex(pointReference)
                    )
                ),
                randomSeed=config.seed + 7,
            ),
        )
    )

    # The noise scenario compares an ordinary single-capture frequency-domain
    # ILC against stronger regularization and four-capture feedback averaging.
    noisyBaselineMetrics = baselineMetrics
    AddRow(
        rows,
        "Noisy-feedback baseline",
        "baseline",
        "32 dB feedback robustness",
        noisyBaselineMetrics,
        noisyBaselineMetrics,
    )
    naiveNoisyResult = RunFrequencyDomainIlc(
        trainingSignal,
        paModel,
        trainingWaveform.sampleRateHz,
        trainingWaveform.bandwidthHz,
        ILCConfig(
            numIterations=config.numIterations,
            learningRate=0.15,
            regularization=1e-3,
            maxAmplitude=maxAmplitude,
            feedbackSnrDb=32.0,
            feedbackAverages=1,
            randomSeed=config.seed + 18,
        ),
    )
    naiveNoisyAnalysisResult = ReportHistory(
        "Naive noisy-feedback ILC",
        naiveNoisyResult,
        trainingAnalysis,
        outputDirectory,
    )
    naiveNoisyMetrics = trainingAnalysis.Analyze(
        paModel.Process(naiveNoisyAnalysisResult.bestInputSignal)
    )
    AddRow(
        rows,
        "Naive noisy-feedback ILC",
        "ILC update law",
        "32 dB feedback robustness",
        naiveNoisyMetrics,
        noisyBaselineMetrics,
    )
    noiseAwareResult = RunFrequencyDomainIlc(
        trainingSignal,
        paModel,
        trainingWaveform.sampleRateHz,
        trainingWaveform.bandwidthHz,
        ILCConfig(
            numIterations=config.numIterations,
            learningRate=0.10,
            regularization=1e-2,
            maxAmplitude=maxAmplitude,
            feedbackSnrDb=32.0,
            feedbackAverages=4,
            randomSeed=config.seed + 8,
        ),
    )
    noiseAwareAnalysisResult = ReportHistory(
        "Noise-aware ILC",
        noiseAwareResult,
        trainingAnalysis,
        outputDirectory,
    )
    noiseAwareMetrics = trainingAnalysis.Analyze(
        paModel.Process(noiseAwareAnalysisResult.bestInputSignal)
    )
    AddRow(
        rows,
        "Noise-aware ILC",
        "ILC update law",
        "32 dB feedback robustness",
        noiseAwareMetrics,
        noisyBaselineMetrics,
    )
    powerEvaluators["Naive noisy-feedback ILC"] = (
        lambda pointReference, pointDrive: RunIlcCurvePoint(
            pointReference,
            pointDrive,
            paModel,
            trainingWaveform,
            config.width,
            "Frequency-domain ILC",
            None,
            ILCConfig(
                numIterations=config.numIterations,
                learningRate=0.15,
                regularization=1e-3,
                maxAmplitude=maxAmplitude,
                feedbackSnrDb=32.0,
                feedbackAverages=1,
                randomSeed=config.seed + 18,
            ),
        )
    )
    powerEvaluators["Noise-aware ILC"] = (
        lambda pointReference, pointDrive: RunIlcCurvePoint(
            pointReference,
            pointDrive,
            paModel,
            trainingWaveform,
            config.width,
            "Frequency-domain ILC",
            None,
            ILCConfig(
                numIterations=config.numIterations,
                learningRate=0.10,
                regularization=1e-2,
                maxAmplitude=maxAmplitude,
                feedbackSnrDb=32.0,
                feedbackAverages=4,
                randomSeed=config.seed + 8,
            ),
        )
    )

    # The IQ scenario compares an ordinary frequency-domain update against the
    # augmented direct-plus-conjugate inverse on exactly the same IQ plant.
    iqPaModel = IQImbalancePA(PaModel(parameters=paParameters))
    iqBaselineOutput = iqPaModel.Process(trainingSignal)
    iqBaselineMetrics = trainingAnalysis.Analyze(iqBaselineOutput)
    AddRow(
        rows,
        "IQ-imbalance baseline",
        "baseline",
        "IQ image impairment",
        iqBaselineMetrics,
        iqBaselineMetrics,
    )
    ordinaryIqResult = RunFrequencyDomainIlc(
        trainingSignal,
        iqPaModel,
        trainingWaveform.sampleRateHz,
        trainingWaveform.bandwidthHz,
        ILCConfig(
            numIterations=config.numIterations,
            learningRate=0.15,
            maxAmplitude=maxAmplitude,
            randomSeed=config.seed + 19,
        ),
    )
    ordinaryIqAnalysisResult = ReportHistory(
        "Frequency-domain ILC on IQ plant",
        ordinaryIqResult,
        trainingAnalysis,
        outputDirectory,
    )
    ordinaryIqMetrics = trainingAnalysis.Analyze(
        iqPaModel.Process(ordinaryIqAnalysisResult.bestInputSignal)
    )
    AddRow(
        rows,
        "Frequency-domain ILC on IQ plant",
        "ILC update law",
        "IQ image impairment",
        ordinaryIqMetrics,
        iqBaselineMetrics,
    )
    augmentedResult = RunAugmentedIqIlc(
        trainingSignal,
        iqPaModel,
        ILCConfig(
            numIterations=config.numIterations,
            learningRate=0.18,
            maxAmplitude=maxAmplitude,
            randomSeed=config.seed + 9,
        ),
    )
    augmentedAnalysisResult = ReportHistory(
        "Augmented IQ ILC",
        augmentedResult,
        trainingAnalysis,
        outputDirectory,
    )
    augmentedMetrics = trainingAnalysis.Analyze(
        iqPaModel.Process(augmentedAnalysisResult.bestInputSignal)
    )
    AddRow(
        rows,
        "Augmented IQ ILC",
        "ILC update law",
        "IQ image impairment",
        augmentedMetrics,
        iqBaselineMetrics,
    )
    powerEvaluators["IQ-imbalance baseline"] = (
        lambda pointReference, _: iqPaModel.Process(pointReference)
    )
    powerEvaluators["Frequency-domain ILC on IQ plant"] = (
        lambda pointReference, pointDrive: RunIlcCurvePoint(
            pointReference,
            pointDrive,
            iqPaModel,
            trainingWaveform,
            config.width,
            "Frequency-domain ILC",
            None,
            ILCConfig(
                numIterations=config.numIterations,
                learningRate=0.15,
                maxAmplitude=maxAmplitude,
                randomSeed=config.seed + 19,
            ),
        )
    )
    powerEvaluators["Augmented IQ ILC"] = (
        lambda pointReference, pointDrive: RunIlcCurvePoint(
            pointReference,
            pointDrive,
            iqPaModel,
            trainingWaveform,
            config.width,
            "Augmented IQ ILC",
            RunAugmentedIqIlc,
            ILCConfig(
                numIterations=config.numIterations,
                learningRate=0.18,
                maxAmplitude=maxAmplitude,
                randomSeed=config.seed + 9,
            ),
        )
    )

    # Fit every deployable model to the same converged ILC labels, then test
    # on a held-out Wi-Fi payload to measure generalization rather than recall.
    validationBaselineOutput = paModel.Process(validationSignal)
    validationBaselineMetrics = validationAnalysis.Analyze(
        validationBaselineOutput
    )
    AddRow(
        rows,
        "Validation baseline",
        "baseline",
        "held-out Wi-Fi packet",
        validationBaselineMetrics,
        validationBaselineMetrics,
    )
    deploymentModels = (
        (
            "ILC label + MP",
            FitGmpPredistorter(
                floatingTrainingSignal,
                interfaceFormat.DecodeComplex(
                    frequencyAnalysisResult.bestInputSignal
                ),
                nonlinearOrders=(1, 3, 5, 7),
                memoryDepth=3,
                crossMemoryDepth=0,
            ),
        ),
        (
            "ILC label + GMP",
            FitGmpPredistorter(
                floatingTrainingSignal,
                interfaceFormat.DecodeComplex(
                    frequencyAnalysisResult.bestInputSignal
                ),
                nonlinearOrders=(1, 3, 5, 7),
                memoryDepth=3,
                crossMemoryDepth=2,
            ),
        ),
        (
            "ILC label + Volterra",
            FitVolterraPredistorter(
                floatingTrainingSignal,
                interfaceFormat.DecodeComplex(
                    frequencyAnalysisResult.bestInputSignal
                ),
                memoryDepth=3,
            ),
        ),
        (
            "ILC label + LUT",
            FitLutPredistorter(
                floatingTrainingSignal,
                interfaceFormat.DecodeComplex(
                    frequencyAnalysisResult.bestInputSignal
                ),
                binCount=64,
            ),
        ),
        (
            "ILC label + NN",
            FitNeuralPredistorter(
                floatingTrainingSignal,
                interfaceFormat.DecodeComplex(
                    frequencyAnalysisResult.bestInputSignal
                ),
                memoryDepth=4,
                hiddenUnitCount=32,
                randomSeed=config.seed + 10,
            ),
        ),
    )
    for methodName, predistorter in deploymentModels:
        methodMetrics = EvaluateDeployment(
            predistorter,
            validationSignal,
            paModel,
            validationAnalysis,
            maxAmplitude,
        )
        AddRow(
            rows,
            methodName,
            "ILC label deployment",
            "held-out Wi-Fi packet",
            methodMetrics,
            validationBaselineMetrics,
        )
        powerEvaluators[methodName] = (
            lambda pointReference,
            _,
            selectedPredistorter=predistorter: paModel.Process(
                interfaceFormat.EncodeComplex(
                    LimitAmplitude(
                        selectedPredistorter.Process(
                            interfaceFormat.DecodeComplex(
                                pointReference
                            )
                        ),
                        maxAmplitude,
                    )
                )
            )
        )
    metadata: Mapping[str, object] = {
        "frameFormat": trainingWaveform.frameFormat,
        "bandwidthMhz": config.bandwidthMhz,
        "sampleRateHz": trainingWaveform.sampleRateHz,
        "mcs": config.mcs,
        "numDataSymbols": config.numDataSymbols,
        "oversampling": trainingWaveform.oversampling,
        "guardIntervalUs": config.guardIntervalUs,
        "outputPowerDbm": config.outputPowerDbm,
        "targetOutputRmsVoltage": powerCalibration.DbmToRms(
            config.outputPowerDbm
        ),
        "normalizedDriveScale": driveScale,
        "maximumOutputPowerDbm": config.maximumOutputPowerDbm,
        "loadResistanceOhm": config.loadResistanceOhm,
        "numIterations": config.numIterations,
        "paModel": config.paModelName,
        "trainingSeed": config.seed,
        "validationSeed": config.seed + 97,
        "outputPowerStartDbm": config.powerStartDbm,
        "outputPowerStopDbm": config.powerStopDbm,
        "powerPointCount": config.powerPointCount,
        "generatePowerEvmCurve": config.generatePowerEvmCurve,
    }
    SaveBenchmarkResults(rows, outputDirectory, metadata)
    powerCurvePaths = None
    if config.generatePowerEvmCurve:
        outputPowerDbmValues = np.linspace(
            config.powerStartDbm,
            config.powerStopDbm,
            config.powerPointCount,
        )
        powerEvmCurve = trainingAnalysis.AnalyzePowerEvmCurve(
            outputPowerDbmValues, powerEvaluators
        )
        powerDataPaths = trainingAnalysis.SavePowerEvmCurveData(
            outputDirectory,
            fileStem="all_ilc_power_evm_curve",
        )
        powerFigurePath = Draw(
            powerEvmFileStem="all_ilc_power_evm_curve"
        ).SavePowerEvmCurve(powerEvmCurve, outputDirectory)
        powerCurvePaths = (*powerDataPaths, powerFigurePath)
    PrintBenchmarkResults(rows)
    if powerCurvePaths is not None:
        powerCsvPath, powerJsonPath, powerFigurePath = powerCurvePaths
        print(f"\nPower-EVM CSV:  {powerCsvPath.resolve()}")
        print(f"Power-EVM JSON: {powerJsonPath.resolve()}")
        print(f"Power-EVM plot: {powerFigurePath.resolve()}")
    return rows

def SaveBenchmarkResults(
    rows: List[BenchmarkRow],
    outputDirectory: Path,
    metadata: Mapping[str, object],
) -> None:
    """Save the complete all-method benchmark as flat CSV and structured JSON.

    Processing details:
        Algorithm: Flatten every immutable benchmark row once, write identical
        row values to CSV, and write JSON containing the same rows plus the
        exact waveform, PA, seed, iteration, and power-sweep metadata.

    Args:
        rows: Benchmark rows accumulated or emitted by the reporting operation.
        outputDirectory: Directory in which result artifacts are written.
        metadata: Reproducibility settings associated with every result row.

    Returns:
        result: None. CSV and JSON summary files are created.
    """

    outputDirectory.mkdir(parents=True, exist_ok=True)
    csvPath = outputDirectory / "all_ilc_metrics.csv"
    jsonPath = outputDirectory / "all_ilc_metrics.json"
    flatRows = [row.ToDict() for row in rows]
    with csvPath.open("w", newline="", encoding="utf-8-sig") as csvFile:
        csvWriter = csv.DictWriter(csvFile, fieldnames=list(flatRows[0].keys()))
        csvWriter.writeheader()
        csvWriter.writerows(flatRows)
    with jsonPath.open("w", encoding="utf-8") as jsonFile:
        json.dump(
            {"metadata": dict(metadata), "results": flatRows},
            jsonFile,
            ensure_ascii=False,
            indent=2,
        )

def PrintBenchmarkResults(rows: List[BenchmarkRow]) -> None:
    """Print a compact all-method SNR, EVM, and worst-ACLR table.

    Processing details:
        Algorithm: Use fixed-width columns for method and scenario names,
        print absolute SNR, EVM percentage, worst-side ACLR, and the signed EVM
        improvement already stored in each row.

    Args:
        rows: Benchmark rows accumulated or emitted by the reporting operation.

    Returns:
        result: None. A deterministic human-readable table is printed.
    """

    header = (
        f"{'Method':<32} {'Scenario':<25} {'SNR':>8} "
        f"{'EVM%':>9} {'ACLR-W':>9} {'dEVM':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.methodName:<32} {row.scenario:<25} "
            f"{row.metrics['snrDb']:>8.2f} "
            f"{row.metrics['evmPercent']:>9.3f} "
            f"{row.metrics['aclrWorstDb']:>9.2f} "
            f"{row.evmImprovementDb:>8.2f}"
        )


def ParseBenchmarkArguments() -> BenchmarkConfig:
    """Parse standalone benchmark command-line options.

    Processing details:
        Algorithm: Define only scenario-level controls, parse one command
        line, convert it into ``BenchmarkConfig``, and validate it before
        returning. Algorithm-internal learning constants remain fixed so
        comparisons stay reproducible.

    Returns:
        result: Validated benchmark configuration ready for
        ``RunAllIlcBenchmark``.
    """

    argumentParser = argparse.ArgumentParser(
        description=(
            "Run classified SISO ILC benchmark scenarios independently of "
            "the production main program."
        )
    )
    argumentParser.add_argument(
        "--format",
        dest="frameFormat",
        default="EHT",
        help="VHT/11ac, HE/11ax, or EHT/11be (default: EHT)",
    )
    argumentParser.add_argument(
        "--bandwidth",
        dest="bandwidthMhz",
        type=int,
        default=20,
        choices=(20, 40, 80, 160),
    )
    argumentParser.add_argument("--mcs", type=int, default=7)
    argumentParser.add_argument(
        "--symbols",
        dest="numDataSymbols",
        type=int,
        default=10,
    )
    argumentParser.add_argument(
        "--sample-rate-hz",
        dest="sampleRateHz",
        type=float,
        default=None,
        help=(
            "Complex-baseband sample rate in Hz; overrides legacy "
            "--oversampling when supplied"
        ),
    )
    argumentParser.add_argument(
        "--oversampling",
        type=int,
        default=4,
        help=(
            "Legacy bandwidth multiplier used only when --sample-rate-hz "
            "is omitted"
        ),
    )
    argumentParser.add_argument(
        "--width",
        type=int,
        default=16,
        help="External I/Q component width; 0 selects floating point",
    )
    argumentParser.add_argument(
        "--guard-interval",
        dest="guardIntervalUs",
        type=float,
        default=0.8,
    )
    argumentParser.add_argument(
        "--output-power-dbm",
        dest="outputPowerDbm",
        type=float,
        default=20.0,
        help=(
            "Absolute nominal PA output power per chain "
            "(default: 20 dBm)"
        ),
    )
    argumentParser.add_argument(
        "--maximum-output-power-dbm",
        dest="maximumOutputPowerDbm",
        type=float,
        default=25.0,
        help="Rated per-PA output-power limit (default: 25 dBm)",
    )
    argumentParser.add_argument(
        "--load-resistance-ohm",
        dest="loadResistanceOhm",
        type=float,
        default=50.0,
        help="Resistive PA port used for dBm conversion (default: 50 ohms)",
    )
    argumentParser.add_argument(
        "--iterations",
        dest="numIterations",
        type=int,
        default=10,
    )
    argumentParser.add_argument(
        "--pa",
        dest="paModelName",
        choices=("wiener", "gmp"),
        default="wiener",
    )
    argumentParser.add_argument("--seed", type=int, default=101)
    argumentParser.add_argument(
        "--power-start-dbm",
        dest="powerStartDbm",
        type=float,
        default=10.0,
        help="First absolute PA output power in the sweep, in dBm",
    )
    argumentParser.add_argument(
        "--power-stop-dbm",
        dest="powerStopDbm",
        type=float,
        default=25.0,
        help="Last absolute PA output power in the sweep, in dBm",
    )
    argumentParser.add_argument(
        "--power-points",
        dest="powerPointCount",
        type=int,
        default=5,
    )
    argumentParser.add_argument(
        "--skip-power-curve",
        dest="generatePowerEvmCurve",
        action="store_false",
        help="Skip the multi-method power-EVM sweep",
    )
    argumentParser.add_argument(
        "--output-dir",
        dest="outputDirectory",
        type=Path,
        default=Path("results/all_ilc_benchmark"),
    )
    arguments = argumentParser.parse_args()
    config = BenchmarkConfig(
        frameFormat=arguments.frameFormat,
        bandwidthMhz=arguments.bandwidthMhz,
        mcs=arguments.mcs,
        numDataSymbols=arguments.numDataSymbols,
        sampleRateHz=arguments.sampleRateHz,
        oversampling=arguments.oversampling,
        width=arguments.width,
        guardIntervalUs=arguments.guardIntervalUs,
        outputPowerDbm=arguments.outputPowerDbm,
        maximumOutputPowerDbm=arguments.maximumOutputPowerDbm,
        loadResistanceOhm=arguments.loadResistanceOhm,
        numIterations=arguments.numIterations,
        paModelName=arguments.paModelName,
        seed=arguments.seed,
        powerStartDbm=arguments.powerStartDbm,
        powerStopDbm=arguments.powerStopDbm,
        powerPointCount=arguments.powerPointCount,
        generatePowerEvmCurve=arguments.generatePowerEvmCurve,
        outputDirectory=arguments.outputDirectory,
    )
    config.Validate()
    return config


def Main() -> int:
    """Run the standalone benchmark and report its artifact directory.

    Processing details:
        Algorithm: Parse the requested scenario controls, execute the complete
        benchmark suite, and print the absolute output path after every result
        has been saved.

    Returns:
        result: Process exit status zero after successful completion.
    """

    config = ParseBenchmarkArguments()
    RunAllIlcBenchmark(config)
    print(f"\nBenchmark results: {config.outputDirectory.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(Main())
