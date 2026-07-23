"""Scenario-based performance benchmark for every supported ILC method.

This test module owns waveform construction, impairment scenarios, metric
comparison, convergence reporting, held-out deployment validation, and the
multi-method power-EVM curve. Production ILC algorithms remain in
``inc.DpdIlc`` and do not depend on this benchmark harness.
"""

import argparse
import csv
import json
from dataclasses import dataclass, replace
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

from inc.Analysis import Analysis, SignalMetrics
from inc.Draw import Draw
from inc.DpdIlc import (
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
from inc.PaModel import IQImbalancePA, PaModel
from inc.waveGen import GenWifi, WifiWaveform


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
    oversampling: int = 4
    guardIntervalUs: float = 0.8
    driveRms: float = 0.24
    numIterations: int = 10
    paModelName: str = "wiener"
    seed: int = 101
    powerStartRms: float = 0.08
    powerStopRms: float = 0.40
    powerPointCount: int = 5
    generatePowerEvmCurve: bool = True
    outputDirectory: Path = Path("results/all_ilc_benchmark")

    def Validate(self) -> None:
        """Validate waveform, iteration, sweep, and output parameters.

        Processing details:
            Algorithm: Reject nonphysical waveform or power settings before
            any long-running scenario starts. PHY-specific format, bandwidth,
            MCS, and guard-interval combinations are validated later by
            ``GenWifi`` using the same configuration.

        Returns:
            result: None. Invalid settings raise ``ValueError`` with the
            parameter responsible for the failure.
        """

        if self.numDataSymbols < 1:
            raise ValueError("numDataSymbols must be positive")
        if self.oversampling < 3:
            raise ValueError(
                "oversampling must be at least three for ACLR analysis"
            )
        if self.driveRms <= 0.0:
            raise ValueError("driveRms must be positive")
        if self.numIterations < 1:
            raise ValueError("numIterations must be positive")
        if self.powerStartRms <= 0.0:
            raise ValueError("powerStartRms must be positive")
        if self.powerStopRms <= self.powerStartRms:
            raise ValueError("powerStopRms must exceed powerStartRms")
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
            flatten the immutable ``SignalMetrics`` record, and merge both
            dictionaries without changing stored numerical values.

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
        rowData.update(self.metrics.ToDict())
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
            snrImprovementDb=metrics.snrDb - baselineMetrics.snrDb,
            # More-negative EVM dB is better, so baseline minus result is a
            # positive improvement.
            evmImprovementDb=baselineMetrics.evmDb - metrics.evmDb,
            aclrImprovementDb=metrics.aclrWorstDb
            - baselineMetrics.aclrWorstDb,
        )
    )

def SaveHistory(
    methodName: str, ilcResult: ILCResult, outputDirectory: Path
) -> None:
    """Save a separate convergence CSV without overwriting other methods.

    Processing details:
        Algorithm: Sanitize the method label into a stable file stem, write
        every immutable iteration record with all three MSE definitions and
        gain diagnostics, then ask ``Draw`` to render the same history.

    Args:
        methodName: Human-readable algorithm or deployment-model label.
        ilcResult: Completed ILC result containing ordered iteration records.
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
                "evmAlignedMse",
                "evmDb",
                "complexGainMagnitudeDb",
                "complexGainPhaseDegrees",
                "inputPeak",
            ),
        )
        csvWriter.writeheader()
        for iterationRecord in ilcResult.history:
            csvWriter.writerow(
                {
                    "iteration": iterationRecord.iteration,
                    "mse": iterationRecord.mse,
                    "errorRms": iterationRecord.errorRms,
                    "nmseDb": iterationRecord.nmseDb,
                    "linearCompensatedMse": (
                        iterationRecord.linearCompensatedMse
                    ),
                    "linearCompensatedNmseDb": (
                        iterationRecord.linearCompensatedNmseDb
                    ),
                    "evmAlignedMse": iterationRecord.evmAlignedMse,
                    "evmDb": iterationRecord.evmDb,
                    "complexGainMagnitudeDb": (
                        iterationRecord.complexGainMagnitudeDb
                    ),
                    "complexGainPhaseDegrees": (
                        iterationRecord.complexGainPhaseDegrees
                    ),
                    "inputPeak": iterationRecord.inputPeak,
                }
            )
    Draw(convergenceFileStem=f"convergence_{safeName}").SaveConvergenceCurve(
        ilcResult.history, outputDirectory
    )

def ReportHistory(
    methodName: str,
    ilcResult: ILCResult,
    resultAnalysis: Analysis,
    outputDirectory: Path,
) -> None:
    """Print and save one method's complete per-iteration MSE history.

    Processing details:
        Algorithm: Use ``Analysis`` for the console table, then serialize the
        same immutable records and render their convergence figure without
        recalculating any metric.

    Args:
        methodName: Human-readable ILC method label.
        ilcResult: Completed ILC result containing ordered history records.
        resultAnalysis: Analysis instance used for consistent presentation.
        outputDirectory: Destination for CSV and PNG result artifacts.

    Returns:
        result: None. Console and file outputs are produced as side effects.
    """

    resultAnalysis.PrintConvergence(
        ilcResult.history, f"{methodName} iteration metrics"
    )
    SaveHistory(methodName, ilcResult, outputDirectory)

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

    predistortedInput = LimitAmplitude(
        predistorter.Process(validationSignal), maxAmplitude
    )
    paOutput = paModel.Process(predistortedInput)
    return resultAnalysis.Analyze(paOutput)

def RunIlcCurvePoint(
    referenceSignal: np.ndarray,
    driveRms: float,
    paModel: Any,
    waveform: WifiWaveform,
    methodName: str,
    methodFunction: Optional[
        Callable[[np.ndarray, Any, ILCConfig], ILCResult]
    ],
    methodConfig: ILCConfig,
) -> np.ndarray:
    """Run one selected ILC method at one power-EVM sweep point.

    Processing details:
        Algorithm: Bind a fresh EVM-MSE evaluator to the waveform after power
        scaling, clone the selected method configuration, dispatch the
        frequency-domain signature separately from ordinary waveform-update
        signatures, and return the selected best PA output.

    Args:
        referenceSignal: Ideal complex baseband samples used as the target or regression input.
        driveRms: Current RMS drive value in the power sweep.
        paModel: PA object exposing Process and SmallSignalGain operations.
        waveform: Wi-Fi metadata defining field locations, FFT sizes, and subcarriers.
        methodName: Human-readable algorithm or deployment-model label.
        methodFunction: Selected ILC update-law callable. ``None`` selects
            the dedicated frequency-domain call path, which also needs the
            waveform sample rate and occupied bandwidth.
        methodConfig: Validated ILC configuration for the selected update law.

    Returns:
        result: Complex PA output learned specifically for this power point.
    """

    del driveRms
    pointAnalysis = Analysis(referenceSignal, waveform)
    pointConfig = replace(
        methodConfig,
        evmMseEvaluator=pointAnalysis.CalculateEvmAlignedMse,
    )
    if methodName == "Frequency-domain ILC":
        return RunFrequencyDomainIlc(
            referenceSignal,
            paModel,
            waveform.sampleRateHz,
            waveform.bandwidthHz,
            pointConfig,
        ).outputSignal
    if methodFunction is None:
        raise ValueError("methodFunction is required for non-frequency ILC")
    return methodFunction(
        referenceSignal, paModel, pointConfig
    ).outputSignal

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
        "oversampling": config.oversampling,
    }
    trainingParameters = dict(sharedWifiParameters)
    trainingParameters["seed"] = config.seed
    validationParameters = dict(sharedWifiParameters)
    validationParameters["seed"] = config.seed + 97
    trainingGenerator = GenWifi(parameters=trainingParameters)
    validationGenerator = GenWifi(parameters=validationParameters)
    trainingWaveform = trainingGenerator.Generate()
    validationWaveform = validationGenerator.Generate()
    trainingSignal = config.driveRms * trainingWaveform.samples
    validationSignal = config.driveRms * validationWaveform.samples
    paParameters = {"modelName": config.paModelName}
    paModel = PaModel(parameters=paParameters)
    trainingAnalysis = Analysis(trainingSignal, trainingWaveform)
    validationAnalysis = Analysis(validationSignal, validationWaveform)
    maxAmplitude = max(2.0, 1.6 * np.max(np.abs(trainingSignal)))

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
        evmMseEvaluator=trainingAnalysis.CalculateEvmAlignedMse,
    )
    complexConfig = ILCConfig(
        numIterations=config.numIterations,
        learningRate=0.15,
        maxAmplitude=maxAmplitude,
        randomSeed=config.seed + 2,
        evmMseEvaluator=trainingAnalysis.CalculateEvmAlignedMse,
    )
    firConfig = ILCConfig(
        numIterations=config.numIterations,
        learningRate=0.15,
        maxAmplitude=maxAmplitude,
        randomSeed=config.seed + 3,
        evmMseEvaluator=trainingAnalysis.CalculateEvmAlignedMse,
    )
    frequencyConfig = ILCConfig(
        numIterations=config.numIterations,
        learningRate=0.15,
        maxAmplitude=maxAmplitude,
        randomSeed=config.seed + 4,
        evmMseEvaluator=trainingAnalysis.CalculateEvmAlignedMse,
    )
    gaussNewtonConfig = ILCConfig(
        numIterations=config.numIterations,
        learningRate=0.65,
        maxAmplitude=maxAmplitude,
        randomSeed=config.seed + 5,
        evmMseEvaluator=trainingAnalysis.CalculateEvmAlignedMse,
    )
    parameterConfig = ILCConfig(
        numIterations=config.numIterations,
        learningRate=0.20,
        maxAmplitude=maxAmplitude,
        randomSeed=config.seed + 6,
        evmMseEvaluator=trainingAnalysis.CalculateEvmAlignedMse,
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
                trainingSignal, paModel, methodConfig
            )
        methodMetrics = trainingAnalysis.Analyze(methodResult.outputSignal)
        AddRow(
            rows,
            methodName,
            "ILC update law",
            "nominal repeated waveform",
            methodMetrics,
            baselineMetrics,
        )
        ReportHistory(
            methodName, methodResult, trainingAnalysis, outputDirectory
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
                selectedName,
                selectedFunction,
                selectedConfig,
            )
        )

    if frequencyResult is None:
        raise RuntimeError("frequency-domain ILC result was not generated")

    # Repeat the physically identical baseline and unconstrained result under
    # the peak scenario label so scenario-filtered reports contain the full
    # baseline-versus-performance-versus-feasibility comparison.
    frequencyMetrics = trainingAnalysis.Analyze(
        frequencyResult.outputSignal
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
    constrainedPeak = 1.05 * np.max(np.abs(trainingSignal))
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
            evmMseEvaluator=trainingAnalysis.CalculateEvmAlignedMse,
        ),
    )
    constrainedMetrics = trainingAnalysis.Analyze(
        constrainedResult.outputSignal
    )
    AddRow(
        rows,
        "Constrained CFR-ILC",
        "ILC update law",
        "peak-constrained waveform",
        constrainedMetrics,
        baselineMetrics,
    )
    ReportHistory(
        "Constrained CFR-ILC",
        constrainedResult,
        trainingAnalysis,
        outputDirectory,
    )
    powerEvaluators["Constrained CFR-ILC"] = (
        lambda pointReference, pointDrive: RunIlcCurvePoint(
            pointReference,
            pointDrive,
            paModel,
            trainingWaveform,
            "Frequency-domain ILC",
            None,
            ILCConfig(
                numIterations=config.numIterations,
                learningRate=0.12,
                maxAmplitude=1.05 * np.max(np.abs(pointReference)),
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
            evmMseEvaluator=trainingAnalysis.CalculateEvmAlignedMse,
        ),
    )
    naiveNoisyMetrics = trainingAnalysis.Analyze(
        naiveNoisyResult.outputSignal
    )
    AddRow(
        rows,
        "Naive noisy-feedback ILC",
        "ILC update law",
        "32 dB feedback robustness",
        naiveNoisyMetrics,
        noisyBaselineMetrics,
    )
    ReportHistory(
        "Naive noisy-feedback ILC",
        naiveNoisyResult,
        trainingAnalysis,
        outputDirectory,
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
            evmMseEvaluator=trainingAnalysis.CalculateEvmAlignedMse,
        ),
    )
    noiseAwareMetrics = trainingAnalysis.Analyze(
        noiseAwareResult.outputSignal
    )
    AddRow(
        rows,
        "Noise-aware ILC",
        "ILC update law",
        "32 dB feedback robustness",
        noiseAwareMetrics,
        noisyBaselineMetrics,
    )
    ReportHistory(
        "Noise-aware ILC",
        noiseAwareResult,
        trainingAnalysis,
        outputDirectory,
    )
    powerEvaluators["Naive noisy-feedback ILC"] = (
        lambda pointReference, pointDrive: RunIlcCurvePoint(
            pointReference,
            pointDrive,
            paModel,
            trainingWaveform,
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
            evmMseEvaluator=trainingAnalysis.CalculateEvmAlignedMse,
        ),
    )
    ordinaryIqMetrics = trainingAnalysis.Analyze(
        ordinaryIqResult.outputSignal
    )
    AddRow(
        rows,
        "Frequency-domain ILC on IQ plant",
        "ILC update law",
        "IQ image impairment",
        ordinaryIqMetrics,
        iqBaselineMetrics,
    )
    ReportHistory(
        "Frequency-domain ILC on IQ plant",
        ordinaryIqResult,
        trainingAnalysis,
        outputDirectory,
    )
    augmentedResult = RunAugmentedIqIlc(
        trainingSignal,
        iqPaModel,
        ILCConfig(
            numIterations=config.numIterations,
            learningRate=0.18,
            maxAmplitude=maxAmplitude,
            randomSeed=config.seed + 9,
            evmMseEvaluator=trainingAnalysis.CalculateEvmAlignedMse,
        ),
    )
    augmentedMetrics = trainingAnalysis.Analyze(
        augmentedResult.outputSignal
    )
    AddRow(
        rows,
        "Augmented IQ ILC",
        "ILC update law",
        "IQ image impairment",
        augmentedMetrics,
        iqBaselineMetrics,
    )
    ReportHistory(
        "Augmented IQ ILC",
        augmentedResult,
        trainingAnalysis,
        outputDirectory,
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
                trainingSignal,
                frequencyResult.learnedInput,
                nonlinearOrders=(1, 3, 5, 7),
                memoryDepth=3,
                crossMemoryDepth=0,
            ),
        ),
        (
            "ILC label + GMP",
            FitGmpPredistorter(
                trainingSignal,
                frequencyResult.learnedInput,
                nonlinearOrders=(1, 3, 5, 7),
                memoryDepth=3,
                crossMemoryDepth=2,
            ),
        ),
        (
            "ILC label + Volterra",
            FitVolterraPredistorter(
                trainingSignal,
                frequencyResult.learnedInput,
                memoryDepth=3,
            ),
        ),
        (
            "ILC label + LUT",
            FitLutPredistorter(
                trainingSignal,
                frequencyResult.learnedInput,
                binCount=64,
            ),
        ),
        (
            "ILC label + NN",
            FitNeuralPredistorter(
                trainingSignal,
                frequencyResult.learnedInput,
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
                LimitAmplitude(
                    selectedPredistorter.Process(pointReference),
                    maxAmplitude,
                )
            )
        )
    metadata: Mapping[str, object] = {
        "frameFormat": trainingWaveform.frameFormat,
        "bandwidthMhz": config.bandwidthMhz,
        "mcs": config.mcs,
        "numDataSymbols": config.numDataSymbols,
        "oversampling": config.oversampling,
        "guardIntervalUs": config.guardIntervalUs,
        "driveRms": config.driveRms,
        "numIterations": config.numIterations,
        "paModel": config.paModelName,
        "trainingSeed": config.seed,
        "validationSeed": config.seed + 97,
        "powerStartRms": config.powerStartRms,
        "powerStopRms": config.powerStopRms,
        "powerPointCount": config.powerPointCount,
        "generatePowerEvmCurve": config.generatePowerEvmCurve,
    }
    SaveBenchmarkResults(rows, outputDirectory, metadata)
    powerCurvePaths = None
    if config.generatePowerEvmCurve:
        powerDriveValues = np.geomspace(
            config.powerStartRms,
            config.powerStopRms,
            config.powerPointCount,
        )
        powerEvmCurve = trainingAnalysis.AnalyzePowerEvmCurve(
            powerDriveValues, powerEvaluators
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
            f"{row.metrics.snrDb:>8.2f} {row.metrics.evmPercent:>9.3f} "
            f"{row.metrics.aclrWorstDb:>9.2f} "
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
        "--oversampling",
        type=int,
        default=4,
    )
    argumentParser.add_argument(
        "--guard-interval",
        dest="guardIntervalUs",
        type=float,
        default=0.8,
    )
    argumentParser.add_argument(
        "--drive",
        dest="driveRms",
        type=float,
        default=0.24,
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
        "--power-start",
        dest="powerStartRms",
        type=float,
        default=0.08,
    )
    argumentParser.add_argument(
        "--power-stop",
        dest="powerStopRms",
        type=float,
        default=0.40,
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
        oversampling=arguments.oversampling,
        guardIntervalUs=arguments.guardIntervalUs,
        driveRms=arguments.driveRms,
        numIterations=arguments.numIterations,
        paModelName=arguments.paModelName,
        seed=arguments.seed,
        powerStartRms=arguments.powerStartRms,
        powerStopRms=arguments.powerStopRms,
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
