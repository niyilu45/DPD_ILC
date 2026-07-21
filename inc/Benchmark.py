"""Unified benchmark for every ILC route described in the project document."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np

from .Analysis import AnalyzeSignal, SignalMetrics
from .DeploymentModels import (
    FitLutPredistorter,
    FitNeuralPredistorter,
    FitVolterraPredistorter,
)
from .DpdIlc import (
    FitGmpPredistorter,
    ILCConfig,
    ILCResult,
    RunFrequencyDomainIlc,
)
from .IlcVariants import (
    RunAugmentedIqIlc,
    RunComplexGainIlc,
    RunDirectionalGaussNewtonIlc,
    RunFirIlc,
    RunParameterDomainIlc,
    RunScalarPIlc,
)
from .PaModel import CreatePaModel, IQImbalancePA
from .waveGen import EHTConfig, GenerateEhtWaveform


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configure a compact but representative all-method comparison."""

    bandwidthMhz: int = 20
    mcs: int = 7
    numDataSymbols: int = 10
    oversampling: int = 4
    guardIntervalUs: float = 0.8
    driveRms: float = 0.24
    numIterations: int = 10
    paModelName: str = "wiener"
    seed: int = 101
    outputDirectory: Path = Path("results/all_ilc_benchmark")


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
        """Convert a row to flat JSON/CSV-compatible values."""

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


def _LimitAmplitude(inputSignal: np.ndarray, maxAmplitude: float) -> np.ndarray:
    """Apply a common safe peak limit to every deployed DPD output."""

    limitedSignal = np.asarray(inputSignal, dtype=np.complex128).copy()
    signalMagnitude = np.abs(limitedSignal)
    overLimit = signalMagnitude > maxAmplitude
    if np.any(overLimit):
        limitedSignal[overLimit] *= maxAmplitude / signalMagnitude[overLimit]
    return limitedSignal


def _AddRow(
    rows: List[BenchmarkRow],
    methodName: str,
    category: str,
    scenario: str,
    metrics: SignalMetrics,
    baselineMetrics: SignalMetrics,
) -> None:
    """Append metrics and consistently signed improvements to the result table."""

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


def _SaveHistory(
    methodName: str, ilcResult: ILCResult, outputDirectory: Path
) -> None:
    """Save a separate convergence CSV without overwriting other methods."""

    safeName = "".join(
        character.lower() if character.isalnum() else "_"
        for character in methodName
    ).strip("_")
    historyPath = outputDirectory / f"convergence_{safeName}.csv"
    with historyPath.open("w", newline="", encoding="utf-8-sig") as csvFile:
        csvWriter = csv.DictWriter(
            csvFile,
            fieldnames=("iteration", "errorRms", "nmseDb", "inputPeak"),
        )
        csvWriter.writeheader()
        for iterationRecord in ilcResult.history:
            csvWriter.writerow(
                {
                    "iteration": iterationRecord.iteration,
                    "errorRms": iterationRecord.errorRms,
                    "nmseDb": iterationRecord.nmseDb,
                    "inputPeak": iterationRecord.inputPeak,
                }
            )


def _EvaluateDeployment(
    predistorter,
    validationSignal: np.ndarray,
    paModel,
    validationWaveform,
    maxAmplitude: float,
) -> SignalMetrics:
    """Evaluate one fitted DPD on a held-out EHT packet."""

    predistortedInput = _LimitAmplitude(
        predistorter.Process(validationSignal), maxAmplitude
    )
    paOutput = paModel.Process(predistortedInput)
    return AnalyzeSignal(validationSignal, paOutput, validationWaveform)


def RunAllIlcBenchmark(config: BenchmarkConfig = BenchmarkConfig()) -> List[BenchmarkRow]:
    """Run every update law and every ILC-label deployment model.

    Waveform update laws use one repeated training packet. Deployment models
    are fitted to frequency-domain ILC labels and evaluated on an independent
    payload generated with a different seed. Augmented ILC uses an IQ-image
    scenario, and noise-aware ILC learns from averaged noisy feedback while
    final metrics are measured on the clean PA output.
    """

    outputDirectory = Path(config.outputDirectory)
    outputDirectory.mkdir(parents=True, exist_ok=True)
    trainingWaveform = GenerateEhtWaveform(
        EHTConfig(
            bandwidthMhz=config.bandwidthMhz,
            mcs=config.mcs,
            numDataSymbols=config.numDataSymbols,
            guardIntervalUs=config.guardIntervalUs,
            oversampling=config.oversampling,
            seed=config.seed,
        )
    )
    validationWaveform = GenerateEhtWaveform(
        EHTConfig(
            bandwidthMhz=config.bandwidthMhz,
            mcs=config.mcs,
            numDataSymbols=config.numDataSymbols,
            guardIntervalUs=config.guardIntervalUs,
            oversampling=config.oversampling,
            seed=config.seed + 97,
        )
    )
    trainingSignal = config.driveRms * trainingWaveform.samples
    validationSignal = config.driveRms * validationWaveform.samples
    paModel = CreatePaModel(config.paModelName)
    maxAmplitude = max(2.0, 1.6 * np.max(np.abs(trainingSignal)))

    baselineOutput = paModel.Process(trainingSignal)
    baselineMetrics = AnalyzeSignal(
        trainingSignal, baselineOutput, trainingWaveform
    )
    rows: List[BenchmarkRow] = []
    _AddRow(
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
        methodMetrics = AnalyzeSignal(
            trainingSignal, methodResult.outputSignal, trainingWaveform
        )
        _AddRow(
            rows,
            methodName,
            "ILC update law",
            "nominal repeated waveform",
            methodMetrics,
            baselineMetrics,
        )
        _SaveHistory(methodName, methodResult, outputDirectory)

    if frequencyResult is None:
        raise RuntimeError("frequency-domain ILC result was not generated")

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
        ),
    )
    constrainedMetrics = AnalyzeSignal(
        trainingSignal, constrainedResult.outputSignal, trainingWaveform
    )
    _AddRow(
        rows,
        "Constrained CFR-ILC",
        "ILC update law",
        "peak-constrained waveform",
        constrainedMetrics,
        baselineMetrics,
    )
    _SaveHistory("Constrained CFR-ILC", constrainedResult, outputDirectory)

    # Noise-aware learning uses a higher regularization and four averaged
    # feedback captures at 32 dB feedback SNR.
    noisyBaselineMetrics = baselineMetrics
    _AddRow(
        rows,
        "Noisy-feedback baseline",
        "baseline",
        "32 dB averaged feedback",
        noisyBaselineMetrics,
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
    noiseAwareMetrics = AnalyzeSignal(
        trainingSignal, noiseAwareResult.outputSignal, trainingWaveform
    )
    _AddRow(
        rows,
        "Noise-aware ILC",
        "ILC update law",
        "32 dB averaged feedback",
        noiseAwareMetrics,
        noisyBaselineMetrics,
    )
    _SaveHistory("Noise-aware ILC", noiseAwareResult, outputDirectory)

    # Augmented ILC is evaluated in the IQ-image scenario for which its
    # conjugate branch is designed.
    iqPaModel = IQImbalancePA(CreatePaModel(config.paModelName))
    iqBaselineOutput = iqPaModel.Process(trainingSignal)
    iqBaselineMetrics = AnalyzeSignal(
        trainingSignal, iqBaselineOutput, trainingWaveform
    )
    _AddRow(
        rows,
        "IQ-imbalance baseline",
        "baseline",
        "IQ image impairment",
        iqBaselineMetrics,
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
    augmentedMetrics = AnalyzeSignal(
        trainingSignal, augmentedResult.outputSignal, trainingWaveform
    )
    _AddRow(
        rows,
        "Augmented IQ ILC",
        "ILC update law",
        "IQ image impairment",
        augmentedMetrics,
        iqBaselineMetrics,
    )
    _SaveHistory("Augmented IQ ILC", augmentedResult, outputDirectory)

    # Fit every deployable model to the same converged ILC labels, then test
    # on a held-out EHT payload to measure generalization rather than recall.
    validationBaselineOutput = paModel.Process(validationSignal)
    validationBaselineMetrics = AnalyzeSignal(
        validationSignal, validationBaselineOutput, validationWaveform
    )
    _AddRow(
        rows,
        "Validation baseline",
        "baseline",
        "held-out EHT packet",
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
        methodMetrics = _EvaluateDeployment(
            predistorter,
            validationSignal,
            paModel,
            validationWaveform,
            maxAmplitude,
        )
        _AddRow(
            rows,
            methodName,
            "ILC label deployment",
            "held-out EHT packet",
            methodMetrics,
            validationBaselineMetrics,
        )
    metadata: Mapping[str, object] = {
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
    }
    SaveBenchmarkResults(rows, outputDirectory, metadata)
    PrintBenchmarkResults(rows)
    return rows


def SaveBenchmarkResults(
    rows: List[BenchmarkRow],
    outputDirectory: Path,
    metadata: Mapping[str, object],
) -> None:
    """Save the complete all-method benchmark as flat CSV and structured JSON."""

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
    """Print a compact all-method SNR, EVM, and worst-ACLR table."""

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
