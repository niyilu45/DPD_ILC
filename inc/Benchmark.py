"""Unified benchmark for every ILC route described in the project document."""

import csv
import json
from collections import ChainMap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np

from .Analysis import Analysis, SignalMetrics, analysisDefaultParameters
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
from .Draw import Draw
from .IlcVariants import (
    RunAugmentedIqIlc,
    RunComplexGainIlc,
    RunDirectionalGaussNewtonIlc,
    RunFirIlc,
    RunParameterDomainIlc,
    RunScalarPIlc,
)
from .PaModel import IQImbalancePA, PaModel, paModelDefaultParameters
from .waveGen import GenWifi, genWifiDefaultParameters


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configure a compact but representative all-method comparison."""

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
            Algorithm: Convert validated in-memory results into a stable reporting format without altering later numerical calculations.

        Returns:
            result: Dict[str, object]. The computed value described by the summary, with documented units, shape, and normalization.
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


def LimitAmplitude(inputSignal: np.ndarray, maxAmplitude: float) -> np.ndarray:
    """Apply a common safe peak limit to every deployed DPD output.

    Processing details:
        Algorithm: Apply the bounded sample-domain transformation without changing array length or causal indexing conventions.

    Args:
        inputSignal: One-dimensional complex baseband samples supplied to the operation.
        maxAmplitude: Maximum allowed complex-envelope magnitude.

    Returns:
        result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
    """

    limitedSignal = np.asarray(inputSignal, dtype=np.complex128).copy()
    signalMagnitude = np.abs(limitedSignal)
    overLimit = signalMagnitude > maxAmplitude
    if np.any(overLimit):
        limitedSignal[overLimit] *= maxAmplitude / signalMagnitude[overLimit]
    return limitedSignal


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
        Algorithm: Carry out the described operation using validated inputs, explicit array-shape handling, and deterministic project conventions.

    Args:
        rows: Benchmark rows accumulated or emitted by the reporting operation.
        methodName: Human-readable algorithm or deployment-model label.
        category: Caller-supplied value consumed according to the function contract.
        scenario: Description of the impairment or validation scenario.
        metrics: Signal-quality metrics calculated for the selected output.
        baselineMetrics: Reference metrics used to calculate improvements.

    Returns:
        result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
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
        Algorithm: Convert validated in-memory results into a stable reporting format without altering later numerical calculations.

    Args:
        methodName: Human-readable algorithm or deployment-model label.
        ilcResult: Caller-supplied value consumed according to the function contract.
        outputDirectory: Directory in which result artifacts are written.

    Returns:
        result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
    """

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


def EvaluateDeployment(
    predistorter,
    validationSignal: np.ndarray,
    paModel,
    resultAnalysis: Analysis,
    maxAmplitude: float,
) -> SignalMetrics:
    """Evaluate one fitted DPD on a held-out Wi-Fi packet.

    Processing details:
        Algorithm: Execute the configured signal-processing path, preserve sample alignment, and return the complete downstream result.

    Args:
        predistorter: Caller-supplied value consumed according to the function contract.
        validationSignal: Independent complex waveform used to evaluate generalization.
        paModel: PA object exposing Process and SmallSignalGain operations.
        resultAnalysis: Caller-supplied value consumed according to the function contract.
        maxAmplitude: Maximum allowed complex-envelope magnitude.

    Returns:
        result: SignalMetrics. The computed value described by the summary, with documented units, shape, and normalization.
    """

    predistortedInput = LimitAmplitude(
        predistorter.Process(validationSignal), maxAmplitude
    )
    paOutput = paModel.Process(predistortedInput)
    return resultAnalysis.Analyze(paOutput)


def RunIlcCurvePoint(
    referenceSignal: np.ndarray,
    driveRms: float,
    paModel,
    waveform,
    methodName: str,
    methodFunction,
    methodConfig: ILCConfig,
) -> np.ndarray:
    """Run one selected ILC method at one power-EVM sweep point.

    Processing details:
        Algorithm: Execute the configured signal-processing path, preserve sample alignment, and return the complete downstream result.

    Args:
        referenceSignal: Ideal complex baseband samples used as the target or regression input.
        driveRms: Current RMS drive value in the power sweep.
        paModel: PA object exposing Process and SmallSignalGain operations.
        waveform: Wi-Fi metadata defining field locations, FFT sizes, and subcarriers.
        methodName: Human-readable algorithm or deployment-model label.
        methodFunction: Selected ILC update-law callable.
        methodConfig: Validated ILC configuration for the selected update law.

    Returns:
        result: np.ndarray. The computed value described by the summary, with documented units, shape, and normalization.
    """

    del driveRms
    if methodName == "Frequency-domain ILC":
        return RunFrequencyDomainIlc(
            referenceSignal,
            paModel,
            waveform.sampleRateHz,
            waveform.bandwidthHz,
            methodConfig,
        ).outputSignal
    return methodFunction(
        referenceSignal, paModel, methodConfig
    ).outputSignal


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
    sharedWifiParameters = {
        "frameFormat": config.frameFormat,
        "bandwidthMhz": config.bandwidthMhz,
        "mcs": config.mcs,
        "numDataSymbols": config.numDataSymbols,
        "guardIntervalUs": config.guardIntervalUs,
        "oversampling": config.oversampling,
    }
    trainingParameters = ChainMap(
        {"seed": config.seed},
        sharedWifiParameters,
        genWifiDefaultParameters,
    )
    validationParameters = ChainMap(
        {"seed": config.seed + 97},
        sharedWifiParameters,
        genWifiDefaultParameters,
    )
    trainingGenerator = GenWifi(parameters=trainingParameters)
    validationGenerator = GenWifi(parameters=validationParameters)
    trainingWaveform = trainingGenerator.Generate()
    validationWaveform = validationGenerator.Generate()
    trainingSignal = config.driveRms * trainingWaveform.samples
    validationSignal = config.driveRms * validationWaveform.samples
    paParameters = ChainMap(
        {"modelName": config.paModelName},
        paModelDefaultParameters,
    )
    analysisParameters = ChainMap({}, analysisDefaultParameters)
    paModel = PaModel(parameters=paParameters)
    trainingAnalysis = Analysis(
        trainingSignal,
        trainingWaveform,
        parameters=analysisParameters,
    )
    validationAnalysis = Analysis(
        validationSignal,
        validationWaveform,
        parameters=analysisParameters,
    )
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
        methodMetrics = trainingAnalysis.Analyze(methodResult.outputSignal)
        AddRow(
            rows,
            methodName,
            "ILC update law",
            "nominal repeated waveform",
            methodMetrics,
            baselineMetrics,
        )
        SaveHistory(methodName, methodResult, outputDirectory)
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
    SaveHistory("Constrained CFR-ILC", constrainedResult, outputDirectory)
    powerEvaluators["Constrained CFR-ILC"] = (
        lambda pointReference, _: RunFrequencyDomainIlc(
            pointReference,
            paModel,
            trainingWaveform.sampleRateHz,
            trainingWaveform.bandwidthHz,
            ILCConfig(
                numIterations=config.numIterations,
                learningRate=0.12,
                maxAmplitude=1.05 * np.max(np.abs(pointReference)),
                randomSeed=config.seed + 7,
            ),
        ).outputSignal
    )

    # Noise-aware learning uses a higher regularization and four averaged
    # feedback captures at 32 dB feedback SNR.
    noisyBaselineMetrics = baselineMetrics
    AddRow(
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
    noiseAwareMetrics = trainingAnalysis.Analyze(
        noiseAwareResult.outputSignal
    )
    AddRow(
        rows,
        "Noise-aware ILC",
        "ILC update law",
        "32 dB averaged feedback",
        noiseAwareMetrics,
        noisyBaselineMetrics,
    )
    SaveHistory("Noise-aware ILC", noiseAwareResult, outputDirectory)
    powerEvaluators["Noise-aware ILC"] = (
        lambda pointReference, _: RunFrequencyDomainIlc(
            pointReference,
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
        ).outputSignal
    )

    # Augmented ILC is evaluated in the IQ-image scenario for which its
    # conjugate branch is designed.
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
    SaveHistory("Augmented IQ ILC", augmentedResult, outputDirectory)
    powerEvaluators["IQ-imbalance baseline"] = (
        lambda pointReference, _: iqPaModel.Process(pointReference)
    )
    powerEvaluators["Augmented IQ ILC"] = (
        lambda pointReference, _: RunAugmentedIqIlc(
            pointReference,
            iqPaModel,
            ILCConfig(
                numIterations=config.numIterations,
                learningRate=0.18,
                maxAmplitude=maxAmplitude,
                randomSeed=config.seed + 9,
            ),
        ).outputSignal
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
        Algorithm: Convert validated in-memory results into a stable reporting format without altering later numerical calculations.

    Args:
        rows: Benchmark rows accumulated or emitted by the reporting operation.
        outputDirectory: Directory in which result artifacts are written.
        metadata: Caller-supplied value consumed according to the function contract.

    Returns:
        result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
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
        Algorithm: Convert validated in-memory results into a stable reporting format without altering later numerical calculations.

    Args:
        rows: Benchmark rows accumulated or emitted by the reporting operation.

    Returns:
        result: None. Completion is communicated through validation, state updates, saved artifacts, printed output, or assertions.
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
