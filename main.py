"""Command-line entry point for the VHT/HE/EHT DPD-ILC simulation project."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from inc.Analysis import Analysis
from inc.Benchmark import BenchmarkConfig, RunAllIlcBenchmark
from inc.DpdIlc import FitGmpPredistorter, ILCConfig, RunFrequencyDomainIlc
from inc.Draw import Draw
from inc.MimoDpd import FitMimoGmpPredistorter, RunMimoFrequencyDomainIlc
from inc.PaModel import MimoPaModel, PaModel
from inc.waveGen import (
    GenWifi,
    NormalizeFrameFormat,
)


def ParseFloatSequence(rawValue: str) -> tuple:
    """Parse a comma-separated list of finite floating-point values.

    Processing details:
        Algorithm: Split on commas, reject empty fields, convert each token,
        and require finite values suitable for per-chain PA settings.

    Args:
        rawValue: Command-line text such as ``0,-3.0,1.5``.

    Returns:
        result: Tuple of floats in physical transmit-chain order.
    """

    try:
        values = tuple(
            float(token.strip()) for token in rawValue.split(",")
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated numeric values"
        ) from error
    if not values or any(not np.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError(
            "per-chain values must be finite and nonempty"
        )
    return values


def ParseOptionalFloatSequence(rawValue: str) -> tuple:
    """Parse positive RMS values while allowing ``none`` per PA chain.

    Processing details:
        Algorithm: Interpret case-insensitive ``none`` as a disabled target,
        parse all other comma-separated tokens as positive finite floats.

    Args:
        rawValue: Text such as ``0.20,none,0.18``.

    Returns:
        result: Tuple containing positive floats or None entries.
    """

    values = []
    for token in rawValue.split(","):
        normalizedToken = token.strip().lower()
        if normalizedToken == "none":
            values.append(None)
            continue
        try:
            numericValue = float(normalizedToken)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "expected positive RMS values or 'none'"
            ) from error
        if not np.isfinite(numericValue) or numericValue <= 0.0:
            raise argparse.ArgumentTypeError(
                "enabled output RMS targets must be positive and finite"
            )
        values.append(numericValue)
    if not values:
        raise argparse.ArgumentTypeError("per-chain RMS list cannot be empty")
    return tuple(values)


def Main() -> int:
    """Generate a Wi-Fi packet, run PA/ILC/DPD stages, and save all metrics.

    Processing details:
        Algorithm: Coordinate validation, waveform generation, PA/ILC execution, metric calculation, and artifact reporting.

    Returns:
        result: int. The computed value described by the summary, with documented units, shape, and normalization.
    """

    argumentParser = argparse.ArgumentParser(
        description=(
            "Simulate VHT/HE/EHT Wi-Fi excitation, a nonlinear PA, "
            "frequency-domain ILC, and a fitted GMP predistorter."
        )
    )
    argumentParser.add_argument(
        "--format",
        dest="frameFormat",
        type=NormalizeFrameFormat,
        choices=("VHT", "HE", "EHT"),
        metavar="FORMAT",
        default=None,
        help=(
            "Wi-Fi format: VHT/11ac, HE/11ax, or EHT/11be "
            "(default: EHT)"
        ),
    )
    argumentParser.add_argument(
        "--bandwidth",
        dest="bandwidthMhz",
        type=int,
        choices=(20, 40, 80, 160),
        default=None,
        help="Wi-Fi channel bandwidth in MHz (default: 80)",
    )
    argumentParser.add_argument(
        "--mcs",
        type=int,
        choices=tuple(range(14)),
        default=None,
        help="MCS index: VHT 0-9, HE 0-11, or EHT 0-13 (default: 9)",
    )
    argumentParser.add_argument(
        "--pa",
        dest="paModelName",
        choices=("wiener", "gmp"),
        default=None,
        help="Nonlinear PA model family (default: wiener)",
    )
    argumentParser.add_argument(
        "--tx-antennas",
        dest="numTransmitAntennas",
        type=int,
        default=None,
        help="Number of physical transmit chains: VHT/HE/EHT 1-8",
    )
    argumentParser.add_argument(
        "--spatial-streams",
        dest="numSpatialStreams",
        type=int,
        default=None,
        help="Number of independent spatial streams, no greater than TX chains",
    )
    argumentParser.add_argument(
        "--spatial-mapping",
        dest="spatialMapping",
        choices=("direct", "dft"),
        default=None,
        help="Stream-to-antenna mapping matrix (default: direct)",
    )
    argumentParser.add_argument(
        "--pa-input-power-db",
        dest="paInputPowerDbPerChain",
        type=ParseFloatSequence,
        default=None,
        metavar="DB0,DB1,...",
        help="Independent input-drive adjustment in dB for every PA chain",
    )
    argumentParser.add_argument(
        "--pa-output-power-db",
        dest="paOutputPowerDbPerChain",
        type=ParseFloatSequence,
        default=None,
        metavar="DB0,DB1,...",
        help="Independent relative output-power adjustment for every PA chain",
    )
    argumentParser.add_argument(
        "--pa-output-rms",
        dest="paTargetOutputRmsPerChain",
        type=ParseOptionalFloatSequence,
        default=None,
        metavar="RMS0,RMS1,...",
        help="Absolute output RMS target per PA; use 'none' to disable a chain target",
    )
    argumentParser.add_argument(
        "--symbols",
        dest="numDataSymbols",
        type=int,
        default=None,
        help="Number of Wi-Fi data OFDM symbols (default: 20)",
    )
    argumentParser.add_argument(
        "--guard-interval",
        dest="guardIntervalUs",
        type=float,
        choices=(0.4, 0.8, 1.6, 3.2),
        default=None,
        help=(
            "Data guard interval in microseconds: VHT 0.4/0.8, "
            "HE/EHT 0.8/1.6/3.2 (default: 0.8)"
        ),
    )
    argumentParser.add_argument(
        "--oversampling",
        type=int,
        choices=(4, 8),
        default=None,
        help="Oversampling factor; ACLR requires at least 4 (default: 4)",
    )
    argumentParser.add_argument(
        "--drive",
        dest="driveRms",
        type=float,
        default=0.24,
        help="RMS PA input drive relative to unit saturation (default: 0.24)",
    )
    argumentParser.add_argument(
        "--power-start",
        dest="powerStartRms",
        type=float,
        default=0.08,
        help="First RMS drive in the power-EVM sweep (default: 0.08)",
    )
    argumentParser.add_argument(
        "--power-stop",
        dest="powerStopRms",
        type=float,
        default=0.40,
        help="Last RMS drive in the power-EVM sweep (default: 0.40)",
    )
    argumentParser.add_argument(
        "--power-points",
        dest="powerPointCount",
        type=int,
        default=7,
        help="Number of logarithmically spaced power-EVM points (default: 7)",
    )
    argumentParser.add_argument(
        "--skip-power-evm-curve",
        dest="skipPowerEvmCurve",
        action="store_true",
        help="Skip the PNG/CSV/JSON multi-method power-EVM sweep",
    )
    argumentParser.add_argument(
        "--iterations",
        dest="numIterations",
        type=int,
        default=8,
        help="Number of ILC iterations (default: 8)",
    )
    argumentParser.add_argument(
        "--learning-rate",
        dest="learningRate",
        type=float,
        default=0.15,
        help="ILC learning gain in the open interval (0, 2) (default: 0.15)",
    )
    argumentParser.add_argument(
        "--regularization",
        type=float,
        default=1e-3,
        help="ILC inverse-response regularization (default: 1e-3)",
    )
    argumentParser.add_argument(
        "--max-amplitude",
        dest="maxAmplitude",
        type=float,
        default=2.0,
        help="Peak constraint for learned and fitted DPD inputs (default: 2.0)",
    )
    argumentParser.add_argument(
        "--feedback-snr",
        dest="feedbackSnrDb",
        type=float,
        default=None,
        help="Optional feedback SNR in dB; omitted means noiseless feedback",
    )
    argumentParser.add_argument(
        "--feedback-averages",
        dest="feedbackAverages",
        type=int,
        default=1,
        help="Repeated feedback captures averaged per ILC step (default: 1)",
    )
    argumentParser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for Wi-Fi data and training fields (default: 7)",
    )
    argumentParser.add_argument(
        "--output-dir",
        dest="outputDirectory",
        type=Path,
        default=Path("results"),
        help="Directory for JSON, CSV, and optional waveform files",
    )
    argumentParser.add_argument(
        "--save-waveforms",
        dest="saveWaveforms",
        action="store_true",
        help="Save reference, PA, ILC, and deployed-DPD arrays as compressed NPZ",
    )
    argumentParser.add_argument(
        "--benchmark-all-ilc",
        dest="benchmarkAllIlc",
        action="store_true",
        help="Run every ILC update law and every ILC-label deployment model",
    )
    arguments = argumentParser.parse_args()

    if arguments.driveRms <= 0.0:
        argumentParser.error("--drive must be positive")
    if arguments.powerStartRms <= 0.0:
        argumentParser.error("--power-start must be positive")
    if arguments.powerStopRms <= arguments.powerStartRms:
        argumentParser.error("--power-stop must exceed --power-start")
    if arguments.powerPointCount < 2:
        argumentParser.error("--power-points must be at least 2")

    # Only explicitly supplied CLI values are passed to the object. GenWifi
    # and PaModel add their immutable default layers internally.
    wifiArgumentNames = (
        "frameFormat",
        "bandwidthMhz",
        "mcs",
        "numDataSymbols",
        "guardIntervalUs",
        "oversampling",
        "seed",
        "numTransmitAntennas",
        "numSpatialStreams",
        "spatialMapping",
    )
    wifiOverrides = {
        argumentName: getattr(arguments, argumentName)
        for argumentName in wifiArgumentNames
        if getattr(arguments, argumentName) is not None
    }
    paOverrides = (
        {}
        if arguments.paModelName is None
        else {"modelName": arguments.paModelName}
    )
    try:
        wifiGenerator = GenWifi(parameters=wifiOverrides)
        useMimoPaFacade = wifiGenerator.numTransmitAntennas > 1 or any(
            value is not None
            for value in (
                arguments.paInputPowerDbPerChain,
                arguments.paOutputPowerDbPerChain,
                arguments.paTargetOutputRmsPerChain,
            )
        )
        if not useMimoPaFacade:
            paModel = PaModel(parameters=paOverrides)
        else:
            mimoPaOverrides = {
                "numTransmitChains": wifiGenerator.numTransmitAntennas,
                "paParametersPerChain": tuple(
                    dict(paOverrides)
                    for _ in range(wifiGenerator.numTransmitAntennas)
                ),
            }
            if arguments.paInputPowerDbPerChain is not None:
                mimoPaOverrides["inputPowerDbPerChain"] = (
                    arguments.paInputPowerDbPerChain
                )
            if arguments.paOutputPowerDbPerChain is not None:
                mimoPaOverrides["outputPowerDbPerChain"] = (
                    arguments.paOutputPowerDbPerChain
                )
            if arguments.paTargetOutputRmsPerChain is not None:
                mimoPaOverrides["targetOutputRmsPerChain"] = (
                    arguments.paTargetOutputRmsPerChain
                )
            paModel = MimoPaModel(parameters=mimoPaOverrides)
    except (TypeError, ValueError) as error:
        argumentParser.error(str(error))

    frameFormat = wifiGenerator.frameFormat
    bandwidthMhz = wifiGenerator.bandwidthMhz
    mcs = wifiGenerator.mcs
    numDataSymbols = wifiGenerator.numDataSymbols
    guardIntervalUs = wifiGenerator.guardIntervalUs
    oversampling = wifiGenerator.oversampling
    seed = wifiGenerator.seed
    paModelName = (
        paModel.modelName
        if isinstance(paModel, PaModel)
        else str(paOverrides.get("modelName", "wiener"))
    )

    if arguments.benchmarkAllIlc:
        if wifiGenerator.numTransmitAntennas > 1:
            argumentParser.error(
                "--benchmark-all-ilc currently requires a SISO waveform"
            )
        benchmarkDirectory = arguments.outputDirectory / "all_ilc_benchmark"
        RunAllIlcBenchmark(
            BenchmarkConfig(
                frameFormat=frameFormat,
                bandwidthMhz=bandwidthMhz,
                mcs=mcs,
                numDataSymbols=numDataSymbols,
                oversampling=oversampling,
                guardIntervalUs=guardIntervalUs,
                driveRms=arguments.driveRms,
                numIterations=arguments.numIterations,
                paModelName=paModelName,
                seed=seed,
                powerStartRms=arguments.powerStartRms,
                powerStopRms=arguments.powerStopRms,
                powerPointCount=arguments.powerPointCount,
                generatePowerEvmCurve=not arguments.skipPowerEvmCurve,
                outputDirectory=benchmarkDirectory,
            )
        )
        print(f"\nAll-ILC results: {benchmarkDirectory.resolve()}")
        return 0

    waveform = wifiGenerator.Generate()
    referenceSignal = arguments.driveRms * waveform.samples
    resultAnalysis = Analysis(referenceSignal, waveform)
    resultDraw = Draw()

    # The first pass establishes the unlinearized baseline at the requested
    # operating point. The same PA instance is reused for every comparison.
    baselineOutput = paModel.Process(referenceSignal)
    ilcConfig = ILCConfig(
        numIterations=arguments.numIterations,
        learningRate=arguments.learningRate,
        regularization=arguments.regularization,
        maxAmplitude=arguments.maxAmplitude,
        feedbackSnrDb=arguments.feedbackSnrDb,
        feedbackAverages=arguments.feedbackAverages,
        randomSeed=seed + 1000,
        evmMseEvaluator=(
            resultAnalysis.CalculateEvmAlignedMse
            if waveform.numTransmitAntennas == 1
            else None
        ),
    )
    if waveform.numTransmitAntennas == 1:
        ilcResult = RunFrequencyDomainIlc(
            referenceSignal,
            paModel,
            waveform.sampleRateHz,
            waveform.bandwidthHz,
            ilcConfig,
        )
    else:
        ilcResult = RunMimoFrequencyDomainIlc(
            referenceSignal,
            paModel,
            waveform.sampleRateHz,
            waveform.bandwidthHz,
            ilcConfig,
        )

    # ILC labels are waveform-specific. Ridge-regression fitting converts them
    # into a causal GMP that can be evaluated on subsequent Wi-Fi packets.
    if waveform.numTransmitAntennas == 1:
        gmpPredistorter = FitGmpPredistorter(
            referenceSignal,
            ilcResult.learnedInput,
            nonlinearOrders=(1, 3, 5, 7),
            memoryDepth=3,
            crossMemoryDepth=2,
            ridgeFactor=1e-6,
        )
    else:
        gmpPredistorter = FitMimoGmpPredistorter(
            referenceSignal,
            ilcResult.learnedInput,
            nonlinearOrders=(1, 3, 5, 7),
            memoryDepth=3,
            crossMemoryDepth=2,
            ridgeFactor=1e-6,
        )
    deployedDpdInput = gmpPredistorter.Process(referenceSignal)
    deployedMagnitude = np.abs(deployedDpdInput)
    overLimit = deployedMagnitude > arguments.maxAmplitude
    if np.any(overLimit):
        deployedDpdInput[overLimit] *= (
            arguments.maxAmplitude / deployedMagnitude[overLimit]
        )
    deployedDpdOutput = paModel.Process(deployedDpdInput)

    resultAnalysis.AnalyzeStages(
        {
            "PA baseline": baselineOutput,
            "Waveform ILC": ilcResult.outputSignal,
            "Fitted GMP DPD": deployedDpdOutput,
        }
    )
    print(
        f"\n{waveform.frameFormat} {bandwidthMhz} MHz | MCS {mcs} "
        f"({waveform.mcsInfo.modulation}, rate {waveform.mcsInfo.codeRate:.3f}) "
        f"| {waveform.numSpatialStreams} spatial stream(s) / "
        f"{waveform.numTransmitAntennas} PA chain(s) | PA {paModelName}\n"
    )
    if isinstance(paModel, MimoPaModel):
        outputRmsText = ", ".join(
            f"PA {chainIndex + 1}={outputRms:.5f} RMS"
            for chainIndex, outputRms in enumerate(
                paModel.GetOutputRmsPerChain()
            )
        )
        print(f"Latest independent PA outputs: {outputRmsText}\n")
    resultAnalysis.Print()
    if waveform.numTransmitAntennas > 1:
        resultAnalysis.PrintMimo()
    if waveform.numTransmitAntennas == 1:
        resultAnalysis.PrintConvergence(
            ilcResult.history, "Waveform ILC iteration metrics"
        )
    else:
        for chainIndex, chainResult in enumerate(ilcResult.chainResults):
            resultAnalysis.PrintConvergence(
                chainResult.history,
                f"PA {chainIndex + 1} ILC iteration metrics",
            )

    runMetadata = {
        "format": waveform.formatName,
        "frameFormat": waveform.frameFormat,
        "bandwidthMhz": bandwidthMhz,
        "sampleRateHz": waveform.sampleRateHz,
        "oversampling": oversampling,
        "mcs": mcs,
        "modulation": waveform.mcsInfo.modulation,
        "codeRate": waveform.mcsInfo.codeRate,
        "numDataSymbols": numDataSymbols,
        "guardIntervalUs": guardIntervalUs,
        "numTransmitAntennas": waveform.numTransmitAntennas,
        "numSpatialStreams": waveform.numSpatialStreams,
        "spatialMapping": waveform.spatialMapping,
        "spatialMappingMatrix": [
            [
                {"real": float(value.real), "imag": float(value.imag)}
                for value in row
            ]
            for row in waveform.spatialMappingMatrix
        ],
        "cyclicShiftsSeconds": waveform.cyclicShiftsSeconds.tolist(),
        "ltfSymbolCount": waveform.ltfSymbolCount,
        "paModel": paModelName,
        "paInputPowerDbPerChain": (
            None
            if not isinstance(paModel, MimoPaModel)
            else list(
                paModel.ResolveNumericSequence(
                    "inputPowerDbPerChain", 0.0
                )
            )
        ),
        "paOutputPowerDbPerChain": (
            None
            if not isinstance(paModel, MimoPaModel)
            else list(
                paModel.ResolveNumericSequence(
                    "outputPowerDbPerChain", 0.0
                )
            )
        ),
        "paTargetOutputRmsPerChain": (
            None
            if not isinstance(paModel, MimoPaModel)
            else list(
                paModel.ResolveNumericSequence(
                    "targetOutputRmsPerChain",
                    0.0,
                    allowNoneEntries=True,
                )
            )
        ),
        "paMeasuredOutputRmsPerChain": (
            None
            if not isinstance(paModel, MimoPaModel)
            else list(paModel.GetOutputRmsPerChain())
        ),
        "driveRms": arguments.driveRms,
        "powerStartRms": arguments.powerStartRms,
        "powerStopRms": arguments.powerStopRms,
        "powerPointCount": arguments.powerPointCount,
        "generatePowerEvmCurve": not arguments.skipPowerEvmCurve,
        "ilcIterations": arguments.numIterations,
        "learningRate": arguments.learningRate,
        "regularization": arguments.regularization,
        "feedbackSnrDb": arguments.feedbackSnrDb,
        "feedbackAverages": arguments.feedbackAverages,
        "seed": seed,
    }
    jsonPath, csvPath = resultAnalysis.Save(
        arguments.outputDirectory, runMetadata
    )
    if waveform.numTransmitAntennas == 1:
        convergencePaths = (
            resultAnalysis.SaveConvergence(
                ilcResult.history, arguments.outputDirectory
            ),
        )
        convergenceFigurePaths = (
            resultDraw.SaveConvergenceCurve(
                ilcResult.history, arguments.outputDirectory
            ),
        )
    else:
        convergencePaths = tuple(
            resultAnalysis.SaveConvergence(
                chainResult.history,
                arguments.outputDirectory / f"pa_chain_{chainIndex + 1}",
            )
            for chainIndex, chainResult in enumerate(ilcResult.chainResults)
        )
        convergenceFigurePaths = tuple(
            resultDraw.SaveConvergenceCurve(
                chainResult.history,
                arguments.outputDirectory / f"pa_chain_{chainIndex + 1}",
            )
            for chainIndex, chainResult in enumerate(ilcResult.chainResults)
        )

    powerCurvePaths = None
    if not arguments.skipPowerEvmCurve:
        # A nominal-reference evaluator cannot score other drive levels. The
        # point sweep therefore falls back to gain-compensated NMSE for ILC
        # best-iteration selection; final EVM remains evaluated against each
        # point's correctly scaled Wi-Fi reference inside Analysis.
        powerSweepIlcConfig = replace(ilcConfig, evmMseEvaluator=None)
        powerDriveValues = np.geomspace(
            arguments.powerStartRms,
            arguments.powerStopRms,
            arguments.powerPointCount,
        )
        methodEvaluators = {
            "PA baseline": lambda pointReference, _: paModel.Process(
                pointReference
            ),
            "Fitted GMP DPD": lambda pointReference, _: paModel.Process(
                gmpPredistorter.Process(pointReference)
            ),
        }
        if waveform.numTransmitAntennas == 1:
            methodEvaluators["Frequency-domain ILC"] = (
                lambda pointReference, _: RunFrequencyDomainIlc(
                    pointReference,
                    paModel,
                    waveform.sampleRateHz,
                    waveform.bandwidthHz,
                    powerSweepIlcConfig,
                ).outputSignal
            )
        else:
            methodEvaluators["Frequency-domain ILC"] = (
                lambda pointReference, _: RunMimoFrequencyDomainIlc(
                    pointReference,
                    paModel,
                    waveform.sampleRateHz,
                    waveform.bandwidthHz,
                    powerSweepIlcConfig,
                ).outputSignal
            )
        resultAnalysis.AnalyzePowerEvmCurve(
            powerDriveValues, methodEvaluators
        )
        powerEvmCurve = resultAnalysis.powerEvmCurve
        powerDataPaths = resultAnalysis.SavePowerEvmCurveData(
            arguments.outputDirectory
        )
        powerFigurePath = resultDraw.SavePowerEvmCurve(
            powerEvmCurve,
            arguments.outputDirectory,
        )
        powerCurvePaths = (*powerDataPaths, powerFigurePath)

    if arguments.saveWaveforms:
        waveformPath = arguments.outputDirectory / "waveforms.npz"
        np.savez_compressed(
            waveformPath,
            referenceSignal=referenceSignal,
            baselineOutput=baselineOutput,
            learnedIlcInput=ilcResult.learnedInput,
            ilcOutput=ilcResult.outputSignal,
            deployedDpdInput=deployedDpdInput,
            deployedDpdOutput=deployedDpdOutput,
        )
        print(f"Waveforms:   {waveformPath.resolve()}")

    print(f"Metrics JSON: {jsonPath.resolve()}")
    print(f"Metrics CSV:  {csvPath.resolve()}")
    for chainIndex, convergencePath in enumerate(convergencePaths):
        historyLabel = (
            "ILC history"
            if len(convergencePaths) == 1
            else f"PA {chainIndex + 1} ILC history"
        )
        print(f"{historyLabel}: {convergencePath.resolve()}")
        print(
            f"{historyLabel} plot: "
            f"{convergenceFigurePaths[chainIndex].resolve()}"
        )
    if powerCurvePaths is not None:
        powerCsvPath, powerJsonPath, powerFigurePath = powerCurvePaths
        print(f"Power-EVM CSV:  {powerCsvPath.resolve()}")
        print(f"Power-EVM JSON: {powerJsonPath.resolve()}")
        print(f"Power-EVM plot: {powerFigurePath.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(Main())
