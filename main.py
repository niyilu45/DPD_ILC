"""Command-line entry point for the HE/EHT DPD-ILC simulation project."""

import argparse
from pathlib import Path

import numpy as np

from inc.Analysis import Analysis
from inc.Benchmark import BenchmarkConfig, RunAllIlcBenchmark
from inc.DpdIlc import FitGmpPredistorter, ILCConfig, RunFrequencyDomainIlc
from inc.PaModel import PaModel
from inc.waveGen import GenWifi


def Main() -> int:
    """Generate a Wi-Fi packet, run PA/ILC/DPD stages, and save all metrics."""

    argumentParser = argparse.ArgumentParser(
        description=(
            "Simulate HE/EHT Wi-Fi excitation, a nonlinear PA, frequency-domain "
            "ILC, and a fitted GMP predistorter."
        )
    )
    argumentParser.add_argument(
        "--format",
        dest="frameFormat",
        choices=("EHT", "HE"),
        default="EHT",
        help="Wi-Fi PHY frame format (default: EHT)",
    )
    argumentParser.add_argument(
        "--bandwidth",
        dest="bandwidthMhz",
        type=int,
        choices=(20, 40, 80, 160),
        default=80,
        help="Wi-Fi channel bandwidth in MHz (default: 80)",
    )
    argumentParser.add_argument(
        "--mcs",
        type=int,
        choices=tuple(range(14)),
        default=11,
        help="MCS index: HE 0-11 or EHT 0-13 (default: 11)",
    )
    argumentParser.add_argument(
        "--pa",
        dest="paModelName",
        choices=("wiener", "gmp"),
        default="wiener",
        help="Nonlinear PA model family (default: wiener)",
    )
    argumentParser.add_argument(
        "--symbols",
        dest="numDataSymbols",
        type=int,
        default=20,
        help="Number of Wi-Fi data OFDM symbols (default: 20)",
    )
    argumentParser.add_argument(
        "--guard-interval",
        dest="guardIntervalUs",
        type=float,
        choices=(0.8, 1.6, 3.2),
        default=0.8,
        help="Data guard interval in microseconds (default: 0.8)",
    )
    argumentParser.add_argument(
        "--oversampling",
        type=int,
        choices=(4, 8),
        default=4,
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
        default=7,
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
    if arguments.numDataSymbols < 1:
        argumentParser.error("--symbols must be positive")
    if arguments.frameFormat == "HE" and arguments.mcs > 11:
        argumentParser.error("HE supports MCS values from 0 through 11")

    if arguments.benchmarkAllIlc:
        benchmarkDirectory = arguments.outputDirectory / "all_ilc_benchmark"
        RunAllIlcBenchmark(
            BenchmarkConfig(
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
                outputDirectory=benchmarkDirectory,
            )
        )
        print(f"\nAll-ILC results: {benchmarkDirectory.resolve()}")
        return 0

    wifiGenerator = GenWifi(
        frameFormat=arguments.frameFormat,
        bandwidthMhz=arguments.bandwidthMhz,
        mcs=arguments.mcs,
        numDataSymbols=arguments.numDataSymbols,
        guardIntervalUs=arguments.guardIntervalUs,
        oversampling=arguments.oversampling,
        seed=arguments.seed,
    )
    waveform = wifiGenerator.Generate()
    referenceSignal = arguments.driveRms * waveform.samples
    paModel = PaModel(modelName=arguments.paModelName)
    resultAnalysis = Analysis(referenceSignal, waveform)

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
        randomSeed=arguments.seed + 1000,
    )
    ilcResult = RunFrequencyDomainIlc(
        referenceSignal,
        paModel,
        waveform.sampleRateHz,
        waveform.bandwidthHz,
        ilcConfig,
    )

    # ILC labels are waveform-specific. Ridge-regression fitting converts them
    # into a causal GMP that can be evaluated on subsequent Wi-Fi packets.
    gmpPredistorter = FitGmpPredistorter(
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
        f"\n{waveform.frameFormat} {arguments.bandwidthMhz} MHz | MCS {arguments.mcs} "
        f"({waveform.mcsInfo.modulation}, rate {waveform.mcsInfo.codeRate:.3f}) "
        f"| PA {arguments.paModelName}\n"
    )
    resultAnalysis.Print()

    runMetadata = {
        "format": waveform.formatName,
        "frameFormat": waveform.frameFormat,
        "bandwidthMhz": arguments.bandwidthMhz,
        "sampleRateHz": waveform.sampleRateHz,
        "oversampling": arguments.oversampling,
        "mcs": arguments.mcs,
        "modulation": waveform.mcsInfo.modulation,
        "codeRate": waveform.mcsInfo.codeRate,
        "numDataSymbols": arguments.numDataSymbols,
        "guardIntervalUs": arguments.guardIntervalUs,
        "paModel": arguments.paModelName,
        "driveRms": arguments.driveRms,
        "ilcIterations": arguments.numIterations,
        "learningRate": arguments.learningRate,
        "regularization": arguments.regularization,
        "feedbackSnrDb": arguments.feedbackSnrDb,
        "feedbackAverages": arguments.feedbackAverages,
        "seed": arguments.seed,
    }
    jsonPath, csvPath = resultAnalysis.Save(
        arguments.outputDirectory, runMetadata
    )
    convergencePath = resultAnalysis.SaveConvergence(
        ilcResult.history, arguments.outputDirectory
    )

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
    print(f"ILC history:  {convergencePath.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(Main())
