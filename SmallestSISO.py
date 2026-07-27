"""Run the smallest SISO ILC example in floating and fixed-point modes."""

from pathlib import Path
from typing import Dict

import numpy as np

from inc.lib.Analysis import Analysis
from inc.lib.DpdIlc import ILCConfig, RunFrequencyDomainIlc
from inc.lib.PaModel import PaModel
from inc.lib.WaveGenWifi import WaveGenWifi
from inc.utils.Draw import Draw
from inc.utils.SigProc import PowerCalibration


def RunSisoMode(
    width: int,
    modeName: str,
    paOutputPowerDbm: float = 20.0,
    maximumOutputPowerDbm: float = 25.0,
) -> Dict[str, object]:
    """Run one complete SISO generation, PA, ILC, and analysis path.

    Processing details:
        Algorithm: Construct all three external signal interfaces with the
        same width, generate an EHT waveform, drive a GMP PA at a normalized
        level corresponding to 20 dBm output backoff, execute frequency-domain
        ILC, independently analyze the baseline and selected ILC waveforms,
        and save one convergence curve under a mode-specific directory.

    Args:
        width: Zero for floating point or a positive I/Q component word width.
        modeName: Human-readable label and result subdirectory name.
        paOutputPowerDbm: Requested average PA output power in dBm.
        maximumOutputPowerDbm: Rated PA output limit in dBm.

    Returns:
        result: Dictionary containing mode information and both metric sets.
    """

    wifiGenerator = WaveGenWifi(
        parameters={
            "frameFormat": "EHT",
            "bandwidthMhz": 20,
            "mcs": 7,
            "numDataSymbols": 10,
            "sampleRateHz": 80.0e6,
            "seed": 101,
            "width": width,
        },
    )
    waveform = wifiGenerator.Generate()
    waveformPower = np.abs(waveform.samples) ** 2
    waveformPeakAmplitude = float(
        np.max(np.abs(waveform.samples))
    )
    waveformMinimumI = float(np.min(waveform.samples.real))
    waveformMaximumI = float(np.max(waveform.samples.real))
    waveformMinimumQ = float(np.min(waveform.samples.imag))
    waveformMaximumQ = float(np.max(waveform.samples.imag))
    waveformPaprDb = float(
        10.0
        * np.log10(
            np.max(waveformPower)
            / max(
                float(np.mean(waveformPower)),
                np.finfo(float).tiny,
            )
        )
    )

    # The requested 20 dBm operating point becomes a drive ratio. In fixed
    # mode the multiplication acts on public integer codes; PaModel rounds
    # those codes and decodes them before normalized floating processing.
    loadResistanceOhm = 50.0
    powerCalibration = PowerCalibration(
        loadResistanceOhm=loadResistanceOhm,
        maximumOutputPowerDbm=maximumOutputPowerDbm,
    )
    driveScale = powerCalibration.OutputPowerToDriveScale(
        paOutputPowerDbm
    )
    referenceSignal = driveScale * waveform.samples

    # WaveGenWifi, PaModel, and Analysis use the same public interface width.
    # Fixed-mode arrays contain integer-valued I/Q codes in a complex128
    # container, while PA and analysis internals use decoded floating values.
    paModel = PaModel(
        parameters={
            "modelName": "gmp",
            "width": width,
        },
    )
    baselineOutput = paModel.Process(referenceSignal)
    resultAnalysis = Analysis(
        referenceSignal,
        waveform,
        parameters={
            "loadResistanceOhm": loadResistanceOhm,
            "width": width,
        },
    )
    baselineMetrics = resultAnalysis.Analyze(baselineOutput)

    ilcConfig = ILCConfig(
        numIterations=8,
        learningRate=0.15,
        regularization=1.0e-3,
        maxAmplitude=2.0,
        randomSeed=1019,
    )
    ilcResult = RunFrequencyDomainIlc(
        referenceSignal,
        paModel,
        waveform.sampleRateHz,
        waveform.bandwidthHz,
        ilcConfig,
    )

    # Analysis remains independent from the ILC update rule. It evaluates
    # every recorded PA output and selects the iteration with the best EVM.
    ilcAnalysisResult = resultAnalysis.AnalyzeIlcHistory(
        ilcResult.history
    )
    selectedIlcInput = ilcAnalysisResult.bestInputSignal
    selectedIlcOutput = paModel.Process(selectedIlcInput)
    selectedMetrics = resultAnalysis.Analyze(selectedIlcOutput)
    resultAnalysis.AnalyzeStages(
        {
            "PA baseline": baselineOutput,
            "Frequency-domain ILC": selectedIlcOutput,
        }
    )
    resultAnalysis.PrintConvergence(
        ilcAnalysisResult.history,
        historyName=f"{modeName} frequency-domain ILC",
    )

    outputDirectory = Path("results") / "smallest_siso" / modeName
    resultAnalysis.SaveConvergence(
        ilcAnalysisResult.history,
        outputDirectory,
    )
    Draw().SaveConvergenceCurve(
        ilcAnalysisResult.history,
        outputDirectory,
        fileStem="frequency_domain_ilc",
    )

    # Convert a copy of the PA code output to the requested physical reporting
    # level without feeding voltage-scaled values back into the code interface.
    calibratedOutput = powerCalibration.ScaleSignalToOutputPower(
        baselineOutput,
        paOutputPowerDbm,
    )
    outputRms = float(
        np.sqrt(np.mean(np.abs(calibratedOutput) ** 2))
    )
    measuredOutputPowerDbm = powerCalibration.RmsToDbm(outputRms)
    result = {
        "mode": modeName,
        "width": width,
        "baselineMetrics": baselineMetrics,
        "selectedIlcMetrics": selectedMetrics,
        "bestIteration": ilcAnalysisResult.bestIteration,
        "waveformPeakAmplitude": waveformPeakAmplitude,
        "waveformMinimumI": waveformMinimumI,
        "waveformMaximumI": waveformMaximumI,
        "waveformMinimumQ": waveformMinimumQ,
        "waveformMaximumQ": waveformMaximumQ,
        "waveformPaprDb": waveformPaprDb,
        "configuredOutputPowerDbm": paOutputPowerDbm,
        "measuredOutputPowerDbm": measuredOutputPowerDbm,
    }
    print(result)
    return result


def Main() -> None:
    """Run and compare the floating-point and default 16-bit examples.

    Processing details:
        Algorithm: Execute the identical SISO scenario twice, first with the
        floating bypass and then with 16-bit public integer-code interfaces,
        before printing the fixed-minus-floating EVM difference.

    Returns:
        result: None. Results are printed and mode-specific plots are saved.
    """

    floatingResult = RunSisoMode(width=0, modeName="floating")
    fixedResult = RunSisoMode(width=16, modeName="fixed_16")
    floatingEvmDb = float(
        floatingResult["selectedIlcMetrics"]["evmDb"]
    )
    fixedEvmDb = float(
        fixedResult["selectedIlcMetrics"]["evmDb"]
    )
    print(
        "Fixed 16-bit minus floating selected EVM: "
        f"{fixedEvmDb - floatingEvmDb:.3f} dB"
    )


if __name__ == "__main__":
    Main()
