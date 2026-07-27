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
    waveformPeakAmplitude = float(
        np.max(np.abs(waveform.samples))
    )
    waveformMinimumI = float(np.min(waveform.samples.real))
    waveformMaximumI = float(np.max(waveform.samples.real))
    waveformMinimumQ = float(np.min(waveform.samples.imag))
    waveformMaximumQ = float(np.max(waveform.samples.imag))

    # WaveGenWifi, PaModel, and Analysis use the same public interface width.
    # Fixed-mode arrays contain integer-valued I/Q codes in a complex128
    # container, while PA and analysis internals use decoded floating values.
    paModel = PaModel(
        parameters={
            "modelName": "gmp",
            "width": width,
        },
    )

    # Closed-loop calibration owns the hidden drive preset. It repeatedly
    # regenerates the original Wi-Fi waveform, measures the actual PA output,
    # and returns the converged PA input without exposing preset corrections.
    loadResistanceOhm = 50.0
    powerCalibration = PowerCalibration(
        paModel=paModel,
        parameters={
            "loadResistanceOhm": loadResistanceOhm,
            "maximumOutputPowerDbm": maximumOutputPowerDbm,
            "outputPowerDbm": paOutputPowerDbm,
            "width": width,
        },
    )
    referenceSignal = powerCalibration.Calibrate(waveform.samples)
    baselineOutput = powerCalibration.GetLastPaOutput()
    resultAnalysis = Analysis(
        referenceSignal,
        waveform,
        parameters={
            "loadResistanceOhm": loadResistanceOhm,
            "maximumOutputPowerDbm": maximumOutputPowerDbm,
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

    # Preserve native ILC measurements. Post-PA power normalization would hide
    # the real compression point and is intentionally not applied.
    ilcAnalysisResult = resultAnalysis.AnalyzeIlcHistory(
        ilcResult.history
    )
    selectedIlcInput = powerCalibration.Calibrate(
        ilcAnalysisResult.bestInputSignal
    )
    selectedIlcOutput = powerCalibration.GetLastPaOutput()
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

    # Analysis and calibration share the same active-burst RMS convention, so
    # leading/trailing padding and long internal off intervals do not reduce
    # the reported PA output power.
    measuredOutputPowerDbm = baselineMetrics["outputPowerDbm"]
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
