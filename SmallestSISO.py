"""Run the smallest SISO ILC example in floating and fixed-point modes."""

from pathlib import Path
from typing import Dict

import numpy as np

from inc.lib.Analysis import Analysis
from inc.lib.Channel import Channel
from inc.lib.DpdIlc import ILCConfig, RunFrequencyDomainIlc
from inc.lib.PaModel import PaModel
from inc.lib.WaveGenWifi import WaveGenWifi
from inc.utils.Draw import Draw


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
        level corresponding to 20 dBm output backoff, pass every observation
        through a zero-degree channel with 10 mV RMS AWGN, execute
        frequency-domain ILC, independently analyze the baseline and selected
        ILC waveforms, and save one convergence curve.

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
    # Both modes use the same built-in GMP coefficient set.  Any result
    # difference therefore comes from the public fixed-point boundary,
    # including code quantization and full-scale clipping when peaks exceed
    # the signed converter range.
    paModel = PaModel(
        parameters={
            "modelName": "gmp",
            "width": width,
        },
    )
    channel = Channel(
        paModel=paModel,
        parameters={
            "sampleMode": "forward",
            "sampleRateHz": waveform.sampleRateHz,
            "phaseDegrees": 0,
            "noiseAmpMv": 10.0,
            "noisePwrDbm": None,
            "noiseSnrDb": None,
            "loadResistanceOhm": 50.0,
            "maximumOutputPowerDbm": maximumOutputPowerDbm,
            "randomSeed": 1019,
            "width": width,
        },
    )

    loadResistanceOhm = 50.0
    # The caller supplies only the arbitrary raw waveform and requested PA
    # output power. Channel owns the complete hidden closed loop: it changes
    # PA input drive, measures clean PA output, converges to the target, and
    # applies the forward receiver phase/noise once after calibration. Because
    # sampleMode is forward, fbOut is an exact copy of chOut in this example.
    baselineOutput, baselineFeedbackOutput = channel.Process(
        waveform.samples,
        outputPowerDbm=paOutputPowerDbm,
    )
    if not np.array_equal(baselineOutput, baselineFeedbackOutput):
        raise RuntimeError("forward sample mode must return identical outputs")
    referenceSignal = channel.GetLastPaInput()
    baselineCalibrationMetrics = (
        channel.GetLastCalibrationMetrics()
    )
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

    # Four iterations are sufficient for this deterministic minimum example
    # to improve both EVM and ACLR in floating and fixed modes.  Longer runs
    # are intentionally left to BenchMark.py, where stopping and metric
    # tradeoffs can be compared instead of hiding them in a minimal script.
    ilcConfig = ILCConfig(
        numIterations=4,
        learningRate=0.15,
        regularization=1.0e-3,
        maxAmplitude=2.0,
        randomSeed=1019,
    )
    # Channel exposes both receiver outputs to ILC. The learning residual and
    # coefficient update always use fbOut; in this forward-mode example it is
    # the exact chOut copy. Selecting sampleMode="fb" would instead route the
    # embedded feedback impairments into the learning observation.
    ilcResult = RunFrequencyDomainIlc(
        referenceSignal,
        channel,
        waveform.sampleRateHz,
        waveform.bandwidthHz,
        ilcConfig,
    )

    # Preserve native ILC measurements. Post-PA power normalization would hide
    # the real compression point and is intentionally not applied.
    ilcAnalysisResult = resultAnalysis.AnalyzeIlcHistory(
        ilcResult.history
    )
    selectedIlcOutput, selectedFeedbackOutput = channel.Process(
        ilcAnalysisResult.bestInputSignal,
        outputPowerDbm=paOutputPowerDbm,
    )
    if not np.array_equal(selectedIlcOutput, selectedFeedbackOutput):
        raise RuntimeError("forward sample mode must return identical outputs")
    selectedCalibrationMetrics = (
        channel.GetLastCalibrationMetrics()
    )
    selectedMetrics = resultAnalysis.Analyze(selectedIlcOutput)
    resultAnalysis.AnalyzeStages(
        {
            "PA + channel baseline": baselineOutput,
            "Frequency-domain ILC + channel": selectedIlcOutput,
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
        "baselinePaCalibration": baselineCalibrationMetrics,
        "selectedPaCalibration": selectedCalibrationMetrics,
        "channelParameters": channel.GetParameters(),
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
