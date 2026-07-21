"""Self-contained project checks that preserve the requested naming style."""

from pathlib import Path
import sys

import numpy as np


projectRoot = Path(__file__).resolve().parents[1]
if str(projectRoot) not in sys.path:
    sys.path.insert(0, str(projectRoot))

from inc.Analysis import Analysis
from inc.DpdIlc import ILCConfig, RunFrequencyDomainIlc
from inc.PaModel import PaModel
from inc.waveGen import GenWifi, ehtMcsTable, heMcsTable


def CheckMcsTables() -> None:
    """Verify the complete HE and EHT MCS ranges."""

    assert set(ehtMcsTable.keys()) == set(range(14))
    assert set(heMcsTable.keys()) == set(range(12))
    assert ehtMcsTable[0].qamOrder == 2
    assert ehtMcsTable[13].qamOrder == 4096
    assert ehtMcsTable[13].codeRate == 5.0 / 6.0
    assert heMcsTable[11].qamOrder == 1024


def CheckWifiFormats() -> None:
    """Verify that each generator instance creates its selected frame format."""

    formatExpectations = {
        "EHT": {
            "L-STF",
            "L-LTF",
            "L-SIG",
            "RL-SIG",
            "U-SIG",
            "EHT-SIG",
            "EHT-STF",
            "EHT-LTF",
            "EHT-Data",
        },
        "HE": {
            "L-STF",
            "L-LTF",
            "L-SIG",
            "RL-SIG",
            "HE-SIG-A",
            "HE-STF",
            "HE-LTF",
            "HE-Data",
        },
    }
    for frameFormat, expectedFields in formatExpectations.items():
        wifiGenerator = GenWifi(
            frameFormat=frameFormat,
            bandwidthMhz=20,
            mcs=0,
            numDataSymbols=2,
            oversampling=1,
        )
        waveform = wifiGenerator.Generate()
        assert waveform.frameFormat == frameFormat
        assert waveform.dataFieldName == f"{frameFormat}-Data"
        assert set(waveform.fieldSlices) == expectedFields

        # Verify common fixed field durations at the configured sample rate.
        assert (
            waveform.fieldSlices["L-STF"].stop
            - waveform.fieldSlices["L-STF"].start
        ) == int(round(8e-6 * waveform.sampleRateHz))
        assert (
            waveform.fieldSlices[f"{frameFormat}-STF"].stop
            - waveform.fieldSlices[f"{frameFormat}-STF"].start
        ) == int(round(4e-6 * waveform.sampleRateHz))


def CheckWifiBandwidths() -> None:
    """Verify nominal FFT and RU data/pilot counts for every bandwidth."""

    expectedValues = {
        20: (256, 234, 8),
        40: (512, 468, 16),
        80: (1024, 980, 16),
        160: (2048, 1960, 32),
    }
    for bandwidthMhz, (
        baseFftLength,
        dataToneCount,
        pilotToneCount,
    ) in expectedValues.items():
        wifiGenerator = GenWifi(
            frameFormat="EHT",
            bandwidthMhz=bandwidthMhz,
            mcs=0,
            numDataSymbols=2,
            oversampling=1,
        )
        waveform = wifiGenerator.Generate()
        assert waveform.fftLength == baseFftLength
        assert waveform.dataSubcarriers.size == dataToneCount
        assert waveform.pilotSubcarriers.size == pilotToneCount


def CheckFormatSpecificMcsValidation() -> None:
    """Verify that HE rejects the EHT-only 4096-QAM MCS values."""

    try:
        GenWifi(frameFormat="HE", mcs=12).Generate()
    except ValueError as error:
        assert "HE MCS" in str(error)
    else:
        raise AssertionError("HE MCS 12 must be rejected")


def CheckIdealMetrics() -> None:
    """Verify that a perfect signal path has effectively zero EVM."""

    for frameFormat, mcs in (("EHT", 13), ("HE", 11)):
        wifiGenerator = GenWifi(
            frameFormat=frameFormat,
            mcs=mcs,
            numDataSymbols=4,
            oversampling=4,
        )
        waveform = wifiGenerator.Generate()
        resultAnalysis = Analysis(waveform.samples, waveform)
        metrics = resultAnalysis.Analyze(waveform.samples)
        assert metrics.snrDb > 250.0
        assert metrics.evmDb < -250.0
        assert metrics.evmPercent < 1e-10


def CheckGuardIntervals() -> None:
    """Verify compatible 2x/4x long-training durations for every GI."""

    expectedLtfDurationUs = {0.8: 13.6, 1.6: 8.0, 3.2: 16.0}
    for frameFormat in ("EHT", "HE"):
        for guardIntervalUs, ltfDurationUs in expectedLtfDurationUs.items():
            wifiGenerator = GenWifi(
                frameFormat=frameFormat,
                bandwidthMhz=20,
                mcs=0,
                numDataSymbols=1,
                guardIntervalUs=guardIntervalUs,
                oversampling=1,
            )
            waveform = wifiGenerator.Generate()
            ltfSlice = waveform.fieldSlices[f"{frameFormat}-LTF"]
            ltfSampleCount = ltfSlice.stop - ltfSlice.start
            assert ltfSampleCount == int(
                round(ltfDurationUs * 1e-6 * waveform.sampleRateHz)
            )


def CheckIlcImprovement() -> None:
    """Verify that ILC reduces reconstruction error for both PA families."""

    wifiGenerator = GenWifi(
        frameFormat="HE",
        bandwidthMhz=20,
        mcs=7,
        numDataSymbols=6,
        oversampling=4,
        seed=31,
    )
    waveform = wifiGenerator.Generate()
    referenceSignal = 0.28 * waveform.samples
    resultAnalysis = Analysis(referenceSignal, waveform)
    for modelName in ("wiener", "gmp"):
        paModel = PaModel(modelName=modelName)
        baselineOutput = paModel.Process(referenceSignal)
        baselineMetrics = resultAnalysis.Analyze(baselineOutput)
        ilcResult = RunFrequencyDomainIlc(
            referenceSignal,
            paModel,
            waveform.sampleRateHz,
            waveform.bandwidthHz,
            ILCConfig(
                numIterations=6,
                learningRate=0.35,
                maxAmplitude=1.25,
            ),
        )
        ilcMetrics = resultAnalysis.Analyze(ilcResult.outputSignal)
        assert ilcMetrics.evmDb < baselineMetrics.evmDb
        assert ilcMetrics.snrDb > baselineMetrics.snrDb


def RunTests() -> None:
    """Run all project checks and report a compact success message."""

    CheckMcsTables()
    CheckWifiFormats()
    CheckWifiBandwidths()
    CheckFormatSpecificMcsValidation()
    CheckIdealMetrics()
    CheckGuardIntervals()
    CheckIlcImprovement()
    print("All DPD-ILC project checks passed.")


if __name__ == "__main__":
    RunTests()
