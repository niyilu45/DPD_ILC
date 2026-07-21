"""Self-contained project checks that preserve the requested naming style."""

from pathlib import Path
import sys

import numpy as np


projectRoot = Path(__file__).resolve().parents[1]
if str(projectRoot) not in sys.path:
    sys.path.insert(0, str(projectRoot))

from inc.Analysis import AnalyzeSignal
from inc.DpdIlc import ILCConfig, RunFrequencyDomainIlc
from inc.PaModel import CreatePaModel
from inc.waveGen import EHTConfig, GenerateEhtWaveform, mcsTable


def CheckMcsTable() -> None:
    """Verify complete EHT MCS 0 through 13 modulation coverage."""

    assert set(mcsTable.keys()) == set(range(14))
    assert mcsTable[0].qamOrder == 2
    assert mcsTable[13].qamOrder == 4096
    assert mcsTable[13].codeRate == 5.0 / 6.0


def CheckEhtBandwidths() -> None:
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
        waveform = GenerateEhtWaveform(
            EHTConfig(
                bandwidthMhz=bandwidthMhz,
                mcs=0,
                numDataSymbols=2,
                oversampling=1,
            )
        )
        assert waveform.fftLength == baseFftLength
        assert waveform.dataSubcarriers.size == dataToneCount
        assert waveform.pilotSubcarriers.size == pilotToneCount
        assert set(waveform.fieldSlices) == {
            "L-STF",
            "L-LTF",
            "L-SIG",
            "RL-SIG",
            "U-SIG",
            "EHT-SIG",
            "EHT-STF",
            "EHT-LTF",
            "EHT-Data",
        }
        # Verify fixed EHT field durations at the configured sample rate.
        assert (
            waveform.fieldSlices["L-STF"].stop
            - waveform.fieldSlices["L-STF"].start
        ) == int(round(8e-6 * waveform.sampleRateHz))
        assert (
            waveform.fieldSlices["EHT-SIG"].stop
            - waveform.fieldSlices["EHT-SIG"].start
        ) == int(round(4e-6 * waveform.sampleRateHz))
        assert (
            waveform.fieldSlices["EHT-STF"].stop
            - waveform.fieldSlices["EHT-STF"].start
        ) == int(round(4e-6 * waveform.sampleRateHz))


def CheckIdealMetrics() -> None:
    """Verify that a perfect signal path has effectively zero EVM."""

    waveform = GenerateEhtWaveform(
        EHTConfig(mcs=13, numDataSymbols=4, oversampling=4)
    )
    metrics = AnalyzeSignal(waveform.samples, waveform.samples, waveform)
    assert metrics.snrDb > 250.0
    assert metrics.evmDb < -250.0
    assert metrics.evmPercent < 1e-10


def CheckGuardIntervals() -> None:
    """Verify compatible 2x/4x EHT-LTF durations for every supported GI."""

    expectedLtfDurationUs = {0.8: 13.6, 1.6: 8.0, 3.2: 16.0}
    for guardIntervalUs, ltfDurationUs in expectedLtfDurationUs.items():
        waveform = GenerateEhtWaveform(
            EHTConfig(
                bandwidthMhz=20,
                mcs=0,
                numDataSymbols=1,
                guardIntervalUs=guardIntervalUs,
                oversampling=1,
            )
        )
        ltfSlice = waveform.fieldSlices["EHT-LTF"]
        ltfSampleCount = ltfSlice.stop - ltfSlice.start
        assert ltfSampleCount == int(
            round(ltfDurationUs * 1e-6 * waveform.sampleRateHz)
        )


def CheckIlcImprovement() -> None:
    """Verify that ILC reduces reconstruction error for both PA families."""

    waveform = GenerateEhtWaveform(
        EHTConfig(
            bandwidthMhz=20,
            mcs=7,
            numDataSymbols=6,
            oversampling=4,
            seed=31,
        )
    )
    referenceSignal = 0.28 * waveform.samples
    for modelName in ("wiener", "gmp"):
        paModel = CreatePaModel(modelName)
        baselineOutput = paModel.Process(referenceSignal)
        baselineMetrics = AnalyzeSignal(
            referenceSignal, baselineOutput, waveform
        )
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
        ilcMetrics = AnalyzeSignal(
            referenceSignal, ilcResult.outputSignal, waveform
        )
        assert ilcMetrics.evmDb < baselineMetrics.evmDb
        assert ilcMetrics.snrDb > baselineMetrics.snrDb


def RunTests() -> None:
    """Run all project checks and report a compact success message."""

    CheckMcsTable()
    CheckEhtBandwidths()
    CheckIdealMetrics()
    CheckGuardIntervals()
    CheckIlcImprovement()
    print("All DPD-ILC project checks passed.")


if __name__ == "__main__":
    RunTests()
