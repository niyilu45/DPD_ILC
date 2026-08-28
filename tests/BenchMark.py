"""Scenario-based performance benchmark for every supported ILC method.

This test module owns waveform construction, impairment scenarios, metric
comparison, convergence reporting, held-out deployment validation, and the
multi-method power-EVM curve. Production ILC algorithms remain in
``inc.lib.DpdIlc`` and do not depend on this benchmark harness.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

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

from inc.lib.Analysis import Analysis, ILCAnalysisResult, SignalMetrics
from inc.lib.Channel import Channel
from inc.lib.ChannelAnalyse import (
    ChannelAnalyse,
    ChannelMeasurementResult,
)
from inc.lib.DpdGmp import (
    AugmentedDpdGmp,
    CouplingAwareDpdGmp,
    CouplingAwareDpdGmpTrainingResult,
    DpdGmp,
    DpdGmpTrainingResult,
)
from inc.lib.DpdLms import DpdLms
from inc.utils.Draw import Draw
from inc.lib.DpdIlc import (
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
from inc.lib.PaModel import (
    GMPConfig,
    IQImbalancePA,
    MimoPaModel,
    PaModel,
    WienerConfig,
)
from inc.lib.TwoToneAnalysis import (
    TwoToneAnalysis,
    TwoToneILCAnalysisResult,
    TwoToneMetrics,
)
from inc.lib.WaveGenTwoTone import TwoToneWaveform, WaveGenTwoTone
from inc.utils.FixedPoint import FixedPoint
from inc.utils.SigProc import PowerCalibration
from inc.lib.WaveGenWifi import WaveGenWifi
from inc.utils.WifiMetadata import WifiWaveform


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
    sampleRateHz: Optional[float] = None
    oversampling: int = 4
    width: int = 16
    guardIntervalUs: float = 0.8
    outputPowerDbm: float = 20.0
    maximumOutputPowerDbm: float = 25.0
    loadResistanceOhm: float = 50.0
    numIterations: int = 10
    paModelName: str = "wiener"
    seed: int = 101
    powerStartDbm: float = 10.0
    powerStopDbm: float = 25.0
    powerPointCount: int = 5
    generatePowerEvmCurve: bool = True
    outputDirectory: Path = Path("results/all_ilc_benchmark")

    def Validate(self) -> None:
        """Validate waveform, iteration, sweep, and output parameters.

        Processing details:
            Algorithm: Reject nonphysical waveform or power settings before
            any long-running scenario starts. PHY-specific format, bandwidth,
            MCS, and guard-interval combinations are validated later by
            ``WaveGenWifi`` using the same configuration.

        Returns:
            result: None. Invalid settings raise ``ValueError`` with the
            parameter responsible for the failure.
        """

        if self.numDataSymbols < 1:
            raise ValueError("numDataSymbols must be positive")
        if (
            not isinstance(self.oversampling, int)
            or isinstance(self.oversampling, bool)
            or self.oversampling < 1
        ):
            raise ValueError("oversampling must be a positive integer")
        if self.sampleRateHz is not None and (
            not isinstance(self.sampleRateHz, (int, float))
            or isinstance(self.sampleRateHz, bool)
            or not np.isfinite(self.sampleRateHz)
            or self.sampleRateHz <= 0.0
        ):
            raise ValueError(
                "sampleRateHz must be finite and positive or None"
            )
        if (
            not isinstance(self.width, int)
            or isinstance(self.width, bool)
            or self.width < 0
            or self.width > 53
        ):
            raise ValueError("width must be an integer from zero through 53")
        bandwidthHz = float(self.bandwidthMhz) * 1.0e6
        effectiveSampleRateHz = (
            bandwidthHz * float(self.oversampling)
            if self.sampleRateHz is None
            else float(self.sampleRateHz)
        )
        if effectiveSampleRateHz < 3.0 * bandwidthHz:
            raise ValueError(
                "sampleRateHz must be at least three times the channel "
                "bandwidth for ACLR analysis"
            )
        powerCalibration = PowerCalibration(
            parameters={
                "loadResistanceOhm": self.loadResistanceOhm,
                "maximumOutputPowerDbm": self.maximumOutputPowerDbm,
                "width": self.width,
            },
        )
        powerCalibration.OutputPowerToDriveScale(
            self.outputPowerDbm
        )
        if self.numIterations < 1:
            raise ValueError("numIterations must be positive")
        if self.paModelName.lower() not in (
            "rapp",
            "wiener",
            "gmp",
            "piecewise_gmp",
            "doherty",
        ):
            raise ValueError(
                "paModelName must be 'rapp', 'wiener', 'gmp', "
                "'piecewise_gmp', or 'doherty'"
            )
        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or self.seed < 0
            or self.seed + 97 >= 2**10
        ):
            raise ValueError(
                "seed must be from zero through 926 so the independent "
                "validation waveform remains inside the 10-bit range"
            )
        powerCalibration.OutputPowerToDriveScale(
            self.powerStartDbm
        )
        powerCalibration.OutputPowerToDriveScale(
            self.powerStopDbm
        )
        if self.powerStopDbm <= self.powerStartDbm:
            raise ValueError("powerStopDbm must exceed powerStartDbm")
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
            merge the ordinary ``SignalMetrics`` dictionary without changing
            stored numerical values.

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
        rowData.update(self.metrics)
        return rowData


@dataclass(frozen=True)
class TwoToneBenchmarkConfig:
    """Configure a deterministic all-SISO-ILC two-tone comparison."""

    sampleRateHz: float = 100.0e6
    toneFrequenciesHz: tuple = (-2.0e6, 2.0e6)
    toneAmplitudes: tuple = (1.0, 1.0)
    tonePhasesDegrees: tuple = (0.0, 0.0)
    numSamples: int = 32768
    rmsLevel: float = 0.5
    width: int = 16
    outputPowerDbm: float = 20.0
    maximumOutputPowerDbm: float = 25.0
    loadResistanceOhm: float = 50.0
    numIterations: int = 10
    paModelName: str = "wiener"
    seed: int = 211
    outputDirectory: Path = Path("results/two_tone_ilc_benchmark")

    def Validate(self) -> None:
        """Validate two-tone generation, PA power, and iteration settings.

        Processing details:
            Algorithm: Delegate waveform-domain constraints to
            ``WaveGenTwoTone``, validate the selected PA model and deterministic
            iteration seed, and use ``PowerCalibration`` to reject output
            targets above the configured normalized PA power ceiling.

        Returns:
            result: None. Invalid settings raise descriptive exceptions.
        """

        WaveGenTwoTone(
            parameters={
                "sampleRateHz": self.sampleRateHz,
                "toneFrequenciesHz": self.toneFrequenciesHz,
                "toneAmplitudes": self.toneAmplitudes,
                "tonePhasesDegrees": self.tonePhasesDegrees,
                "numSamples": self.numSamples,
                "rmsLevel": self.rmsLevel,
                "width": self.width,
            }
        )
        if self.paModelName.lower() not in (
            "rapp",
            "wiener",
            "gmp",
            "piecewise_gmp",
            "doherty",
        ):
            raise ValueError(
                "paModelName must be 'rapp', 'wiener', 'gmp', "
                "'piecewise_gmp', or 'doherty'"
            )
        if (
            not isinstance(self.numIterations, int)
            or isinstance(self.numIterations, bool)
            or self.numIterations < 1
        ):
            raise ValueError("numIterations must be a positive integer")
        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer")
        PowerCalibration(
            parameters={
                "loadResistanceOhm": self.loadResistanceOhm,
                "maximumOutputPowerDbm": self.maximumOutputPowerDbm,
                "outputPowerDbm": self.outputPowerDbm,
                "width": self.width,
            }
        )


@dataclass(frozen=True)
class TwoToneBenchmarkRow:
    """Store one two-tone method result and IM improvement over baseline."""

    methodName: str
    category: str
    metrics: TwoToneMetrics
    im3ImprovementDb: float
    im5ImprovementDb: float
    im7ImprovementDb: float

    def ToDict(self) -> Dict[str, object]:
        """Flatten one two-tone benchmark result for CSV and JSON output.

        Processing details:
            Algorithm: Copy method identity and positive-is-better suppression
            improvements, then merge the ordinary IM metric dictionary without
            recalculating spectral values or changing their dBc convention.

        Returns:
            result: Flat serialization-ready benchmark dictionary.
        """

        rowData: Dict[str, object] = {
            "methodName": self.methodName,
            "category": self.category,
            "im3ImprovementDb": self.im3ImprovementDb,
            "im5ImprovementDb": self.im5ImprovementDb,
            "im7ImprovementDb": self.im7ImprovementDb,
        }
        rowData.update(self.metrics)
        return rowData


@dataclass(frozen=True)
class PaCharacterizationConfig:
    """Configure repeatable two-tone characterization of every PA family."""

    sampleRateHz: float = 200.0e6
    frequencyCentersHz: Tuple[float, ...] = (
        -40.0e6,
        -30.0e6,
        -20.0e6,
        -10.0e6,
        0.0,
        10.0e6,
        20.0e6,
        30.0e6,
        40.0e6,
    )
    frequencyToneSpacingHz: float = 2.0e6
    memoryToneSpacingsHz: Tuple[float, ...] = (
        0.5e6,
        1.0e6,
        2.0e6,
        4.0e6,
        8.0e6,
        12.0e6,
    )
    dynamicToneSpacingHz: float = 4.0e6
    powerSweepDbm: Tuple[float, ...] = (
        10.0,
        15.0,
        20.0,
        23.0,
        25.0,
    )
    numSamples: int = 16384
    settlingSamples: int = 256
    smallSignalRmsLevel: float = 0.05
    nonlinearRmsLevel: float = 0.5
    outputPowerDbm: float = 20.0
    maximumOutputPowerDbm: float = 25.0
    loadResistanceOhm: float = 50.0
    width: int = 0
    paModelNames: Tuple[str, ...] = (
        "rapp",
        "wiener",
        "gmp",
        "doherty",
    )
    runDpdGmpBenchmark: bool = True
    outputDirectory: Path = Path("results/pa_characterization")

    def Validate(self) -> None:
        """Validate all frequency, power, model, record, and width settings.

        Processing details:
            Algorithm: Check scalar domains and ordered nonempty sequences,
            require unique supported PA names, instantiate representative
            frequency-sweep and spacing-sweep two-tone generators to verify
            Nyquist/IM7 limits, and use the shared power converter to reject
            an unreachable nonlinear output-power target.

        Returns:
            result: None. Invalid characterization settings raise an error.
        """

        for parameterName, parameterValue in (
            ("sampleRateHz", self.sampleRateHz),
            ("frequencyToneSpacingHz", self.frequencyToneSpacingHz),
            ("dynamicToneSpacingHz", self.dynamicToneSpacingHz),
            ("smallSignalRmsLevel", self.smallSignalRmsLevel),
            ("nonlinearRmsLevel", self.nonlinearRmsLevel),
            ("loadResistanceOhm", self.loadResistanceOhm),
        ):
            if (
                not isinstance(parameterValue, (int, float))
                or isinstance(parameterValue, bool)
                or not np.isfinite(parameterValue)
                or float(parameterValue) <= 0.0
            ):
                raise ValueError(
                    f"{parameterName} must be finite and positive"
                )
        if (
            not isinstance(self.numSamples, int)
            or isinstance(self.numSamples, bool)
            or self.numSamples < 512
        ):
            raise ValueError(
                "numSamples must be an integer no smaller than 512"
            )
        if not isinstance(self.runDpdGmpBenchmark, bool):
            raise TypeError("runDpdGmpBenchmark must be a boolean")
        if (
            not isinstance(self.settlingSamples, int)
            or isinstance(self.settlingSamples, bool)
            or self.settlingSamples < 0
            or 2 * self.settlingSamples + 64 > self.numSamples
        ):
            raise ValueError(
                "settlingSamples must leave at least 64 analyzed samples"
            )
        if (
            not 0.0 < float(self.smallSignalRmsLevel) <= 1.0
            or not 0.0 < float(self.nonlinearRmsLevel) <= 1.0
        ):
            raise ValueError(
                "smallSignalRmsLevel and nonlinearRmsLevel "
                "must not exceed one"
            )
        if (
            not isinstance(self.frequencyCentersHz, tuple)
            or len(self.frequencyCentersHz) < 2
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isfinite(value)
                for value in self.frequencyCentersHz
            )
            or any(
                laterValue <= earlierValue
                for earlierValue, laterValue in zip(
                    self.frequencyCentersHz[:-1],
                    self.frequencyCentersHz[1:],
                )
            )
        ):
            raise ValueError(
                "frequencyCentersHz must be a strictly increasing "
                "finite tuple with at least two values"
            )
        if (
            not isinstance(self.memoryToneSpacingsHz, tuple)
            or len(self.memoryToneSpacingsHz) < 2
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isfinite(value)
                or float(value) <= 0.0
                for value in self.memoryToneSpacingsHz
            )
            or any(
                laterValue <= earlierValue
                for earlierValue, laterValue in zip(
                    self.memoryToneSpacingsHz[:-1],
                    self.memoryToneSpacingsHz[1:],
                )
            )
        ):
            raise ValueError(
                "memoryToneSpacingsHz must be a strictly increasing "
                "positive tuple with at least two values"
            )
        if not any(
            np.isclose(
                self.dynamicToneSpacingHz,
                spacingHz,
                rtol=0.0,
                atol=np.finfo(float).eps
                * max(1.0, abs(float(spacingHz))),
            )
            for spacingHz in self.memoryToneSpacingsHz
        ):
            raise ValueError(
                "dynamicToneSpacingHz must be one memoryToneSpacingsHz value"
            )
        for parameterName, parameterValue in (
            ("outputPowerDbm", self.outputPowerDbm),
            ("maximumOutputPowerDbm", self.maximumOutputPowerDbm),
        ):
            if (
                not isinstance(parameterValue, (int, float))
                or isinstance(parameterValue, bool)
                or not np.isfinite(parameterValue)
            ):
                raise ValueError(f"{parameterName} must be finite")
        if self.outputPowerDbm > self.maximumOutputPowerDbm:
            raise ValueError(
                "outputPowerDbm cannot exceed maximumOutputPowerDbm"
            )
        if (
            not isinstance(self.powerSweepDbm, tuple)
            or len(self.powerSweepDbm) < 2
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isfinite(value)
                for value in self.powerSweepDbm
            )
            or any(
                laterValue <= earlierValue
                for earlierValue, laterValue in zip(
                    self.powerSweepDbm[:-1],
                    self.powerSweepDbm[1:],
                )
            )
            or any(
                float(value) > self.maximumOutputPowerDbm
                for value in self.powerSweepDbm
            )
        ):
            raise ValueError(
                "powerSweepDbm must be a strictly increasing finite "
                "tuple whose values do not exceed maximumOutputPowerDbm"
            )
        normalizedModelNames = tuple(
            str(modelName).strip().lower()
            for modelName in self.paModelNames
        )
        if (
            not self.paModelNames
            or any(
                modelName
                not in (
                    "rapp",
                    "wiener",
                    "gmp",
                    "piecewise_gmp",
                    "doherty",
                )
                for modelName in normalizedModelNames
            )
            or len(set(normalizedModelNames))
            != len(normalizedModelNames)
        ):
            raise ValueError(
                "paModelNames must contain unique Rapp, Wiener, GMP, "
                "piecewise GMP, or Doherty names"
            )
        for centerFrequencyHz in (
            self.frequencyCentersHz[0],
            self.frequencyCentersHz[-1],
        ):
            halfSpacingHz = 0.5 * self.frequencyToneSpacingHz
            WaveGenTwoTone(
                parameters={
                    "sampleRateHz": self.sampleRateHz,
                    "toneFrequenciesHz": (
                        centerFrequencyHz - halfSpacingHz,
                        centerFrequencyHz + halfSpacingHz,
                    ),
                    "numSamples": self.numSamples,
                    "rmsLevel": self.smallSignalRmsLevel,
                    "width": self.width,
                }
            )
        for spacingHz in (
            self.memoryToneSpacingsHz[0],
            self.memoryToneSpacingsHz[-1],
        ):
            WaveGenTwoTone(
                parameters={
                    "sampleRateHz": self.sampleRateHz,
                    "toneFrequenciesHz": (
                        -0.5 * spacingHz,
                        0.5 * spacingHz,
                    ),
                    "numSamples": self.numSamples,
                    "rmsLevel": self.nonlinearRmsLevel,
                    "width": self.width,
                }
            )
        powerCalibration = PowerCalibration(
            parameters={
                "loadResistanceOhm": self.loadResistanceOhm,
                "maximumOutputPowerDbm": self.maximumOutputPowerDbm,
                "outputPowerDbm": self.outputPowerDbm,
                "width": self.width,
            }
        )
        powerCalibration.OutputPowerToDriveScale(self.outputPowerDbm)
        for powerDbm in self.powerSweepDbm:
            powerCalibration.OutputPowerToDriveScale(powerDbm)


@dataclass(frozen=True)
class PaFrequencyResponsePoint:
    """Store one exact-tone complex small-signal response sample."""

    modelName: str
    frequencyHz: float
    gainDb: float
    phaseDegrees: float

    def ToDict(self) -> Dict[str, object]:
        """Convert one frequency-response point to a stable flat mapping.

        Processing details:
            Algorithm: Copy model identity and all finite physical scalars
            without rewrapping phase or changing units.

        Returns:
            result: CSV/JSON-ready frequency-response dictionary.
        """

        return {
            "modelName": self.modelName,
            "frequencyHz": self.frequencyHz,
            "gainDb": self.gainDb,
            "phaseDegrees": self.phaseDegrees,
        }


@dataclass(frozen=True)
class PaMemoryEffectPoint:
    """Store one equal-power two-tone spacing and its nonlinear products."""

    modelName: str
    toneSpacingHz: float
    outputPowerDbm: float
    im3LowerDbc: float
    im3UpperDbc: float
    im3WorstDbc: float
    im5WorstDbc: float
    im7WorstDbc: float

    def ToDict(self) -> Dict[str, object]:
        """Convert one spacing measurement to a flat result dictionary.

        Processing details:
            Algorithm: Copy the measured lower/upper IM3, worse-side odd-order
            products, actual PA output power, and spacing without recalculation.

        Returns:
            result: CSV/JSON-ready memory-effect dictionary.
        """

        return {
            "modelName": self.modelName,
            "toneSpacingHz": self.toneSpacingHz,
            "outputPowerDbm": self.outputPowerDbm,
            "im3LowerDbc": self.im3LowerDbc,
            "im3UpperDbc": self.im3UpperDbc,
            "im3WorstDbc": self.im3WorstDbc,
            "im5WorstDbc": self.im5WorstDbc,
            "im7WorstDbc": self.im7WorstDbc,
            "im3AsymmetryDb": (
                self.im3UpperDbc - self.im3LowerDbc
            ),
        }


@dataclass(frozen=True)
class PaCharacterizationSummary:
    """Summarize frequency, memory, and nominal nonlinear PA features."""

    modelName: str
    averageGainDb: float
    gainRippleDb: float
    groupDelayNs: float
    phaseNonlinearityDegrees: float
    im3SpacingVariationDb: float
    maximumIm3AsymmetryDb: float
    dynamicGainHysteresisDb: float
    dynamicPhaseHysteresisDegrees: float
    nominalIm3Dbc: float
    nominalIm5Dbc: float
    nominalIm7Dbc: float

    def ToDict(self) -> Dict[str, object]:
        """Return all summary features with their documented units.

        Processing details:
            Algorithm: Copy the immutable model label and scalar features
            exactly as measured or derived by the benchmark.

        Returns:
            result: Flat summary dictionary for tables and grouped plots.
        """

        return {
            "modelName": self.modelName,
            "averageGainDb": self.averageGainDb,
            "gainRippleDb": self.gainRippleDb,
            "groupDelayNs": self.groupDelayNs,
            "phaseNonlinearityDegrees": (
                self.phaseNonlinearityDegrees
            ),
            "im3SpacingVariationDb": self.im3SpacingVariationDb,
            "maximumIm3AsymmetryDb": self.maximumIm3AsymmetryDb,
            "dynamicGainHysteresisDb": (
                self.dynamicGainHysteresisDb
            ),
            "dynamicPhaseHysteresisDegrees": (
                self.dynamicPhaseHysteresisDegrees
            ),
            "nominalIm3Dbc": self.nominalIm3Dbc,
            "nominalIm5Dbc": self.nominalIm5Dbc,
            "nominalIm7Dbc": self.nominalIm7Dbc,
        }


@dataclass(frozen=True)
class PaPowerSweepPoint:
    """Store one PA feature measurement at a controlled output power."""

    modelName: str
    targetOutputPowerDbm: float
    measuredOutputPowerDbm: float
    im3WorstDbc: float
    im5WorstDbc: float
    im7WorstDbc: float
    dynamicGainHysteresisDb: float
    dynamicPhaseHysteresisDegrees: float

    def ToDict(self) -> Dict[str, object]:
        """Flatten one controlled-power nonlinear and memory measurement.

        Processing details:
            Algorithm: Copy target/actual dBm, worse-side odd-order products,
            and dynamic rising/falling loop separation without changing signs
            or units.

        Returns:
            result: CSV/JSON-ready power-sweep point dictionary.
        """

        return {
            "modelName": self.modelName,
            "targetOutputPowerDbm": self.targetOutputPowerDbm,
            "measuredOutputPowerDbm": self.measuredOutputPowerDbm,
            "im3WorstDbc": self.im3WorstDbc,
            "im5WorstDbc": self.im5WorstDbc,
            "im7WorstDbc": self.im7WorstDbc,
            "dynamicGainHysteresisDb": (
                self.dynamicGainHysteresisDb
            ),
            "dynamicPhaseHysteresisDegrees": (
                self.dynamicPhaseHysteresisDegrees
            ),
        }


@dataclass(frozen=True)
class PaDpdRecommendation:
    """Store one measurement-backed DPD design recommendation."""

    modelName: str
    testName: str
    measuredEvidence: str
    dpdArchitecture: str
    dpdConfiguration: str
    trainingStrategy: str
    acceptanceCriteria: str

    def ToDict(self) -> Dict[str, object]:
        """Flatten one recommendation without changing its measured basis.

        Processing details:
            Algorithm: Copy the PA identity, test category, measured evidence,
            proposed DPD structure, initial configuration, training strategy,
            and validation gate into a stable table row.

        Returns:
            result: CSV/JSON-ready DPD recommendation dictionary.
        """

        return {
            "modelName": self.modelName,
            "testName": self.testName,
            "measuredEvidence": self.measuredEvidence,
            "dpdArchitecture": self.dpdArchitecture,
            "dpdConfiguration": self.dpdConfiguration,
            "trainingStrategy": self.trainingStrategy,
            "acceptanceCriteria": self.acceptanceCriteria,
        }


@dataclass(frozen=True)
class PaCharacterizationResult:
    """Store PA measurements, summaries, and DPD design recommendations."""

    frequencyResponse: Tuple[PaFrequencyResponsePoint, ...]
    memoryEffect: Tuple[PaMemoryEffectPoint, ...]
    powerSweep: Tuple[PaPowerSweepPoint, ...]
    summaries: Tuple[PaCharacterizationSummary, ...]
    recommendations: Tuple[PaDpdRecommendation, ...] = ()

    def ToDict(self) -> Dict[str, object]:
        """Convert every immutable result table to JSON-ready dictionaries.

        Processing details:
            Algorithm: Preserve record ordering and delegate unit-stable
            flattening to each point or summary object without recalculating
            any PA metric.

        Returns:
            result: Mapping containing all frequency, memory, power, and
                summary records.
        """

        return {
            "frequencyResponse": [
                point.ToDict() for point in self.frequencyResponse
            ],
            "memoryEffect": [
                point.ToDict() for point in self.memoryEffect
            ],
            "powerSweep": [
                point.ToDict() for point in self.powerSweep
            ],
            "summaries": [
                summary.ToDict() for summary in self.summaries
            ],
            "recommendations": [
                recommendation.ToDict()
                for recommendation in self.recommendations
            ],
        }


@dataclass(frozen=True)
class DpdGmpBenchmarkConfig:
    """Configure the PA-analysis-driven DPD-GMP improvement benchmark."""

    frameFormat: str = "EHT"
    bandwidthMhz: int = 20
    sampleRateHz: float = 80.0e6
    mcs: int = 7
    numDataSymbols: int = 4
    seed: int = 321
    validationSeed: int = 987
    toneFrequenciesHz: Tuple[float, float] = (-2.0e6, 2.0e6)
    toneNumSamples: int = 8192
    stressOutputPowerDbm: float = 15.0
    optimizedOutputPowerDbm: float = 12.0
    trainingPowerDbm: Tuple[float, ...] = (10.0, 12.0, 14.0)
    maximumOutputPowerDbm: float = 25.0
    loadResistanceOhm: float = 50.0
    numIterations: int = 8
    width: int = 0
    outputDirectory: Path = Path("results/dpd_gmp_benchmark")

    def Validate(self) -> None:
        """Validate waveform, power, ILC-label, and output settings.

        Processing details:
            Algorithm: Instantiate representative Wi-Fi and two-tone
            generators with distinct training and validation seeds, require
            ordered power anchors containing the optimized point and below the
            rated ceiling, validate iteration/record counts, and use
            PowerCalibration for dBm/width constraints.

        Returns:
            result: None. Invalid benchmark controls raise an exception.
        """

        if (
            not isinstance(self.bandwidthMhz, int)
            or isinstance(self.bandwidthMhz, bool)
            or self.bandwidthMhz not in (20, 40, 80, 160)
            or not isinstance(self.sampleRateHz, (int, float))
            or isinstance(self.sampleRateHz, bool)
            or not np.isfinite(self.sampleRateHz)
            or self.sampleRateHz
            < 3.0 * self.bandwidthMhz * 1.0e6
        ):
            raise ValueError(
                "bandwidthMhz must be supported and sampleRateHz must be "
                "at least three times its channel bandwidth for DPD-GMP "
                "ACLR analysis"
            )
        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or not isinstance(self.validationSeed, int)
            or isinstance(self.validationSeed, bool)
            or self.validationSeed == self.seed
        ):
            raise ValueError(
                "seed and validationSeed must be distinct integers"
            )
        for waveformSeed in (self.seed, self.validationSeed):
            WaveGenWifi(
                parameters={
                    "frameFormat": self.frameFormat,
                    "bandwidthMhz": self.bandwidthMhz,
                    "sampleRateHz": self.sampleRateHz,
                    "mcs": self.mcs,
                    "numDataSymbols": self.numDataSymbols,
                    "seed": waveformSeed,
                    "width": self.width,
                }
            )
        WaveGenTwoTone(
            parameters={
                "sampleRateHz": self.sampleRateHz,
                "toneFrequenciesHz": self.toneFrequenciesHz,
                "numSamples": self.toneNumSamples,
                "width": self.width,
            }
        )
        if (
            not isinstance(self.trainingPowerDbm, tuple)
            or len(self.trainingPowerDbm) < 3
            or any(
                not isinstance(powerDbm, (int, float))
                or isinstance(powerDbm, bool)
                or not np.isfinite(powerDbm)
                for powerDbm in self.trainingPowerDbm
            )
            or any(
                laterPowerDbm <= earlierPowerDbm
                for earlierPowerDbm, laterPowerDbm in zip(
                    self.trainingPowerDbm[:-1],
                    self.trainingPowerDbm[1:],
                )
            )
            or self.optimizedOutputPowerDbm
            not in self.trainingPowerDbm
        ):
            raise ValueError(
                "trainingPowerDbm must be an increasing tuple with at "
                "least three points including optimizedOutputPowerDbm"
            )
        if self.stressOutputPowerDbm <= self.optimizedOutputPowerDbm:
            raise ValueError(
                "stressOutputPowerDbm must exceed optimizedOutputPowerDbm"
            )
        for outputPowerDbm in (
            *self.trainingPowerDbm,
            self.stressOutputPowerDbm,
        ):
            PowerCalibration(
                parameters={
                    "outputPowerDbm": outputPowerDbm,
                    "maximumOutputPowerDbm": (
                        self.maximumOutputPowerDbm
                    ),
                    "loadResistanceOhm": self.loadResistanceOhm,
                    "width": self.width,
                }
            )
        if (
            not isinstance(self.numIterations, int)
            or isinstance(self.numIterations, bool)
            or self.numIterations < 2
        ):
            raise ValueError(
                "numIterations must be an integer no smaller than two"
            )
        if (
            not isinstance(self.toneNumSamples, int)
            or isinstance(self.toneNumSamples, bool)
            or self.toneNumSamples < 1024
        ):
            raise ValueError(
                "toneNumSamples must be an integer no smaller than 1024"
            )


@dataclass(frozen=True)
class DpdGmpStageResult:
    """Store RF, label-fit, conditioning, and robustness stage metrics."""

    stageName: str
    improvementCategory: str
    targetOutputPowerDbm: float
    measuredOutputPowerDbm: float
    evmDb: float
    evmPercent: float
    aclrWorstDb: float
    im3WorstDbc: float
    im5WorstDbc: float
    im7WorstDbc: float
    labelNmseDb: Optional[float]
    peakWeightedLabelNmseDb: Optional[float]
    regularizedConditionNumber: Optional[float]
    coefficientNorm: Optional[float]
    worstPowerLabelNmseDb: Optional[float]
    worstPowerEvmDb: Optional[float]
    worstPowerAclrDb: Optional[float]
    modelDescription: str

    def ToDict(self) -> Dict[str, object]:
        """Flatten one DPD-GMP stage without changing metric conventions.

        Processing details:
            Algorithm: Copy stage identity, equal-power RF metrics, optional
            coefficient/label diagnostics, robustness metrics, and the exact
            model description into one stable row.

        Returns:
            result: CSV/JSON-ready stage result mapping.
        """

        return {
            "stageName": self.stageName,
            "improvementCategory": self.improvementCategory,
            "targetOutputPowerDbm": self.targetOutputPowerDbm,
            "measuredOutputPowerDbm": self.measuredOutputPowerDbm,
            "evmDb": self.evmDb,
            "evmPercent": self.evmPercent,
            "aclrWorstDb": self.aclrWorstDb,
            "im3WorstDbc": self.im3WorstDbc,
            "im5WorstDbc": self.im5WorstDbc,
            "im7WorstDbc": self.im7WorstDbc,
            "labelNmseDb": self.labelNmseDb,
            "peakWeightedLabelNmseDb": (
                self.peakWeightedLabelNmseDb
            ),
            "regularizedConditionNumber": (
                self.regularizedConditionNumber
            ),
            "coefficientNorm": self.coefficientNorm,
            "worstPowerLabelNmseDb": self.worstPowerLabelNmseDb,
            "worstPowerEvmDb": self.worstPowerEvmDb,
            "worstPowerAclrDb": self.worstPowerAclrDb,
            "modelDescription": self.modelDescription,
        }


@dataclass(frozen=True)
class DpdGmpImprovementComparison:
    """Store one before/after test of a concrete DPD-GMP improvement."""

    improvementName: str
    beforeStage: str
    afterStage: str
    targetMetric: str
    beforeValue: float
    afterValue: float
    improvementValue: float
    expectedDirection: str
    expectationMet: bool
    methodDetails: str

    def ToDict(self) -> Dict[str, object]:
        """Return one auditable improvement comparison dictionary.

        Processing details:
            Algorithm: Copy the compared stages, target metric, raw values,
            consistently positive improvement, expected direction, pass/fail
            result, and implementation details.

        Returns:
            result: CSV/JSON-ready improvement record.
        """

        return {
            "improvementName": self.improvementName,
            "beforeStage": self.beforeStage,
            "afterStage": self.afterStage,
            "targetMetric": self.targetMetric,
            "beforeValue": self.beforeValue,
            "afterValue": self.afterValue,
            "improvementValue": self.improvementValue,
            "expectedDirection": self.expectedDirection,
            "expectationMet": self.expectationMet,
            "methodDetails": self.methodDetails,
        }


@dataclass(frozen=True)
class DpdGmpBenchmarkResult:
    """Store complete DPD-GMP stages and expected improvement checks."""

    stages: Tuple[DpdGmpStageResult, ...]
    comparisons: Tuple[DpdGmpImprovementComparison, ...]

    def ToDict(self) -> Dict[str, object]:
        """Convert all DPD-GMP benchmark records to ordinary mappings.

        Processing details:
            Algorithm: Preserve stage/comparison order and delegate each flat
            record conversion without recalculating metrics.

        Returns:
            result: Mapping containing stage and comparison row lists.
        """

        return {
            "stages": [stage.ToDict() for stage in self.stages],
            "comparisons": [
                comparison.ToDict()
                for comparison in self.comparisons
            ],
        }


def ResolveBenchmarkOutputFullScaleAmplitude(plant: Any) -> float:
    """Resolve a benchmark plant's physical output-code full scale.

    Processing details:
        Algorithm: Inspect an object or bound callback through the common
        ``outputFullScaleAmplitude`` protocol, preserve the historical unit
        full scale for third-party plants that do not expose it, and validate
        the result through ``FixedPoint``.

    Args:
        plant: PA, PA cascade, calibrator, or third-party plant adapter.

    Returns:
        result: Positive physical I/Q component full-scale amplitude.
    """

    protocolOwner = getattr(plant, "__self__", None)
    if protocolOwner is None:
        protocolOwner = plant
    width = int(getattr(protocolOwner, "width", 0))
    return FixedPoint(
        width,
        getattr(protocolOwner, "outputFullScaleAmplitude", 1.0),
    ).fullScaleAmplitude


class DpdGmpPaCascade:
    """Expose DpdGmp plus PA as one output-format-aware calibration plant."""

    def __init__(self, predistorter: DpdGmp, paModel: PaModel) -> None:
        """Store equal-width DPD and PA objects.

        Processing details:
            Algorithm: Require matching public widths and retain both stages
            for deterministic DPD-then-PA processing.

        Args:
            predistorter: Trained SISO DpdGmp object.
            paModel: SISO PaModel under compensation.

        Returns:
            result: None. The cascade is ready for PowerCalibration.
        """

        if predistorter.width != paModel.width:
            raise ValueError(
                "predistorter and paModel must use the same public width"
            )
        self.predistorter = predistorter
        self.paModel = paModel
        self.width = predistorter.width

    @property
    def OutputFullScaleAmplitude(self) -> float:
        """Forward the physical PA-output code full scale.

        Processing details:
            Algorithm: Resolve the live value from the final PA stage so a
            calibrator or analyzer does not mistake an expanded PA output
            range for the normalized DPD input range.

        Returns:
            result: Positive physical I/Q component full-scale amplitude.
        """

        return ResolveBenchmarkOutputFullScaleAmplitude(self.paModel)

    outputFullScaleAmplitude = OutputFullScaleAmplitude

    def Process(self, inputSignal: np.ndarray) -> np.ndarray:
        """Apply DPD followed by the physical PA model.

        Processing details:
            Algorithm: Preserve the public data convention across both object
            boundaries and return the PA output without post-scaling.

        Args:
            inputSignal: Desired public waveform before DPD.

        Returns:
            result: Public PA output produced by DPD then PA.
        """

        return self.paModel.Process(
            self.predistorter.Process(inputSignal)
        )


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
            snrImprovementDb=(
                metrics["snrDb"] - baselineMetrics["snrDb"]
            ),
            # More-negative EVM dB is better, so baseline minus result is a
            # positive improvement.
            evmImprovementDb=(
                baselineMetrics["evmDb"] - metrics["evmDb"]
            ),
            aclrImprovementDb=(
                metrics["aclrWorstDb"]
                - baselineMetrics["aclrWorstDb"]
            ),
        )
    )


def AddTwoToneRow(
    rows: List[TwoToneBenchmarkRow],
    methodName: str,
    category: str,
    metrics: TwoToneMetrics,
    baselineMetrics: TwoToneMetrics,
) -> None:
    """Append one consistently signed two-tone comparison result.

    Processing details:
        Algorithm: Subtract each method's worse-side dBc value from the
        matching baseline value so stronger suppression is a positive
        improvement, then append one immutable row with absolute metrics.

    Args:
        rows: Caller-owned list receiving the new comparison row.
        methodName: Human-readable ILC or baseline label.
        category: Classified benchmark group for reporting.
        metrics: Independently measured two-tone output metrics.
        baselineMetrics: Unlinearized PA metrics at equal output power.

    Returns:
        result: None. The supplied result list is extended by one row.
    """

    rows.append(
        TwoToneBenchmarkRow(
            methodName=methodName,
            category=category,
            metrics=metrics,
            im3ImprovementDb=(
                baselineMetrics["im3WorstDbc"]
                - metrics["im3WorstDbc"]
            ),
            im5ImprovementDb=(
                baselineMetrics["im5WorstDbc"]
                - metrics["im5WorstDbc"]
            ),
            im7ImprovementDb=(
                baselineMetrics["im7WorstDbc"]
                - metrics["im7WorstDbc"]
            ),
        )
    )


def CalculateDynamicHysteresis(
    inputSignal: np.ndarray,
    outputSignal: np.ndarray,
    width: int,
    settlingSamples: int,
    outputFullScaleAmplitude: float = 1.0,
) -> Tuple[float, float]:
    """Measure rising/falling envelope gain and phase loop separation.

    Processing details:
        Algorithm: Decode the source and PA-output waveforms with their
        respective physical full scales, remove settling edges,
        calculate the instantaneous complex gain away from envelope nulls,
        classify samples by positive or negative envelope slope, compare
        median gain and circular-mean phase in common amplitude bins, and
        return the RMS separation of the two trajectories. A memoryless PA
        follows one trajectory, while electrical memory opens a hysteresis
        loop because output also depends on recent envelope history.

    Args:
        inputSignal: Public two-tone waveform actually presented to the PA.
        outputSignal: Matching public PA output waveform.
        width: Shared public I/Q component width.
        settlingSamples: Equal number of edge samples discarded at both ends.
        outputFullScaleAmplitude: Physical component magnitude represented by
            the PA output's nominal full-scale code. The default preserves
            third-party plants that use the historical unit output range.

    Returns:
        result: RMS gain-loop separation in dB and phase-loop separation in
            degrees.
    """

    inputInterfaceFormat = FixedPoint(width)
    outputInterfaceFormat = FixedPoint(
        width, outputFullScaleAmplitude
    )
    decodedInput = inputInterfaceFormat.DecodeComplex(inputSignal).reshape(-1)
    decodedOutput = outputInterfaceFormat.DecodeComplex(
        outputSignal
    ).reshape(-1)
    if (
        decodedInput.size != decodedOutput.size
        or decodedInput.size == 0
        or not np.all(np.isfinite(decodedInput))
        or not np.all(np.isfinite(decodedOutput))
    ):
        raise ValueError(
            "inputSignal and outputSignal must be finite equal-length vectors"
        )
    if settlingSamples > 0:
        decodedInput = decodedInput[
            settlingSamples:-settlingSamples
        ]
        decodedOutput = decodedOutput[
            settlingSamples:-settlingSamples
        ]
    inputMagnitude = np.abs(decodedInput)
    maximumMagnitude = float(np.max(inputMagnitude))
    if maximumMagnitude <= np.finfo(float).tiny:
        raise ValueError("inputSignal must contain a nonzero envelope")
    normalizedMagnitude = inputMagnitude / maximumMagnitude
    envelopeSlope = np.gradient(normalizedMagnitude)
    validSamples = normalizedMagnitude >= 0.10
    complexGain = decodedOutput[validSamples] / decodedInput[validSamples]
    validMagnitude = normalizedMagnitude[validSamples]
    validSlope = envelopeSlope[validSamples]
    gainDifferencesDb = []
    phaseDifferencesDegrees = []
    amplitudeEdges = np.linspace(0.10, 1.00, 10)
    for lowerEdge, upperEdge in zip(
        amplitudeEdges[:-1], amplitudeEdges[1:]
    ):
        binMask = (
            (validMagnitude >= lowerEdge)
            & (validMagnitude < upperEdge)
        )
        risingMask = binMask & (validSlope > 0.0)
        fallingMask = binMask & (validSlope < 0.0)
        if np.count_nonzero(risingMask) < 8 or np.count_nonzero(
            fallingMask
        ) < 8:
            continue
        risingGainDb = 20.0 * np.log10(
            max(
                float(np.median(np.abs(complexGain[risingMask]))),
                np.finfo(float).tiny,
            )
        )
        fallingGainDb = 20.0 * np.log10(
            max(
                float(np.median(np.abs(complexGain[fallingMask]))),
                np.finfo(float).tiny,
            )
        )
        risingPhase = np.angle(
            np.mean(
                np.exp(1j * np.angle(complexGain[risingMask]))
            )
        )
        fallingPhase = np.angle(
            np.mean(
                np.exp(1j * np.angle(complexGain[fallingMask]))
            )
        )
        wrappedPhaseDifference = np.angle(
            np.exp(1j * (risingPhase - fallingPhase))
        )
        gainDifferencesDb.append(risingGainDb - fallingGainDb)
        phaseDifferencesDegrees.append(
            float(np.rad2deg(wrappedPhaseDifference))
        )
    if len(gainDifferencesDb) < 2:
        raise RuntimeError(
            "dynamic hysteresis requires at least two populated "
            "rising/falling envelope bins"
        )
    return (
        float(
            np.sqrt(np.mean(np.asarray(gainDifferencesDb) ** 2))
        ),
        float(
            np.sqrt(
                np.mean(
                    np.asarray(phaseDifferencesDegrees) ** 2
                )
            )
        ),
    )


def MeasurePaFrequencyResponse(
    config: PaCharacterizationConfig,
    modelName: str,
) -> Tuple[PaFrequencyResponsePoint, ...]:
    """Sweep exact low-power tone pairs through one PA model.

    Processing details:
        Algorithm: Generate a low-RMS pair around every configured center,
        process it without output-power normalization so the drive remains
        common, project input and output at both exact tone frequencies,
        divide their complex coefficients to obtain H(f), sort all samples,
        and unwrap phase along frequency.

    Args:
        config: Validated characterization controls.
        modelName: Rapp, Wiener, GMP, or Doherty model family.

    Returns:
        result: Ordered exact-frequency gain and unwrapped-phase points.
    """

    paModel = PaModel(
        parameters={
            "modelName": modelName,
            "width": config.width,
        }
    )
    outputFullScaleAmplitude = ResolveBenchmarkOutputFullScaleAmplitude(
        paModel
    )
    rawPoints = []
    halfSpacingHz = 0.5 * config.frequencyToneSpacingHz
    for centerFrequencyHz in config.frequencyCentersHz:
        waveform = WaveGenTwoTone(
            parameters={
                "sampleRateHz": config.sampleRateHz,
                "toneFrequenciesHz": (
                    centerFrequencyHz - halfSpacingHz,
                    centerFrequencyHz + halfSpacingHz,
                ),
                "numSamples": config.numSamples,
                "rmsLevel": config.smallSignalRmsLevel,
                "width": config.width,
            }
        ).Generate()
        paOutput = paModel.Process(waveform.samples)
        resultAnalysis = TwoToneAnalysis(
            waveform,
            parameters={
                "settlingSamples": config.settlingSamples,
                "width": config.width,
                "outputFullScaleAmplitude": outputFullScaleAmplitude,
            },
        )
        inputInterfaceFormat = FixedPoint(config.width)
        outputInterfaceFormat = FixedPoint(
            config.width, outputFullScaleAmplitude
        )
        decodedInput = inputInterfaceFormat.DecodeComplex(
            waveform.samples
        )
        decodedOutput = outputInterfaceFormat.DecodeComplex(paOutput)
        steadyInput = (
            decodedInput
            if config.settlingSamples == 0
            else decodedInput[
                config.settlingSamples:-config.settlingSamples
            ]
        )
        steadyOutput = (
            decodedOutput
            if config.settlingSamples == 0
            else decodedOutput[
                config.settlingSamples:-config.settlingSamples
            ]
        )
        analysisWindow = resultAnalysis.BuildAnalysisWindow(
            steadyInput.size
        )
        for toneFrequencyHz in waveform.toneFrequenciesHz:
            inputCoefficient = (
                resultAnalysis.CalculateToneCoefficient(
                    steadyInput,
                    toneFrequencyHz,
                    analysisWindow,
                )
            )
            outputCoefficient = (
                resultAnalysis.CalculateToneCoefficient(
                    steadyOutput,
                    toneFrequencyHz,
                    analysisWindow,
                )
            )
            if abs(inputCoefficient) <= np.finfo(float).tiny:
                raise RuntimeError(
                    "frequency-response input tone is below numeric floor"
                )
            complexResponse = outputCoefficient / inputCoefficient
            rawPoints.append(
                (
                    float(toneFrequencyHz),
                    float(
                        20.0
                        * np.log10(
                            max(
                                abs(complexResponse),
                                np.finfo(float).tiny,
                            )
                        )
                    ),
                    float(np.angle(complexResponse)),
                )
            )
    rawPoints.sort(key=lambda point: point[0])
    unwrappedPhases = np.unwrap(
        np.asarray([point[2] for point in rawPoints], dtype=float)
    )
    return tuple(
        PaFrequencyResponsePoint(
            modelName=modelName,
            frequencyHz=frequencyHz,
            gainDb=gainDb,
            phaseDegrees=float(np.rad2deg(unwrappedPhases[pointIndex])),
        )
        for pointIndex, (frequencyHz, gainDb, _) in enumerate(rawPoints)
    )


def MeasurePaMemoryEffect(
    config: PaCharacterizationConfig,
    modelName: str,
) -> Tuple[
    Tuple[PaMemoryEffectPoint, ...],
    float,
    float,
]:
    """Sweep tone spacing at equal PA output power for one PA family.

    Processing details:
        Algorithm: Generate symmetric tone pairs, use the closed input-drive
        loop to hold the actual PA output at the same dBm for every spacing,
        measure exact IM3/IM5/IM7 products, and at the designated spacing
        compare rising/falling dynamic gain and phase trajectories.

    Args:
        config: Validated characterization controls.
        modelName: Rapp, Wiener, GMP, or Doherty model family.

    Returns:
        result: Ordered spacing points plus dynamic gain and phase hysteresis.
    """

    paModel = PaModel(
        parameters={
            "modelName": modelName,
            "width": config.width,
        }
    )
    powerCalibration = PowerCalibration(
        paModel=paModel,
        parameters={
            "loadResistanceOhm": config.loadResistanceOhm,
            "maximumOutputPowerDbm": config.maximumOutputPowerDbm,
            "outputPowerDbm": config.outputPowerDbm,
            "width": config.width,
        },
    )
    memoryPoints = []
    dynamicGainHysteresisDb = None
    dynamicPhaseHysteresisDegrees = None
    for toneSpacingHz in config.memoryToneSpacingsHz:
        waveform = WaveGenTwoTone(
            parameters={
                "sampleRateHz": config.sampleRateHz,
                "toneFrequenciesHz": (
                    -0.5 * toneSpacingHz,
                    0.5 * toneSpacingHz,
                ),
                "numSamples": config.numSamples,
                "rmsLevel": config.nonlinearRmsLevel,
                "width": config.width,
            }
        ).Generate()
        paInput = powerCalibration.Calibrate(waveform.samples)
        paOutput = powerCalibration.GetLastPaOutput()
        resultAnalysis = TwoToneAnalysis(
            waveform,
            parameters={
                "settlingSamples": config.settlingSamples,
                "loadResistanceOhm": config.loadResistanceOhm,
                "maximumOutputPowerDbm": (
                    config.maximumOutputPowerDbm
                ),
                "width": config.width,
                "outputFullScaleAmplitude": (
                    powerCalibration.outputFullScaleAmplitude
                ),
            },
        )
        metrics = resultAnalysis.Analyze(paOutput)
        memoryPoints.append(
            PaMemoryEffectPoint(
                modelName=modelName,
                toneSpacingHz=float(toneSpacingHz),
                outputPowerDbm=metrics["outputPowerDbm"],
                im3LowerDbc=metrics["im3LowerDbc"],
                im3UpperDbc=metrics["im3UpperDbc"],
                im3WorstDbc=metrics["im3WorstDbc"],
                im5WorstDbc=metrics["im5WorstDbc"],
                im7WorstDbc=metrics["im7WorstDbc"],
            )
        )
        if np.isclose(
            toneSpacingHz,
            config.dynamicToneSpacingHz,
            rtol=0.0,
            atol=np.finfo(float).eps
            * max(1.0, abs(float(toneSpacingHz))),
        ):
            (
                dynamicGainHysteresisDb,
                dynamicPhaseHysteresisDegrees,
            ) = CalculateDynamicHysteresis(
                paInput,
                paOutput,
                config.width,
                config.settlingSamples,
                powerCalibration.outputFullScaleAmplitude,
            )
    if (
        dynamicGainHysteresisDb is None
        or dynamicPhaseHysteresisDegrees is None
    ):
        raise RuntimeError(
            "dynamicToneSpacingHz was not evaluated"
        )
    return (
        tuple(memoryPoints),
        float(dynamicGainHysteresisDb),
        float(dynamicPhaseHysteresisDegrees),
    )


def MeasurePaPowerSweep(
    config: PaCharacterizationConfig,
    modelName: str,
) -> Tuple[PaPowerSweepPoint, ...]:
    """Measure nonlinear and dynamic-memory features versus PA output power.

    Processing details:
        Algorithm: Hold tone spacing, waveform statistics, record length, and
        PA parameters fixed; step through configured target dBm values; close
        the input-drive loop independently at every point; then calculate
        actual output power, IM3/IM5/IM7, and rising/falling AM-AM/AM-PM
        hysteresis from the accepted PA input and output.

    Args:
        config: Validated characterization controls and power sweep.
        modelName: Rapp, Wiener, GMP, or Doherty model family.

    Returns:
        result: Ordered controlled-power PA feature points.
    """

    halfSpacingHz = 0.5 * config.dynamicToneSpacingHz
    waveform = WaveGenTwoTone(
        parameters={
            "sampleRateHz": config.sampleRateHz,
            "toneFrequenciesHz": (
                -halfSpacingHz,
                halfSpacingHz,
            ),
            "numSamples": config.numSamples,
            "rmsLevel": config.nonlinearRmsLevel,
            "width": config.width,
        }
    ).Generate()
    paModel = PaModel(
        parameters={
            "modelName": modelName,
            "width": config.width,
        }
    )
    powerCalibration = PowerCalibration(
        paModel=paModel,
        parameters={
            "loadResistanceOhm": config.loadResistanceOhm,
            "maximumOutputPowerDbm": config.maximumOutputPowerDbm,
            "outputPowerDbm": config.powerSweepDbm[0],
            "width": config.width,
        },
    )
    resultAnalysis = TwoToneAnalysis(
        waveform,
        parameters={
            "settlingSamples": config.settlingSamples,
            "loadResistanceOhm": config.loadResistanceOhm,
            "maximumOutputPowerDbm": config.maximumOutputPowerDbm,
            "width": config.width,
            "outputFullScaleAmplitude": (
                powerCalibration.outputFullScaleAmplitude
            ),
        },
    )
    powerPoints = []
    for targetOutputPowerDbm in config.powerSweepDbm:
        powerCalibration.UpdateParameters(
            outputPowerDbm=targetOutputPowerDbm
        )
        paInput = powerCalibration.Calibrate(waveform.samples)
        paOutput = powerCalibration.GetLastPaOutput()
        metrics = resultAnalysis.Analyze(paOutput)
        (
            dynamicGainHysteresisDb,
            dynamicPhaseHysteresisDegrees,
        ) = CalculateDynamicHysteresis(
            paInput,
            paOutput,
            config.width,
            config.settlingSamples,
            powerCalibration.outputFullScaleAmplitude,
        )
        powerPoints.append(
            PaPowerSweepPoint(
                modelName=modelName,
                targetOutputPowerDbm=float(targetOutputPowerDbm),
                measuredOutputPowerDbm=metrics["outputPowerDbm"],
                im3WorstDbc=metrics["im3WorstDbc"],
                im5WorstDbc=metrics["im5WorstDbc"],
                im7WorstDbc=metrics["im7WorstDbc"],
                dynamicGainHysteresisDb=(
                    dynamicGainHysteresisDb
                ),
                dynamicPhaseHysteresisDegrees=(
                    dynamicPhaseHysteresisDegrees
                ),
            )
        )
    return tuple(powerPoints)


def SummarizePaCharacterization(
    modelName: str,
    frequencyPoints: Tuple[PaFrequencyResponsePoint, ...],
    memoryPoints: Tuple[PaMemoryEffectPoint, ...],
    dynamicGainHysteresisDb: float,
    dynamicPhaseHysteresisDegrees: float,
    nominalToneSpacingHz: float,
) -> PaCharacterizationSummary:
    """Reduce detailed sweeps to interpretable PA feature scalars.

    Processing details:
        Algorithm: Calculate gain peak-to-peak ripple, fit unwrapped phase
        versus frequency for average group delay, measure residual phase
        curvature, quantify IM3 variation and side asymmetry over tone
        spacing, and copy IM3/IM5/IM7 at the nominal dynamic spacing.

    Args:
        modelName: PA family represented by all supplied points.
        frequencyPoints: Ordered small-signal complex response samples.
        memoryPoints: Ordered equal-power spacing-sweep results.
        dynamicGainHysteresisDb: Rising/falling RMS gain separation.
        dynamicPhaseHysteresisDegrees: Rising/falling RMS phase separation.
        nominalToneSpacingHz: Spacing used for nominal IM comparison.

    Returns:
        result: Compact immutable feature summary for one PA model.
    """

    if len(frequencyPoints) < 2 or len(memoryPoints) < 2:
        raise ValueError(
            "summary requires at least two frequency and memory points"
        )
    frequencyHz = np.asarray(
        [point.frequencyHz for point in frequencyPoints], dtype=float
    )
    gainDb = np.asarray(
        [point.gainDb for point in frequencyPoints], dtype=float
    )
    phaseRadians = np.deg2rad(
        np.asarray(
            [point.phaseDegrees for point in frequencyPoints],
            dtype=float,
        )
    )
    centeredFrequencyHz = frequencyHz - float(np.mean(frequencyHz))
    phaseDesignMatrix = np.column_stack(
        (
            centeredFrequencyHz,
            np.ones(frequencyHz.size, dtype=float),
        )
    )
    phaseSlope, phaseIntercept = np.linalg.lstsq(
        phaseDesignMatrix,
        phaseRadians,
        rcond=None,
    )[0]
    fittedPhase = (
        phaseSlope * centeredFrequencyHz + phaseIntercept
    )
    phaseResidual = phaseRadians - fittedPhase
    im3WorstDbc = np.asarray(
        [point.im3WorstDbc for point in memoryPoints], dtype=float
    )
    im3AsymmetryDb = np.asarray(
        [
            point.im3UpperDbc - point.im3LowerDbc
            for point in memoryPoints
        ],
        dtype=float,
    )
    nominalPoint = min(
        memoryPoints,
        key=lambda point: abs(
            point.toneSpacingHz - nominalToneSpacingHz
        ),
    )
    return PaCharacterizationSummary(
        modelName=modelName,
        averageGainDb=float(np.mean(gainDb)),
        gainRippleDb=float(np.ptp(gainDb)),
        groupDelayNs=float(
            -phaseSlope / (2.0 * np.pi) * 1.0e9
        ),
        phaseNonlinearityDegrees=float(
            np.rad2deg(np.ptp(phaseResidual))
        ),
        im3SpacingVariationDb=float(np.ptp(im3WorstDbc)),
        maximumIm3AsymmetryDb=float(
            np.max(np.abs(im3AsymmetryDb))
        ),
        dynamicGainHysteresisDb=dynamicGainHysteresisDb,
        dynamicPhaseHysteresisDegrees=(
            dynamicPhaseHysteresisDegrees
        ),
        nominalIm3Dbc=nominalPoint.im3WorstDbc,
        nominalIm5Dbc=nominalPoint.im5WorstDbc,
        nominalIm7Dbc=nominalPoint.im7WorstDbc,
    )


def BuildPaDpdRecommendations(
    result: PaCharacterizationResult,
    config: PaCharacterizationConfig,
) -> Tuple[PaDpdRecommendation, ...]:
    """Translate measured PA features into concrete DPD design guidance.

    Processing details:
        Algorithm: Derive a linear-equalizer tap budget from frequency
        ripple/phase curvature, classify spectral and dynamic memory severity,
        identify the first power point with strong IM3 or hysteresis, detect
        nonmonotonic power behavior, and combine those measurements with each
        PA architecture to recommend an initial DPD structure, configuration,
        training plan, and quantitative validation gate for every test class.

    Args:
        result: Complete measured PA frequency, memory, power, and summary
            result before recommendations are attached.
        config: Exact characterization settings defining nominal power and
            power-sweep coverage.

    Returns:
        result: Five ordered measurement-backed recommendations per PA model.
    """

    summaryByModel = {
        summary.modelName: summary for summary in result.summaries
    }
    recommendations = []
    minimumLinearTapsByModel = {
        "rapp": 1,
        "wiener": 5,
        "gmp": 7,
        "doherty": 5,
    }
    for configuredModelName in config.paModelNames:
        modelName = configuredModelName.strip().lower()
        if modelName not in summaryByModel:
            raise ValueError(
                f"missing PA characterization summary for {modelName}"
            )
        summary = summaryByModel[modelName]
        powerPoints = sorted(
            (
                point
                for point in result.powerSweep
                if point.modelName == modelName
            ),
            key=lambda point: point.measuredOutputPowerDbm,
        )
        if len(powerPoints) < 2:
            raise ValueError(
                f"at least two power points are required for {modelName}"
            )
        frequencyPenalty = (
            4
            if (
                summary.gainRippleDb > 0.50
                or summary.phaseNonlinearityDegrees > 1.00
            )
            else 0
        )
        linearTapCount = (
            minimumLinearTapsByModel[modelName] + frequencyPenalty
        )
        memorySeverity = "strong" if (
            summary.im3SpacingVariationDb > 1.0
            or summary.maximumIm3AsymmetryDb > 1.0
            or summary.dynamicGainHysteresisDb > 0.30
            or summary.dynamicPhaseHysteresisDegrees > 3.0
        ) else "moderate" if (
            summary.im3SpacingVariationDb > 0.25
            or summary.maximumIm3AsymmetryDb > 0.25
            or summary.dynamicGainHysteresisDb > 0.10
            or summary.dynamicPhaseHysteresisDegrees > 1.0
        ) else "weak"
        nonlinearSeverity = (
            "severe"
            if summary.nominalIm3Dbc > -20.0
            else "moderate"
            if summary.nominalIm3Dbc > -35.0
            else "mild"
        )
        detectedKneePoint = next(
            (
                point
                for point in powerPoints
                if (
                    point.im3WorstDbc > -30.0
                    or point.dynamicGainHysteresisDb > 0.25
                    or point.dynamicPhaseHysteresisDegrees > 1.0
                )
            ),
            None,
        )
        kneeDetected = detectedKneePoint is not None
        kneePoint = (
            powerPoints[-1]
            if detectedKneePoint is None
            else detectedKneePoint
        )
        im3PowerTrend = np.diff(
            np.asarray(
                [point.im3WorstDbc for point in powerPoints],
                dtype=float,
            )
        )
        significantIm3PowerTrend = im3PowerTrend[
            np.abs(im3PowerTrend) >= 0.25
        ]
        nonmonotonicPowerTrend = bool(
            significantIm3PowerTrend.size >= 2
            and np.any(
                significantIm3PowerTrend[:-1]
                * significantIm3PowerTrend[1:]
                < 0.0
            )
        )
        frequencyArchitecture = {
            "rapp": (
                "Memoryless LUT or odd-order polynomial with no delayed "
                "basis terms; add an FIR only for an external measured path."
            ),
            "wiener": (
                "Short complex FIR pre-equalizer followed by a "
                "memory-polynomial nonlinear stage."
            ),
            "gmp": (
                "Complex FIR or frequency-domain pre-equalizer followed by "
                "GMP so linear and nonlinear memory remain jointly modeled."
            ),
            "doherty": (
                "Power-conditioned complex FIR followed by a branch-aware "
                "or piecewise nonlinear DPD."
            ),
        }[modelName]
        frequencyTraining = {
            "rapp": (
                "Verify the low-drive response is flat and zero-delay, then "
                "skip linear equalizer identification and train only the "
                "static AM-AM inverse."
            ),
            "wiener": (
                "Estimate the inverse linear response at low drive, freeze "
                "the FIR, then identify nonlinear coefficients at nominal "
                "power."
            ),
            "gmp": (
                "Jointly fit FIR and GMP on multitone or wideband data with "
                "ridge regularization; retain edge-frequency validation."
            ),
            "doherty": (
                "Initialize from the carrier-only low-power response, then "
                "repeat identification above peaking turn-on instead of "
                "reusing one fixed inverse."
            ),
        }[modelName]
        recommendations.append(
            PaDpdRecommendation(
                modelName=modelName,
                testName="frequency_response",
                measuredEvidence=(
                    f"Gain ripple={summary.gainRippleDb:.3f} dB, "
                    f"group delay={summary.groupDelayNs:.3f} ns, "
                    "residual phase="
                    f"{summary.phaseNonlinearityDegrees:.3f} deg."
                ),
                dpdArchitecture=frequencyArchitecture,
                dpdConfiguration=(
                    f"Start with {linearTapCount} complex FIR taps; "
                    "normalize the center tap and regularize the remaining "
                    "tap energy."
                ),
                trainingStrategy=frequencyTraining,
                acceptanceCriteria=(
                    "On an independent frequency sweep, require residual "
                    "gain ripple <=0.10 dB and residual phase curvature "
                    "<=0.50 deg before nonlinear training is accepted."
                ),
            )
        )
        memoryArchitecture = {
            "rapp": (
                "Strictly memoryless LUT or polynomial inverse; delayed GMP "
                "terms are unnecessary unless another system block adds "
                "measured memory."
            ),
            "wiener": (
                "Memory polynomial after the linear FIR; add GMP cross terms "
                "only if validation exposes spacing-dependent residual IM."
            ),
            "gmp": (
                "Full lagging-and-leading GMP or frequency-selective ILC "
                "with explicit envelope cross-memory branches."
            ),
            "doherty": (
                "Shallow branch-aware memory polynomial with separate "
                "carrier-only and carrier-plus-peaking regions."
            ),
        }[modelName]
        memoryConfiguration = {
            "rapp": (
                "Use main memory depth 1 and cross-memory depth 0; retain "
                "only same-sample amplitude basis terms."
            ),
            "wiener": (
                "Use odd orders (1,3,5,7), main memory depth 3, and zero "
                "cross-memory depth initially."
            ),
            "gmp": (
                "Use odd orders (1,3,5,7,9), main memory depth 5-7, and "
                "lagging/leading envelope cross-memory depth 3-5."
            ),
            "doherty": (
                "Use odd orders (1,3,5,7), depth 3 per region, and a smooth "
                "gate around the peaking turn-on amplitude."
            ),
        }[modelName]
        recommendations.append(
            PaDpdRecommendation(
                modelName=modelName,
                testName="memory_effect",
                measuredEvidence=(
                    f"Memory severity={memorySeverity}; IM3 spacing "
                    f"variation={summary.im3SpacingVariationDb:.3f} dB "
                    "and maximum sideband asymmetry="
                    f"{summary.maximumIm3AsymmetryDb:.3f} dB."
                ),
                dpdArchitecture=memoryArchitecture,
                dpdConfiguration=memoryConfiguration,
                trainingStrategy=(
                    "Train with at least the minimum, nominal, and maximum "
                    "tested tone spacings using equal output-power weighting; "
                    "select memory depth on a held-out spacing."
                ),
                acceptanceCriteria=(
                    "Require residual IM3 spacing variation <0.50 dB and "
                    "upper/lower IM3 asymmetry <0.50 dB without worsening "
                    "IM5 or IM7 by more than 1 dB."
                ),
            )
        )
        dynamicArchitecture = {
            "rapp": (
                "Use one static inverse curve because equal-amplitude rising "
                "and falling samples have the same output by construction."
            ),
            "wiener": (
                "Keep the static nonlinear inverse and short memory "
                "polynomial unless the dynamic loop grows at deployment "
                "power."
            ),
            "gmp": (
                "Use bidirectional envelope GMP terms or a compact recurrent "
                "envelope-state branch in addition to the main polynomial."
            ),
            "doherty": (
                "Use a smooth mixture-of-experts DPD whose state includes "
                "envelope direction near peaking activation."
            ),
        }[modelName]
        dynamicConfiguration = {
            "rapp": (
                "Set memory depth 1, cross-memory depth 0, and disable any "
                "recurrent or envelope-state branch."
            ),
            "wiener": (
                "Start with depth 2-3 and strong ridge regularization on all "
                "delayed nonlinear terms."
            ),
            "gmp": (
                "Retain at least 3 lagging and 3 leading envelope delays; "
                "weight AM-PM residuals explicitly in coefficient fitting."
            ),
            "doherty": (
                "Use two smoothly blended coefficient sets and constrain "
                "their boundary value and first derivative to be continuous."
            ),
        }[modelName]
        recommendations.append(
            PaDpdRecommendation(
                modelName=modelName,
                testName="dynamic_hysteresis",
                measuredEvidence=(
                    "Dynamic AM-AM RMS="
                    f"{summary.dynamicGainHysteresisDb:.3f} dB and "
                    "dynamic AM-PM RMS="
                    f"{summary.dynamicPhaseHysteresisDegrees:.3f} deg."
                ),
                dpdArchitecture=dynamicArchitecture,
                dpdConfiguration=dynamicConfiguration,
                trainingStrategy=(
                    "Balance rising and falling envelope samples in every "
                    "amplitude bin; do not let dense low-amplitude samples "
                    "dominate the regression."
                ),
                acceptanceCriteria=(
                    "Require dynamic AM-AM RMS <0.10 dB and dynamic AM-PM "
                    "RMS <1.0 deg on a held-out two-tone phase and spacing."
                ),
            )
        )
        nonlinearArchitecture = {
            "rapp": (
                "Monotonic memoryless LUT or low-order odd polynomial that "
                "approximates the inverse Rapp AM-AM curve below saturation."
            ),
            "wiener": (
                "Odd-order memory polynomial or GMP with the measured linear "
                "FIR handled separately."
            ),
            "gmp": (
                "Regularized high-order GMP trained with peak-aware weighting "
                "and explicit input-amplitude projection."
            ),
            "doherty": (
                "Piecewise carrier/peaking DPD, such as a smooth LUT plus "
                "GMP residual corrector, rather than one global polynomial."
            ),
        }[modelName]
        nonlinearConfiguration = {
            "rapp": (
                "Start with an amplitude LUT or orders (1,3,5,7), memory "
                "depth 1, cross-depth 0, and no conjugate branch."
            ),
            "wiener": (
                "Start with orders (1,3,5,7), depth 3, and remove order 7 "
                "only if held-out IM7 remains below the target."
            ),
            "gmp": (
                "Start with orders (1,3,5,7,9), depth 5, cross-depth 3, "
                "coefficient normalization, and ridge-factor search."
            ),
            "doherty": (
                "Use orders (1,3,5,7) per operating region, a continuous "
                "transition gate, and separate carrier/peaking gain anchors."
            ),
        }[modelName]
        nonlinearTrainingStrategy = {
            "rapp": (
                "Fit uniformly populated amplitude bins across the usable "
                "range and stop below the near-saturation region where the "
                "inverse slope becomes ill conditioned."
            ),
            "wiener": (
                "Identify at nominal power, then add maximum-power samples "
                "with peak-aware weighting; optimize waveform error and "
                "IM3/IM5/IM7 at equal measured output power."
            ),
            "gmp": (
                f"Begin about 5 dB below the {config.outputPowerDbm:.1f} dBm "
                "severe-distortion point, fit a stable regularized inverse, "
                "and increase power in small steps only while held-out error "
                "continues to improve."
            ),
            "doherty": (
                "Collect balanced samples below, inside, and above peaking "
                "turn-on; fit regional inverses jointly and penalize boundary "
                "discontinuity while monitoring every odd-order product."
            ),
        }[modelName]
        recommendations.append(
            PaDpdRecommendation(
                modelName=modelName,
                testName="nominal_nonlinearity",
                measuredEvidence=(
                    f"Nonlinearity severity={nonlinearSeverity} at "
                    f"{config.outputPowerDbm:.1f} dBm: IM3="
                    f"{summary.nominalIm3Dbc:.2f}, IM5="
                    f"{summary.nominalIm5Dbc:.2f}, IM7="
                    f"{summary.nominalIm7Dbc:.2f} dBc."
                ),
                dpdArchitecture=nonlinearArchitecture,
                dpdConfiguration=nonlinearConfiguration,
                trainingStrategy=nonlinearTrainingStrategy,
                acceptanceCriteria=(
                    "Require at least 10 dB IM3 improvement at equal output "
                    "power, no IM5/IM7 regression larger than 1 dB, and no "
                    "increase in peak input beyond the configured limit."
                ),
            )
        )
        powerArchitecture = {
            "rapp": (
                "One memoryless inverse spanning all invertible power points; "
                "use a power-indexed LUT only if deployment changes the Rapp "
                "parameters themselves."
            ),
            "wiener": (
                "Power-conditioned memory polynomial with coefficient "
                "interpolation between linear and compression regions."
            ),
            "gmp": (
                "Multi-operating-point GMP or coefficient bank indexed by "
                "measured output power and envelope RMS."
            ),
            "doherty": (
                "Carrier-only/carrier-plus-peaking mixture-of-experts DPD "
                "with a physically aligned smooth transition gate."
            ),
        }[modelName]
        configuredPowerText = ", ".join(
            f"{float(powerDbm):g}"
            for powerDbm in config.powerSweepDbm
        )
        powerConfiguration = {
            "rapp": (
                f"Use {configuredPowerText} dBm as validation anchors for one "
                "shared static curve, and enforce a drive ceiling before the "
                "measured saturation knee."
            ),
            "wiener": (
                f"Use coefficient anchors at {configuredPowerText} dBm; "
                "apply stronger peak projection above the detected knee "
                "when a knee is present."
            ),
            "gmp": (
                f"Use all measured anchors ({configuredPowerText} dBm), "
                "normalize each power block, and regularize coefficient "
                "changes between adjacent blocks."
            ),
            "doherty": (
                f"Retain the measured anchors ({configuredPowerText} dBm), "
                "add dense anchors at the detected knee +/-1 dB when present, "
                "and constrain regional outputs and slopes to join "
                "continuously."
            ),
        }[modelName]
        kneeEvidence = (
            "First strong-distortion point="
            f"{kneePoint.measuredOutputPowerDbm:.2f} dBm "
            f"(IM3={kneePoint.im3WorstDbc:.2f} dBc)"
            if kneeDetected
            else (
                "No strong-distortion threshold crossed; highest tested "
                f"point={kneePoint.measuredOutputPowerDbm:.2f} dBm "
                f"(IM3={kneePoint.im3WorstDbc:.2f} dBc)"
            )
        )
        recommendations.append(
            PaDpdRecommendation(
                modelName=modelName,
                testName="output_power",
                measuredEvidence=(
                    f"{kneeEvidence}; "
                    "IM3 power trend="
                    f"{'nonmonotonic' if nonmonotonicPowerTrend else 'monotonic'}."
                ),
                dpdArchitecture=powerArchitecture,
                dpdConfiguration=powerConfiguration,
                trainingStrategy=(
                    "Train and validate every configured output-power point, "
                    "up-weight the detected knee and maximum-power point, and "
                    "interpolate coefficients only inside the measured range."
                ),
                acceptanceCriteria=(
                    "At every power point require improved EVM and worst-side "
                    "ACLR on Wi-Fi plus non-regressing IM3/IM5/IM7 on two-tone; "
                    "reject improvements caused by PA-output rescaling."
                ),
            )
        )
    return tuple(recommendations)


def SavePaCharacterizationResults(
    result: PaCharacterizationResult,
    config: PaCharacterizationConfig,
) -> Tuple[Path, Path, Path, Path, Path, Path]:
    """Save measurement tables, DPD recommendations, and combined JSON.

    Processing details:
        Algorithm: Create the configured directory, flatten immutable records
        once, write five stable UTF-8 CSV tables, and serialize the same data
        with every reproducibility parameter in one structured JSON file.

    Args:
        result: Complete calculated PA characterization result.
        config: Exact benchmark controls used to create the result.

    Returns:
            result: Paths to frequency, memory, power, summary,
                recommendation CSV, and combined JSON.
    """

    outputDirectory = Path(config.outputDirectory)
    outputDirectory.mkdir(parents=True, exist_ok=True)
    frequencyPath = outputDirectory / "pa_frequency_response.csv"
    memoryPath = outputDirectory / "pa_memory_effect.csv"
    powerPath = outputDirectory / "pa_power_sweep.csv"
    summaryPath = outputDirectory / "pa_characterization_summary.csv"
    recommendationPath = (
        outputDirectory / "pa_dpd_recommendations.csv"
    )
    jsonPath = outputDirectory / "pa_characterization.json"
    resultData = result.ToDict()
    frequencyRows = resultData["frequencyResponse"]
    memoryRows = resultData["memoryEffect"]
    powerRows = resultData["powerSweep"]
    summaryRows = resultData["summaries"]
    recommendationRows = resultData["recommendations"]
    for outputPath, rows in (
        (frequencyPath, frequencyRows),
        (memoryPath, memoryRows),
        (powerPath, powerRows),
        (summaryPath, summaryRows),
        (recommendationPath, recommendationRows),
    ):
        if not rows:
            raise ValueError(
                "PA characterization result tables cannot be empty"
            )
        with outputPath.open(
            "w", newline="", encoding="utf-8-sig"
        ) as csvFile:
            csvWriter = csv.DictWriter(
                csvFile, fieldnames=tuple(rows[0])
            )
            csvWriter.writeheader()
            csvWriter.writerows(rows)
    document = {
        "metadata": {
            "sampleRateHz": config.sampleRateHz,
            "frequencyCentersHz": list(
                config.frequencyCentersHz
            ),
            "frequencyToneSpacingHz": (
                config.frequencyToneSpacingHz
            ),
            "memoryToneSpacingsHz": list(
                config.memoryToneSpacingsHz
            ),
            "dynamicToneSpacingHz": config.dynamicToneSpacingHz,
            "powerSweepDbm": list(config.powerSweepDbm),
            "numSamples": config.numSamples,
            "settlingSamples": config.settlingSamples,
            "smallSignalRmsLevel": config.smallSignalRmsLevel,
            "nonlinearRmsLevel": config.nonlinearRmsLevel,
            "outputPowerDbm": config.outputPowerDbm,
            "maximumOutputPowerDbm": (
                config.maximumOutputPowerDbm
            ),
            "loadResistanceOhm": config.loadResistanceOhm,
            "width": config.width,
            "paModelNames": list(config.paModelNames),
            "runDpdGmpBenchmark": config.runDpdGmpBenchmark,
            "dpdGmpResultDirectory": (
                "dpd_gmp"
                if config.runDpdGmpBenchmark
                else None
            ),
        },
        "frequencyResponse": frequencyRows,
        "memoryEffect": memoryRows,
        "powerSweep": powerRows,
        "summaries": summaryRows,
        "recommendations": recommendationRows,
    }
    with jsonPath.open("w", encoding="utf-8") as jsonFile:
        json.dump(document, jsonFile, indent=2, ensure_ascii=False)
    return (
        frequencyPath,
        memoryPath,
        powerPath,
        summaryPath,
        recommendationPath,
        jsonPath,
    )


def PrintPaCharacterizationResults(
    summaries: Tuple[PaCharacterizationSummary, ...],
) -> None:
    """Print one compact comparison table for all characterized PA models.

    Processing details:
        Algorithm: Preserve requested model order and print gain ripple,
        group delay, IM3 spacing variation/asymmetry, dynamic gain/phase
        hysteresis, and nominal IM3 using fixed-width columns.

    Args:
        summaries: Nonempty ordered model feature summaries.

    Returns:
        result: None. The comparison table is written to standard output.
    """

    header = (
        f"{'PA':<10} {'GainRip':>8} {'Delay(ns)':>10} "
        f"{'IM3var':>8} {'IM3asym':>9} {'DynG':>8} "
        f"{'DynP':>8} {'IM3':>9}"
    )
    print(header)
    print("-" * len(header))
    for summary in summaries:
        print(
            f"{summary.modelName:<10} "
            f"{summary.gainRippleDb:>8.3f} "
            f"{summary.groupDelayNs:>10.3f} "
            f"{summary.im3SpacingVariationDb:>8.3f} "
            f"{summary.maximumIm3AsymmetryDb:>9.3f} "
            f"{summary.dynamicGainHysteresisDb:>8.3f} "
            f"{summary.dynamicPhaseHysteresisDegrees:>8.3f} "
            f"{summary.nominalIm3Dbc:>9.2f}"
        )


def PrintPaDpdRecommendations(
    recommendations: Tuple[PaDpdRecommendation, ...],
) -> None:
    """Print a compact index of all measurement-backed DPD recommendations.

    Processing details:
        Algorithm: Preserve PA and test ordering, print the measured evidence
        followed by the selected DPD architecture, and leave full parameter,
        training, and acceptance details in CSV/JSON.

    Args:
        recommendations: Ordered recommendations created from measured data.

    Returns:
        result: None. A compact design-guidance table is printed.
    """

    if not recommendations:
        raise ValueError("recommendations cannot be empty")
    print("\nDPD design recommendations")
    for recommendation in recommendations:
        print(
            f"- {recommendation.modelName}/"
            f"{recommendation.testName}: "
            f"{recommendation.measuredEvidence} "
            f"{recommendation.dpdArchitecture}"
        )


def RunPaCharacterizationBenchmark(
    config: Optional[PaCharacterizationConfig] = None,
) -> PaCharacterizationResult:
    """Characterize Wiener, GMP, and Doherty PA frequency and memory effects.

    Processing details:
        Algorithm: Run a common low-drive two-tone frequency sweep and an
        equal-output-power tone-spacing sweep for every requested PA family,
        derive linear response, group delay, spectral memory, dynamic
        hysteresis, nominal IM3/IM5/IM7, and output-power-dependent features,
        save raw/summary data, delegate all comparison figures to ``Draw``,
        and optionally run the independent staged DPD-GMP benchmark in a
        child result directory after PA recommendations are available.

    Args:
        config: Optional complete characterization setup. None uses defaults.

    Returns:
        result: Detailed immutable frequency, memory, and power points,
            per-model summaries, and per-test DPD recommendations.
    """

    if config is None:
        config = PaCharacterizationConfig()
    config.Validate()
    allFrequencyPoints = []
    allMemoryPoints = []
    allPowerPoints = []
    summaries = []
    for configuredModelName in config.paModelNames:
        modelName = configuredModelName.strip().lower()
        frequencyPoints = MeasurePaFrequencyResponse(
            config, modelName
        )
        (
            memoryPoints,
            dynamicGainHysteresisDb,
            dynamicPhaseHysteresisDegrees,
        ) = MeasurePaMemoryEffect(config, modelName)
        summary = SummarizePaCharacterization(
            modelName,
            frequencyPoints,
            memoryPoints,
            dynamicGainHysteresisDb,
            dynamicPhaseHysteresisDegrees,
            config.dynamicToneSpacingHz,
        )
        powerPoints = MeasurePaPowerSweep(config, modelName)
        allFrequencyPoints.extend(frequencyPoints)
        allMemoryPoints.extend(memoryPoints)
        allPowerPoints.extend(powerPoints)
        summaries.append(summary)
    measurementResult = PaCharacterizationResult(
        frequencyResponse=tuple(allFrequencyPoints),
        memoryEffect=tuple(allMemoryPoints),
        powerSweep=tuple(allPowerPoints),
        summaries=tuple(summaries),
    )
    recommendations = BuildPaDpdRecommendations(
        measurementResult,
        config,
    )
    result = PaCharacterizationResult(
        frequencyResponse=measurementResult.frequencyResponse,
        memoryEffect=measurementResult.memoryEffect,
        powerSweep=measurementResult.powerSweep,
        summaries=measurementResult.summaries,
        recommendations=recommendations,
    )
    SavePaCharacterizationResults(result, config)
    frequencyByModel = {
        modelName: [
            point.ToDict()
            for point in result.frequencyResponse
            if point.modelName == modelName
        ]
        for modelName in (
            summary.modelName for summary in result.summaries
        )
    }
    memoryByModel = {
        modelName: [
            point.ToDict()
            for point in result.memoryEffect
            if point.modelName == modelName
        ]
        for modelName in (
            summary.modelName for summary in result.summaries
        )
    }
    summaryByModel = {
        summary.modelName: summary.ToDict()
        for summary in result.summaries
    }
    powerByModel = {
        modelName: [
            point.ToDict()
            for point in result.powerSweep
            if point.modelName == modelName
        ]
        for modelName in (
            summary.modelName for summary in result.summaries
        )
    }
    resultDraw = Draw()
    resultDraw.SavePaFrequencyResponse(
        frequencyByModel,
        config.outputDirectory,
    )
    resultDraw.SavePaMemoryEffect(
        memoryByModel,
        summaryByModel,
        config.outputDirectory,
    )
    resultDraw.SavePaNonlinearityComparison(
        summaryByModel,
        config.outputDirectory,
    )
    resultDraw.SavePaPowerCharacteristics(
        powerByModel,
        config.outputDirectory,
    )
    PrintPaCharacterizationResults(result.summaries)
    PrintPaDpdRecommendations(result.recommendations)
    if config.runDpdGmpBenchmark:
        # PA characterization identifies the nonlinear and memory mechanisms;
        # the nested benchmark then verifies concrete GMP inverse-design
        # responses to those mechanisms with independent RF metrics.
        gmpPowerPoints = sorted(
            (
                point
                for point in result.powerSweep
                if point.modelName == "gmp"
            ),
            key=lambda point: point.targetOutputPowerDbm,
        )
        strongDistortionPoints = tuple(
            point
            for point in gmpPowerPoints
            if point.im3WorstDbc > -30.0
        )
        measuredStressPowerDbm = (
            strongDistortionPoints[0].targetOutputPowerDbm
            if strongDistortionPoints
            else min(15.0, config.maximumOutputPowerDbm)
        )
        # The documented benchmark caps the stress point at 15 dBm because
        # the default GMP is already poorly invertible there. A measured
        # earlier knee is retained instead of being hidden by that cap.
        stressOutputPowerDbm = min(
            measuredStressPowerDbm,
            15.0,
            config.maximumOutputPowerDbm,
        )
        optimizedOutputPowerDbm = stressOutputPowerDbm - 3.0
        RunDpdGmpBenchmark(
            DpdGmpBenchmarkConfig(
                stressOutputPowerDbm=stressOutputPowerDbm,
                optimizedOutputPowerDbm=optimizedOutputPowerDbm,
                trainingPowerDbm=(
                    optimizedOutputPowerDbm - 2.0,
                    optimizedOutputPowerDbm,
                    optimizedOutputPowerDbm + 2.0,
                ),
                maximumOutputPowerDbm=config.maximumOutputPowerDbm,
                loadResistanceOhm=config.loadResistanceOhm,
                width=config.width,
                outputDirectory=(
                    Path(config.outputDirectory) / "dpd_gmp"
                ),
            )
        )
    return result


def GenerateDpdGmpIlcLabel(
    config: DpdGmpBenchmarkConfig,
    waveform: WifiWaveform,
    paModel: PaModel,
    outputPowerDbm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate one equal-power Wi-Fi reference and converged ILC input label.

    Processing details:
        Algorithm: Close the unlinearized PA drive loop at the requested
        output power, use the calibrated desired waveform as the ILC target,
        run frequency-domain ILC with synchronization and common-gain
        alignment enabled by the production implementation, and return the
        desired waveform together with its learned PA-input label.

    Args:
        config: Validated DPD-GMP benchmark controls.
        waveform: Deterministic Wi-Fi frame used for training.
        paModel: GMP PA whose inverse is learned.
        outputPowerDbm: Conducted PA output-power target in dBm.

    Returns:
        result: Calibrated desired waveform and matching ILC-learned input.
    """

    powerCalibration = PowerCalibration(
        paModel=paModel,
        parameters={
            "loadResistanceOhm": config.loadResistanceOhm,
            "maximumOutputPowerDbm": config.maximumOutputPowerDbm,
            "outputPowerDbm": outputPowerDbm,
            "width": config.width,
        },
    )
    referenceSignal = powerCalibration.Calibrate(waveform.samples)
    floatingReference = FixedPoint(config.width).DecodeComplex(
        referenceSignal
    )
    maximumInputMagnitude = max(
        1.5,
        2.0 * float(np.max(np.abs(floatingReference))),
    )
    ilcResult = RunFrequencyDomainIlc(
        referenceSignal,
        paModel,
        waveform.sampleRateHz,
        waveform.bandwidthHz,
        ILCConfig(
            numIterations=config.numIterations,
            learningRate=0.10,
            regularization=1.0e-3,
            maxAmplitude=maximumInputMagnitude,
        ),
    )
    return referenceSignal, ilcResult.learnedInput


def EvaluateDpdGmpWifiStage(
    config: DpdGmpBenchmarkConfig,
    waveform: WifiWaveform,
    paModel: PaModel,
    outputPowerDbm: float,
    predistorter: Optional[DpdGmp] = None,
) -> SignalMetrics:
    """Measure one baseline or DPD-GMP Wi-Fi stage at equal PA output power.

    Processing details:
        Algorithm: Bind PowerCalibration either to the PA alone or to the
        DPD-then-PA cascade, adjust only the plant input until native PA
        output reaches the requested dBm, then analyze that output against
        the calibrated desired Wi-Fi waveform. No post-PA rescaling occurs.

    Args:
        config: Validated benchmark power and interface controls.
        waveform: Wi-Fi metadata and original samples.
        paModel: GMP PA used by every comparison.
        outputPowerDbm: Equal conducted output-power target.
        predistorter: Optional trained DpdGmp; None measures the PA baseline.

    Returns:
        result: Independent Analysis SNR, EVM, ACLR, and output-power mapping.
    """

    calibratedPlant: Any = (
        paModel
        if predistorter is None
        else DpdGmpPaCascade(predistorter, paModel)
    )
    powerCalibration = PowerCalibration(
        paModel=calibratedPlant,
        parameters={
            "loadResistanceOhm": config.loadResistanceOhm,
            "maximumOutputPowerDbm": config.maximumOutputPowerDbm,
            "outputPowerDbm": outputPowerDbm,
            "width": config.width,
        },
    )
    desiredInput = powerCalibration.Calibrate(waveform.samples)
    paOutput = powerCalibration.GetLastPaOutput()
    resultAnalysis = Analysis(
        desiredInput,
        waveform,
        parameters={
            "maximumOutputPowerDbm": config.maximumOutputPowerDbm,
            "loadResistanceOhm": config.loadResistanceOhm,
            "width": config.width,
            "outputFullScaleAmplitude": (
                powerCalibration.outputFullScaleAmplitude
            ),
        },
    )
    return resultAnalysis.Analyze(paOutput)


def EvaluateDpdGmpTwoToneStage(
    config: DpdGmpBenchmarkConfig,
    waveform: TwoToneWaveform,
    paModel: PaModel,
    outputPowerDbm: float,
    predistorter: Optional[DpdGmp] = None,
) -> TwoToneMetrics:
    """Measure IM3, IM5, and IM7 for one baseline or DPD-GMP stage.

    Processing details:
        Algorithm: Reuse the same equal-output-power calibration convention
        as the Wi-Fi test, then project the native PA output onto exact
        fundamental and odd-order intermodulation frequencies.

    Args:
        config: Validated benchmark controls.
        waveform: TwoToneWaveform descriptor and periodic samples.
        paModel: Common GMP PA under test.
        outputPowerDbm: Conducted output-power target in dBm.
        predistorter: Optional trained DpdGmp; None selects the PA baseline.

    Returns:
        result: TwoToneAnalysis output-power and IM product dictionary.
    """

    calibratedPlant: Any = (
        paModel
        if predistorter is None
        else DpdGmpPaCascade(predistorter, paModel)
    )
    powerCalibration = PowerCalibration(
        paModel=calibratedPlant,
        parameters={
            "loadResistanceOhm": config.loadResistanceOhm,
            "maximumOutputPowerDbm": config.maximumOutputPowerDbm,
            "outputPowerDbm": outputPowerDbm,
            "width": config.width,
        },
    )
    powerCalibration.Calibrate(waveform.samples)
    paOutput = powerCalibration.GetLastPaOutput()
    resultAnalysis = TwoToneAnalysis(
        waveform,
        parameters={
            "loadResistanceOhm": config.loadResistanceOhm,
            "maximumOutputPowerDbm": config.maximumOutputPowerDbm,
            "width": config.width,
            "outputFullScaleAmplitude": (
                powerCalibration.outputFullScaleAmplitude
            ),
        },
    )
    return resultAnalysis.Analyze(paOutput)


def CalculateDpdGmpLabelMetrics(
    predistorter: DpdGmp,
    referenceSignals: Tuple[np.ndarray, ...],
    learnedInputs: Tuple[np.ndarray, ...],
    optimizedPowerIndex: int,
) -> Tuple[float, float, float]:
    """Calculate ordinary, peak-weighted, and worst-power label NMSE.

    Processing details:
        Algorithm: Evaluate the optimized-power pair with uniform weights,
        evaluate it again with a squared normalized-envelope weight, and
        select the least-negative ordinary NMSE across all power anchors.

    Args:
        predistorter: Trained GMP coefficient set.
        referenceSignals: Desired Wi-Fi waveforms at all training powers.
        learnedInputs: Matching ILC-learned PA-input labels.
        optimizedPowerIndex: Index of the nominal optimized operating point.

    Returns:
        result: Nominal NMSE, peak-weighted NMSE, and worst-power NMSE in dB.
    """

    selectedReference = referenceSignals[optimizedPowerIndex]
    selectedLabel = learnedInputs[optimizedPowerIndex]
    floatingReference = FixedPoint(
        predistorter.width
    ).DecodeComplex(selectedReference)
    peakMagnitude = max(
        float(np.max(np.abs(floatingReference))),
        np.finfo(float).tiny,
    )
    peakWeights = np.maximum(
        np.abs(floatingReference) / peakMagnitude,
        0.05,
    ) ** 2
    labelNmseDb = predistorter.CalculateNmse(
        selectedReference,
        selectedLabel,
    )
    peakWeightedLabelNmseDb = predistorter.CalculateNmse(
        selectedReference,
        selectedLabel,
        peakWeights,
    )
    worstPowerLabelNmseDb = max(
        predistorter.CalculateNmse(referenceSignal, learnedInput)
        for referenceSignal, learnedInput in zip(
            referenceSignals,
            learnedInputs,
        )
    )
    return (
        labelNmseDb,
        peakWeightedLabelNmseDb,
        worstPowerLabelNmseDb,
    )


def EvaluateDpdGmpPowerRobustness(
    config: DpdGmpBenchmarkConfig,
    waveform: WifiWaveform,
    paModel: PaModel,
    predistorter: DpdGmp,
) -> Tuple[float, float]:
    """Return worst Wi-Fi EVM and ACLR across all configured power anchors.

    Processing details:
        Algorithm: Recalibrate the complete DPD-plus-PA cascade independently
        at every power, select the highest EVM dB as worst modulation quality,
        and select the lowest positive ACLR as worst spectral containment.

    Args:
        config: Validated multi-power benchmark controls.
        waveform: Common Wi-Fi evaluation frame.
        paModel: GMP PA used by the cascade.
        predistorter: DPD model being checked for power robustness.

    Returns:
        result: Worst EVM dB and worst ACLR dB over the configured powers.
    """

    powerMetrics = tuple(
        EvaluateDpdGmpWifiStage(
            config,
            waveform,
            paModel,
            outputPowerDbm,
            predistorter,
        )
        for outputPowerDbm in config.trainingPowerDbm
    )
    return (
        max(metrics["evmDb"] for metrics in powerMetrics),
        min(metrics["aclrWorstDb"] for metrics in powerMetrics),
    )


def BuildDpdGmpStageResult(
    stageName: str,
    improvementCategory: str,
    outputPowerDbm: float,
    wifiMetrics: SignalMetrics,
    twoToneMetrics: TwoToneMetrics,
    modelDescription: str,
    predistorter: Optional[DpdGmp] = None,
    trainingResult: Optional[DpdGmpTrainingResult] = None,
    referenceSignals: Tuple[np.ndarray, ...] = tuple(),
    learnedInputs: Tuple[np.ndarray, ...] = tuple(),
    optimizedPowerIndex: int = 0,
    robustnessMetrics: Optional[Tuple[float, float]] = None,
) -> DpdGmpStageResult:
    """Combine independently measured RF and optional training diagnostics.

    Processing details:
        Algorithm: Copy Wi-Fi and two-tone metrics directly; for trained
        stages calculate nominal, peak-aware, and power-robust label errors,
        attach solver conditioning and coefficient norm, and retain optional
        measured worst-power Wi-Fi quality.

    Args:
        stageName: Stable benchmark stage label.
        improvementCategory: Physical issue addressed by this stage.
        outputPowerDbm: Equal-power target used for displayed RF metrics.
        wifiMetrics: Analysis result for the native PA output.
        twoToneMetrics: TwoToneAnalysis result for the same stage.
        modelDescription: Human-readable DPD structure and training method.
        predistorter: Optional trained DpdGmp model.
        trainingResult: Optional diagnostics from its latest fit.
        referenceSignals: Desired waveforms used for label evaluation.
        learnedInputs: Matching ILC labels used for label evaluation.
        optimizedPowerIndex: Nominal operating-point index.
        robustnessMetrics: Optional measured worst EVM and ACLR pair.

    Returns:
        result: One immutable, serialization-ready stage record.
    """

    labelNmseDb: Optional[float] = None
    peakWeightedLabelNmseDb: Optional[float] = None
    worstPowerLabelNmseDb: Optional[float] = None
    coefficientNorm: Optional[float] = None
    regularizedConditionNumber: Optional[float] = None
    if predistorter is not None:
        if (
            not referenceSignals
            or len(referenceSignals) != len(learnedInputs)
        ):
            raise ValueError(
                "trained stages require matching nonempty label segments"
            )
        (
            labelNmseDb,
            peakWeightedLabelNmseDb,
            worstPowerLabelNmseDb,
        ) = CalculateDpdGmpLabelMetrics(
            predistorter,
            referenceSignals,
            learnedInputs,
            optimizedPowerIndex,
        )
        coefficientNorm = float(
            np.linalg.norm(predistorter.GetCoefficients())
        )
        if trainingResult is not None:
            regularizedConditionNumber = (
                trainingResult.regularizedConditionNumber
            )
    return DpdGmpStageResult(
        stageName=stageName,
        improvementCategory=improvementCategory,
        targetOutputPowerDbm=float(outputPowerDbm),
        measuredOutputPowerDbm=wifiMetrics["outputPowerDbm"],
        evmDb=wifiMetrics["evmDb"],
        evmPercent=wifiMetrics["evmPercent"],
        aclrWorstDb=wifiMetrics["aclrWorstDb"],
        im3WorstDbc=twoToneMetrics["im3WorstDbc"],
        im5WorstDbc=twoToneMetrics["im5WorstDbc"],
        im7WorstDbc=twoToneMetrics["im7WorstDbc"],
        labelNmseDb=labelNmseDb,
        peakWeightedLabelNmseDb=peakWeightedLabelNmseDb,
        regularizedConditionNumber=regularizedConditionNumber,
        coefficientNorm=coefficientNorm,
        worstPowerLabelNmseDb=worstPowerLabelNmseDb,
        worstPowerEvmDb=(
            None if robustnessMetrics is None else robustnessMetrics[0]
        ),
        worstPowerAclrDb=(
            None if robustnessMetrics is None else robustnessMetrics[1]
        ),
        modelDescription=modelDescription,
    )


def BuildDpdGmpImprovementComparisons(
    stages: Tuple[DpdGmpStageResult, ...],
) -> Tuple[DpdGmpImprovementComparison, ...]:
    """Create auditable before/after checks for every proposed improvement.

    Processing details:
        Algorithm: Resolve stages by stable names, calculate positive-is-better
        differences with the correct direction for each metric, and mark an
        expectation successful only when the measured target metric improves.

    Args:
        stages: Complete ordered DPD-GMP stage result tuple.

    Returns:
        result: Ordered concrete improvements with measured pass/fail flags.
    """

    stageByName = {stage.stageName: stage for stage in stages}

    def AddLowerIsBetter(
        comparisonRows: List[DpdGmpImprovementComparison],
        improvementName: str,
        beforeStage: str,
        afterStage: str,
        targetMetric: str,
        beforeValue: float,
        afterValue: float,
        methodDetails: str,
    ) -> None:
        """Append one check whose target value must decrease.

        Processing details:
            Algorithm: Subtract the after value from the before value so a
            positive result consistently means improvement, retain both raw
            values, and mark the expectation only for a strict decrease.
        """

        improvementValue = beforeValue - afterValue
        comparisonRows.append(
            DpdGmpImprovementComparison(
                improvementName=improvementName,
                beforeStage=beforeStage,
                afterStage=afterStage,
                targetMetric=targetMetric,
                beforeValue=float(beforeValue),
                afterValue=float(afterValue),
                improvementValue=float(improvementValue),
                expectedDirection="lower",
                expectationMet=bool(improvementValue > 0.0),
                methodDetails=methodDetails,
            )
        )

    comparisonRows: List[DpdGmpImprovementComparison] = []
    baselineNominal = stageByName["PA baseline nominal"]
    basicNominal = stageByName["Basic DPD-GMP nominal"]
    AddLowerIsBetter(
        comparisonRows,
        "Enable basic DPD-GMP for Wi-Fi modulation",
        baselineNominal.stageName,
        basicNominal.stageName,
        "evmDb",
        baselineNominal.evmDb,
        basicNominal.evmDb,
        "Fit orders 1/3/5, three main-memory taps, and one cross-memory "
        f"tap to the {basicNominal.targetOutputPowerDbm:g} dBm ILC "
        "input label.",
    )
    AddLowerIsBetter(
        comparisonRows,
        "Enable basic DPD-GMP for two-tone intermodulation",
        baselineNominal.stageName,
        basicNominal.stageName,
        "im3WorstDbc",
        baselineNominal.im3WorstDbc,
        basicNominal.im3WorstDbc,
        "Apply the Wi-Fi-trained basic GMP coefficients without retraining "
        "to verify that the inverse suppresses a physical IM3 product.",
    )
    basicStress = stageByName["Basic DPD-GMP stress"]
    AddLowerIsBetter(
        comparisonRows,
        "Back off the severe compression operating point",
        basicStress.stageName,
        basicNominal.stageName,
        "evmDb",
        basicStress.evmDb,
        basicNominal.evmDb,
        "Reduce the target PA output from "
        f"{basicStress.targetOutputPowerDbm:g} to "
        f"{basicNominal.targetOutputPowerDbm:g} dBm while preserving the "
        "same basic coefficient structure, then recalibrate the cascade.",
    )
    memoryExpanded = stageByName["Memory-expanded DPD-GMP"]
    if (
        basicNominal.labelNmseDb is None
        or memoryExpanded.labelNmseDb is None
    ):
        raise RuntimeError("label NMSE is required for structure comparison")
    AddLowerIsBetter(
        comparisonRows,
        "Expand nonlinear order and memory structure",
        basicNominal.stageName,
        memoryExpanded.stageName,
        "labelNmseDb",
        basicNominal.labelNmseDb,
        memoryExpanded.labelNmseDb,
        "Increase maximum odd order from five to seven, main memory from "
        "three to five, and cross-memory depth from one to three.",
    )
    peakWeighted = stageByName["Peak-weighted DPD-GMP"]
    if (
        memoryExpanded.peakWeightedLabelNmseDb is None
        or peakWeighted.peakWeightedLabelNmseDb is None
    ):
        raise RuntimeError(
            "peak-weighted NMSE is required for weighting comparison"
        )
    AddLowerIsBetter(
        comparisonRows,
        "Add envelope-peak-aware sample weighting",
        memoryExpanded.stageName,
        peakWeighted.stageName,
        "peakWeightedLabelNmseDb",
        memoryExpanded.peakWeightedLabelNmseDb,
        peakWeighted.peakWeightedLabelNmseDb,
        "Multiply each regression sample by the squared normalized envelope "
        "magnitude before solving the GMP normal equations.",
    )
    regularized = stageByName["Regularized DPD-GMP"]
    if (
        peakWeighted.regularizedConditionNumber is None
        or regularized.regularizedConditionNumber is None
    ):
        raise RuntimeError(
            "condition numbers are required for ridge comparison"
        )
    conditionImprovementDb = 10.0 * np.log10(
        peakWeighted.regularizedConditionNumber
        / regularized.regularizedConditionNumber
    )
    comparisonRows.append(
        DpdGmpImprovementComparison(
            improvementName="Increase ridge stabilization",
            beforeStage=peakWeighted.stageName,
            afterStage=regularized.stageName,
            targetMetric="regularizedConditionNumber",
            beforeValue=peakWeighted.regularizedConditionNumber,
            afterValue=regularized.regularizedConditionNumber,
            improvementValue=float(conditionImprovementDb),
            expectedDirection="lower",
            expectationMet=bool(conditionImprovementDb > 0.0),
            methodDetails=(
                "Increase ridgeFactor from 1e-6 to 1e-4 while retaining "
                "peak weighting; improvementValue is 10log10 of the "
                "before/after condition-number ratio."
            ),
        )
    )
    multiPower = stageByName["Multi-power DPD-GMP"]
    if (
        regularized.worstPowerLabelNmseDb is None
        or multiPower.worstPowerLabelNmseDb is None
        or regularized.worstPowerAclrDb is None
        or multiPower.worstPowerAclrDb is None
    ):
        raise RuntimeError(
            "multi-power robustness metrics are required for comparison"
        )
    AddLowerIsBetter(
        comparisonRows,
        "Joint multi-power coefficient training",
        regularized.stageName,
        multiPower.stageName,
        "worstPowerLabelNmseDb",
        regularized.worstPowerLabelNmseDb,
        multiPower.worstPowerLabelNmseDb,
        multiPower.modelDescription
        + "; accumulate each segment without joining memory across frame "
        "boundaries.",
    )
    aclrGuardrailDb = 0.10
    aclrChangeDb = (
        multiPower.worstPowerAclrDb
        - regularized.worstPowerAclrDb
    )
    comparisonRows.append(
        DpdGmpImprovementComparison(
            improvementName="Joint multi-power ACLR robustness",
            beforeStage=regularized.stageName,
            afterStage=multiPower.stageName,
            targetMetric="worstPowerAclrDb",
            beforeValue=regularized.worstPowerAclrDb,
            afterValue=multiPower.worstPowerAclrDb,
            improvementValue=float(aclrChangeDb),
            expectedDirection="no decrease beyond 0.10 dB",
            expectationMet=bool(aclrChangeDb >= -aclrGuardrailDb),
            methodDetails=(
                "Evaluate both coefficient sets on the independent Wi-Fi "
                "validation frame at equal output powers, retain the "
                "minimum ACLR across 10/12/14 dBm, and allow at most "
                "0.10 dB degradation near the source-waveform ACLR floor."
            ),
        )
    )
    return tuple(comparisonRows)


def SaveDpdGmpBenchmarkResults(
    result: DpdGmpBenchmarkResult,
    config: DpdGmpBenchmarkConfig,
) -> Tuple[Path, Path, Path, Path]:
    """Save DPD-GMP stage data, improvement checks, JSON, and one figure.

    Processing details:
        Algorithm: Serialize flat stage and comparison tables, preserve the
        complete configuration and nested records in JSON, and delegate all
        graphical presentation to Draw.

    Args:
        result: Completed DPD-GMP benchmark result.
        config: Controls used to generate the result.

    Returns:
        result: Paths to stage CSV, comparison CSV, JSON, and PNG artifacts.
    """

    outputDirectory = Path(config.outputDirectory)
    outputDirectory.mkdir(parents=True, exist_ok=True)
    stagePath = outputDirectory / "dpd_gmp_stage_metrics.csv"
    comparisonPath = (
        outputDirectory / "dpd_gmp_improvement_comparison.csv"
    )
    jsonPath = outputDirectory / "dpd_gmp_benchmark.json"
    stageRows = [stage.ToDict() for stage in result.stages]
    comparisonRows = [
        comparison.ToDict() for comparison in result.comparisons
    ]
    with stagePath.open("w", newline="", encoding="utf-8-sig") as csvFile:
        csvWriter = csv.DictWriter(
            csvFile,
            fieldnames=list(stageRows[0]),
        )
        csvWriter.writeheader()
        csvWriter.writerows(stageRows)
    with comparisonPath.open(
        "w", newline="", encoding="utf-8-sig"
    ) as csvFile:
        csvWriter = csv.DictWriter(
            csvFile,
            fieldnames=list(comparisonRows[0]),
        )
        csvWriter.writeheader()
        csvWriter.writerows(comparisonRows)
    document = {
        "configuration": {
            "frameFormat": config.frameFormat,
            "bandwidthMhz": config.bandwidthMhz,
            "sampleRateHz": config.sampleRateHz,
            "mcs": config.mcs,
            "numDataSymbols": config.numDataSymbols,
            "seed": config.seed,
            "validationSeed": config.validationSeed,
            "toneFrequenciesHz": list(config.toneFrequenciesHz),
            "toneNumSamples": config.toneNumSamples,
            "stressOutputPowerDbm": config.stressOutputPowerDbm,
            "optimizedOutputPowerDbm": config.optimizedOutputPowerDbm,
            "trainingPowerDbm": list(config.trainingPowerDbm),
            "maximumOutputPowerDbm": config.maximumOutputPowerDbm,
            "loadResistanceOhm": config.loadResistanceOhm,
            "numIterations": config.numIterations,
            "width": config.width,
        },
        **result.ToDict(),
    }
    with jsonPath.open("w", encoding="utf-8") as jsonFile:
        json.dump(document, jsonFile, indent=2, ensure_ascii=False)
    figurePath = Draw().SaveDpdGmpPerformance(
        stageRows,
        outputDirectory,
    )
    return stagePath, comparisonPath, jsonPath, figurePath


def PrintDpdGmpBenchmarkResults(
    result: DpdGmpBenchmarkResult,
) -> None:
    """Print compact DPD-GMP stage and expected-improvement summaries.

    Processing details:
        Algorithm: Preserve benchmark ordering, show equal-power EVM/ACLR/IM3
        and label NMSE, then print every improvement magnitude and pass state.

    Args:
        result: Complete DPD-GMP benchmark result.

    Returns:
        result: None. Human-readable tables are written to standard output.
    """

    header = (
        f"{'Stage':<31} {'Pout':>7} {'EVM':>9} "
        f"{'ACLR':>8} {'IM3':>9} {'Label':>9}"
    )
    print("\nDPD-GMP performance stages")
    print(header)
    print("-" * len(header))
    for stage in result.stages:
        labelText = (
            "n/a"
            if stage.labelNmseDb is None
            else f"{stage.labelNmseDb:.2f}"
        )
        print(
            f"{stage.stageName:<31} "
            f"{stage.measuredOutputPowerDbm:>7.2f} "
            f"{stage.evmDb:>9.2f} "
            f"{stage.aclrWorstDb:>8.2f} "
            f"{stage.im3WorstDbc:>9.2f} "
            f"{labelText:>9}"
        )
    print("\nMeasured DPD-GMP improvement checks")
    for comparison in result.comparisons:
        status = "PASS" if comparison.expectationMet else "FAIL"
        print(
            f"- [{status}] {comparison.improvementName}: "
            f"{comparison.improvementValue:.3f}"
        )


def RunDpdGmpBenchmark(
    config: Optional[DpdGmpBenchmarkConfig] = None,
) -> DpdGmpBenchmarkResult:
    """Run staged, PA-analysis-driven DPD-GMP performance improvements.

    Processing details:
        Algorithm: Generate ILC input labels at 10/12/14 dBm from one Wi-Fi
        frame, train basic, memory-expanded, peak-weighted, stabilized, and
        multi-power GMP models, measure native PA output at equal power on a
        distinct validation frame and two-tone stimulus, verify each intended
        metric change, and save all tables and figures.

    Args:
        config: Optional complete benchmark setup. None uses internal defaults.

    Returns:
        result: Ordered stages and auditable improvement comparisons.
    """

    if config is None:
        config = DpdGmpBenchmarkConfig()
    config.Validate()
    trainingWifiWaveform = WaveGenWifi(
        parameters={
            "frameFormat": config.frameFormat,
            "bandwidthMhz": config.bandwidthMhz,
            "sampleRateHz": config.sampleRateHz,
            "mcs": config.mcs,
            "numDataSymbols": config.numDataSymbols,
            "seed": config.seed,
            "width": config.width,
        }
    ).Generate()
    validationWifiWaveform = WaveGenWifi(
        parameters={
            "frameFormat": config.frameFormat,
            "bandwidthMhz": config.bandwidthMhz,
            "sampleRateHz": config.sampleRateHz,
            "mcs": config.mcs,
            "numDataSymbols": config.numDataSymbols,
            "seed": config.validationSeed,
            "width": config.width,
        }
    ).Generate()
    twoToneWaveform = WaveGenTwoTone(
        parameters={
            "sampleRateHz": config.sampleRateHz,
            "toneFrequenciesHz": config.toneFrequenciesHz,
            "numSamples": config.toneNumSamples,
            "rmsLevel": 0.5,
            "width": config.width,
        }
    ).Generate()
    paModel = PaModel(
        parameters={
            "modelName": "gmp",
            "width": config.width,
        }
    )
    labelPairs = tuple(
        GenerateDpdGmpIlcLabel(
            config,
            trainingWifiWaveform,
            paModel,
            outputPowerDbm,
        )
        for outputPowerDbm in config.trainingPowerDbm
    )
    referenceSignals = tuple(pair[0] for pair in labelPairs)
    learnedInputs = tuple(pair[1] for pair in labelPairs)
    optimizedPowerIndex = config.trainingPowerDbm.index(
        config.optimizedOutputPowerDbm
    )
    optimizedReference = referenceSignals[optimizedPowerIndex]
    optimizedLearnedInput = learnedInputs[optimizedPowerIndex]

    basicDpd = DpdGmp(
        parameters={
            "nonlinearOrders": (1, 3, 5),
            "memoryDepth": 3,
            "crossMemoryDepth": 1,
            "ridgeFactor": 1.0e-5,
            "peakWeightExponent": 0.0,
            "maximumOutputMagnitude": 1.5,
            "width": config.width,
        }
    )
    basicTraining = basicDpd.FitFromIlc(
        optimizedReference,
        optimizedLearnedInput,
    )
    memoryDpd = DpdGmp(
        parameters={
            "nonlinearOrders": (1, 3, 5, 7),
            "memoryDepth": 5,
            "crossMemoryDepth": 3,
            "ridgeFactor": 1.0e-5,
            "peakWeightExponent": 0.0,
            "maximumOutputMagnitude": 1.5,
            "width": config.width,
        }
    )
    memoryTraining = memoryDpd.FitFromIlc(
        optimizedReference,
        optimizedLearnedInput,
    )
    peakDpd = DpdGmp(
        parameters={
            "nonlinearOrders": (1, 3, 5, 7),
            "memoryDepth": 5,
            "crossMemoryDepth": 3,
            "ridgeFactor": 1.0e-6,
            "peakWeightExponent": 2.0,
            "maximumOutputMagnitude": 1.5,
            "width": config.width,
        }
    )
    peakTraining = peakDpd.FitFromIlc(
        optimizedReference,
        optimizedLearnedInput,
    )
    regularizedDpd = DpdGmp(
        parameters={
            "nonlinearOrders": (1, 3, 5, 7),
            "memoryDepth": 5,
            "crossMemoryDepth": 3,
            "ridgeFactor": 1.0e-4,
            "peakWeightExponent": 2.0,
            "maximumOutputMagnitude": 1.5,
            "width": config.width,
        }
    )
    regularizedTraining = regularizedDpd.FitFromIlc(
        optimizedReference,
        optimizedLearnedInput,
    )
    multiPowerDpd = DpdGmp(
        parameters={
            "nonlinearOrders": (1, 3, 5, 7),
            "memoryDepth": 5,
            "crossMemoryDepth": 3,
            "ridgeFactor": 1.0e-4,
            "peakWeightExponent": 2.0,
            "maximumOutputMagnitude": 1.5,
            "width": config.width,
        }
    )
    segmentWeights = tuple(
        2.0
        if np.isclose(
            powerDbm,
            config.optimizedOutputPowerDbm,
            rtol=0.0,
            atol=np.finfo(float).eps
            * max(1.0, abs(config.optimizedOutputPowerDbm)),
        )
        else 1.0
        for powerDbm in config.trainingPowerDbm
    )
    multiPowerTraining = multiPowerDpd.FitSegments(
        referenceSignals,
        learnedInputs,
        segmentWeights=segmentWeights,
    )

    baselineNominalWifi = EvaluateDpdGmpWifiStage(
        config,
        validationWifiWaveform,
        paModel,
        config.optimizedOutputPowerDbm,
    )
    baselineNominalTone = EvaluateDpdGmpTwoToneStage(
        config,
        twoToneWaveform,
        paModel,
        config.optimizedOutputPowerDbm,
    )
    baselineStressWifi = EvaluateDpdGmpWifiStage(
        config,
        validationWifiWaveform,
        paModel,
        config.stressOutputPowerDbm,
    )
    baselineStressTone = EvaluateDpdGmpTwoToneStage(
        config,
        twoToneWaveform,
        paModel,
        config.stressOutputPowerDbm,
    )

    modelStages = (
        (
            "Basic DPD-GMP nominal",
            "Baseline inverse",
            basicDpd,
            basicTraining,
            (
                "Orders 1/3/5, memoryDepth=3, crossMemoryDepth=1, "
                "ridgeFactor=1e-5, nominal-power ILC label"
            ),
        ),
        (
            "Memory-expanded DPD-GMP",
            "Memory and nonlinear order",
            memoryDpd,
            memoryTraining,
            (
                "Orders 1/3/5/7, memoryDepth=5, crossMemoryDepth=3, "
                "ridgeFactor=1e-5, nominal-power ILC label"
            ),
        ),
        (
            "Peak-weighted DPD-GMP",
            "Envelope peaks",
            peakDpd,
            peakTraining,
            (
                "Expanded structure, ridgeFactor=1e-6, "
                "peakWeightExponent=2, nominal-power ILC label"
            ),
        ),
        (
            "Regularized DPD-GMP",
            "Numerical stability",
            regularizedDpd,
            regularizedTraining,
            (
                "Expanded peak-weighted structure, ridgeFactor=1e-4, "
                "nominal-power ILC label"
            ),
        ),
        (
            "Multi-power DPD-GMP",
            "Power robustness",
            multiPowerDpd,
            multiPowerTraining,
            (
                "Expanded peak-weighted structure, ridgeFactor=1e-4, "
                "joint "
                + "/".join(
                    f"{powerDbm:g}"
                    for powerDbm in config.trainingPowerDbm
                )
                + " dBm labels with "
                + "/".join(
                    f"{segmentWeight:g}"
                    for segmentWeight in segmentWeights
                )
                + " segment weights"
            ),
        ),
    )
    stages: List[DpdGmpStageResult] = [
        BuildDpdGmpStageResult(
            "PA baseline stress",
            "Severe compression reference",
            config.stressOutputPowerDbm,
            baselineStressWifi,
            baselineStressTone,
            "Unlinearized GMP PA",
        ),
        BuildDpdGmpStageResult(
            "PA baseline nominal",
            "Equal-power reference",
            config.optimizedOutputPowerDbm,
            baselineNominalWifi,
            baselineNominalTone,
            "Unlinearized GMP PA",
        ),
    ]
    basicStressWifi = EvaluateDpdGmpWifiStage(
        config,
        validationWifiWaveform,
        paModel,
        config.stressOutputPowerDbm,
        basicDpd,
    )
    basicStressTone = EvaluateDpdGmpTwoToneStage(
        config,
        twoToneWaveform,
        paModel,
        config.stressOutputPowerDbm,
        basicDpd,
    )
    stages.append(
        BuildDpdGmpStageResult(
            "Basic DPD-GMP stress",
            "Severe compression",
            config.stressOutputPowerDbm,
            basicStressWifi,
            basicStressTone,
            modelStages[0][4],
            basicDpd,
            basicTraining,
            referenceSignals,
            learnedInputs,
            optimizedPowerIndex,
        )
    )
    for (
        stageName,
        improvementCategory,
        predistorter,
        trainingResult,
        modelDescription,
    ) in modelStages:
        wifiMetrics = EvaluateDpdGmpWifiStage(
            config,
            validationWifiWaveform,
            paModel,
            config.optimizedOutputPowerDbm,
            predistorter,
        )
        twoToneMetrics = EvaluateDpdGmpTwoToneStage(
            config,
            twoToneWaveform,
            paModel,
            config.optimizedOutputPowerDbm,
            predistorter,
        )
        robustnessMetrics = (
            EvaluateDpdGmpPowerRobustness(
                config,
                validationWifiWaveform,
                paModel,
                predistorter,
            )
            if stageName in (
                "Regularized DPD-GMP",
                "Multi-power DPD-GMP",
            )
            else None
        )
        stages.append(
            BuildDpdGmpStageResult(
                stageName,
                improvementCategory,
                config.optimizedOutputPowerDbm,
                wifiMetrics,
                twoToneMetrics,
                modelDescription,
                predistorter,
                trainingResult,
                referenceSignals,
                learnedInputs,
                optimizedPowerIndex,
                robustnessMetrics,
            )
        )
    stageTuple = tuple(stages)
    result = DpdGmpBenchmarkResult(
        stages=stageTuple,
        comparisons=BuildDpdGmpImprovementComparisons(stageTuple),
    )
    SaveDpdGmpBenchmarkResults(result, config)
    PrintDpdGmpBenchmarkResults(result)
    return result


@dataclass(frozen=True)
class ChannelAnalysisBenchmarkConfig:
    """Configure channel measurement and coupling-aware DPD comparison."""

    frameFormat: str = "EHT"
    bandwidthMhz: int = 20
    sampleRateHz: float = 80.0e6
    mcs: int = 7
    numDataSymbols: int = 4
    seed: int = 517
    outputPowerDbm: float = 13.0
    maximumOutputPowerDbm: float = 25.0
    loadResistanceOhm: float = 50.0
    fftLength: int = 2048
    impulseLength: int = 64
    numIterations: int = 10
    width: int = 0
    outputDirectory: Path = Path("results/channel_analysis")

    def Validate(self) -> None:
        """Validate waveform, measurement, power, and learning controls.

        Processing details:
            Algorithm: Instantiate the production Wi-Fi generator and channel
            analyzer with the requested values, validate the absolute output
            power through PowerCalibration, and reject too-short learning or
            waveform records before the multi-stage benchmark starts.

        Returns:
            result: None. Invalid settings raise descriptive exceptions.
        """

        if (
            not isinstance(self.numDataSymbols, int)
            or isinstance(self.numDataSymbols, bool)
            or self.numDataSymbols < 2
        ):
            raise ValueError(
                "numDataSymbols must be an integer no smaller than two"
            )
        if (
            not isinstance(self.numIterations, int)
            or isinstance(self.numIterations, bool)
            or self.numIterations < 2
        ):
            raise ValueError(
                "numIterations must be an integer no smaller than two"
            )
        WaveGenWifi(
            parameters={
                "frameFormat": self.frameFormat,
                "bandwidthMhz": self.bandwidthMhz,
                "sampleRateHz": self.sampleRateHz,
                "mcs": self.mcs,
                "numDataSymbols": self.numDataSymbols,
                "seed": self.seed,
                "width": self.width,
            }
        )
        ChannelAnalyse(
            parameters={
                "sampleRateHz": self.sampleRateHz,
                "channelBandwidthHz": self.bandwidthMhz * 1.0e6,
                "fftLength": self.fftLength,
                "impulseLength": self.impulseLength,
                "width": self.width,
            }
        )
        PowerCalibration(
            parameters={
                "outputPowerDbm": self.outputPowerDbm,
                "maximumOutputPowerDbm": self.maximumOutputPowerDbm,
                "loadResistanceOhm": self.loadResistanceOhm,
                "width": self.width,
            }
        )


@dataclass(frozen=True)
class ChannelDpdStageResult:
    """Store one equal-reference MIMO DPD performance stage."""

    stageName: str
    methodDescription: str
    evmDb: float
    evmPercent: float
    normalizedMseDb: float
    aclrWorstDb: float
    residualCouplingDb: float

    def ToDict(self) -> Dict[str, object]:
        """Convert one immutable comparison stage to a flat mapping.

        Processing details:
            Algorithm: Copy the method identity and all aggregate waveform,
            spectral, and residual-coupling metrics without recalculation.

        Returns:
            result: CSV/JSON-compatible performance row.
        """

        return {
            "stageName": self.stageName,
            "methodDescription": self.methodDescription,
            "evmDb": self.evmDb,
            "evmPercent": self.evmPercent,
            "normalizedMseDb": self.normalizedMseDb,
            "aclrWorstDb": self.aclrWorstDb,
            "residualCouplingDb": self.residualCouplingDb,
        }


@dataclass(frozen=True)
class ChannelDpdImprovement:
    """Store the measured benefit of coupling-aware DPD over independent DPD."""

    metricName: str
    beforeValue: float
    afterValue: float
    improvementValue: float
    expectedDirection: str
    expectationMet: bool

    def ToDict(self) -> Dict[str, object]:
        """Convert one before/after assertion to a flat result mapping.

        Processing details:
            Algorithm: Preserve raw values, the consistently positive
            improvement convention, expected direction, and pass/fail flag.

        Returns:
            result: JSON/CSV-compatible comparison record.
        """

        return {
            "metricName": self.metricName,
            "beforeValue": self.beforeValue,
            "afterValue": self.afterValue,
            "improvementValue": self.improvementValue,
            "expectedDirection": self.expectedDirection,
            "expectationMet": self.expectationMet,
        }


@dataclass(frozen=True)
class IqGmpStageResult:
    """Store one equal-power IQ-imbalance DPD comparison point."""

    methodName: str
    outputPowerDbm: float
    measuredOutputPowerDbm: float
    evmDb: float
    evmPercent: float
    irrDb: float
    aclrWorstDb: float

    def ToDict(self) -> Dict[str, object]:
        """Convert one IQ-GMP curve point to a flat result mapping.

        Processing details:
            Algorithm: Copy method identity, requested/measured power, EVM,
            signed image-relative IRR, and ACLR without changing metric sign
            conventions.

        Returns:
            result: CSV/JSON-compatible curve row.
        """

        return {
            "methodName": self.methodName,
            "outputPowerDbm": self.outputPowerDbm,
            "measuredOutputPowerDbm": self.measuredOutputPowerDbm,
            "evmDb": self.evmDb,
            "evmPercent": self.evmPercent,
            "irrDb": self.irrDb,
            "aclrWorstDb": self.aclrWorstDb,
        }


@dataclass(frozen=True)
class ChannelAnalysisBenchmarkResult:
    """Store channel measurements, DPD stages, and expected improvements."""

    prePaMeasurement: ChannelMeasurementResult
    postPaMeasurement: ChannelMeasurementResult
    stages: Tuple[ChannelDpdStageResult, ...]
    improvements: Tuple[ChannelDpdImprovement, ...]
    trainingResult: CouplingAwareDpdGmpTrainingResult
    iqImbalanceStages: Tuple[IqGmpStageResult, ...]

    def ToDict(self) -> Dict[str, object]:
        """Convert the complete benchmark result to nested plain mappings.

        Processing details:
            Algorithm: Serialize pre/post channel summaries, every DPD stage,
            all expectation checks, and per-chain training diagnostics while
            retaining large complex frequency arrays outside JSON.

        Returns:
            result: JSON-compatible benchmark record.
        """

        return {
            "prePaMeasurement": self.prePaMeasurement.ToDict(),
            "postPaMeasurement": self.postPaMeasurement.ToDict(),
            "stages": [stage.ToDict() for stage in self.stages],
            "improvements": [
                improvement.ToDict()
                for improvement in self.improvements
            ],
            "trainingResult": self.trainingResult.ToDict(),
            "iqImbalanceStages": [
                stage.ToDict()
                for stage in self.iqImbalanceStages
            ],
        }


def BuildChannelAnalysisPaBank(width: int) -> MimoPaModel:
    """Construct two different nonlinear PA branches for channel testing.

    Processing details:
        Algorithm: Bind a full-reference-strength GMP PA to chain zero and a
        Wiener PA to chain one so this explicit stress benchmark remains
        stable when the ordinary user-facing GMP default is retuned, cannot
        benefit from an accidental common inverse model, and preserves the
        selected public I/Q convention at the MIMO boundary.

    Args:
        width: Public floating or fixed-point component width.

    Returns:
        result: Two-chain independent nonlinear PA bank.
    """

    return MimoPaModel(
        parameters={
            "numTransmitChains": 2,
            "paParametersPerChain": (
                {
                    "modelName": "gmp",
                    "gmpConfig": GMPConfig(nonlinearScale=1.0),
                },
                {"modelName": "wiener"},
            ),
            "width": width,
        }
    )


def BuildChannelAnalysisPlant(
    config: ChannelAnalysisBenchmarkConfig,
) -> Channel:
    """Construct the asymmetric frequency-selective coupled MIMO plant.

    Processing details:
        Algorithm: Place direction-dependent complex FIR leakage with
        different integer and fractional delays both before and after two
        different PAs.  Disable noise and feedback impairments so every
        before/after difference is attributable to measured coupling-aware
        training and compensation.

    Args:
        config: Validated channel benchmark controls.

    Returns:
        result: Floating or fixed public Channel with a bound MIMO PA bank.
    """

    prePaCouplingPaths: Tuple[Mapping[str, object], ...] = (
        {
            "sourceChain": 0,
            "destinationChain": 1,
            "gainDb": -15.0,
            "phaseDegrees": 35.0,
            "integerDelaySamples": 2,
            "fractionalDelaySamples": 0.25,
            "firTaps": (
                1.0 + 0.0j,
                0.28 - 0.12j,
            ),
        },
        {
            "sourceChain": 1,
            "destinationChain": 0,
            "gainDb": -18.0,
            "phaseDegrees": -42.0,
            "integerDelaySamples": 1,
            "fractionalDelaySamples": -0.20,
            "firTaps": (
                1.0 + 0.0j,
                -0.18 + 0.09j,
            ),
        },
    )
    postPaCouplingPaths: Tuple[Mapping[str, object], ...] = (
        {
            "sourceChain": 0,
            "destinationChain": 1,
            "gainDb": -13.0,
            "phaseDegrees": -25.0,
            "integerDelaySamples": 1,
            "fractionalDelaySamples": 0.15,
            "firTaps": (
                1.0 + 0.0j,
                0.22 + 0.08j,
            ),
        },
        {
            "sourceChain": 1,
            "destinationChain": 0,
            "gainDb": -17.0,
            "phaseDegrees": 55.0,
            "integerDelaySamples": 3,
            "fractionalDelaySamples": -0.10,
            "firTaps": (
                1.0 + 0.0j,
                -0.16 - 0.05j,
            ),
        },
    )
    return Channel(
        paModel=BuildChannelAnalysisPaBank(config.width),
        parameters={
            "sampleMode": "forward",
            "sampleRateHz": config.sampleRateHz,
            "phaseDegrees": 0,
            "noiseAmpMv": None,
            "noisePwrDbm": None,
            "noiseSnrDb": None,
            "prePaCouplingPaths": prePaCouplingPaths,
            "postPaCouplingPaths": postPaCouplingPaths,
            "maximumOutputPowerDbm": config.maximumOutputPowerDbm,
            "loadResistanceOhm": config.loadResistanceOhm,
            "width": config.width,
        },
    )


def GenerateChannelAnalysisReferences(
    config: ChannelAnalysisBenchmarkConfig,
) -> Tuple[Tuple[WifiWaveform, ...], np.ndarray]:
    """Generate two independent equal-power Wi-Fi reference waveforms.

    Processing details:
        Algorithm: Generate equal-format packets from decorrelated seeds,
        scale each active burst to the same absolute conducted-power target,
        decode public samples once, and stack the two physical-chain columns.

    Args:
        config: Validated waveform and power controls.

    Returns:
        result: Per-chain metadata tuple and normalized reference matrix.
    """

    waveforms = []
    referenceColumns = []
    for chainIndex in range(2):
        waveform = WaveGenWifi(
            parameters={
                "frameFormat": config.frameFormat,
                "bandwidthMhz": config.bandwidthMhz,
                "sampleRateHz": config.sampleRateHz,
                "mcs": config.mcs,
                "numDataSymbols": config.numDataSymbols,
                "seed": config.seed + 97 * chainIndex,
                "width": config.width,
            }
        ).Generate()
        powerCalibration = PowerCalibration(
            parameters={
                "outputPowerDbm": config.outputPowerDbm,
                "maximumOutputPowerDbm": (
                    config.maximumOutputPowerDbm
                ),
                "loadResistanceOhm": config.loadResistanceOhm,
                "width": config.width,
            }
        )
        scaledReference = powerCalibration.ScaleSignalToOutputPower(
            waveform.samples,
            config.outputPowerDbm,
        )
        waveforms.append(waveform)
        referenceColumns.append(
            FixedPoint(config.width).DecodeComplex(
                scaledReference
            )
        )
    if len({column.size for column in referenceColumns}) != 1:
        raise ValueError(
            "channel benchmark Wi-Fi references must have equal lengths"
        )
    return tuple(waveforms), np.column_stack(referenceColumns)


def GenerateChannelPaInputLabels(
    config: ChannelAnalysisBenchmarkConfig,
    paOutputTargets: np.ndarray,
    paModels: Sequence[PaModel],
) -> np.ndarray:
    """Generate independent ILC labels at the actual inputs of both PAs.

    Processing details:
        Algorithm: Run the synchronized frequency-domain ILC against each
        physical PA alone using the post-deembedded target for that branch,
        and stack the converged predistorted inputs.  These labels describe
        signals after the pre-PA coupling network, which is canceled only at
        deployment time.

    Args:
        config: Validated sample-rate, bandwidth, and iteration controls.
        paOutputTargets: Desired individual PA-output matrix.
        paModels: Ordered internal floating PA facades from the MIMO bank.

    Returns:
        result: Normalized samples-by-chain PA-input label matrix.
    """

    targetMatrix = np.asarray(
        paOutputTargets, dtype=np.complex128
    )
    if (
        targetMatrix.ndim != 2
        or targetMatrix.shape[1] != len(paModels)
        or targetMatrix.shape[0] == 0
        or not np.all(np.isfinite(targetMatrix))
    ):
        raise ValueError(
            "paOutputTargets must contain one finite column per PA"
        )
    learnedColumns = []
    for chainIndex, paModel in enumerate(paModels):
        targetColumn = targetMatrix[:, chainIndex]
        maximumInputMagnitude = max(
            1.5,
            2.5 * float(np.max(np.abs(targetColumn))),
        )
        ilcResult = RunFrequencyDomainIlc(
            targetColumn,
            paModel,
            config.sampleRateHz,
            config.bandwidthMhz * 1.0e6,
            ILCConfig(
                numIterations=config.numIterations,
                learningRate=0.12,
                regularization=1.0e-3,
                maxAmplitude=maximumInputMagnitude,
            ),
        )
        learnedColumns.append(
            np.asarray(
                ilcResult.learnedInput, dtype=np.complex128
            )
        )
    return np.column_stack(learnedColumns)


def BuildChannelDpdModels(
    chainCount: int,
) -> Tuple[DpdGmp, ...]:
    """Construct equal-structure floating GMP models for all PA branches.

    Processing details:
        Algorithm: Create one independent seventh-order, four-sample-memory
        DpdGmp per chain with cross-envelope memory, ridge stabilization, and
        no public quantization inside the benchmark learning loop.

    Args:
        chainCount: Number of independent physical PA inverse models.

    Returns:
        result: Ordered identity-initialized DPD model tuple.
    """

    if (
        not isinstance(chainCount, int)
        or isinstance(chainCount, bool)
        or chainCount < 1
    ):
        raise ValueError("chainCount must be a positive integer")
    return tuple(
        DpdGmp(
            parameters={
                "nonlinearOrders": (1, 3, 5, 7),
                "memoryDepth": 4,
                "crossMemoryDepth": 3,
                "ridgeFactor": 3.0e-5,
                "peakWeightExponent": 1.0,
                "maximumOutputMagnitude": 2.0,
                "width": 0,
            }
        )
        for _ in range(chainCount)
    )


def EvaluateChannelDpdStage(
    config: ChannelAnalysisBenchmarkConfig,
    channel: Channel,
    waveforms: Sequence[WifiWaveform],
    referenceSignal: np.ndarray,
    dacInputSignal: np.ndarray,
    stageName: str,
    methodDescription: str,
) -> ChannelDpdStageResult:
    """Evaluate one raw-DAC strategy through the same coupled nonlinear plant.

    Processing details:
        Algorithm: Run the complete pre-coupling, independent PA, post-
        coupling cascade once, analyze every observed chain against its known
        Wi-Fi reference, aggregate worst EVM and ACLR, calculate common-gain-
        aligned matrix NMSE, and project each output onto the other chain to
        estimate residual linear leakage.

    Args:
        config: Validated analysis and physical-power settings.
        channel: Common coupled nonlinear plant.
        waveforms: Per-chain Wi-Fi metadata matching reference columns.
        referenceSignal: Desired final samples-by-chain outputs.
        dacInputSignal: Raw samples-by-chain input before pre-PA coupling.
        stageName: Display label for this strategy.
        methodDescription: Exact compensation operations used by this stage.

    Returns:
        result: Aggregate EVM, NMSE, ACLR, and residual coupling metrics.
    """

    referenceMatrix = np.asarray(
        referenceSignal, dtype=np.complex128
    )
    measuredMatrix = np.asarray(
        channel.ProcessOutputPathsFloating(dacInputSignal)[0],
        dtype=np.complex128,
    )
    if measuredMatrix.shape != referenceMatrix.shape:
        raise ValueError(
            "coupled plant output must match the reference matrix shape"
        )
    evmDbValues = []
    evmPercentValues = []
    aclrValues = []
    residualPower = 0.0
    alignedReferencePower = 0.0
    residualCouplingValues = []
    numericFloor = np.finfo(float).tiny
    for chainIndex, waveform in enumerate(waveforms):
        referenceColumn = referenceMatrix[:, chainIndex]
        measuredColumn = measuredMatrix[:, chainIndex]
        metrics = Analysis(
            referenceColumn,
            waveform,
            parameters={
                "maximumOutputPowerDbm": (
                    config.maximumOutputPowerDbm
                ),
                "loadResistanceOhm": config.loadResistanceOhm,
                "width": 0,
            },
        ).Analyze(measuredColumn)
        evmDbValues.append(float(metrics["evmDb"]))
        evmPercentValues.append(float(metrics["evmPercent"]))
        aclrValues.append(float(metrics["aclrWorstDb"]))
        complexGain = np.vdot(
            referenceColumn, measuredColumn
        ) / max(
            float(np.vdot(referenceColumn, referenceColumn).real),
            numericFloor,
        )
        residual = measuredColumn - complexGain * referenceColumn
        residualPower += float(np.vdot(residual, residual).real)
        alignedReferencePower += float(
            np.vdot(
                complexGain * referenceColumn,
                complexGain * referenceColumn,
            ).real
        )
        otherChain = 1 - chainIndex
        otherReference = referenceMatrix[:, otherChain]
        leakageCoefficient = np.vdot(
            otherReference, residual
        ) / max(
            float(np.vdot(otherReference, otherReference).real),
            numericFloor,
        )
        residualCouplingValues.append(
            20.0
            * np.log10(
                max(float(np.abs(leakageCoefficient)), numericFloor)
                / max(float(np.abs(complexGain)), numericFloor)
            )
        )
    normalizedMseDb = 10.0 * np.log10(
        max(residualPower, numericFloor)
        / max(alignedReferencePower, numericFloor)
    )
    return ChannelDpdStageResult(
        stageName=stageName,
        methodDescription=methodDescription,
        evmDb=float(max(evmDbValues)),
        evmPercent=float(max(evmPercentValues)),
        normalizedMseDb=float(normalizedMseDb),
        aclrWorstDb=float(min(aclrValues)),
        residualCouplingDb=float(max(residualCouplingValues)),
    )


def BuildChannelDpdImprovements(
    stages: Sequence[ChannelDpdStageResult],
) -> Tuple[ChannelDpdImprovement, ...]:
    """Compare measured coupling-aware DPD with independent per-chain DPD.

    Processing details:
        Algorithm: Locate the named before and after stages, convert lower-
        is-better EVM/NMSE/leakage to consistently positive improvement
        values, and require those three coupling objectives to improve.  Keep
        the signed ACLR delta visible while applying a one-decibel regression
        guard because uncanceled same-channel leakage raises the independent
        stage's wanted-band denominator and is not an ACLR benefit. The three
        direct coupling objectives must still improve.

    Args:
        stages: Ordered benchmark stage results.

    Returns:
        result: Four auditable before/after expectation records.
    """

    stageByName = {stage.stageName: stage for stage in stages}
    beforeStage = stageByName["Independent DPD-GMP"]
    afterStage = stageByName["Coupling-aware DPD-GMP"]
    lowerMetricValues = (
        (
            "EVM dB",
            beforeStage.evmDb,
            afterStage.evmDb,
        ),
        (
            "Normalized MSE dB",
            beforeStage.normalizedMseDb,
            afterStage.normalizedMseDb,
        ),
        (
            "Residual coupling dB",
            beforeStage.residualCouplingDb,
            afterStage.residualCouplingDb,
        ),
    )
    improvements = [
        ChannelDpdImprovement(
            metricName=metricName,
            beforeValue=beforeValue,
            afterValue=afterValue,
            improvementValue=beforeValue - afterValue,
            expectedDirection="lower",
            expectationMet=afterValue < beforeValue,
        )
        for metricName, beforeValue, afterValue in lowerMetricValues
    ]
    maximumAllowedAclrRegressionDb = 1.0
    improvements.append(
        ChannelDpdImprovement(
            metricName="Worst ACLR dB",
            beforeValue=beforeStage.aclrWorstDb,
            afterValue=afterStage.aclrWorstDb,
            improvementValue=(
                afterStage.aclrWorstDb - beforeStage.aclrWorstDb
            ),
            expectedDirection="no more than 1.0 dB lower",
            expectationMet=(
                afterStage.aclrWorstDb
                >= beforeStage.aclrWorstDb
                - maximumAllowedAclrRegressionDb
            ),
        )
    )
    return tuple(improvements)


def EvaluateIqGmpStage(
    config: ChannelAnalysisBenchmarkConfig,
    waveform: WifiWaveform,
    iqPaModel: IQImbalancePA,
    outputPowerDbm: float,
    methodName: str,
    predistorter: Optional[DpdGmp] = None,
) -> IqGmpStageResult:
    """Evaluate one IQ-imbalance compensation method at equal output power.

    Processing details:
        Algorithm: Calibrate either the IQ-impaired PA or a DPD-plus-PA
        cascade to the requested native output power, retain the calibrated
        waveform as the ideal reference, and calculate EVM, IRR, and ACLR
        from the unscaled PA output through the common Analysis path.

    Args:
        config: Validated power, impedance, and interface controls.
        waveform: Common deterministic Wi-Fi waveform and metadata.
        iqPaModel: Widely linear IQ-impaired PA used by every method.
        outputPowerDbm: Requested native plant output power in dBm.
        methodName: Stable curve and table label.
        predistorter: Optional conventional or augmented GMP DPD.

    Returns:
        result: One comparable power, EVM, IRR, and ACLR record.
    """

    calibratedPlant: Any = (
        iqPaModel
        if predistorter is None
        else DpdGmpPaCascade(predistorter, iqPaModel)
    )
    powerCalibration = PowerCalibration(
        paModel=calibratedPlant,
        parameters={
            "loadResistanceOhm": config.loadResistanceOhm,
            "maximumOutputPowerDbm": config.maximumOutputPowerDbm,
            "outputPowerDbm": outputPowerDbm,
            "width": 0,
        },
    )
    referenceSignal = powerCalibration.Calibrate(
        waveform.samples
    )
    measuredSignal = powerCalibration.GetLastPaOutput()
    metrics = Analysis(
        referenceSignal,
        waveform,
        parameters={
            "maximumOutputPowerDbm": config.maximumOutputPowerDbm,
            "loadResistanceOhm": config.loadResistanceOhm,
            "width": 0,
        },
    ).Analyze(measuredSignal)
    return IqGmpStageResult(
        methodName=methodName,
        outputPowerDbm=float(outputPowerDbm),
        measuredOutputPowerDbm=metrics["outputPowerDbm"],
        evmDb=metrics["evmDb"],
        evmPercent=metrics["evmPercent"],
        irrDb=metrics["irrDb"],
        aclrWorstDb=metrics["aclrWorstDb"],
    )


def RunIqGmpPowerSweep(
    config: ChannelAnalysisBenchmarkConfig,
) -> Tuple[IqGmpStageResult, ...]:
    """Compare baseline, conventional GMP, and augmented GMP versus power.

    Processing details:
        Algorithm: Generate one Wi-Fi frame, construct a nearly linear PA
        followed by a known conjugate IQ image path, derive the exact
        widely-linear inverse labels, fit ordinary and augmented GMP models
        to the same labels, and evaluate all three methods at five equal
        native output-power points. The isolated linear PA makes any IRR
        difference attributable to image-model structure rather than PA
        compression.

    Args:
        config: Validated channel-analysis benchmark configuration.

    Returns:
        result: Method-major tuple containing all equal-power curve points.
    """

    waveform = WaveGenWifi(
        parameters={
            "frameFormat": config.frameFormat,
            "bandwidthMhz": config.bandwidthMhz,
            "sampleRateHz": config.sampleRateHz,
            "mcs": config.mcs,
            "numDataSymbols": config.numDataSymbols,
            "seed": config.seed + 211,
            "width": 0,
        }
    ).Generate()
    directCoefficient = 1.0 + 0.0j
    imageCoefficient = 0.08 * np.exp(1j * 0.40)
    linearPa = PaModel(
        wienerConfig=WienerConfig(
            linearTaps=(1.0 + 0.0j,),
            linearGain=1.0,
            saturationAmplitude=1000.0,
            rappSmoothness=3.0,
            ampmCoefficient=0.0,
        ),
        parameters={
            "modelName": "wiener",
            "width": 0,
        },
    )
    iqPaModel = IQImbalancePA(
        linearPa,
        directCoefficient=directCoefficient,
        imageCoefficient=imageCoefficient,
    )
    inverseDenominator = (
        np.abs(directCoefficient) ** 2
        - np.abs(imageCoefficient) ** 2
    )
    referenceSignal = np.asarray(
        waveform.samples, dtype=np.complex128
    )
    inverseLabels = (
        np.conj(directCoefficient) * referenceSignal
        - imageCoefficient * np.conj(referenceSignal)
    ) / inverseDenominator
    dpdParameters = {
        "nonlinearOrders": (1, 3),
        "memoryDepth": 2,
        "crossMemoryDepth": 1,
        "ridgeFactor": 1.0e-9,
        "maximumOutputMagnitude": None,
        "width": 0,
    }
    conventionalDpd = DpdGmp(parameters=dpdParameters)
    augmentedDpd = AugmentedDpdGmp(parameters=dpdParameters)
    conventionalDpd.Fit(referenceSignal, inverseLabels)
    augmentedDpd.Fit(referenceSignal, inverseLabels)
    highestPowerDbm = config.maximumOutputPowerDbm - 3.0
    lowestPowerDbm = highestPowerDbm - 14.0
    powerValuesDbm = tuple(
        float(powerValue)
        for powerValue in np.linspace(
            lowestPowerDbm,
            highestPowerDbm,
            5,
        )
    )
    stageResults = []
    for methodName, predistorter in (
        ("IQ-impaired PA", None),
        ("Conventional GMP", conventionalDpd),
        ("Augmented GMP", augmentedDpd),
    ):
        for outputPowerDbm in powerValuesDbm:
            stageResults.append(
                EvaluateIqGmpStage(
                    config,
                    waveform,
                    iqPaModel,
                    outputPowerDbm,
                    methodName,
                    predistorter,
                )
            )
    return tuple(stageResults)


def SaveChannelAnalysisResults(
    result: ChannelAnalysisBenchmarkResult,
    config: ChannelAnalysisBenchmarkConfig,
) -> None:
    """Save channel paths, frequency responses, DPD stages, JSON, and figure.

    Processing details:
        Algorithm: Create the output directory, serialize scalar summaries,
        flatten pre/post path and DPD comparison tables, save every occupied-
        band complex transfer entry as magnitude/phase CSV, and delegate the
        comparison figure to Draw without recalculating physical metrics.

    Args:
        result: Complete measured and compensated benchmark result.
        config: Validated output and reproducibility settings.

    Returns:
        result: None. JSON, CSV, and PNG artifacts are created.
    """

    outputDirectory = Path(config.outputDirectory)
    outputDirectory.mkdir(parents=True, exist_ok=True)
    with (
        outputDirectory / "channel_analysis.json"
    ).open("w", encoding="utf-8") as jsonFile:
        json.dump(
            {
                "configuration": {
                    "frameFormat": config.frameFormat,
                    "bandwidthMhz": config.bandwidthMhz,
                    "sampleRateHz": config.sampleRateHz,
                    "mcs": config.mcs,
                    "numDataSymbols": config.numDataSymbols,
                    "seed": config.seed,
                    "outputPowerDbm": config.outputPowerDbm,
                    "maximumOutputPowerDbm": (
                        config.maximumOutputPowerDbm
                    ),
                    "loadResistanceOhm": config.loadResistanceOhm,
                    "fftLength": config.fftLength,
                    "impulseLength": config.impulseLength,
                    "numIterations": config.numIterations,
                    "width": config.width,
                },
                **result.ToDict(),
            },
            jsonFile,
            ensure_ascii=False,
            indent=2,
        )
    pathRows = []
    for measurement in (
        result.prePaMeasurement,
        result.postPaMeasurement,
    ):
        for pathMeasurement in measurement.paths:
            pathRows.append(
                {
                    "stageName": measurement.stageName,
                    **pathMeasurement.ToDict(),
                }
            )
    with (
        outputDirectory / "channel_path_measurements.csv"
    ).open("w", newline="", encoding="utf-8-sig") as csvFile:
        csvWriter = csv.DictWriter(
            csvFile, fieldnames=list(pathRows[0].keys())
        )
        csvWriter.writeheader()
        csvWriter.writerows(pathRows)
    stageRows = [stage.ToDict() for stage in result.stages]
    with (
        outputDirectory / "channel_dpd_comparison.csv"
    ).open("w", newline="", encoding="utf-8-sig") as csvFile:
        csvWriter = csv.DictWriter(
            csvFile, fieldnames=list(stageRows[0].keys())
        )
        csvWriter.writeheader()
        csvWriter.writerows(stageRows)
    improvementRows = [
        improvement.ToDict()
        for improvement in result.improvements
    ]
    with (
        outputDirectory / "channel_dpd_improvements.csv"
    ).open("w", newline="", encoding="utf-8-sig") as csvFile:
        csvWriter = csv.DictWriter(
            csvFile, fieldnames=list(improvementRows[0].keys())
        )
        csvWriter.writeheader()
        csvWriter.writerows(improvementRows)
    iqStageRows = [
        stage.ToDict()
        for stage in result.iqImbalanceStages
    ]
    with (
        outputDirectory / "iq_gmp_comparison.csv"
    ).open("w", newline="", encoding="utf-8-sig") as csvFile:
        csvWriter = csv.DictWriter(
            csvFile, fieldnames=list(iqStageRows[0].keys())
        )
        csvWriter.writeheader()
        csvWriter.writerows(iqStageRows)
    frequencyRows = []
    for measurement in (
        result.prePaMeasurement,
        result.postPaMeasurement,
    ):
        occupiedMask = (
            np.abs(measurement.frequencyBinsHz)
            <= measurement.channelBandwidthHz / 2.0
        )
        for frequencyIndex in np.flatnonzero(occupiedMask):
            for destinationChain in range(
                measurement.frequencyResponse.shape[1]
            ):
                for sourceChain in range(
                    measurement.frequencyResponse.shape[2]
                ):
                    transferValue = measurement.frequencyResponse[
                        frequencyIndex,
                        destinationChain,
                        sourceChain,
                    ]
                    frequencyRows.append(
                        {
                            "stageName": measurement.stageName,
                            "frequencyHz": float(
                                measurement.frequencyBinsHz[
                                    frequencyIndex
                                ]
                            ),
                            "sourceChain": sourceChain,
                            "destinationChain": destinationChain,
                            "magnitudeDb": float(
                                20.0
                                * np.log10(
                                    max(
                                        float(np.abs(transferValue)),
                                        np.finfo(float).tiny,
                                    )
                                )
                            ),
                            "phaseDegrees": float(
                                np.degrees(np.angle(transferValue))
                            ),
                        }
                    )
    with (
        outputDirectory / "channel_frequency_response.csv"
    ).open("w", newline="", encoding="utf-8-sig") as csvFile:
        csvWriter = csv.DictWriter(
            csvFile, fieldnames=list(frequencyRows[0].keys())
        )
        csvWriter.writeheader()
        csvWriter.writerows(frequencyRows)
    Draw().SaveChannelAnalysis(
        {
            "pre-PA": result.prePaMeasurement,
            "post-PA": result.postPaMeasurement,
        },
        stageRows,
        outputDirectory,
    )
    Draw().SaveIqGmpComparison(
        iqStageRows,
        outputDirectory,
    )


def PrintChannelAnalysisResults(
    result: ChannelAnalysisBenchmarkResult,
) -> None:
    """Print compact channel measurements and DPD before/after comparisons.

    Processing details:
        Algorithm: Print the worst flatness, coupling, and condition number
        for both measured networks, then print each DPD stage and every
        expected improvement with an explicit pass/fail label.

    Args:
        result: Completed channel analysis benchmark.

    Returns:
        result: None. Human-readable results are written to standard output.
    """

    print("\nChannel measurements")
    for measurement in (
        result.prePaMeasurement,
        result.postPaMeasurement,
    ):
        print(
            f"{measurement.stageName:<9} "
            f"direct-flat={measurement.worstDirectFlatnessDb:>6.3f} dB  "
            f"path-flat={measurement.worstDetectedPathFlatnessDb:>6.3f} dB  "
            f"coupling={measurement.worstCouplingDb:>7.3f} dB  "
            f"cond={measurement.worstConditionNumber:>7.3f}"
        )
    print("\nCoupled DPD-GMP comparison")
    for stage in result.stages:
        print(
            f"{stage.stageName:<28} "
            f"EVM={stage.evmDb:>8.3f} dB  "
            f"NMSE={stage.normalizedMseDb:>8.3f} dB  "
            f"ACLR={stage.aclrWorstDb:>7.3f} dB  "
            f"leakage={stage.residualCouplingDb:>8.3f} dB"
        )
    print("\nExpected improvements")
    for improvement in result.improvements:
        status = "PASS" if improvement.expectationMet else "FAIL"
        print(
            f"{status:<4} {improvement.metricName:<24} "
            f"{improvement.improvementValue:>8.3f} dB"
        )
    print("\nIQ imbalance and augmented-GMP comparison")
    for stage in result.iqImbalanceStages:
        print(
            f"{stage.methodName:<18} "
            f"Pout={stage.measuredOutputPowerDbm:>6.2f} dBm  "
            f"EVM={stage.evmDb:>8.3f} dB  "
            f"IRR={stage.irrDb:>8.3f} dBc"
        )


def RunChannelAnalysisBenchmark(
    config: Optional[ChannelAnalysisBenchmarkConfig] = None,
) -> ChannelAnalysisBenchmarkResult:
    """Measure channel properties and verify coupling-aware DPD-GMP.

    Processing details:
        Algorithm: Construct one asymmetric two-PA coupled plant, measure its
        pre/post linear networks using orthogonal impulse probes, generate two
        independent equal-power Wi-Fi references, compare the raw PA,
        independently trained DPD, post-only de-embedding, and complete
        measured pre/post compensation, then save all values and figures.

    Args:
        config: Optional complete channel-analysis setup.

    Returns:
        result: Measurements, DPD stages, training diagnostics, and checks.
    """

    if config is None:
        config = ChannelAnalysisBenchmarkConfig()
    config.Validate()
    channel = BuildChannelAnalysisPlant(config)
    analyzer = ChannelAnalyse(
        parameters={
            "sampleRateHz": config.sampleRateHz,
            "channelBandwidthHz": config.bandwidthMhz * 1.0e6,
            "fftLength": config.fftLength,
            "impulseLength": config.impulseLength,
            "width": 0,
        }
    )
    prePaMeasurement = analyzer.Measure(
        channel.ApplyPrePaCoupling,
        2,
        "pre-PA",
    )
    postPaMeasurement = analyzer.Measure(
        channel.ApplyPostPaCoupling,
        2,
        "post-PA",
    )
    waveforms, referenceSignal = GenerateChannelAnalysisReferences(
        config
    )
    paBank = channel.paModel
    if not isinstance(paBank, MimoPaModel):
        raise TypeError(
            "channel analysis plant must bind a MimoPaModel"
        )
    paModels = tuple(paBank.paModels)

    baselineStage = EvaluateChannelDpdStage(
        config,
        channel,
        waveforms,
        referenceSignal,
        referenceSignal,
        "Coupled PA baseline",
        "No nonlinear or coupling compensation.",
    )

    independentModels = BuildChannelDpdModels(2)
    independentLabels = GenerateChannelPaInputLabels(
        config,
        referenceSignal,
        paModels,
    )
    for chainIndex, dpdModel in enumerate(independentModels):
        dpdModel.FitFromIlc(
            referenceSignal[:, chainIndex],
            independentLabels[:, chainIndex],
        )
    independentDacInput = np.column_stack(
        tuple(
            dpdModel.ProcessFloating(
                referenceSignal[:, chainIndex]
            )
            for chainIndex, dpdModel in enumerate(independentModels)
        )
    )
    independentStage = EvaluateChannelDpdStage(
        config,
        channel,
        waveforms,
        referenceSignal,
        independentDacInput,
        "Independent DPD-GMP",
        (
            "Per-chain PA inverse training without measured pre/post "
            "coupling de-embedding."
        ),
    )

    couplingAwareModels = BuildChannelDpdModels(2)
    couplingAwareDpd = CouplingAwareDpdGmp(
        couplingAwareModels,
        prePaMeasurement,
        postPaMeasurement,
        parameters={
            "compensatePrePaCoupling": True,
            "compensatePostPaCoupling": True,
            "inverseRegularization": 1.0e-8,
            "maximumInverseGainDb": 18.0,
            "width": 0,
        },
    )
    paOutputTargets = couplingAwareDpd.BuildPaOutputTargets(
        referenceSignal
    )
    couplingAwareLabels = GenerateChannelPaInputLabels(
        config,
        paOutputTargets,
        paModels,
    )
    trainingResult = couplingAwareDpd.FitCoupledSegments(
        (referenceSignal,),
        (couplingAwareLabels,),
    )

    postOnlyDpd = CouplingAwareDpdGmp(
        couplingAwareModels,
        prePaMeasurement,
        postPaMeasurement,
        parameters={
            "compensatePrePaCoupling": False,
            "compensatePostPaCoupling": True,
            "inverseRegularization": 1.0e-8,
            "maximumInverseGainDb": 18.0,
            "width": 0,
        },
    )
    postOnlyStage = EvaluateChannelDpdStage(
        config,
        channel,
        waveforms,
        referenceSignal,
        postOnlyDpd.ProcessFloating(referenceSignal),
        "Post-deembedded DPD-GMP",
        (
            "Measured post-PA target de-embedding without pre-PA "
            "DAC coupling cancellation."
        ),
    )
    couplingAwareStage = EvaluateChannelDpdStage(
        config,
        channel,
        waveforms,
        referenceSignal,
        couplingAwareDpd.ProcessFloating(referenceSignal),
        "Coupling-aware DPD-GMP",
        (
            "Measured post-PA target de-embedding, per-PA GMP fitting, "
            "and measured pre-PA DAC coupling cancellation."
        ),
    )
    stages = (
        baselineStage,
        independentStage,
        postOnlyStage,
        couplingAwareStage,
    )
    iqImbalanceStages = RunIqGmpPowerSweep(config)
    result = ChannelAnalysisBenchmarkResult(
        prePaMeasurement=prePaMeasurement,
        postPaMeasurement=postPaMeasurement,
        stages=stages,
        improvements=BuildChannelDpdImprovements(stages),
        trainingResult=trainingResult,
        iqImbalanceStages=iqImbalanceStages,
    )
    SaveChannelAnalysisResults(result, config)
    PrintChannelAnalysisResults(result)
    return result


def SaveHistory(
    methodName: str,
    analysisResult: ILCAnalysisResult,
    outputDirectory: Path,
) -> None:
    """Save a separate convergence CSV without overwriting other methods.

    Processing details:
        Algorithm: Sanitize the method label into a stable file stem, write
        every post-ILC performance record without recalculation, then ask
        ``Draw`` to render the same independently analyzed history.

    Args:
        methodName: Human-readable algorithm or deployment-model label.
        analysisResult: Post-ILC RF performance history and best candidate.
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
                "outputPowerDbm",
                "snrDb",
                "evmAlignedMse",
                "evmDb",
                "evmPercent",
                "aclrLowerDb",
                "aclrUpperDb",
                "aclrWorstDb",
                "complexGainMagnitudeDb",
                "complexGainPhaseDegrees",
                "feedbackIntegerDelaySamples",
                "feedbackFractionalDelaySamples",
                "feedbackCarrierFrequencyOffsetHz",
                "feedbackSamplingFrequencyOffsetPpm",
                "feedbackComplexGainMagnitudeDb",
                "feedbackComplexGainPhaseDegrees",
                "inputPeak",
            ),
        )
        csvWriter.writeheader()
        for iterationRecord in analysisResult.history:
            csvWriter.writerow(iterationRecord.ToDict())
    Draw(convergenceFileStem=f"convergence_{safeName}").SaveConvergenceCurve(
        analysisResult.history, outputDirectory
    )

def ReportHistory(
    methodName: str,
    ilcResult: ILCResult,
    resultAnalysis: Analysis,
    outputDirectory: Path,
) -> ILCAnalysisResult:
    """Print and save one method's complete per-iteration MSE history.

    Processing details:
        Algorithm: Feed the native PA observations into ``Analysis`` without
        post-PA amplitude normalization, select the EVM-best measured
        candidate outside the algorithm, then print, serialize, and render.

    Args:
        methodName: Human-readable ILC method label.
        ilcResult: Completed ILC result containing ordered history records.
        resultAnalysis: Analysis instance used for consistent presentation.
        outputDirectory: Destination for CSV and PNG result artifacts.
    Returns:
        result: Post-ILC history and the EVM-best measured candidate.
    """

    analysisResult = resultAnalysis.AnalyzeIlcHistory(
        ilcResult.history
    )
    resultAnalysis.PrintConvergence(
        analysisResult.history, f"{methodName} iteration metrics"
    )
    SaveHistory(methodName, analysisResult, outputDirectory)
    return analysisResult

def EvaluateDeployment(
    predistorter: Any,
    validationSignal: np.ndarray,
    paModel: Any,
    resultAnalysis: Analysis,
    maxAmplitude: float,
    powerCalibration: PowerCalibration,
    outputPowerDbm: float,
) -> SignalMetrics:
    """Evaluate one fitted DPD on a held-out Wi-Fi packet.

    Processing details:
        Algorithm: Run the fitted predistorter on an independent packet,
        project the complex envelope onto the configured amplitude disk, then
        let ``PowerCalibration`` repeatedly drive the unchanged PA until its
        measured active-burst output reaches the configured target.

    Args:
        predistorter: Fitted model exposing a ``Process`` method.
        validationSignal: Independent complex waveform used to evaluate generalization.
        paModel: PA object exposing Process and SmallSignalGain operations.
        resultAnalysis: Analyzer bound to the independent validation packet.
        maxAmplitude: Maximum allowed complex-envelope magnitude.
        powerCalibration: Shared floating/fixed active-burst calibrator.
        outputPowerDbm: Scenario conducted output-power target.

    Returns:
        result: SNR, EVM, and ACLR of the held-out PA output.
    """

    interfaceFormat = FixedPoint(int(getattr(paModel, "width", 0)))
    floatingValidationSignal = interfaceFormat.DecodeComplex(
        validationSignal
    )
    floatingPredistortedInput = LimitAmplitude(
        predistorter.Process(floatingValidationSignal), maxAmplitude
    )
    predistortedInput = interfaceFormat.EncodeComplex(
        floatingPredistortedInput
    )
    powerCalibration.UpdateParameters(outputPowerDbm=outputPowerDbm)
    powerCalibration.SetPaModel(paModel)
    powerCalibration.Calibrate(predistortedInput)
    paOutput = powerCalibration.GetLastPaOutput()
    return resultAnalysis.Analyze(paOutput)

def RunIlcCurvePoint(
    referenceSignal: np.ndarray,
    outputPowerDbm: float,
    paModel: Any,
    waveform: WifiWaveform,
    width: int,
    maximumOutputPowerDbm: float,
    methodName: str,
    methodFunction: Optional[Callable[..., ILCResult]],
    methodConfig: ILCConfig,
) -> np.ndarray:
    """Run one selected ILC method at one power-EVM sweep point.

    Processing details:
        Algorithm: Run ILC without any RF metric callback, analyze every
        native PA output without post-PA scaling, and return the externally
        selected minimum-EVM measured candidate. The outer power-sweep
        calibrator changes ``referenceSignal`` and reruns this complete plant.

    Args:
        referenceSignal: Ideal complex baseband samples used as the target or regression input.
        outputPowerDbm: Current absolute PA output power in dBm.
        paModel: PA object exposing Process and SmallSignalGain operations.
        waveform: Wi-Fi metadata defining field locations, FFT sizes, and subcarriers.
        width: External I/Q component width shared by all benchmark modules.
        maximumOutputPowerDbm: Rated normalized full-scale output power.
        methodName: Human-readable algorithm or deployment-model label.
        methodFunction: Selected ILC update-law callable. ``None`` selects
            the dedicated frequency-domain call path, which also needs the
            waveform sample rate and occupied bandwidth.
        methodConfig: Validated ILC configuration for the selected update law.

    Returns:
        result: Complex PA output learned specifically for this power point.
    """

    pointAnalysis = Analysis(
        referenceSignal,
        waveform,
        parameters={
            "maximumOutputPowerDbm": maximumOutputPowerDbm,
            "width": width,
            "outputFullScaleAmplitude": (
                ResolveBenchmarkOutputFullScaleAmplitude(paModel)
            ),
        },
    )
    if methodName == "Frequency-domain ILC":
        methodResult = RunFrequencyDomainIlc(
            referenceSignal,
            paModel,
            waveform.sampleRateHz,
            waveform.bandwidthHz,
            methodConfig,
        )
    else:
        if methodFunction is None:
            raise ValueError(
                "methodFunction is required for non-frequency ILC"
            )
        methodResult = methodFunction(
            referenceSignal,
            paModel,
            methodConfig,
        )
    analysisResult = pointAnalysis.AnalyzeIlcHistory(
        methodResult.history
    )
    return paModel.Process(analysisResult.bestInputSignal)


def RunTwoToneIlcBenchmark(
    config: Optional[TwoToneBenchmarkConfig] = None,
) -> List[TwoToneBenchmarkRow]:
    """Compare every applicable SISO ILC method using IM3, IM5, and IM7.

    Processing details:
        Algorithm: Generate one deterministic complex two-tone record, close
        the PA input-power loop to a common actual output dBm, run scalar,
        complex-gain, FIR, frequency-domain, directional Gauss-Newton,
        parameter-domain, and augmented-IQ ILC under a shared iteration budget,
        independently select each method's best native iteration by the largest
        remaining IM product, recalibrate that learned waveform to equal PA
        output power, and save tabular plus graphical comparisons.

    Args:
        config: Optional two-tone scenario configuration. None creates defaults
            inside this function.

    Returns:
        result: Ordered baseline and all-method two-tone benchmark rows.
    """

    if config is None:
        config = TwoToneBenchmarkConfig()
    config.Validate()
    outputDirectory = Path(config.outputDirectory)
    outputDirectory.mkdir(parents=True, exist_ok=True)
    waveform = WaveGenTwoTone(
        parameters={
            "sampleRateHz": config.sampleRateHz,
            "toneFrequenciesHz": config.toneFrequenciesHz,
            "toneAmplitudes": config.toneAmplitudes,
            "tonePhasesDegrees": config.tonePhasesDegrees,
            "numSamples": config.numSamples,
            "rmsLevel": config.rmsLevel,
            "width": config.width,
        }
    ).Generate()
    paModel = PaModel(
        parameters={
            "modelName": config.paModelName,
            "width": config.width,
        }
    )
    powerCalibration = PowerCalibration(
        paModel=paModel,
        parameters={
            "loadResistanceOhm": config.loadResistanceOhm,
            "maximumOutputPowerDbm": config.maximumOutputPowerDbm,
            "outputPowerDbm": config.outputPowerDbm,
            "width": config.width,
        },
    )
    referenceSignal = powerCalibration.Calibrate(waveform.samples)
    baselineOutput = powerCalibration.GetLastPaOutput()
    resultAnalysis = TwoToneAnalysis(
        waveform,
        parameters={
            "loadResistanceOhm": config.loadResistanceOhm,
            "maximumOutputPowerDbm": config.maximumOutputPowerDbm,
            "width": config.width,
            "outputFullScaleAmplitude": (
                powerCalibration.outputFullScaleAmplitude
            ),
        },
    )
    baselineMetrics = resultAnalysis.Analyze(baselineOutput)
    rows: List[TwoToneBenchmarkRow] = []
    AddTwoToneRow(
        rows,
        "PA baseline",
        "A: unlinearized baseline",
        baselineMetrics,
        baselineMetrics,
    )
    floatingReference = FixedPoint(config.width).DecodeComplex(
        referenceSignal
    )
    maxAmplitude = max(
        1.2,
        1.8 * float(np.max(np.abs(floatingReference))),
    )
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
    augmentedConfig = ILCConfig(
        numIterations=config.numIterations,
        learningRate=0.15,
        maxAmplitude=maxAmplitude,
        randomSeed=config.seed + 7,
    )
    methodRuns: tuple = (
        (
            "Scalar P ILC",
            lambda targetSignal, plant, selectedConfig: RunScalarPIlc(
                targetSignal,
                plant,
                selectedConfig,
                waveform.sampleRateHz,
            ),
            scalarConfig,
        ),
        (
            "Complex-gain ILC",
            lambda targetSignal, plant, selectedConfig: RunComplexGainIlc(
                targetSignal,
                plant,
                selectedConfig,
                waveform.sampleRateHz,
            ),
            complexConfig,
        ),
        (
            "FIR ILC",
            lambda targetSignal, plant, selectedConfig: RunFirIlc(
                targetSignal,
                plant,
                selectedConfig,
                17,
                waveform.sampleRateHz,
            ),
            firConfig,
        ),
        (
            "Frequency-domain ILC",
            lambda targetSignal, plant, selectedConfig: RunFrequencyDomainIlc(
                targetSignal,
                plant,
                waveform.sampleRateHz,
                waveform.ilcBandwidthHz,
                selectedConfig,
            ),
            frequencyConfig,
        ),
        (
            "Directional Gauss-Newton ILC",
            lambda targetSignal,
            plant,
            selectedConfig: RunDirectionalGaussNewtonIlc(
                targetSignal,
                plant,
                selectedConfig,
                1.0e-3,
                waveform.sampleRateHz,
            ),
            gaussNewtonConfig,
        ),
        (
            "Parameter-domain MP ILC",
            lambda targetSignal, plant, selectedConfig: RunParameterDomainIlc(
                targetSignal,
                plant,
                selectedConfig,
                (1, 3, 5, 7),
                3,
                waveform.sampleRateHz,
            ),
            parameterConfig,
        ),
        (
            "Augmented IQ ILC",
            lambda targetSignal, plant, selectedConfig: RunAugmentedIqIlc(
                targetSignal,
                plant,
                selectedConfig,
                waveform.sampleRateHz,
            ),
            augmentedConfig,
        ),
    )
    for methodName, methodRunner, methodConfig in methodRuns:
        methodResult: ILCResult = methodRunner(
            referenceSignal,
            paModel,
            methodConfig,
        )
        methodAnalysis: TwoToneILCAnalysisResult = (
            resultAnalysis.AnalyzeIlcHistory(methodResult.history)
        )
        historyStem = (
            methodName.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )
        resultAnalysis.SaveIlcHistory(
            methodAnalysis,
            outputDirectory / "histories",
            historyStem,
        )
        powerCalibration.Calibrate(methodAnalysis.bestInputSignal)
        equalPowerOutput = powerCalibration.GetLastPaOutput()
        methodMetrics = resultAnalysis.Analyze(equalPowerOutput)
        AddTwoToneRow(
            rows,
            methodName,
            "B: applicable SISO ILC methods",
            methodMetrics,
            baselineMetrics,
        )
    metadata = {
        "sampleRateHz": waveform.sampleRateHz,
        "toneFrequenciesHz": list(waveform.toneFrequenciesHz),
        "toneAmplitudes": list(waveform.toneAmplitudes),
        "tonePhasesDegrees": list(waveform.tonePhasesDegrees),
        "numSamples": waveform.numSamples,
        "rmsLevel": waveform.rmsLevel,
        "width": waveform.width,
        "ilcBandwidthHz": waveform.ilcBandwidthHz,
        "outputPowerDbm": config.outputPowerDbm,
        "maximumOutputPowerDbm": config.maximumOutputPowerDbm,
        "loadResistanceOhm": config.loadResistanceOhm,
        "numIterations": config.numIterations,
        "paModel": config.paModelName,
        "seed": config.seed,
        "selectionMetric": "minimum worstIntermodulationDbc",
        "powerCalibrationMode": "closedLoopInputDrive",
    }
    SaveTwoToneBenchmarkResults(rows, outputDirectory, metadata)
    metricsByMethod = {
        row.methodName: row.metrics
        for row in rows
    }
    Draw().SaveTwoToneImdComparison(
        metricsByMethod,
        outputDirectory,
        fileStem="all_ilc_two_tone_imd",
    )
    PrintTwoToneBenchmarkResults(rows)
    return rows


def SaveTwoToneBenchmarkResults(
    rows: List[TwoToneBenchmarkRow],
    outputDirectory: Path,
    metadata: Mapping[str, object],
) -> Tuple[Path, Path]:
    """Save all-method two-tone results as matching CSV and JSON files.

    Processing details:
        Algorithm: Flatten each immutable row exactly once, create the output
        directory, write a stable CSV table, and serialize identical rows with
        reproducibility metadata in structured JSON.

    Args:
        rows: Nonempty ordered two-tone result rows.
        outputDirectory: Destination directory for both summary files.
        metadata: Exact waveform, PA, ILC, and calibration settings.

    Returns:
        result: CSV path followed by JSON path.
    """

    if not rows:
        raise ValueError("rows cannot be empty")
    outputPath = Path(outputDirectory)
    outputPath.mkdir(parents=True, exist_ok=True)
    csvPath = outputPath / "all_ilc_two_tone_metrics.csv"
    jsonPath = outputPath / "all_ilc_two_tone_metrics.json"
    flatRows = [row.ToDict() for row in rows]
    with csvPath.open("w", newline="", encoding="utf-8-sig") as csvFile:
        csvWriter = csv.DictWriter(
            csvFile, fieldnames=list(flatRows[0].keys())
        )
        csvWriter.writeheader()
        csvWriter.writerows(flatRows)
    with jsonPath.open("w", encoding="utf-8") as jsonFile:
        json.dump(
            {"metadata": dict(metadata), "results": flatRows},
            jsonFile,
            ensure_ascii=False,
            indent=2,
        )
    return csvPath, jsonPath


def PrintTwoToneBenchmarkResults(
    rows: List[TwoToneBenchmarkRow],
) -> None:
    """Print a compact all-method IM3, IM5, and IM7 comparison table.

    Processing details:
        Algorithm: Preserve benchmark row order and print worse-side absolute
        dBc plus positive-is-better suppression improvements for each odd order
        using fixed-width columns.

    Args:
        rows: Ordered baseline and ILC result rows.

    Returns:
        result: None. The comparison table is written to standard output.
    """

    header = (
        f"{'Method':<32} {'IM3':>9} {'IM5':>9} {'IM7':>9} "
        f"{'dIM3':>8} {'dIM5':>8} {'dIM7':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.methodName:<32} "
            f"{row.metrics['im3WorstDbc']:>9.2f} "
            f"{row.metrics['im5WorstDbc']:>9.2f} "
            f"{row.metrics['im7WorstDbc']:>9.2f} "
            f"{row.im3ImprovementDb:>8.2f} "
            f"{row.im5ImprovementDb:>8.2f} "
            f"{row.im7ImprovementDb:>8.2f}"
        )


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
        "width": config.width,
    }
    if config.sampleRateHz is None:
        sharedWifiParameters["oversampling"] = config.oversampling
    else:
        sharedWifiParameters["sampleRateHz"] = config.sampleRateHz
    trainingParameters = dict(sharedWifiParameters)
    trainingParameters["seed"] = config.seed
    validationParameters = dict(sharedWifiParameters)
    validationParameters["seed"] = config.seed + 97
    trainingGenerator = WaveGenWifi(parameters=trainingParameters)
    validationGenerator = WaveGenWifi(parameters=validationParameters)
    trainingWaveform = trainingGenerator.Generate()
    validationWaveform = validationGenerator.Generate()
    interfaceFormat = FixedPoint(config.width)
    paParameters = {
        "modelName": config.paModelName,
        "width": config.width,
    }
    paModel = PaModel(parameters=paParameters)
    powerCalibration = PowerCalibration(
        paModel=paModel,
        parameters={
            "loadResistanceOhm": config.loadResistanceOhm,
            "maximumOutputPowerDbm": config.maximumOutputPowerDbm,
            "outputPowerDbm": config.outputPowerDbm,
            "width": config.width,
        },
    )
    trainingSignal = powerCalibration.Calibrate(
        trainingWaveform.samples
    )
    baselineOutput = powerCalibration.GetLastPaOutput()
    validationSignal = powerCalibration.Calibrate(
        validationWaveform.samples
    )
    validationBaselineOutput = (
        powerCalibration.GetLastPaOutput()
    )
    floatingTrainingSignal = interfaceFormat.DecodeComplex(trainingSignal)
    trainingAnalysis = Analysis(
        trainingSignal,
        trainingWaveform,
        parameters={
            "loadResistanceOhm": config.loadResistanceOhm,
            "maximumOutputPowerDbm": config.maximumOutputPowerDbm,
            "width": config.width,
            "outputFullScaleAmplitude": (
                powerCalibration.outputFullScaleAmplitude
            ),
        },
    )
    validationAnalysis = Analysis(
        validationSignal,
        validationWaveform,
        parameters={
            "loadResistanceOhm": config.loadResistanceOhm,
            "maximumOutputPowerDbm": config.maximumOutputPowerDbm,
            "width": config.width,
            "outputFullScaleAmplitude": (
                powerCalibration.outputFullScaleAmplitude
            ),
        },
    )
    maxAmplitude = max(
        2.0, 1.6 * np.max(np.abs(floatingTrainingSignal))
    )

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
    frequencyAnalysisResult = None
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
                trainingSignal,
                paModel,
                methodConfig,
            )
        methodAnalysisResult = ReportHistory(
            methodName,
            methodResult,
            trainingAnalysis,
            outputDirectory,
        )
        powerCalibration.Calibrate(
            methodAnalysisResult.bestInputSignal
        )
        methodOutput = powerCalibration.GetLastPaOutput()
        methodMetrics = trainingAnalysis.Analyze(methodOutput)
        if methodName == "Frequency-domain ILC":
            frequencyAnalysisResult = methodAnalysisResult
        AddRow(
            rows,
            methodName,
            "ILC update law",
            "nominal repeated waveform",
            methodMetrics,
            baselineMetrics,
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
                config.width,
                config.maximumOutputPowerDbm,
                selectedName,
                selectedFunction,
                selectedConfig,
            )
        )

    if frequencyResult is None or frequencyAnalysisResult is None:
        raise RuntimeError("frequency-domain ILC result was not generated")

    # Repeat the physically identical baseline and unconstrained result under
    # the peak scenario label so scenario-filtered reports contain the full
    # baseline-versus-performance-versus-feasibility comparison.
    powerCalibration.Calibrate(
        frequencyAnalysisResult.bestInputSignal
    )
    frequencyOutput = powerCalibration.GetLastPaOutput()
    frequencyMetrics = trainingAnalysis.Analyze(frequencyOutput)
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
    constrainedPeak = 1.05 * np.max(
        np.abs(floatingTrainingSignal)
    )
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
    constrainedAnalysisResult = ReportHistory(
        "Constrained CFR-ILC",
        constrainedResult,
        trainingAnalysis,
        outputDirectory,
    )
    powerCalibration.Calibrate(
        constrainedAnalysisResult.bestInputSignal
    )
    constrainedOutput = powerCalibration.GetLastPaOutput()
    constrainedMetrics = trainingAnalysis.Analyze(constrainedOutput)
    AddRow(
        rows,
        "Constrained CFR-ILC",
        "ILC update law",
        "peak-constrained waveform",
        constrainedMetrics,
        baselineMetrics,
    )
    powerEvaluators["Constrained CFR-ILC"] = (
        lambda pointReference, pointDrive: RunIlcCurvePoint(
            pointReference,
            pointDrive,
            paModel,
            trainingWaveform,
            config.width,
            config.maximumOutputPowerDbm,
            "Frequency-domain ILC",
            None,
            ILCConfig(
                numIterations=config.numIterations,
                learningRate=0.12,
                maxAmplitude=1.05
                * np.max(
                    np.abs(
                        interfaceFormat.DecodeComplex(pointReference)
                    )
                ),
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
        ),
    )
    naiveNoisyAnalysisResult = ReportHistory(
        "Naive noisy-feedback ILC",
        naiveNoisyResult,
        trainingAnalysis,
        outputDirectory,
    )
    powerCalibration.Calibrate(
        naiveNoisyAnalysisResult.bestInputSignal
    )
    naiveNoisyOutput = powerCalibration.GetLastPaOutput()
    naiveNoisyMetrics = trainingAnalysis.Analyze(naiveNoisyOutput)
    AddRow(
        rows,
        "Naive noisy-feedback ILC",
        "ILC update law",
        "32 dB feedback robustness",
        naiveNoisyMetrics,
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
    noiseAwareAnalysisResult = ReportHistory(
        "Noise-aware ILC",
        noiseAwareResult,
        trainingAnalysis,
        outputDirectory,
    )
    powerCalibration.Calibrate(
        noiseAwareAnalysisResult.bestInputSignal
    )
    noiseAwareOutput = powerCalibration.GetLastPaOutput()
    noiseAwareMetrics = trainingAnalysis.Analyze(noiseAwareOutput)
    AddRow(
        rows,
        "Noise-aware ILC",
        "ILC update law",
        "32 dB feedback robustness",
        noiseAwareMetrics,
        noisyBaselineMetrics,
    )
    powerEvaluators["Naive noisy-feedback ILC"] = (
        lambda pointReference, pointDrive: RunIlcCurvePoint(
            pointReference,
            pointDrive,
            paModel,
            trainingWaveform,
            config.width,
            config.maximumOutputPowerDbm,
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
            config.width,
            config.maximumOutputPowerDbm,
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
    powerCalibration.SetPaModel(iqPaModel)
    powerCalibration.Calibrate(trainingSignal)
    iqBaselineOutput = powerCalibration.GetLastPaOutput()
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
        ),
    )
    ordinaryIqAnalysisResult = ReportHistory(
        "Frequency-domain ILC on IQ plant",
        ordinaryIqResult,
        trainingAnalysis,
        outputDirectory,
    )
    powerCalibration.Calibrate(
        ordinaryIqAnalysisResult.bestInputSignal
    )
    ordinaryIqOutput = powerCalibration.GetLastPaOutput()
    ordinaryIqMetrics = trainingAnalysis.Analyze(ordinaryIqOutput)
    AddRow(
        rows,
        "Frequency-domain ILC on IQ plant",
        "ILC update law",
        "IQ image impairment",
        ordinaryIqMetrics,
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
    augmentedAnalysisResult = ReportHistory(
        "Augmented IQ ILC",
        augmentedResult,
        trainingAnalysis,
        outputDirectory,
    )
    powerCalibration.Calibrate(
        augmentedAnalysisResult.bestInputSignal
    )
    augmentedOutput = powerCalibration.GetLastPaOutput()
    augmentedMetrics = trainingAnalysis.Analyze(augmentedOutput)
    AddRow(
        rows,
        "Augmented IQ ILC",
        "ILC update law",
        "IQ image impairment",
        augmentedMetrics,
        iqBaselineMetrics,
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
            config.width,
            config.maximumOutputPowerDbm,
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
            config.width,
            config.maximumOutputPowerDbm,
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
    powerCalibration.SetPaModel(paModel)
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
                floatingTrainingSignal,
                interfaceFormat.DecodeComplex(
                    frequencyAnalysisResult.bestInputSignal
                ),
                nonlinearOrders=(1, 3, 5, 7),
                memoryDepth=3,
                crossMemoryDepth=0,
            ),
        ),
        (
            "ILC label + GMP",
            FitGmpPredistorter(
                floatingTrainingSignal,
                interfaceFormat.DecodeComplex(
                    frequencyAnalysisResult.bestInputSignal
                ),
                nonlinearOrders=(1, 3, 5, 7),
                memoryDepth=3,
                crossMemoryDepth=2,
            ),
        ),
        (
            "ILC label + Volterra",
            FitVolterraPredistorter(
                floatingTrainingSignal,
                interfaceFormat.DecodeComplex(
                    frequencyAnalysisResult.bestInputSignal
                ),
                memoryDepth=3,
            ),
        ),
        (
            "ILC label + LUT",
            FitLutPredistorter(
                floatingTrainingSignal,
                interfaceFormat.DecodeComplex(
                    frequencyAnalysisResult.bestInputSignal
                ),
                binCount=64,
            ),
        ),
        (
            "ILC label + NN",
            FitNeuralPredistorter(
                floatingTrainingSignal,
                interfaceFormat.DecodeComplex(
                    frequencyAnalysisResult.bestInputSignal
                ),
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
            powerCalibration,
            config.outputPowerDbm,
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
                interfaceFormat.EncodeComplex(
                    LimitAmplitude(
                        selectedPredistorter.Process(
                            interfaceFormat.DecodeComplex(
                                pointReference
                            )
                        ),
                        maxAmplitude,
                    )
                )
            )
        )
    metadata: Mapping[str, object] = {
        "frameFormat": trainingWaveform.frameFormat,
        "bandwidthMhz": config.bandwidthMhz,
        "sampleRateHz": trainingWaveform.sampleRateHz,
        "mcs": config.mcs,
        "numDataSymbols": config.numDataSymbols,
        "oversampling": trainingWaveform.oversampling,
        "guardIntervalUs": config.guardIntervalUs,
        "outputPowerDbm": config.outputPowerDbm,
        "targetOutputRmsVoltage": powerCalibration.DbmToRms(
            config.outputPowerDbm
        ),
        "powerCalibrationMode": "closedLoopInputDrive",
        "maximumOutputPowerDbm": config.maximumOutputPowerDbm,
        "loadResistanceOhm": config.loadResistanceOhm,
        "numIterations": config.numIterations,
        "paModel": config.paModelName,
        "trainingSeed": config.seed,
        "validationSeed": config.seed + 97,
        "outputPowerStartDbm": config.powerStartDbm,
        "outputPowerStopDbm": config.powerStopDbm,
        "powerPointCount": config.powerPointCount,
        "generatePowerEvmCurve": config.generatePowerEvmCurve,
    }
    SaveBenchmarkResults(rows, outputDirectory, metadata)
    powerCurvePaths = None
    if config.generatePowerEvmCurve:
        outputPowerDbmValues = np.linspace(
            config.powerStartDbm,
            config.powerStopDbm,
            config.powerPointCount,
        )
        powerEvmCurve = trainingAnalysis.AnalyzePowerEvmCurve(
            outputPowerDbmValues, powerEvaluators
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
            f"{row.metrics['snrDb']:>8.2f} "
            f"{row.metrics['evmPercent']:>9.3f} "
            f"{row.metrics['aclrWorstDb']:>9.2f} "
            f"{row.evmImprovementDb:>8.2f}"
        )


@dataclass(frozen=True)
class DpdLmsBenchmarkConfig:
    """Configure deterministic stationary and drifting GMP-label tests."""

    numSamples: int = 8192
    seed: int = 907
    learningRate: float = 0.10
    outputDirectory: Path = Path("results/dpd_lms_benchmark")

    def Validate(self) -> None:
        """Validate sample count, seed, step size, and output path.

        Processing details:
            Algorithm: Require enough chronological samples to expose LMS
            convergence, an integer random seed, a finite NLMS step in the
            theoretical open interval from zero to two, and a Path output.

        Returns:
            result: None. Invalid benchmark settings raise an exception.
        """

        if (
            not isinstance(self.numSamples, int)
            or isinstance(self.numSamples, bool)
            or self.numSamples < 512
        ):
            raise ValueError("numSamples must be an integer at least 512")
        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
        ):
            raise TypeError("seed must be an integer")
        if (
            not isinstance(self.learningRate, (int, float))
            or isinstance(self.learningRate, bool)
            or not np.isfinite(self.learningRate)
            or not 0.0 < float(self.learningRate) < 2.0
        ):
            raise ValueError("learningRate must be finite in (0, 2)")
        if not isinstance(self.outputDirectory, Path):
            raise TypeError("outputDirectory must be a pathlib.Path")


@dataclass(frozen=True)
class DpdLmsBenchmarkResult:
    """Store batch, sample-adaptive, and PA-drift tracking comparisons."""

    sampleCount: int
    updateCountPerFrame: int
    batchStationaryNmseDb: float
    lmsStationaryNmseDb: float
    staleBatchDriftNmseDb: float
    lmsBeforeTrackingNmseDb: float
    lmsAfterTrackingNmseDb: float
    trackingImprovementDb: float

    def ToDict(self) -> Dict[str, object]:
        """Return one flat auditable LMS comparison record.

        Processing details:
            Algorithm: Copy immutable counts and NMSE values without
            retraining or changing improvement signs.

        Returns:
            result: Serialization-ready benchmark dictionary.
        """

        return {
            "sampleCount": self.sampleCount,
            "updateCountPerFrame": self.updateCountPerFrame,
            "batchStationaryNmseDb": self.batchStationaryNmseDb,
            "lmsStationaryNmseDb": self.lmsStationaryNmseDb,
            "staleBatchDriftNmseDb": self.staleBatchDriftNmseDb,
            "lmsBeforeTrackingNmseDb": (
                self.lmsBeforeTrackingNmseDb
            ),
            "lmsAfterTrackingNmseDb": self.lmsAfterTrackingNmseDb,
            "trackingImprovementDb": self.trackingImprovementDb,
        }


def BuildDpdLmsTarget(
    referenceSignal: np.ndarray,
    linearCoefficient: complex,
    cubicCoefficient: complex,
) -> np.ndarray:
    """Create one known memoryless GMP label stream.

    Processing details:
        Algorithm: Apply a complex first-order term plus a complex cubic
        envelope term so both batch regression and chronological NLMS see the
        same exact two-feature target without PA or synchronization ambiguity.

    Args:
        referenceSignal: Normalized finite complex training vector.
        linearCoefficient: Known coefficient multiplying the direct sample.
        cubicCoefficient: Known coefficient multiplying x times magnitude
            squared.

    Returns:
        result: Exact complex target vector for the controlled benchmark.
    """

    complexReference = np.asarray(
        referenceSignal, dtype=np.complex128
    ).reshape(-1)
    return (
        complex(linearCoefficient) * complexReference
        + complex(cubicCoefficient)
        * complexReference
        * np.abs(complexReference) ** 2
    )


def SaveDpdLmsBenchmarkResult(
    result: DpdLmsBenchmarkResult,
    config: DpdLmsBenchmarkConfig,
) -> Tuple[Path, Path]:
    """Save identical LMS benchmark values to CSV and JSON.

    Processing details:
        Algorithm: Create the configured directory, flatten the immutable
        result once, write one CSV row, and store configuration plus the same
        result dictionary in JSON for reproducible comparison.

    Args:
        result: Completed stationary and drift benchmark.
        config: Exact controls used to generate the result.

    Returns:
        result: CSV and JSON artifact paths in that order.
    """

    outputDirectory = Path(config.outputDirectory)
    outputDirectory.mkdir(parents=True, exist_ok=True)
    csvPath = outputDirectory / "dpd_lms_benchmark.csv"
    jsonPath = outputDirectory / "dpd_lms_benchmark.json"
    resultRow = result.ToDict()
    with csvPath.open("w", newline="", encoding="utf-8-sig") as csvFile:
        csvWriter = csv.DictWriter(
            csvFile,
            fieldnames=list(resultRow),
        )
        csvWriter.writeheader()
        csvWriter.writerow(resultRow)
    with jsonPath.open("w", encoding="utf-8") as jsonFile:
        json.dump(
            {
                "configuration": {
                    "numSamples": config.numSamples,
                    "seed": config.seed,
                    "learningRate": config.learningRate,
                },
                "result": resultRow,
            },
            jsonFile,
            indent=2,
            ensure_ascii=False,
        )
    return csvPath, jsonPath


def PrintDpdLmsBenchmarkResult(
    result: DpdLmsBenchmarkResult,
) -> None:
    """Print batch accuracy and sample-adaptive drift tracking.

    Processing details:
        Algorithm: Display the stored stationary NMSE values, stale-model
        drift errors, post-tracking error, update count, and positive tracking
        improvement without recalculating any metric.

    Args:
        result: Completed LMS benchmark result.

    Returns:
        result: None. A compact deterministic table is printed.
    """

    print("\nDPD-LMS sample-update benchmark")
    print(
        f"Batch stationary NMSE:       "
        f"{result.batchStationaryNmseDb:9.3f} dB"
    )
    print(
        f"NLMS stationary NMSE:        "
        f"{result.lmsStationaryNmseDb:9.3f} dB"
    )
    print(
        f"Stale batch drift NMSE:       "
        f"{result.staleBatchDriftNmseDb:9.3f} dB"
    )
    print(
        f"NLMS before tracking NMSE:    "
        f"{result.lmsBeforeTrackingNmseDb:9.3f} dB"
    )
    print(
        f"NLMS after tracking NMSE:     "
        f"{result.lmsAfterTrackingNmseDb:9.3f} dB"
    )
    print(
        f"Tracking improvement:         "
        f"{result.trackingImprovementDb:9.3f} dB"
    )
    print(
        f"Coefficient updates per frame: "
        f"{result.updateCountPerFrame}"
    )


def RunDpdLmsBenchmark(
    config: Optional[DpdLmsBenchmarkConfig] = None,
) -> DpdLmsBenchmarkResult:
    """Compare batch GMP fitting with chronological NLMS and drift tracking.

    Processing details:
        Algorithm: Generate one deterministic normalized complex frame, build
        exact stationary and changed two-term GMP labels, fit a batch ridge
        model and one sample-adaptive model on the stationary frame, measure
        both before PA drift, evaluate stale coefficients on changed labels,
        run one chronological NLMS tracking pass, require material improvement,
        then save and print the auditable result.

    Args:
        config: Optional benchmark controls. None uses internal defaults.

    Returns:
        result: Stationary accuracy and drift-tracking NMSE comparison.
    """

    if config is None:
        config = DpdLmsBenchmarkConfig()
    config.Validate()
    randomGenerator = np.random.default_rng(config.seed)
    referenceSignal = (
        randomGenerator.standard_normal(config.numSamples)
        + 1j
        * randomGenerator.standard_normal(config.numSamples)
    )
    referenceSignal *= 0.25 / np.sqrt(
        np.mean(np.abs(referenceSignal) ** 2)
    )
    stationaryTarget = BuildDpdLmsTarget(
        referenceSignal,
        1.03 + 0.01j,
        0.18 - 0.04j,
    )
    driftTarget = BuildDpdLmsTarget(
        referenceSignal,
        0.97 - 0.02j,
        0.30 + 0.06j,
    )
    commonParameters = {
        "nonlinearOrders": (1, 3),
        "memoryDepth": 1,
        "crossMemoryDepth": 0,
        "maximumOutputMagnitude": None,
        "width": 0,
    }
    batchDpd = DpdGmp(
        parameters={
            **commonParameters,
            "ridgeFactor": 1.0e-10,
        }
    )
    batchDpd.Fit(referenceSignal, stationaryTarget)
    batchStationaryNmseDb = batchDpd.CalculateNmse(
        referenceSignal,
        stationaryTarget,
    )
    staleBatchDriftNmseDb = batchDpd.CalculateNmse(
        referenceSignal,
        driftTarget,
    )
    lmsDpd = DpdLms(
        parameters={
            **commonParameters,
            "adaptationMode": "nlms",
            "learningRate": config.learningRate,
            "featureScaleMode": "frame",
            "coefficientCommitMode": "frame",
            "leakageFactor": 0.0,
        }
    )
    stationaryTraining = lmsDpd.UpdateFromLabels(
        referenceSignal,
        stationaryTarget,
    )
    lmsStationaryNmseDb = lmsDpd.CalculateNmse(
        referenceSignal,
        stationaryTarget,
    )
    lmsBeforeTrackingNmseDb = lmsDpd.CalculateNmse(
        referenceSignal,
        driftTarget,
    )
    driftTraining = lmsDpd.UpdateFromLabels(
        referenceSignal,
        driftTarget,
    )
    lmsAfterTrackingNmseDb = lmsDpd.CalculateNmse(
        referenceSignal,
        driftTarget,
    )
    trackingImprovementDb = (
        lmsBeforeTrackingNmseDb - lmsAfterTrackingNmseDb
    )
    if trackingImprovementDb <= 20.0:
        raise AssertionError(
            "sample NLMS did not materially track the changed GMP target"
        )
    if (
        stationaryTraining.updateCount != config.numSamples
        or driftTraining.updateCount != config.numSamples
    ):
        raise AssertionError(
            "updateDecimation=1 must update every nonzero sample"
        )
    result = DpdLmsBenchmarkResult(
        sampleCount=config.numSamples,
        updateCountPerFrame=driftTraining.updateCount,
        batchStationaryNmseDb=batchStationaryNmseDb,
        lmsStationaryNmseDb=lmsStationaryNmseDb,
        staleBatchDriftNmseDb=staleBatchDriftNmseDb,
        lmsBeforeTrackingNmseDb=lmsBeforeTrackingNmseDb,
        lmsAfterTrackingNmseDb=lmsAfterTrackingNmseDb,
        trackingImprovementDb=trackingImprovementDb,
    )
    SaveDpdLmsBenchmarkResult(result, config)
    PrintDpdLmsBenchmarkResult(result)
    return result


def ParseBenchmarkArguments() -> Union[
    BenchmarkConfig,
    TwoToneBenchmarkConfig,
    PaCharacterizationConfig,
    DpdGmpBenchmarkConfig,
    DpdLmsBenchmarkConfig,
    ChannelAnalysisBenchmarkConfig,
]:
    """Parse Wi-Fi, PA, DPD-GMP, or channel benchmark command-line options.

    Processing details:
        Algorithm: Define only scenario-level controls, parse one command
        line, convert it into ``BenchmarkConfig``, and validate it before
        returning. Algorithm-internal learning constants remain fixed so
            comparisons stay reproducible. The mutually exclusive
            mode switches select ILC/IM, multi-model PA characterization,
            staged GMP predistortion, or channel/coupling-aware DPD
            validation and bypass the ordinary Wi-Fi suite.

    Returns:
        result: Validated configuration ready for the selected benchmark.
    """

    argumentParser = argparse.ArgumentParser(
        description=(
            "Run classified SISO ILC benchmark scenarios independently of "
            "the production main program."
        )
    )
    modeGroup = argumentParser.add_mutually_exclusive_group()
    modeGroup.add_argument(
        "--two-tone",
        dest="twoTone",
        action="store_true",
        help="Run the IM3/IM5/IM7 two-tone benchmark instead of Wi-Fi",
    )
    modeGroup.add_argument(
        "--pa-analyse",
        dest="paAnalyse",
        action="store_true",
        help=(
            "Characterize Wiener, GMP, and Doherty PA frequency/memory "
            "features with two-tone sweeps"
        ),
    )
    modeGroup.add_argument(
        "--dpd-gmp",
        dest="dpdGmp",
        action="store_true",
        help=(
            "Run the staged PA-analysis-driven DPD-GMP benchmark with "
            "Wi-Fi and two-tone comparisons"
        ),
    )
    modeGroup.add_argument(
        "--dpd-lms",
        dest="dpdLms",
        action="store_true",
        help=(
            "Compare batch GMP fitting with sample-by-sample NLMS "
            "stationary accuracy and coefficient tracking"
        ),
    )
    modeGroup.add_argument(
        "--channel-analyse",
        dest="channelAnalyse",
        action="store_true",
        help=(
            "Measure MIMO channel flatness/coupling/delay and compare "
            "independent with coupling-aware DPD-GMP"
        ),
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
        "--sample-rate-hz",
        dest="sampleRateHz",
        type=float,
        default=None,
        help=(
            "Complex-baseband sample rate in Hz; overrides legacy "
            "--oversampling when supplied"
        ),
    )
    argumentParser.add_argument(
        "--oversampling",
        type=int,
        default=4,
        help=(
            "Legacy bandwidth multiplier used only when --sample-rate-hz "
            "is omitted"
        ),
    )
    argumentParser.add_argument(
        "--width",
        type=int,
        default=None,
        help=(
            "External I/Q component width; 0 selects floating point "
            "(default: 0 for PA analysis, 16 for ILC benchmarks)"
        ),
    )
    argumentParser.add_argument(
        "--guard-interval",
        dest="guardIntervalUs",
        type=float,
        default=0.8,
    )
    argumentParser.add_argument(
        "--output-power-dbm",
        dest="outputPowerDbm",
        type=float,
        default=20.0,
        help=(
            "Absolute nominal PA output power per chain "
            "(default: 20 dBm)"
        ),
    )
    argumentParser.add_argument(
        "--maximum-output-power-dbm",
        dest="maximumOutputPowerDbm",
        type=float,
        default=25.0,
        help="Rated per-PA output-power limit (default: 25 dBm)",
    )
    argumentParser.add_argument(
        "--load-resistance-ohm",
        dest="loadResistanceOhm",
        type=float,
        default=50.0,
        help="Resistive PA port used for dBm conversion (default: 50 ohms)",
    )
    argumentParser.add_argument(
        "--iterations",
        dest="numIterations",
        type=int,
        default=None,
        help=(
            "Iteration count (default: 8 for DPD-GMP, 10 for ILC "
            "comparison modes)"
        ),
    )
    argumentParser.add_argument(
        "--pa",
        dest="paModelName",
        choices=(
            "rapp",
            "wiener",
            "gmp",
            "piecewise_gmp",
            "doherty",
        ),
        default="wiener",
    )
    argumentParser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Override the selected scenario's deterministic default seed"
        ),
    )
    argumentParser.add_argument(
        "--validation-seed",
        dest="validationSeed",
        type=int,
        default=None,
        help=(
            "Independent Wi-Fi validation seed for --dpd-gmp "
            "(default: 987)"
        ),
    )
    argumentParser.add_argument(
        "--power-start-dbm",
        dest="powerStartDbm",
        type=float,
        default=10.0,
        help="First absolute PA output power in the sweep, in dBm",
    )
    argumentParser.add_argument(
        "--power-stop-dbm",
        dest="powerStopDbm",
        type=float,
        default=25.0,
        help="Last absolute PA output power in the sweep, in dBm",
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
        default=None,
    )
    argumentParser.add_argument(
        "--tone-lower-hz",
        dest="toneLowerHz",
        type=float,
        default=-2.0e6,
        help="Lower complex-baseband two-tone frequency in Hz",
    )
    argumentParser.add_argument(
        "--tone-upper-hz",
        dest="toneUpperHz",
        type=float,
        default=2.0e6,
        help="Upper complex-baseband two-tone frequency in Hz",
    )
    argumentParser.add_argument(
        "--tone-samples",
        dest="toneNumSamples",
        type=int,
        default=None,
        help=(
            "Number of samples in the repeated two-tone record "
            "(default: 16384 for PA analysis, 32768 for two-tone ILC)"
        ),
    )
    argumentParser.add_argument(
        "--tone-rms-level",
        dest="toneRmsLevel",
        type=float,
        default=0.5,
        help="Generated two-tone RMS before PA power calibration",
    )
    arguments = argumentParser.parse_args()
    if arguments.channelAnalyse:
        channelOutputDirectory = (
            Path("results/channel_analysis")
            if arguments.outputDirectory is None
            else arguments.outputDirectory
        )
        channelConfig = ChannelAnalysisBenchmarkConfig(
            frameFormat=arguments.frameFormat,
            bandwidthMhz=arguments.bandwidthMhz,
            sampleRateHz=(
                80.0e6
                if arguments.sampleRateHz is None
                else arguments.sampleRateHz
            ),
            mcs=arguments.mcs,
            numDataSymbols=arguments.numDataSymbols,
            seed=(
                517
                if arguments.seed is None
                else arguments.seed
            ),
            outputPowerDbm=arguments.outputPowerDbm,
            maximumOutputPowerDbm=arguments.maximumOutputPowerDbm,
            loadResistanceOhm=arguments.loadResistanceOhm,
            numIterations=(
                10
                if arguments.numIterations is None
                else arguments.numIterations
            ),
            width=0 if arguments.width is None else arguments.width,
            outputDirectory=channelOutputDirectory,
        )
        channelConfig.Validate()
        return channelConfig
    if arguments.dpdLms:
        dpdLmsOutputDirectory = (
            Path("results/dpd_lms_benchmark")
            if arguments.outputDirectory is None
            else arguments.outputDirectory
        )
        dpdLmsConfig = DpdLmsBenchmarkConfig(
            numSamples=(
                8192
                if arguments.toneNumSamples is None
                else arguments.toneNumSamples
            ),
            seed=(
                907
                if arguments.seed is None
                else arguments.seed
            ),
            outputDirectory=dpdLmsOutputDirectory,
        )
        dpdLmsConfig.Validate()
        return dpdLmsConfig
    if arguments.dpdGmp:
        dpdGmpOutputDirectory = (
            Path("results/dpd_gmp_benchmark")
            if arguments.outputDirectory is None
            else arguments.outputDirectory
        )
        dpdGmpConfig = DpdGmpBenchmarkConfig(
            sampleRateHz=(
                80.0e6
                if arguments.sampleRateHz is None
                else arguments.sampleRateHz
            ),
            seed=(
                321
                if arguments.seed is None
                else arguments.seed
            ),
            validationSeed=(
                987
                if arguments.validationSeed is None
                else arguments.validationSeed
            ),
            maximumOutputPowerDbm=arguments.maximumOutputPowerDbm,
            loadResistanceOhm=arguments.loadResistanceOhm,
            numIterations=(
                8
                if arguments.numIterations is None
                else arguments.numIterations
            ),
            width=0 if arguments.width is None else arguments.width,
            outputDirectory=dpdGmpOutputDirectory,
        )
        dpdGmpConfig.Validate()
        return dpdGmpConfig
    if arguments.paAnalyse:
        paAnalysisOutputDirectory = (
            Path("results/pa_characterization")
            if arguments.outputDirectory is None
            else arguments.outputDirectory
        )
        paAnalysisConfig = PaCharacterizationConfig(
            sampleRateHz=(
                200.0e6
                if arguments.sampleRateHz is None
                else arguments.sampleRateHz
            ),
            numSamples=(
                16384
                if arguments.toneNumSamples is None
                else arguments.toneNumSamples
            ),
            width=0 if arguments.width is None else arguments.width,
            outputPowerDbm=arguments.outputPowerDbm,
            maximumOutputPowerDbm=arguments.maximumOutputPowerDbm,
            loadResistanceOhm=arguments.loadResistanceOhm,
            outputDirectory=paAnalysisOutputDirectory,
        )
        paAnalysisConfig.Validate()
        return paAnalysisConfig
    if arguments.twoTone:
        twoToneOutputDirectory = (
            Path("results/two_tone_ilc_benchmark")
            if arguments.outputDirectory is None
            else arguments.outputDirectory
        )
        twoToneConfig = TwoToneBenchmarkConfig(
            sampleRateHz=(
                100.0e6
                if arguments.sampleRateHz is None
                else arguments.sampleRateHz
            ),
            toneFrequenciesHz=(
                arguments.toneLowerHz,
                arguments.toneUpperHz,
            ),
            numSamples=(
                32768
                if arguments.toneNumSamples is None
                else arguments.toneNumSamples
            ),
            rmsLevel=arguments.toneRmsLevel,
            width=16 if arguments.width is None else arguments.width,
            outputPowerDbm=arguments.outputPowerDbm,
            maximumOutputPowerDbm=arguments.maximumOutputPowerDbm,
            loadResistanceOhm=arguments.loadResistanceOhm,
            numIterations=(
                10
                if arguments.numIterations is None
                else arguments.numIterations
            ),
            paModelName=arguments.paModelName,
            seed=(
                211
                if arguments.seed is None
                else arguments.seed
            ),
            outputDirectory=twoToneOutputDirectory,
        )
        twoToneConfig.Validate()
        return twoToneConfig
    wifiOutputDirectory = (
        Path("results/all_ilc_benchmark")
        if arguments.outputDirectory is None
        else arguments.outputDirectory
    )
    config = BenchmarkConfig(
        frameFormat=arguments.frameFormat,
        bandwidthMhz=arguments.bandwidthMhz,
        mcs=arguments.mcs,
        numDataSymbols=arguments.numDataSymbols,
        sampleRateHz=arguments.sampleRateHz,
        oversampling=arguments.oversampling,
        width=16 if arguments.width is None else arguments.width,
        guardIntervalUs=arguments.guardIntervalUs,
        outputPowerDbm=arguments.outputPowerDbm,
        maximumOutputPowerDbm=arguments.maximumOutputPowerDbm,
        loadResistanceOhm=arguments.loadResistanceOhm,
        numIterations=(
            10
            if arguments.numIterations is None
            else arguments.numIterations
        ),
        paModelName=arguments.paModelName,
        seed=101 if arguments.seed is None else arguments.seed,
        powerStartDbm=arguments.powerStartDbm,
        powerStopDbm=arguments.powerStopDbm,
        powerPointCount=arguments.powerPointCount,
        generatePowerEvmCurve=arguments.generatePowerEvmCurve,
        outputDirectory=wifiOutputDirectory,
    )
    config.Validate()
    return config


def Main() -> int:
    """Run the standalone benchmark and report its artifact directory.

    Processing details:
        Algorithm: Parse the requested scenario controls, execute the complete
            Wi-Fi, ILC two-tone, PA-characterization, DPD-GMP, or channel
            analysis suite, and print the absolute output path after every
            result has been saved.

    Returns:
        result: Process exit status zero after successful completion.
    """

    config = ParseBenchmarkArguments()
    if isinstance(config, ChannelAnalysisBenchmarkConfig):
        RunChannelAnalysisBenchmark(config)
    elif isinstance(config, DpdLmsBenchmarkConfig):
        RunDpdLmsBenchmark(config)
    elif isinstance(config, DpdGmpBenchmarkConfig):
        RunDpdGmpBenchmark(config)
    elif isinstance(config, PaCharacterizationConfig):
        RunPaCharacterizationBenchmark(config)
    elif isinstance(config, TwoToneBenchmarkConfig):
        RunTwoToneIlcBenchmark(config)
    else:
        RunAllIlcBenchmark(config)
    print(f"\nBenchmark results: {config.outputDirectory.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(Main())
