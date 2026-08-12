"""Regenerate the PA electrothermal parameter-effect documentation figures."""

from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def ConfigurePlotStyle() -> None:
    """Apply one readable and deterministic style to every thermal figure.

    Processing details:
        Algorithm: Set a light background, restrained grid, readable font sizes,
        and a fixed export resolution without depending on host user settings.

    Returns:
        result: None. Later Matplotlib figures inherit the selected settings.
    """

    plt.rcParams.update(
        {
            "figure.facecolor": "#f5f7fb",
            "axes.facecolor": "#f5f7fb",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
            "font.size": 10.5,
            "axes.titlesize": 12.0,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.0,
            "savefig.dpi": 180,
        }
    )


def CalculateStepRise(
    timeSec: np.ndarray,
    dissipatedPowerW: float,
    resistanceValuesCPerW: Sequence[float],
    timeConstantValuesSec: Sequence[float],
) -> np.ndarray:
    """Calculate the exact Foster temperature rise for a power step.

    Processing details:
        Algorithm: Evaluate each parallel thermal branch response
        ``P * R * (1 - exp(-t/tau))`` and sum all branch temperature rises.

    Args:
        timeSec: Nonnegative observation-time array in seconds.
        dissipatedPowerW: Constant power applied after time zero in watts.
        resistanceValuesCPerW: Thermal resistance of each Foster branch.
        timeConstantValuesSec: Matching thermal time constant of each branch.

    Returns:
        result: Junction-to-ambient temperature rise in degrees Celsius.
    """

    temperatureRiseC = np.zeros_like(timeSec, dtype=float)
    for resistanceCPerW, timeConstantSec in zip(
        resistanceValuesCPerW,
        timeConstantValuesSec,
    ):
        temperatureRiseC += (
            float(dissipatedPowerW)
            * float(resistanceCPerW)
            * (1.0 - np.exp(-timeSec / float(timeConstantSec)))
        )
    return temperatureRiseC


def CalculateEfficiency(
    outputPowerDbm: np.ndarray,
    minimumEfficiency: float,
    peakEfficiency: float,
    kneePowerDbm: float,
) -> np.ndarray:
    """Evaluate the smooth output-power-dependent efficiency equation.

    Processing details:
        Algorithm: Convert dBm to watts, divide by the configured knee power,
        and apply the bounded rational transition used by ``ThermalConfig``.

    Args:
        outputPowerDbm: RF output-power samples in dBm.
        minimumEfficiency: Low-power efficiency bound as a fraction.
        peakEfficiency: High-power asymptotic efficiency as a fraction.
        kneePowerDbm: Half-transition output power in dBm.

    Returns:
        result: Drain-efficiency fraction at every requested power.
    """

    outputPowerW = 10.0 ** ((outputPowerDbm - 30.0) / 10.0)
    kneePowerW = 10.0 ** ((float(kneePowerDbm) - 30.0) / 10.0)
    normalizedPower = outputPowerW / kneePowerW
    return minimumEfficiency + (
        peakEfficiency - minimumEfficiency
    ) * normalizedPower / (1.0 + normalizedPower)


def SimulatePulseTemperature(
    totalTimeSec: float,
    timeStepSec: float,
    pulsePeriodSec: float,
    dutyCycle: float,
    onPowerW: float,
    idlePowerW: float,
    resistanceValuesCPerW: Sequence[float],
    timeConstantValuesSec: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate causal Foster heating for a periodic RF burst sequence.

    Processing details:
        Algorithm: Build a rectangular on/off power sequence, advance every
        branch with its exact zero-order-hold coefficient, and sum branch rises.

    Args:
        totalTimeSec: Total simulation duration in seconds.
        timeStepSec: Thermal update duration in seconds.
        pulsePeriodSec: Repeating RF burst period in seconds.
        dutyCycle: Active fraction of each burst period.
        onPowerW: Dissipated power while RF is active.
        idlePowerW: Dissipated power while RF is inactive.
        resistanceValuesCPerW: Foster thermal resistance vector.
        timeConstantValuesSec: Matching Foster time-constant vector.

    Returns:
        result: Time and junction-temperature-rise arrays.
    """

    timeSec = np.arange(0.0, totalTimeSec, timeStepSec)
    branchRiseC = np.zeros(len(resistanceValuesCPerW), dtype=float)
    temperatureRiseC = np.zeros_like(timeSec)
    decayValues = np.exp(
        -timeStepSec / np.asarray(timeConstantValuesSec, dtype=float)
    )
    resistanceArray = np.asarray(resistanceValuesCPerW, dtype=float)
    activeDurationSec = pulsePeriodSec * dutyCycle
    for sampleIndex, currentTimeSec in enumerate(timeSec):
        isActive = np.mod(currentTimeSec, pulsePeriodSec) < activeDurationSec
        dissipatedPowerW = onPowerW if isActive else idlePowerW
        branchRiseC = (
            decayValues * branchRiseC
            + resistanceArray
            * (1.0 - decayValues)
            * dissipatedPowerW
        )
        temperatureRiseC[sampleIndex] = float(np.sum(branchRiseC))
    return timeSec, temperatureRiseC


def SaveThermalNetworkEffects(outputDirectory: Path) -> None:
    """Draw how resistance, time constant, branches, and update interval act.

    Processing details:
        Algorithm: Generate four shared-scale parameter sweeps from analytical
        RC equations and an exact versus coarse zero-order-hold comparison.

    Args:
        outputDirectory: Existing or creatable documentation image directory.

    Returns:
        result: None. A PNG file is written with a stable documentation name.
    """

    timeSec = np.logspace(-6.0, 1.0, 700)
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 7.4))
    resistanceAxis, timeAxis, branchAxis, updateAxis = axes.ravel()

    for resistanceCPerW in (10.0, 20.0, 40.0):
        riseC = CalculateStepRise(
            timeSec,
            1.0,
            (resistanceCPerW,),
            (0.05,),
        )
        resistanceAxis.semilogx(
            timeSec,
            riseC,
            linewidth=2.0,
            label=f"Rth = {resistanceCPerW:.0f} C/W",
        )
    resistanceAxis.set_title("A. Thermal resistance sets final temperature rise")
    resistanceAxis.set_xlabel("Time (s)")
    resistanceAxis.set_ylabel("Temperature rise for 1 W (C)")
    resistanceAxis.legend()

    for timeConstantSec in (0.005, 0.05, 0.5):
        riseC = CalculateStepRise(
            timeSec,
            1.0,
            (30.0,),
            (timeConstantSec,),
        )
        timeAxis.semilogx(
            timeSec,
            riseC,
            linewidth=2.0,
            label=f"tau = {timeConstantSec:g} s",
        )
    timeAxis.set_title("B. Time constant sets response speed, not final rise")
    timeAxis.set_xlabel("Time (s)")
    timeAxis.set_ylabel("Temperature rise for 1 W (C)")
    timeAxis.legend()

    modelDefinitions = (
        ("single RC", (30.0,), (0.05,)),
        ("fast + slow", (10.0, 20.0), (0.0002, 0.5)),
        ("3-pole Foster", (2.0, 8.0, 20.0), (0.00005, 0.005, 0.5)),
    )
    for modelLabel, resistanceValues, timeConstantValues in modelDefinitions:
        branchAxis.semilogx(
            timeSec,
            CalculateStepRise(
                timeSec,
                1.0,
                resistanceValues,
                timeConstantValues,
            ),
            linewidth=2.0,
            label=modelLabel,
        )
    branchAxis.set_title("C. Branch distribution creates fast and slow knees")
    branchAxis.set_xlabel("Time (s)")
    branchAxis.set_ylabel("Temperature rise for 1 W (C)")
    branchAxis.legend()

    linearTimeSec = np.linspace(0.0, 0.2, 600)
    exactRiseC = CalculateStepRise(
        linearTimeSec,
        1.0,
        (30.0,),
        (0.05,),
    )
    updateAxis.plot(
        linearTimeSec,
        exactRiseC,
        linewidth=2.2,
        label="continuous reference",
    )
    for updateDurationSec in (0.01, 0.05):
        updateTimesSec = np.arange(0.0, 0.2001, updateDurationSec)
        updateRiseC = CalculateStepRise(
            updateTimesSec,
            1.0,
            (30.0,),
            (0.05,),
        )
        updateAxis.step(
            updateTimesSec,
            updateRiseC,
            where="post",
            linewidth=1.8,
            label=f"update = {updateDurationSec:g} s",
        )
    updateAxis.set_title("D. Update interval changes visible stair steps only")
    updateAxis.set_xlabel("Time (s)")
    updateAxis.set_ylabel("Temperature rise for 1 W (C)")
    updateAxis.legend()

    figure.suptitle(
        "Thermal-network parameter effects (analytical schematic)",
        fontsize=16.0,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(
        outputDirectory / "thermal_network_parameter_effects.png",
        bbox_inches="tight",
    )
    plt.close(figure)


def SaveHeatSourceEffects(outputDirectory: Path) -> None:
    """Draw efficiency, knee, reference-power, idle-power, and duty effects.

    Processing details:
        Algorithm: Evaluate the exact configured efficiency law, convert its
        RF output to heat, and show independent scaling by duty and idle power.

    Args:
        outputDirectory: Existing or creatable documentation image directory.

    Returns:
        result: None. A PNG file is written with a stable documentation name.
    """

    outputPowerDbm = np.linspace(0.0, 25.0, 300)
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 7.4))
    efficiencyAxis, kneeAxis, referenceAxis, dutyAxis = axes.ravel()

    efficiencyDefinitions = (
        ("wide range: 10%-45%", 0.10, 0.45),
        ("higher low-power eta: 20%-45%", 0.20, 0.45),
        ("higher peak eta: 10%-60%", 0.10, 0.60),
    )
    for label, minimumEfficiency, peakEfficiency in efficiencyDefinitions:
        efficiencyAxis.plot(
            outputPowerDbm,
            100.0
            * CalculateEfficiency(
                outputPowerDbm,
                minimumEfficiency,
                peakEfficiency,
                15.0,
            ),
            linewidth=2.0,
            label=label,
        )
    efficiencyAxis.set_title("A. Minimum and peak efficiency bound the curve")
    efficiencyAxis.set_xlabel("RF output power (dBm)")
    efficiencyAxis.set_ylabel("Drain efficiency (%)")
    efficiencyAxis.legend()

    for kneePowerDbm in (10.0, 15.0, 20.0):
        kneeAxis.plot(
            outputPowerDbm,
            100.0
            * CalculateEfficiency(
                outputPowerDbm,
                0.10,
                0.45,
                kneePowerDbm,
            ),
            linewidth=2.0,
            label=f"knee = {kneePowerDbm:.0f} dBm",
        )
    kneeAxis.set_title("B. Efficiency knee shifts the transition in power")
    kneeAxis.set_xlabel("RF output power (dBm)")
    kneeAxis.set_ylabel("Drain efficiency (%)")
    kneeAxis.legend()

    normalizedEnvelope = np.linspace(0.0, 1.0, 300)
    for referencePowerDbm in (20.0, 22.0, 25.0):
        physicalPowerW = (
            10.0 ** ((referencePowerDbm - 30.0) / 10.0)
            * normalizedEnvelope**2
        )
        efficiency = CalculateEfficiency(
            10.0 * np.log10(np.maximum(physicalPowerW, 1.0e-15)) + 30.0,
            0.10,
            0.45,
            15.0,
        )
        dissipatedPowerW = 0.15 + physicalPowerW * (1.0 / efficiency - 1.0)
        referenceAxis.plot(
            normalizedEnvelope,
            dissipatedPowerW,
            linewidth=2.0,
            label=f"reference = {referencePowerDbm:.0f} dBm",
        )
    referenceAxis.set_title("C. Reference dBm maps normalized output to heat")
    referenceAxis.set_xlabel("Normalized output magnitude |y|")
    referenceAxis.set_ylabel("Estimated dissipated power (W)")
    referenceAxis.legend()

    dutyPercent = np.linspace(0.0, 100.0, 200)
    for idlePowerW in (0.0, 0.15, 0.4):
        meanDissipationW = (
            dutyPercent / 100.0 * 1.2
            + (1.0 - dutyPercent / 100.0) * idlePowerW
        )
        dutyAxis.plot(
            dutyPercent,
            meanDissipationW,
            linewidth=2.0,
            label=f"idle = {idlePowerW:g} W",
        )
    dutyAxis.set_title("D. Duty cycle mixes active heat with idle bias heat")
    dutyAxis.set_xlabel("RF active duty cycle (%)")
    dutyAxis.set_ylabel("Mean dissipated power (W)")
    dutyAxis.legend()

    figure.suptitle(
        "Heat-source parameter effects (configured behavioral law)",
        fontsize=16.0,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(
        outputDirectory / "thermal_heat_source_parameter_effects.png",
        bbox_inches="tight",
    )
    plt.close(figure)


def SaveElectricalDriftEffects(outputDirectory: Path) -> None:
    """Draw each temperature-to-electrical coefficient independently.

    Processing details:
        Algorithm: Sweep junction temperature through the implemented gain,
        phase, saturation, and nonlinear-envelope equations while holding all
        unrelated coefficients fixed at zero.

    Args:
        outputDirectory: Existing or creatable documentation image directory.

    Returns:
        result: None. A PNG file is written with a stable documentation name.
    """

    junctionTemperatureC = np.linspace(25.0, 125.0, 250)
    temperatureDeltaC = junctionTemperatureC - 25.0
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 7.4))
    gainAxis, phaseAxis, saturationAxis, nonlinearAxis = axes.ravel()

    for gainCoefficientDbPerC in (-0.005, -0.012, -0.025):
        gainAxis.plot(
            junctionTemperatureC,
            gainCoefficientDbPerC * temperatureDeltaC,
            linewidth=2.0,
            label=f"kG = {gainCoefficientDbPerC:g} dB/C",
        )
    gainAxis.set_title("A. Gain coefficient creates linear dB drift")
    gainAxis.set_xlabel("Junction temperature (C)")
    gainAxis.set_ylabel("Small-signal gain change (dB)")
    gainAxis.legend()

    for phaseCoefficientDegreePerC in (-0.03, 0.03, 0.08):
        phaseAxis.plot(
            junctionTemperatureC,
            phaseCoefficientDegreePerC * temperatureDeltaC,
            linewidth=2.0,
            label=f"kPhase = {phaseCoefficientDegreePerC:g} deg/C",
        )
    phaseAxis.set_title("B. Phase coefficient rotates the whole waveform")
    phaseAxis.set_xlabel("Junction temperature (C)")
    phaseAxis.set_ylabel("Common phase change (degree)")
    phaseAxis.legend()

    for saturationCoefficientPerC in (-0.0005, -0.0015, -0.003):
        saturationScale = np.maximum(
            0.05,
            1.0 + saturationCoefficientPerC * temperatureDeltaC,
        )
        saturationAxis.plot(
            junctionTemperatureC,
            saturationScale,
            linewidth=2.0,
            label=f"kSat = {saturationCoefficientPerC:g} /C",
        )
    saturationAxis.set_title("C. Saturation coefficient moves the compression scale")
    saturationAxis.set_xlabel("Junction temperature (C)")
    saturationAxis.set_ylabel("Relative saturation scale")
    saturationAxis.legend()

    outputMagnitude = np.linspace(0.0, 1.4, 300)
    selectedTemperatureDeltaC = 75.0
    for nonlinearCoefficientPerC in (0.0, 0.001, 0.002, 0.004):
        nonlinearScale = max(
            0.0,
            nonlinearCoefficientPerC * selectedTemperatureDeltaC,
        )
        envelopeScale = 1.0 / (
            1.0 + nonlinearScale * outputMagnitude**2
        )
        nonlinearAxis.plot(
            outputMagnitude,
            envelopeScale,
            linewidth=2.0,
            label=f"kNL = {nonlinearCoefficientPerC:g} /C",
        )
    nonlinearAxis.set_title("D. Nonlinearity coefficient bends high amplitudes first")
    nonlinearAxis.set_xlabel("Base electrical output magnitude |y0|")
    nonlinearAxis.set_ylabel("Additional thermal envelope scale")
    nonlinearAxis.legend()

    figure.suptitle(
        "Temperature-to-electrical parameter effects (Tref = 25 C)",
        fontsize=16.0,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(
        outputDirectory / "thermal_electrical_parameter_effects.png",
        bbox_inches="tight",
    )
    plt.close(figure)


def SaveOperatingScenarioEffects(outputDirectory: Path) -> None:
    """Draw burst-period, ambient, initial-state, and mutual-heating effects.

    Processing details:
        Algorithm: Simulate pulse trains with equal mean duty but unequal burst
        periods, then plot direct algebraic effects of ambient, initial state,
        and a two-chain steady mutual thermal-resistance matrix.

    Args:
        outputDirectory: Existing or creatable documentation image directory.

    Returns:
        result: None. A PNG file is written with a stable documentation name.
    """

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 7.4))
    burstAxis, ambientAxis, initialAxis, couplingAxis = axes.ravel()
    resistanceValues = (2.0, 8.0, 20.0)
    timeConstantValues = (0.00005, 0.005, 0.5)
    for pulsePeriodSec in (0.002, 0.02, 0.1):
        timeSec, temperatureRiseC = SimulatePulseTemperature(
            totalTimeSec=0.5,
            timeStepSec=0.00005,
            pulsePeriodSec=pulsePeriodSec,
            dutyCycle=0.5,
            onPowerW=1.5,
            idlePowerW=0.15,
            resistanceValuesCPerW=resistanceValues,
            timeConstantValuesSec=timeConstantValues,
        )
        burstAxis.plot(
            1000.0 * timeSec,
            25.0 + temperatureRiseC,
            linewidth=1.5,
            label=f"period = {1000.0 * pulsePeriodSec:g} ms",
        )
    burstAxis.set_title("A. Equal duty can produce unequal temperature ripple")
    burstAxis.set_xlabel("Time (ms)")
    burstAxis.set_ylabel("Junction temperature (C)")
    burstAxis.legend()

    observationTimeSec = np.linspace(0.0, 0.5, 300)
    commonRiseC = CalculateStepRise(
        observationTimeSec,
        1.0,
        resistanceValues,
        timeConstantValues,
    )
    for ambientTemperatureC in (0.0, 25.0, 60.0):
        ambientAxis.plot(
            observationTimeSec,
            ambientTemperatureC + commonRiseC,
            linewidth=2.0,
            label=f"ambient = {ambientTemperatureC:.0f} C",
        )
    ambientAxis.set_title("B. Ambient temperature vertically shifts junction temperature")
    ambientAxis.set_xlabel("Time (s)")
    ambientAxis.set_ylabel("Junction temperature (C)")
    ambientAxis.legend()

    resistanceTotalCPerW = float(np.sum(resistanceValues))
    equivalentTimeConstantSec = 0.15
    for initialJunctionTemperatureC in (25.0, 50.0, 85.0):
        steadyTemperatureC = 25.0 + resistanceTotalCPerW * 1.0
        junctionTemperatureC = steadyTemperatureC + (
            initialJunctionTemperatureC - steadyTemperatureC
        ) * np.exp(-observationTimeSec / equivalentTimeConstantSec)
        initialAxis.plot(
            observationTimeSec,
            junctionTemperatureC,
            linewidth=2.0,
            label=f"initial Tj = {initialJunctionTemperatureC:.0f} C",
        )
    initialAxis.set_title("C. Initial Tj changes only transient starting state")
    initialAxis.set_xlabel("Time (s)")
    initialAxis.set_ylabel("Junction temperature (C)")
    initialAxis.legend()

    sourcePowerW = np.linspace(0.0, 2.0, 200)
    for mutualResistanceCPerW in (0.0, 1.0, 3.0, 6.0):
        couplingAxis.plot(
            sourcePowerW,
            mutualResistanceCPerW * sourcePowerW,
            linewidth=2.0,
            label=f"Rmutual = {mutualResistanceCPerW:g} C/W",
        )
    couplingAxis.set_title("D. Mutual thermal resistance maps neighbor heat to Tj")
    couplingAxis.set_xlabel("Neighbor PA dissipated power (W)")
    couplingAxis.set_ylabel("Victim PA temperature offset (C)")
    couplingAxis.legend()

    figure.suptitle(
        "Thermal operating-condition effects (behavioral examples)",
        fontsize=16.0,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(
        outputDirectory / "thermal_operating_parameter_effects.png",
        bbox_inches="tight",
    )
    plt.close(figure)


def SaveBoundaryParameterEffects(outputDirectory: Path) -> None:
    """Draw sampling, activity threshold, reference, and safety parameters.

    Processing details:
        Algorithm: Convert update samples to physical time at several sample
        rates, classify a deterministic burst envelope with relative thresholds,
        shift the zero-drift reference temperature, and show the hard thermal
        safety boundary independently from the physical temperature trajectory.

    Args:
        outputDirectory: Existing or creatable documentation image directory.

    Returns:
        result: None. A PNG file is written with a stable documentation name.
    """

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 7.4))
    sampleRateAxis, thresholdAxis, referenceAxis, maximumAxis = axes.ravel()

    updateSamples = np.asarray((64, 128, 256, 512, 1024), dtype=float)
    for sampleRateHz in (20.0e6, 80.0e6, 320.0e6):
        updateTimeUs = 1.0e6 * updateSamples / sampleRateHz
        sampleRateAxis.plot(
            updateSamples,
            updateTimeUs,
            marker="o",
            linewidth=2.0,
            label=f"sample rate = {sampleRateHz / 1.0e6:.0f} MHz",
        )
    sampleRateAxis.set_title("A. Sample rate converts update samples to physical time")
    sampleRateAxis.set_xlabel("Thermal update interval (samples)")
    sampleRateAxis.set_ylabel("Thermal update time (us)")
    sampleRateAxis.legend()

    sampleIndex = np.arange(1600)
    envelope = np.zeros(sampleIndex.size, dtype=float)
    envelope[200:1400] = (
        0.03
        + 0.97
        * np.abs(
            np.sin(
                np.pi * np.arange(1200, dtype=float) / 1199.0
            )
        )
    )
    thresholdAxis.plot(
        sampleIndex,
        20.0 * np.log10(np.maximum(envelope, 1.0e-4)),
        color="#4c78a8",
        linewidth=1.8,
        label="example burst envelope",
    )
    for thresholdDb, lineStyle in ((-60.0, "--"), (-30.0, "-."), (-10.0, ":")):
        thresholdAxis.axhline(
            thresholdDb,
            linestyle=lineStyle,
            linewidth=1.8,
            label=f"active threshold = {thresholdDb:.0f} dB",
        )
    thresholdAxis.set_ylim(-82.0, 3.0)
    thresholdAxis.set_title("B. Active threshold separates RF-on samples from idle")
    thresholdAxis.set_xlabel("Sample index")
    thresholdAxis.set_ylabel("Envelope relative to peak (dB)")
    thresholdAxis.legend()

    junctionTemperatureC = np.linspace(0.0, 125.0, 300)
    gainCoefficientDbPerC = -0.012
    for referenceTemperatureC in (0.0, 25.0, 60.0):
        gainChangeDb = gainCoefficientDbPerC * (
            junctionTemperatureC - referenceTemperatureC
        )
        referenceAxis.plot(
            junctionTemperatureC,
            gainChangeDb,
            linewidth=2.0,
            label=f"reference temperature = {referenceTemperatureC:.0f} C",
        )
    referenceAxis.axhline(0.0, color="#555555", linewidth=1.0)
    referenceAxis.set_title("C. Reference temperature moves the zero-drift crossing")
    referenceAxis.set_xlabel("Junction temperature (C)")
    referenceAxis.set_ylabel("Configured gain change (dB)")
    referenceAxis.legend()

    observationTimeSec = np.linspace(0.0, 1.0, 400)
    modeledTemperatureC = 25.0 + 145.0 * (
        1.0 - np.exp(-observationTimeSec / 0.35)
    )
    maximumAxis.plot(
        observationTimeSec,
        modeledTemperatureC,
        linewidth=2.2,
        label="unlimited model trajectory",
    )
    for maximumTemperatureC in (100.0, 125.0, 150.0):
        maximumAxis.axhline(
            maximumTemperatureC,
            linestyle="--",
            linewidth=1.8,
            label=f"maximum Tj = {maximumTemperatureC:.0f} C",
        )
    maximumAxis.set_title("D. Maximum junction temperature is a stop limit only")
    maximumAxis.set_xlabel("Open-loop test time (s)")
    maximumAxis.set_ylabel("Junction temperature (C)")
    maximumAxis.legend()

    figure.suptitle(
        "Thermal configuration-boundary effects (behavioral examples)",
        fontsize=16.0,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(
        outputDirectory / "thermal_boundary_parameter_effects.png",
        bbox_inches="tight",
    )
    plt.close(figure)


def GenerateThermalFigures(outputDirectory: Path) -> None:
    """Regenerate every detailed PA thermal parameter-effect figure.

    Processing details:
        Algorithm: Create the destination, apply deterministic style, and call
        each focused renderer so documentation images remain reproducible.

    Args:
        outputDirectory: Destination directory for all generated PNG files.

    Returns:
        result: None. Four documented figures are replaced atomically per file.
    """

    outputDirectory.mkdir(parents=True, exist_ok=True)
    ConfigurePlotStyle()
    SaveThermalNetworkEffects(outputDirectory)
    SaveHeatSourceEffects(outputDirectory)
    SaveElectricalDriftEffects(outputDirectory)
    SaveOperatingScenarioEffects(outputDirectory)
    SaveBoundaryParameterEffects(outputDirectory)


if __name__ == "__main__":
    GenerateThermalFigures(Path(__file__).resolve().parent)
