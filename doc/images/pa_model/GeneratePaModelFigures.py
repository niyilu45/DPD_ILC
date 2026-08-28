"""Generate schematic PA parameter-effect figures used by ``doc/PaModel.md``.

The curves intentionally emphasize which physical region each parameter controls.
They are explanatory sketches rather than parameter sweeps or fitted device data.
"""

from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def CalculateRappGain(
    inputAmplitude: np.ndarray,
    linearGain: float,
    saturationAmplitude: float,
    rappSmoothness: float,
) -> np.ndarray:
    """Return the Rapp voltage-gain curve for a nonnegative amplitude grid.

    The implementation mirrors the project model: ``saturationAmplitude`` sets
    the input-referred knee and ``rappSmoothness`` sets the knee sharpness.
    """

    normalizedPower = np.power(
        inputAmplitude / saturationAmplitude,
        2.0 * rappSmoothness,
    )
    return linearGain / np.power(
        1.0 + normalizedPower,
        1.0 / (2.0 * rappSmoothness),
    )


def ConfigureAxes(axis: plt.Axes, title: str) -> None:
    """Apply a common visual style to one explanatory panel."""

    axis.set_title(title, loc="left", fontsize=12, fontweight="bold")
    axis.set_xlabel("Normalized input amplitude")
    axis.set_ylabel("Voltage gain (dB)")
    axis.grid(True, alpha=0.22)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def PlotRappPanel(axis: plt.Axes, inputAmplitude: np.ndarray) -> None:
    """Draw how each Rapp parameter changes a distinct part of the gain curve."""

    nominalGain = CalculateRappGain(inputAmplitude, 1.0, 1.0, 3.0)
    earlyKneeGain = CalculateRappGain(inputAmplitude, 1.0, 0.78, 3.0)
    softKneeGain = CalculateRappGain(inputAmplitude, 1.0, 1.0, 1.4)

    axis.plot(inputAmplitude, 20.0 * np.log10(nominalGain), linewidth=2.5, label="nominal")
    axis.plot(
        inputAmplitude,
        20.0 * np.log10(earlyKneeGain),
        linestyle="--",
        linewidth=2.0,
        label="lower saturationAmplitude",
    )
    axis.plot(
        inputAmplitude,
        20.0 * np.log10(softKneeGain),
        linestyle=":",
        linewidth=2.3,
        label="lower rappSmoothness",
    )
    axis.axvline(1.0, color="#555555", alpha=0.35, linewidth=1.2)
    axis.annotate(
        "linearGain shifts the whole curve",
        xy=(0.25, -0.03),
        xytext=(0.42, -2.2),
        arrowprops={"arrowstyle": "->", "color": "#444444"},
    )
    axis.annotate(
        "saturationAmplitude moves the knee",
        xy=(0.78, -1.0),
        xytext=(0.10, -6.0),
        arrowprops={"arrowstyle": "->", "color": "#444444"},
    )
    axis.annotate(
        "rappSmoothness controls knee sharpness",
        xy=(1.04, -2.6),
        xytext=(1.10, -7.2),
        arrowprops={"arrowstyle": "->", "color": "#444444"},
    )
    axis.text(
        0.57,
        0.08,
        "No state, no FIR, no intrinsic AM-PM",
        transform=axis.transAxes,
        color="#16794b",
        fontweight="bold",
    )
    axis.set_ylim(-10.5, 0.8)
    axis.legend(loc="lower left", frameon=False, fontsize=9)
    ConfigureAxes(axis, "A. Rapp: a strictly memoryless SSPA reference")


def PlotWienerPanel(axis: plt.Axes, inputAmplitude: np.ndarray) -> None:
    """Draw a schematic Wiener static compression curve and memory controls."""

    staticGain = 1.03 - 0.22 * inputAmplitude**2 + 0.035 * inputAmplitude**4
    axis.plot(
        inputAmplitude,
        20.0 * np.log10(np.maximum(staticGain, 1.0e-6)),
        linewidth=2.5,
        color="#d97706",
    )
    axis.annotate(
        "amAmCoefficients set static curvature",
        xy=(1.16, 20.0 * np.log10(1.03 - 0.22 * 1.16**2 + 0.035 * 1.16**4)),
        xytext=(0.56, -5.9),
        arrowprops={"arrowstyle": "->", "color": "#444444"},
    )
    axis.text(
        0.03,
        0.12,
        "linearTaps -> frequency response and group delay\namPmCoefficients -> phase versus amplitude",
        transform=axis.transAxes,
        color="#8a4f05",
        fontweight="bold",
    )
    axis.set_ylim(-10.5, 1.0)
    ConfigureAxes(axis, "B. Wiener: linear memory followed by static AM-AM/AM-PM")


def PlotGmpPanel(axis: plt.Axes, inputAmplitude: np.ndarray) -> None:
    """Draw the default GMP curve beside its full-strength stress reference."""

    referenceSteadyStateCoefficients = {
        1: 1.261692 + 0.014052j,
        3: -0.291144 + 0.054204j,
        5: 0.031812 - 0.022452j,
        7: -0.000168 + 0.002784j,
    }
    nonlinearScale = 0.135
    defaultSteadyStateCoefficients = {
        nonlinearOrder: coefficient
        * (1.0 if nonlinearOrder == 1 else nonlinearScale)
        for nonlinearOrder, coefficient in (
            referenceSteadyStateCoefficients.items()
        )
    }
    defaultSteadyOutput = sum(
        coefficient * inputAmplitude**nonlinearOrder
        for nonlinearOrder, coefficient in (
            defaultSteadyStateCoefficients.items()
        )
    )
    referenceSteadyOutput = sum(
        coefficient * inputAmplitude**nonlinearOrder
        for nonlinearOrder, coefficient in (
            referenceSteadyStateCoefficients.items()
        )
    )
    defaultGainDb = 20.0 * np.log10(
        np.maximum(np.abs(defaultSteadyOutput) / inputAmplitude, 1.0e-6)
    )
    referenceGainDb = 20.0 * np.log10(
        np.maximum(
            np.abs(referenceSteadyOutput) / inputAmplitude,
            1.0e-6,
        )
    )
    dynamicHalfWidthDb = 0.025 * np.minimum(inputAmplitude, 1.5)
    axis.plot(
        inputAmplitude,
        defaultGainDb,
        linewidth=2.5,
        color="#2563eb",
        label="default nonlinearScale=0.135",
    )
    axis.plot(
        inputAmplitude,
        referenceGainDb,
        linewidth=1.8,
        linestyle="--",
        color="#64748b",
        label="nonlinearScale=1.0 stress reference",
    )
    axis.fill_between(
        inputAmplitude,
        defaultGainDb - dynamicHalfWidthDb,
        defaultGainDb + dynamicHalfWidthDb,
        color="#60a5fa",
        alpha=0.22,
        label="mild default memory trajectory",
    )
    annotationAmplitude = 1.05
    annotationOutput = sum(
        coefficient * annotationAmplitude**nonlinearOrder
        for nonlinearOrder, coefficient in (
            defaultSteadyStateCoefficients.items()
        )
    )
    annotationGainDb = 20.0 * np.log10(
        abs(annotationOutput) / annotationAmplitude
    )
    axis.annotate(
        "nonlinearScale adjusts orders 3 and above",
        xy=(annotationAmplitude, annotationGainDb),
        xytext=(0.10, -2.4),
        arrowprops={"arrowstyle": "->", "color": "#444444"},
    )
    axis.text(
        0.03,
        0.08,
        "order 1 preserves the small-signal gain\n"
        "memoryDepth -> zero-sum main residual\n"
        "crossMemoryDepth -> zero-sum envelope-history residual",
        transform=axis.transAxes,
        color="#174ea6",
        fontweight="bold",
    )
    axis.set_ylim(-5.0, 3.0)
    axis.legend(loc="lower right", frameon=False, fontsize=9)
    ConfigureAxes(axis, "C. GMP: polynomial order and delayed-envelope memory")


def PlotDohertyPanel(axis: plt.Axes, inputAmplitude: np.ndarray) -> None:
    """Draw the carrier-only and carrier-plus-peaking regions of a Doherty PA."""

    turnOnAmplitude = 0.72
    transitionWidth = 0.18
    peakingActivation = np.clip(
        (inputAmplitude - turnOnAmplitude) / transitionWidth,
        0.0,
        1.0,
    )
    carrierGain = 0.98 - 0.17 * inputAmplitude**2
    loadModulation = 0.10 * peakingActivation
    peakingGain = 0.08 * peakingActivation
    totalGain = np.maximum(carrierGain + loadModulation + peakingGain, 1.0e-6)

    axis.plot(inputAmplitude, 20.0 * np.log10(totalGain), linewidth=2.5, color="#7c3aed")
    axis.axvline(turnOnAmplitude, color="#7c3aed", linestyle="--", alpha=0.55)
    axis.axvspan(
        turnOnAmplitude,
        turnOnAmplitude + transitionWidth,
        color="#c4b5fd",
        alpha=0.28,
    )
    axis.text(
        0.10,
        -2.6,
        "Carrier only",
        color="#5b21b6",
        fontweight="bold",
    )
    axis.text(
        0.84,
        -2.6,
        "Peaking turn-on\nand load modulation",
        color="#5b21b6",
        fontweight="bold",
    )
    axis.annotate(
        "turnOnAmplitude moves the second knee\ntransitionWidth changes its width",
        xy=(turnOnAmplitude + 0.08, -0.5),
        xytext=(1.05, -5.5),
        arrowprops={"arrowstyle": "->", "color": "#444444"},
    )
    axis.set_ylim(-9.0, 1.0)
    ConfigureAxes(axis, "D. Doherty: two branches create two operating regions")


def GeneratePaModelFigures(outputDirectory: Path = None) -> Tuple[Path]:
    """Generate and return the paths of all PA-model explanatory figures."""

    resolvedDirectory = Path(outputDirectory) if outputDirectory else Path(__file__).parent
    resolvedDirectory.mkdir(parents=True, exist_ok=True)
    inputAmplitude = np.linspace(0.04, 1.55, 500)

    figure, axes = plt.subplots(4, 1, figsize=(11.0, 18.0), constrained_layout=True)
    PlotRappPanel(axes[0], inputAmplitude)
    PlotWienerPanel(axes[1], inputAmplitude)
    PlotGmpPanel(axes[2], inputAmplitude)
    PlotDohertyPanel(axes[3], inputAmplitude)
    figure.suptitle(
        "How PA configuration parameters shape the modeled response",
        fontsize=16,
        fontweight="bold",
    )

    outputPath = resolvedDirectory / "pa_gain_parameter_effects.png"
    figure.savefig(outputPath, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return (outputPath,)


if __name__ == "__main__":
    for generatedPath in GeneratePaModelFigures():
        print(generatedPath)
