"""Generate deterministic frequency-selective I/Q imbalance figures and data."""

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def BuildRecommendedIqProfiles() -> Dict[str, Dict[str, object]]:
    """Return the documented flat, mild, moderate, and severe I/Q profiles.

    Processing details:
        Algorithm: Construct fresh dictionaries on every call so callers cannot
        mutate shared module state. Every explicit FIR is the complete effective
        response of its widely-linear branch, matching ``Channel`` semantics.

    Returns:
        result: Ordered profile mapping containing labels, styles, and FIR taps.
    """

    return {
        "flat_reference": {
            "label": "Flat reference",
            "color": "#64748b",
            "lineStyle": "--",
            "directFirTaps": (1.0 + 0.0j,),
            "imageFirTaps": (0.010 + 0.0j,),
        },
        "mild_frequency_selective": {
            "label": "Mild / calibrated residual",
            "color": "#15803d",
            "lineStyle": "-",
            "directFirTaps": (
                0.999 + 0.0j,
                0.004 - 0.003j,
                -0.001 + 0.001j,
            ),
            "imageFirTaps": (
                0.004 + 0.002j,
                -0.0015 + 0.001j,
                0.0005 - 0.0005j,
            ),
        },
        "moderate_edge_degradation": {
            "label": "Moderate / edge degradation",
            "color": "#d97706",
            "lineStyle": "-",
            "directFirTaps": (0.997 + 0.0j, 0.003 + 0.0j),
            "imageFirTaps": (0.019 + 0.0j, -0.009 + 0.0j),
        },
        "severe_asymmetric_stress": {
            "label": "Severe / asymmetric stress",
            "color": "#dc2626",
            "lineStyle": "-",
            "directFirTaps": (
                0.985 + 0.0j,
                0.025 - 0.018j,
                -0.008 + 0.006j,
            ),
            "imageFirTaps": (
                0.050 + 0.028j,
                -0.024 + 0.017j,
                0.010 - 0.008j,
            ),
        },
    }


def CalculateIqFrequencyResponses(
    frequencyHz: np.ndarray,
    sampleRateHz: float,
    directFirTaps: Sequence[complex],
    imageFirTaps: Sequence[complex],
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate direct response at f and image response at mirror frequency.

    Processing details:
        Algorithm: Evaluate the causal direct FIR at ``f`` and the conjugate
        branch FIR at ``-f``. The latter uses the positive exponential because
        an input tone at ``f`` creates its image basis at ``-f``.

    Args:
        frequencyHz: Input-tone frequency offsets from the carrier in hertz.
        sampleRateHz: Complex-baseband sample rate in hertz.
        directFirTaps: Complete effective direct-branch causal FIR.
        imageFirTaps: Complete effective conjugate-image causal FIR.

    Returns:
        result: Tuple ``(A(f), B(-f))`` on the requested frequency grid.
    """

    frequencyVector = np.asarray(frequencyHz, dtype=float).reshape(-1)
    directTaps = np.asarray(directFirTaps, dtype=np.complex128).reshape(-1)
    imageTaps = np.asarray(imageFirTaps, dtype=np.complex128).reshape(-1)
    directDelays = np.arange(directTaps.size, dtype=float)
    imageDelays = np.arange(imageTaps.size, dtype=float)
    directResponse = np.sum(
        directTaps.reshape(1, -1)
        * np.exp(
            -1j
            * 2.0
            * np.pi
            * frequencyVector.reshape(-1, 1)
            * directDelays.reshape(1, -1)
            / float(sampleRateHz)
        ),
        axis=1,
    )
    imageAtMirrorResponse = np.sum(
        imageTaps.reshape(1, -1)
        * np.exp(
            1j
            * 2.0
            * np.pi
            * frequencyVector.reshape(-1, 1)
            * imageDelays.reshape(1, -1)
            / float(sampleRateHz)
        ),
        axis=1,
    )
    return directResponse, imageAtMirrorResponse


def CalculateIrrDbCurve(
    frequencyHz: np.ndarray,
    sampleRateHz: float,
    directFirTaps: Sequence[complex],
    imageFirTaps: Sequence[complex],
) -> np.ndarray:
    """Calculate expected image-relative level versus input-tone frequency.

    Processing details:
        Algorithm: Evaluate ``20*log10(abs(B(-f)/A(f)))`` with a numerical
        floor only at zero magnitude. Negative dBc is expected and more
        negative values indicate better image rejection.

    Args:
        frequencyHz: Input-tone frequency offsets from carrier in hertz.
        sampleRateHz: Complex-baseband sample rate in hertz.
        directFirTaps: Complete effective direct-branch causal FIR.
        imageFirTaps: Complete effective conjugate-image causal FIR.

    Returns:
        result: Expected image-to-desired level in dBc at every frequency.
    """

    directResponse, imageAtMirrorResponse = CalculateIqFrequencyResponses(
        frequencyHz,
        sampleRateHz,
        directFirTaps,
        imageFirTaps,
    )
    magnitudeFloor = np.finfo(float).tiny
    return 20.0 * np.log10(
        np.maximum(np.abs(imageAtMirrorResponse), magnitudeFloor)
        / np.maximum(np.abs(directResponse), magnitudeFloor)
    )


def ConfigurePlotStyle() -> None:
    """Apply one deterministic documentation-figure style.

    Processing details:
        Algorithm: Configure a light canvas, restrained grid, readable labels,
        and fixed export resolution without using host-specific style files.

    Returns:
        result: None. Later Matplotlib figures inherit the selected settings.
    """

    plt.rcParams.update(
        {
            "figure.facecolor": "#f8fafc",
            "axes.facecolor": "#f8fafc",
            "axes.grid": True,
            "grid.alpha": 0.23,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10.0,
            "axes.titlesize": 12.0,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.0,
            "savefig.dpi": 180,
        }
    )


def GenerateIqIrrFigures(
    outputDirectory: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Generate the expected-IRR plot and its exact numeric CSV data.

    Processing details:
        Algorithm: Evaluate every documented profile over the full normalized
        Nyquist interval for 80 Msps, plot image-relative level and direct gain,
        then export the same arrays to CSV so documentation values are auditable.

    Args:
        outputDirectory: Optional destination. None selects this script folder.

    Returns:
        result: Paths to the generated PNG figure and CSV numeric data.
    """

    resolvedDirectory = (
        Path(outputDirectory)
        if outputDirectory is not None
        else Path(__file__).parent
    )
    resolvedDirectory.mkdir(parents=True, exist_ok=True)
    sampleRateHz = 80.0e6
    frequencyHz = np.linspace(-40.0e6, 40.0e6, 2001)
    frequencyMhz = frequencyHz / 1.0e6
    profiles = BuildRecommendedIqProfiles()
    ConfigurePlotStyle()
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11.2, 9.0),
        sharex=True,
        constrained_layout=True,
    )
    csvColumns = [frequencyMhz]
    csvHeaders = ["frequency_mhz"]
    for profileName, profile in profiles.items():
        directFirTaps = profile["directFirTaps"]
        imageFirTaps = profile["imageFirTaps"]
        directResponse, imageAtMirrorResponse = (
            CalculateIqFrequencyResponses(
                frequencyHz,
                sampleRateHz,
                directFirTaps,
                imageFirTaps,
            )
        )
        irrDb = CalculateIrrDbCurve(
            frequencyHz,
            sampleRateHz,
            directFirTaps,
            imageFirTaps,
        )
        directGainDb = 20.0 * np.log10(np.abs(directResponse))
        imageGainDb = 20.0 * np.log10(np.abs(imageAtMirrorResponse))
        axes[0].plot(
            frequencyMhz,
            irrDb,
            color=str(profile["color"]),
            linestyle=str(profile["lineStyle"]),
            linewidth=2.2,
            label=str(profile["label"]),
        )
        axes[1].plot(
            frequencyMhz,
            directGainDb,
            color=str(profile["color"]),
            linestyle=str(profile["lineStyle"]),
            linewidth=2.0,
            label=str(profile["label"]),
        )
        csvHeaders.extend(
            (
                f"{profileName}_irr_db",
                f"{profileName}_direct_gain_db",
                f"{profileName}_image_gain_db",
            )
        )
        csvColumns.extend((irrDb, directGainDb, imageGainDb))

    axes[0].axhline(-40.0, color="#475569", linestyle=":", linewidth=1.0)
    axes[0].axhline(-30.0, color="#475569", linestyle=":", linewidth=1.0)
    axes[0].axhline(-20.0, color="#475569", linestyle=":", linewidth=1.0)
    axes[0].set_title("Expected image relative level versus input-tone frequency", loc="left")
    axes[0].set_ylabel("Image / desired (dBc)")
    axes[0].set_ylim(-55.0, -15.0)
    axes[0].legend(loc="lower left", frameon=False, ncol=2)
    axes[0].text(
        39.0,
        -53.0,
        "more negative is better",
        color="#334155",
        ha="right",
    )
    axes[1].set_title("Direct-branch gain ripple from the same FIR profiles", loc="left")
    axes[1].set_xlabel("Input-tone offset from carrier (MHz), Fs = 80 MHz")
    axes[1].set_ylabel("Direct gain (dB)")
    axes[1].set_ylim(-0.60, 0.20)
    axes[1].legend(loc="lower left", frameon=False, ncol=2)
    for axis in axes:
        axis.axvline(0.0, color="#64748b", linewidth=0.9, alpha=0.5)
        axis.set_xlim(-40.0, 40.0)

    figure.suptitle(
        "Frequency-selective widely-linear I/Q imbalance profiles",
        fontsize=15.0,
        fontweight="bold",
    )
    figurePath = resolvedDirectory / "iq_irr_frequency_profiles.png"
    figure.savefig(figurePath, bbox_inches="tight")
    plt.close(figure)

    csvPath = resolvedDirectory / "iq_irr_frequency_profiles.csv"
    csvMatrix = np.column_stack(csvColumns)
    np.savetxt(
        csvPath,
        csvMatrix,
        delimiter=",",
        header=",".join(csvHeaders),
        comments="",
        fmt="%.12g",
    )
    return figurePath, csvPath


if __name__ == "__main__":
    for generatedPath in GenerateIqIrrFigures():
        print(generatedPath)
