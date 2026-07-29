"""Object-oriented plotting utilities for DPD-ILC result curves."""

from collections import ChainMap
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .ConfigUtils import (
    FilterRecognizedParameters,
    RecognizedParameterView,
)

# Support ``inc.utils`` and the compatibility ``utils`` package used when
# callers place the ``inc`` directory on sys.path.
if __package__ and "." in __package__:
    from ..lib.Analysis import PowerEvmCurve
else:
    from lib.Analysis import PowerEvmCurve


class Draw:
    """Configure and render plots without performing metric calculations.

    The class owns only visualization settings and Matplotlib operations.
    Numerical curve generation and CSV/JSON serialization remain in
    ``Analysis``, which keeps calculation and presentation responsibilities
    independent.
    """

    def __init__(
        self,
        parameters: Optional[Mapping[str, object]] = None,
        **parameterOverrides: object,
    ) -> None:
        """Initialize live plotting parameters with ChainMap precedence.

        Processing details:
            Algorithm: Define immutable plotting defaults inside this
            constructor, then layer constructor overrides and a caller-owned
            mapping ahead of them so callers never repeat default values.

        Args:
            parameters: Optional external mapping layered ahead of defaults.
            parameterOverrides: Highest-priority local plotting overrides.

        Returns:
            result: None. The validated settings are retained by the object.
        """

        self.defaultParameters: Mapping[str, object] = MappingProxyType(
            {
                "powerEvmFileStem": "power_evm_curve",
                "convergenceFileStem": "ilc_convergence",
                "imdFileStem": "two_tone_imd_comparison",
                "paFrequencyFileStem": "pa_frequency_response",
                "paMemoryFileStem": "pa_memory_effect",
                "paNonlinearityFileStem": (
                    "pa_nonlinearity_comparison"
                ),
                "paPowerFileStem": "pa_power_characteristics",
                "dpdGmpFileStem": "dpd_gmp_performance",
                "figureWidthInches": 10.5,
                "figureHeightInches": 6.2,
                "figureDpi": 180,
                "lineWidth": 1.8,
                "markerSize": 5.0,
                "legendColumnThreshold": 6,
                "plotTitle": "Power-EVM comparison",
                "convergencePlotTitle": "ILC MSE convergence",
                "imdPlotTitle": "Two-tone ILC intermodulation comparison",
                "paFrequencyPlotTitle": (
                    "PA small-signal frequency response"
                ),
                "paMemoryPlotTitle": (
                    "PA two-tone memory-effect comparison"
                ),
                "paNonlinearityPlotTitle": (
                    "PA nominal intermodulation comparison"
                ),
                "paPowerPlotTitle": (
                    "PA output-power-dependent characteristics"
                ),
                "dpdGmpPlotTitle": (
                    "PA-analysis-driven DPD-GMP improvements"
                ),
                "xAxisLabel": "PA output power per chain (dBm)",
                "yAxisLabel": "RMS EVM (dB, lower is better)",
                "convergenceXAxisLabel": "ILC iteration",
                "convergenceYAxisLabel": (
                    "Normalized error / EVM (dB, lower is better)"
                ),
                "imdYAxisLabel": (
                    "Worst-side intermodulation (dBc, lower is better)"
                ),
            }
        )
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping or None")
        externalParameters: Mapping[str, object] = (
            {}
            if parameters is None
            else RecognizedParameterView(
                parameters,
                self.defaultParameters,
                "Draw",
            )
        )
        recognizedOverrides = FilterRecognizedParameters(
            parameterOverrides,
            self.defaultParameters,
            "Draw",
        )
        self.parameters: ChainMap[str, object] = ChainMap(
            recognizedOverrides,
            externalParameters,
            self.defaultParameters,
        )
        self.ValidateParameters()

    def GetParameters(self) -> Dict[str, object]:
        """Return a flattened snapshot of resolved plotting parameters.

        Processing details:
            Algorithm: Resolve every ChainMap layer using normal mapping
            precedence and copy the result so callers cannot mutate internal
            configuration through the returned dictionary.

        Returns:
            result: Dictionary containing all effective drawing settings.
        """

        return dict(self.parameters)

    def UpdateParameters(self, **parameterOverrides: object) -> None:
        """Apply validated highest-priority plotting parameter overrides.

        Processing details:
            Algorithm: Update the local ChainMap layer transactionally and
            restore its previous state if any new value fails validation.

        Args:
            parameterOverrides: Plotting values to place in the local layer.

        Returns:
            result: None. Valid values become active for subsequent plots.
        """

        recognizedOverrides = FilterRecognizedParameters(
            parameterOverrides,
            self.defaultParameters,
            "Draw.UpdateParameters",
        )
        previousOverrides = dict(self.parameters.maps[0])
        self.parameters.maps[0].update(recognizedOverrides)
        try:
            self.ValidateParameters()
        except (TypeError, ValueError):
            self.parameters.maps[0].clear()
            self.parameters.maps[0].update(previousOverrides)
            raise

    def ValidateParameters(self) -> None:
        """Validate every resolved plotting parameter and file-name setting.

        Processing details:
            Algorithm: Check string fields, positive dimensions, integer DPI,
            and legend layout constraints in a deterministic order. Unknown
            keys have already been warned about and filtered at the boundary.

        Returns:
            result: None. Invalid configuration raises a descriptive error.
        """

        for parameterName in (
            "powerEvmFileStem",
            "convergenceFileStem",
            "imdFileStem",
            "paFrequencyFileStem",
            "paMemoryFileStem",
            "paNonlinearityFileStem",
            "paPowerFileStem",
            "dpdGmpFileStem",
        ):
            fileStem = self.parameters[parameterName]
            if not isinstance(fileStem, str):
                raise TypeError(f"{parameterName} must be a string")
            if not fileStem or any(
                character in fileStem for character in '<>:"/\\|?*'
            ):
                raise ValueError(
                    f"{parameterName} must be a valid simple file name"
                )

        positiveFloatNames = (
            "figureWidthInches",
            "figureHeightInches",
            "lineWidth",
            "markerSize",
        )
        for parameterName in positiveFloatNames:
            parameterValue = self.parameters[parameterName]
            if not isinstance(parameterValue, (int, float)) or isinstance(
                parameterValue, bool
            ):
                raise TypeError(f"{parameterName} must be numeric")
            if float(parameterValue) <= 0.0:
                raise ValueError(f"{parameterName} must be positive")

        for parameterName in ("figureDpi", "legendColumnThreshold"):
            parameterValue = self.parameters[parameterName]
            if not isinstance(parameterValue, int) or isinstance(
                parameterValue, bool
            ):
                raise TypeError(f"{parameterName} must be an integer")
            if parameterValue < 1:
                raise ValueError(f"{parameterName} must be positive")

        for parameterName in (
            "plotTitle",
            "convergencePlotTitle",
            "xAxisLabel",
            "yAxisLabel",
            "convergenceXAxisLabel",
            "convergenceYAxisLabel",
            "imdPlotTitle",
            "imdYAxisLabel",
            "paFrequencyPlotTitle",
            "paMemoryPlotTitle",
            "paNonlinearityPlotTitle",
            "paPowerPlotTitle",
            "dpdGmpPlotTitle",
        ):
            parameterValue = self.parameters[parameterName]
            if not isinstance(parameterValue, str):
                raise TypeError(f"{parameterName} must be a string")
            if not parameterValue:
                raise ValueError(f"{parameterName} cannot be empty")

    def ValidatePowerEvmCurve(self, powerEvmCurve: PowerEvmCurve) -> None:
        """Check that a power-EVM curve is complete and drawable.

        Processing details:
            Algorithm: Verify the result type, common vector length, finite
            horizontal coordinates, at least one method, and one finite EVM
            vector per method before Matplotlib allocates a figure.

        Args:
            powerEvmCurve: Calculated multi-method curve supplied by Analysis.

        Returns:
            result: None. Malformed or non-finite data raises an error.
        """

        if not isinstance(powerEvmCurve, PowerEvmCurve):
            raise TypeError("powerEvmCurve must be a PowerEvmCurve")
        pointCount = powerEvmCurve.outputPowerDbmValues.size
        if pointCount < 2:
            raise ValueError("powerEvmCurve must contain at least two points")
        if (
            powerEvmCurve.driveScaleValues.size != pointCount
            or powerEvmCurve.targetOutputRmsValues.size != pointCount
        ):
            raise ValueError("power-EVM coordinate arrays must have equal length")
        if not np.all(np.isfinite(powerEvmCurve.outputPowerDbmValues)):
            raise ValueError("power-EVM output powers must be finite")
        if np.any(np.diff(powerEvmCurve.outputPowerDbmValues) <= 0.0):
            raise ValueError(
                "power-EVM output powers must be strictly increasing in dBm"
            )
        if (
            not np.all(np.isfinite(powerEvmCurve.driveScaleValues))
            or np.any(powerEvmCurve.driveScaleValues <= 0.0)
            or not np.all(
                np.isfinite(powerEvmCurve.targetOutputRmsValues)
            )
            or np.any(powerEvmCurve.targetOutputRmsValues <= 0.0)
        ):
            raise ValueError(
                "power-EVM drive scales and RMS targets must be positive"
            )
        if not powerEvmCurve.evmDbByMethod:
            raise ValueError("powerEvmCurve must contain at least one method")
        for methodName, evmDbValues in powerEvmCurve.evmDbByMethod.items():
            if not isinstance(methodName, str) or not methodName:
                raise ValueError("power-EVM method names must be non-empty")
            if np.asarray(evmDbValues).size != pointCount:
                raise ValueError(
                    f"power-EVM method '{methodName}' has an invalid length"
                )
            if not np.all(np.isfinite(evmDbValues)):
                raise ValueError(
                    f"power-EVM method '{methodName}' contains non-finite data"
                )

    def CreatePowerEvmFigure(
        self, powerEvmCurve: PowerEvmCurve
    ) -> Any:
        """Create one comparison figure containing every EVM method curve.

        Processing details:
            Algorithm: Assign deterministic marker and line-style cycles,
            label both axes, enable a reading grid, and move large legends
            outside the axes to preserve the data region.

        Args:
            powerEvmCurve: Calculated output-power and per-method EVM vectors.

        Returns:
            result: Matplotlib Figure ready for display or file output.
        """

        self.ValidateParameters()
        self.ValidatePowerEvmCurve(powerEvmCurve)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise RuntimeError(
                "matplotlib is required to create the power-EVM figure"
            ) from error

        figure, axes = plt.subplots(
            figsize=(
                float(self.parameters["figureWidthInches"]),
                float(self.parameters["figureHeightInches"]),
            )
        )
        markerStyles = ("o", "s", "^", "D", "v", "P", "X", "<", ">")
        lineStyles = ("-", "--", "-.", ":")
        methodNames = list(powerEvmCurve.evmDbByMethod)
        for methodIndex, methodName in enumerate(methodNames):
            axes.plot(
                powerEvmCurve.outputPowerDbmValues,
                powerEvmCurve.evmDbByMethod[methodName],
                label=methodName,
                marker=markerStyles[methodIndex % len(markerStyles)],
                linestyle=lineStyles[
                    (methodIndex // len(markerStyles)) % len(lineStyles)
                ],
                linewidth=float(self.parameters["lineWidth"]),
                markersize=float(self.parameters["markerSize"]),
            )
        axes.set_xlabel(str(self.parameters["xAxisLabel"]))
        axes.set_ylabel(str(self.parameters["yAxisLabel"]))
        axes.set_title(str(self.parameters["plotTitle"]))
        axes.grid(True, which="both", linestyle=":", linewidth=0.7)
        if len(methodNames) <= int(self.parameters["legendColumnThreshold"]):
            axes.legend(loc="best")
        else:
            axes.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
            )
        figure.tight_layout()
        return figure

    def SavePowerEvmCurve(
        self,
        powerEvmCurve: PowerEvmCurve,
        outputDirectory: Path,
        fileStem: Optional[str] = None,
    ) -> Path:
        """Render and save a power-EVM comparison as a PNG image.

        Processing details:
            Algorithm: Resolve the configured or per-call file stem, create
            the output directory, build the figure, save it at the configured
            DPI, and close Matplotlib resources even if file output fails.

        Args:
            powerEvmCurve: Curve calculated by Analysis.
            outputDirectory: Directory in which the PNG image is written.
            fileStem: Optional per-call name overriding the ChainMap value.

        Returns:
            result: Path to the generated PNG comparison figure.
        """

        self.ValidateParameters()
        selectedFileStem = (
            str(self.parameters["powerEvmFileStem"])
            if fileStem is None
            else fileStem
        )
        if not isinstance(selectedFileStem, str):
            raise TypeError("fileStem must be a string")
        if not selectedFileStem or any(
            character in selectedFileStem for character in '<>:"/\\|?*'
        ):
            raise ValueError("fileStem must be a valid simple file name")

        outputPath = Path(outputDirectory)
        outputPath.mkdir(parents=True, exist_ok=True)
        figurePath = outputPath / f"{selectedFileStem}.png"
        figure = self.CreatePowerEvmFigure(powerEvmCurve)
        try:
            figure.savefig(
                figurePath,
                dpi=int(self.parameters["figureDpi"]),
                bbox_inches="tight",
            )
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)
        return figurePath

    def ValidateTwoToneMetrics(
        self, metricsByMethod: Mapping[str, Mapping[str, float]]
    ) -> None:
        """Validate a multi-method IM3, IM5, and IM7 comparison mapping.

        Processing details:
            Algorithm: Require at least one nonempty method name and finite
            worse-side dBc values for all three requested odd orders before any
            visualization backend or output file is created.

        Args:
            metricsByMethod: Mapping from method labels to metric dictionaries.

        Returns:
            result: None. Missing or nonfinite comparison data raises an error.
        """

        if not isinstance(metricsByMethod, Mapping) or not metricsByMethod:
            raise ValueError("metricsByMethod must be a nonempty mapping")
        requiredMetricNames = (
            "im3WorstDbc",
            "im5WorstDbc",
            "im7WorstDbc",
        )
        for methodName, methodMetrics in metricsByMethod.items():
            if not isinstance(methodName, str) or not methodName:
                raise ValueError("two-tone method names must be nonempty")
            if not isinstance(methodMetrics, Mapping):
                raise TypeError("every two-tone metric value must be a mapping")
            for metricName in requiredMetricNames:
                if metricName not in methodMetrics:
                    raise ValueError(
                        f"method '{methodName}' is missing {metricName}"
                    )
                metricValue = methodMetrics[metricName]
                if (
                    not isinstance(metricValue, (int, float))
                    or isinstance(metricValue, bool)
                    or not np.isfinite(metricValue)
                ):
                    raise ValueError(
                        f"method '{methodName}' has invalid {metricName}"
                    )

    def CreateTwoToneImdFigure(
        self, metricsByMethod: Mapping[str, Mapping[str, float]]
    ) -> Any:
        """Create one grouped-bar IM3, IM5, and IM7 comparison figure.

        Processing details:
            Algorithm: Preserve method insertion order, place three adjacent
            order bars at each method, label values in dBc with the documented
            more-negative-is-better convention, and rotate method labels for a
            readable single-figure all-ILC comparison.

        Args:
            metricsByMethod: Mapping from method names to two-tone metrics.

        Returns:
            result: Matplotlib Figure ready for display or PNG output.
        """

        self.ValidateParameters()
        self.ValidateTwoToneMetrics(metricsByMethod)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise RuntimeError(
                "matplotlib is required to create the two-tone IMD figure"
            ) from error
        methodNames = list(metricsByMethod)
        methodPositions = np.arange(len(methodNames), dtype=float)
        barWidth = 0.24
        figure, axes = plt.subplots(
            figsize=(
                float(self.parameters["figureWidthInches"]),
                float(self.parameters["figureHeightInches"]),
            )
        )
        for orderIndex, nonlinearOrder in enumerate((3, 5, 7)):
            metricName = f"im{nonlinearOrder}WorstDbc"
            orderValues = [
                float(metricsByMethod[methodName][metricName])
                for methodName in methodNames
            ]
            axes.bar(
                methodPositions + (orderIndex - 1) * barWidth,
                orderValues,
                width=barWidth,
                label=f"IM{nonlinearOrder}",
            )
        axes.set_xticks(methodPositions)
        axes.set_xticklabels(methodNames, rotation=25, ha="right")
        axes.set_ylabel(str(self.parameters["imdYAxisLabel"]))
        axes.set_title(str(self.parameters["imdPlotTitle"]))
        axes.grid(True, axis="y", linestyle=":", linewidth=0.7)
        axes.legend(loc="best")
        figure.tight_layout()
        return figure

    def SaveTwoToneImdComparison(
        self,
        metricsByMethod: Mapping[str, Mapping[str, float]],
        outputDirectory: Path,
        fileStem: Optional[str] = None,
    ) -> Path:
        """Render and save the all-method two-tone IMD comparison as PNG.

        Processing details:
            Algorithm: Resolve and validate the configured or per-call file
            stem, create the destination directory, build the grouped-bar
            figure, save it at configured DPI, and always close its resources.

        Args:
            metricsByMethod: Mapping from method labels to IM metric dictionaries.
            outputDirectory: Destination directory for the PNG artifact.
            fileStem: Optional simple filename overriding ``imdFileStem``.

        Returns:
            result: Path to the saved comparison figure.
        """

        selectedFileStem = (
            str(self.parameters["imdFileStem"])
            if fileStem is None
            else fileStem
        )
        if (
            not isinstance(selectedFileStem, str)
            or not selectedFileStem
            or any(
                character in selectedFileStem for character in '<>:"/\\|?*'
            )
        ):
            raise ValueError("fileStem must be a valid simple file name")
        outputPath = Path(outputDirectory)
        outputPath.mkdir(parents=True, exist_ok=True)
        figurePath = outputPath / f"{selectedFileStem}.png"
        figure = self.CreateTwoToneImdFigure(metricsByMethod)
        try:
            figure.savefig(
                figurePath,
                dpi=int(self.parameters["figureDpi"]),
                bbox_inches="tight",
            )
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)
        return figurePath

    def ValidatePaSeries(
        self,
        dataByModel: Mapping[
            str, Sequence[Mapping[str, object]]
        ],
        requiredFieldNames: Sequence[str],
        dataName: str,
    ) -> None:
        """Validate model-grouped finite PA characterization point series.

        Processing details:
            Algorithm: Require a nonempty model mapping, nonempty point
            sequence per model, every requested scalar field, and finite
            numeric values before any plot allocates resources.

        Args:
            dataByModel: Model names mapped to ordered point dictionaries.
            requiredFieldNames: Numeric fields required in every point.
            dataName: Human-readable series name for validation errors.

        Returns:
            result: None. Malformed or nonfinite data raises an exception.
        """

        if not isinstance(dataByModel, Mapping) or not dataByModel:
            raise ValueError(f"{dataName} must be a nonempty mapping")
        if (
            isinstance(requiredFieldNames, (str, bytes))
            or not requiredFieldNames
        ):
            raise ValueError(
                "requiredFieldNames must be a nonempty sequence"
            )
        for modelName, modelPoints in dataByModel.items():
            if not isinstance(modelName, str) or not modelName:
                raise ValueError("PA model names must be nonempty strings")
            if (
                isinstance(modelPoints, (str, bytes))
                or not isinstance(modelPoints, Sequence)
                or len(modelPoints) == 0
            ):
                raise ValueError(
                    f"{dataName} for '{modelName}' must be nonempty"
                )
            for pointIndex, modelPoint in enumerate(modelPoints):
                if not isinstance(modelPoint, Mapping):
                    raise TypeError(
                        f"{dataName}[{modelName}][{pointIndex}] "
                        "must be a mapping"
                    )
                for fieldName in requiredFieldNames:
                    fieldValue = modelPoint.get(fieldName)
                    if (
                        not isinstance(fieldValue, (int, float))
                        or isinstance(fieldValue, bool)
                        or not np.isfinite(fieldValue)
                    ):
                        raise ValueError(
                            f"{dataName}[{modelName}][{pointIndex}]."
                            f"{fieldName} must be finite"
                        )

    def ValidatePaSummary(
        self,
        summaryByModel: Mapping[str, Mapping[str, object]],
    ) -> None:
        """Validate scalar PA memory and nonlinear summary dictionaries.

        Processing details:
            Algorithm: Require each model to provide finite dynamic
            hysteresis and nominal IM3/IM5/IM7 values used by the two bar
            comparisons.

        Args:
            summaryByModel: Model names mapped to summary dictionaries.

        Returns:
            result: None. Missing or nonfinite summary values raise an error.
        """

        if not isinstance(summaryByModel, Mapping) or not summaryByModel:
            raise ValueError("summaryByModel must be a nonempty mapping")
        requiredFields = (
            "dynamicGainHysteresisDb",
            "dynamicPhaseHysteresisDegrees",
            "nominalIm3Dbc",
            "nominalIm5Dbc",
            "nominalIm7Dbc",
        )
        for modelName, modelSummary in summaryByModel.items():
            if (
                not isinstance(modelName, str)
                or not modelName
                or not isinstance(modelSummary, Mapping)
            ):
                raise TypeError(
                    "each PA summary must have a nonempty model name "
                    "and mapping value"
                )
            for fieldName in requiredFields:
                fieldValue = modelSummary.get(fieldName)
                if (
                    not isinstance(fieldValue, (int, float))
                    or isinstance(fieldValue, bool)
                    or not np.isfinite(fieldValue)
                ):
                    raise ValueError(
                        f"summaryByModel[{modelName}].{fieldName} "
                        "must be finite"
                    )

    def CreatePaFrequencyResponseFigure(
        self,
        frequencyResponseByModel: Mapping[
            str, Sequence[Mapping[str, object]]
        ],
    ) -> Any:
        """Create aligned PA small-signal gain and phase response panels.

        Processing details:
            Algorithm: Sort each model's exact-tone points by frequency and
            draw gain and already unwrapped phase on shared MHz coordinates
            with consistent model labels, markers, grids, and titles.

        Args:
            frequencyResponseByModel: Model-grouped response point mappings.

        Returns:
            result: Matplotlib Figure containing gain and phase panels.
        """

        self.ValidateParameters()
        self.ValidatePaSeries(
            frequencyResponseByModel,
            ("frequencyHz", "gainDb", "phaseDegrees"),
            "frequencyResponseByModel",
        )
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise RuntimeError(
                "matplotlib is required for PA frequency-response plots"
            ) from error
        figure, axes = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(
                float(self.parameters["figureWidthInches"]),
                float(self.parameters["figureHeightInches"]),
            ),
        )
        markerStyles = ("o", "s", "^", "D", "v")
        for modelIndex, (modelName, modelPoints) in enumerate(
            frequencyResponseByModel.items()
        ):
            sortedPoints = sorted(
                modelPoints,
                key=lambda point: float(point["frequencyHz"]),
            )
            frequencyMhz = np.asarray(
                [
                    float(point["frequencyHz"]) * 1.0e-6
                    for point in sortedPoints
                ]
            )
            axes[0].plot(
                frequencyMhz,
                [float(point["gainDb"]) for point in sortedPoints],
                label=modelName,
                marker=markerStyles[
                    modelIndex % len(markerStyles)
                ],
                linewidth=float(self.parameters["lineWidth"]),
                markersize=float(self.parameters["markerSize"]),
            )
            axes[1].plot(
                frequencyMhz,
                [
                    float(point["phaseDegrees"])
                    for point in sortedPoints
                ],
                label=modelName,
                marker=markerStyles[
                    modelIndex % len(markerStyles)
                ],
                linewidth=float(self.parameters["lineWidth"]),
                markersize=float(self.parameters["markerSize"]),
            )
        axes[0].set_ylabel("Small-signal gain (dB)")
        axes[1].set_ylabel("Unwrapped phase (degrees)")
        axes[1].set_xlabel("Complex-baseband frequency (MHz)")
        axes[0].set_title(
            str(self.parameters["paFrequencyPlotTitle"])
        )
        for selectedAxes in axes:
            selectedAxes.grid(
                True, which="both", linestyle=":", linewidth=0.7
            )
            selectedAxes.legend(loc="best")
        figure.tight_layout()
        return figure

    def SavePaFrequencyResponse(
        self,
        frequencyResponseByModel: Mapping[
            str, Sequence[Mapping[str, object]]
        ],
        outputDirectory: Path,
        fileStem: Optional[str] = None,
    ) -> Path:
        """Save the PA small-signal gain/phase comparison PNG.

        Processing details:
            Algorithm: Resolve a safe filename, create the output directory,
            delegate all rendering to CreatePaFrequencyResponseFigure, save at
            configured DPI, and close resources on every exit path.

        Args:
            frequencyResponseByModel: Model-grouped frequency points.
            outputDirectory: Destination directory for the PNG.
            fileStem: Optional simple filename overriding the default.

        Returns:
            result: Path to the saved frequency-response image.
        """

        selectedFileStem = (
            str(self.parameters["paFrequencyFileStem"])
            if fileStem is None
            else fileStem
        )
        if (
            not isinstance(selectedFileStem, str)
            or not selectedFileStem
            or any(
                character in selectedFileStem
                for character in '<>:"/\\|?*'
            )
        ):
            raise ValueError("fileStem must be a valid simple file name")
        outputPath = Path(outputDirectory)
        outputPath.mkdir(parents=True, exist_ok=True)
        figurePath = outputPath / f"{selectedFileStem}.png"
        figure = self.CreatePaFrequencyResponseFigure(
            frequencyResponseByModel
        )
        try:
            figure.savefig(
                figurePath,
                dpi=int(self.parameters["figureDpi"]),
                bbox_inches="tight",
            )
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)
        return figurePath

    def CreatePaMemoryEffectFigure(
        self,
        memoryEffectByModel: Mapping[
            str, Sequence[Mapping[str, object]]
        ],
        summaryByModel: Mapping[str, Mapping[str, object]],
    ) -> Any:
        """Create spectral and dynamic PA memory-effect comparison panels.

        Processing details:
            Algorithm: Plot worse-side IM3 and absolute upper/lower IM3
            asymmetry against tone spacing, then compare rising/falling
            envelope gain and phase hysteresis as grouped model bars.

        Args:
            memoryEffectByModel: Model-grouped spacing-sweep measurements.
            summaryByModel: Model-grouped dynamic hysteresis summaries.

        Returns:
            result: Matplotlib Figure with two spectral and two dynamic panels.
        """

        self.ValidateParameters()
        self.ValidatePaSeries(
            memoryEffectByModel,
            (
                "toneSpacingHz",
                "im3LowerDbc",
                "im3UpperDbc",
                "im3WorstDbc",
            ),
            "memoryEffectByModel",
        )
        self.ValidatePaSummary(summaryByModel)
        if tuple(memoryEffectByModel) != tuple(summaryByModel):
            raise ValueError(
                "memoryEffectByModel and summaryByModel model order "
                "must match"
            )
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise RuntimeError(
                "matplotlib is required for PA memory-effect plots"
            ) from error
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(
                float(self.parameters["figureWidthInches"]),
                float(self.parameters["figureHeightInches"]) * 1.25,
            ),
        )
        markerStyles = ("o", "s", "^", "D", "v")
        for modelIndex, (modelName, modelPoints) in enumerate(
            memoryEffectByModel.items()
        ):
            sortedPoints = sorted(
                modelPoints,
                key=lambda point: float(point["toneSpacingHz"]),
            )
            spacingMhz = np.asarray(
                [
                    float(point["toneSpacingHz"]) * 1.0e-6
                    for point in sortedPoints
                ]
            )
            markerStyle = markerStyles[
                modelIndex % len(markerStyles)
            ]
            axes[0, 0].plot(
                spacingMhz,
                [
                    float(point["im3WorstDbc"])
                    for point in sortedPoints
                ],
                label=modelName,
                marker=markerStyle,
                linewidth=float(self.parameters["lineWidth"]),
                markersize=float(self.parameters["markerSize"]),
            )
            axes[0, 1].plot(
                spacingMhz,
                [
                    abs(
                        float(point["im3UpperDbc"])
                        - float(point["im3LowerDbc"])
                    )
                    for point in sortedPoints
                ],
                label=modelName,
                marker=markerStyle,
                linewidth=float(self.parameters["lineWidth"]),
                markersize=float(self.parameters["markerSize"]),
            )
        modelNames = list(summaryByModel)
        modelPositions = np.arange(len(modelNames), dtype=float)
        axes[1, 0].bar(
            modelPositions,
            [
                float(
                    summaryByModel[modelName][
                        "dynamicGainHysteresisDb"
                    ]
                )
                for modelName in modelNames
            ],
        )
        axes[1, 1].bar(
            modelPositions,
            [
                float(
                    summaryByModel[modelName][
                        "dynamicPhaseHysteresisDegrees"
                    ]
                )
                for modelName in modelNames
            ],
        )
        axes[0, 0].set_title("Worst-side IM3 versus tone spacing")
        axes[0, 0].set_ylabel("IM3 (dBc)")
        axes[0, 1].set_title("Upper/lower IM3 asymmetry")
        axes[0, 1].set_ylabel("Absolute asymmetry (dB)")
        for selectedAxes in axes[0, :]:
            selectedAxes.set_xlabel("Tone spacing (MHz)")
            selectedAxes.legend(loc="best")
        axes[1, 0].set_title("Dynamic AM-AM hysteresis")
        axes[1, 0].set_ylabel("RMS rising/falling separation (dB)")
        axes[1, 1].set_title("Dynamic AM-PM hysteresis")
        axes[1, 1].set_ylabel(
            "RMS rising/falling separation (degrees)"
        )
        for selectedAxes in axes[1, :]:
            selectedAxes.set_xticks(modelPositions)
            selectedAxes.set_xticklabels(modelNames)
        for selectedAxes in axes.reshape(-1):
            selectedAxes.grid(
                True, axis="y", linestyle=":", linewidth=0.7
            )
        figure.suptitle(
            str(self.parameters["paMemoryPlotTitle"])
        )
        figure.tight_layout()
        return figure

    def SavePaMemoryEffect(
        self,
        memoryEffectByModel: Mapping[
            str, Sequence[Mapping[str, object]]
        ],
        summaryByModel: Mapping[str, Mapping[str, object]],
        outputDirectory: Path,
        fileStem: Optional[str] = None,
    ) -> Path:
        """Save the four-panel spectral/dynamic PA memory comparison.

        Processing details:
            Algorithm: Validate a safe filename, create the destination,
            render through CreatePaMemoryEffectFigure, save at configured DPI,
            and always close the Matplotlib figure.

        Args:
            memoryEffectByModel: Model-grouped tone-spacing measurements.
            summaryByModel: Model-grouped dynamic memory summaries.
            outputDirectory: Destination directory for the PNG.
            fileStem: Optional simple filename overriding the default.

        Returns:
            result: Path to the saved memory-effect image.
        """

        selectedFileStem = (
            str(self.parameters["paMemoryFileStem"])
            if fileStem is None
            else fileStem
        )
        if (
            not isinstance(selectedFileStem, str)
            or not selectedFileStem
            or any(
                character in selectedFileStem
                for character in '<>:"/\\|?*'
            )
        ):
            raise ValueError("fileStem must be a valid simple file name")
        outputPath = Path(outputDirectory)
        outputPath.mkdir(parents=True, exist_ok=True)
        figurePath = outputPath / f"{selectedFileStem}.png"
        figure = self.CreatePaMemoryEffectFigure(
            memoryEffectByModel, summaryByModel
        )
        try:
            figure.savefig(
                figurePath,
                dpi=int(self.parameters["figureDpi"]),
                bbox_inches="tight",
            )
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)
        return figurePath

    def CreatePaNonlinearityComparisonFigure(
        self,
        summaryByModel: Mapping[str, Mapping[str, object]],
    ) -> Any:
        """Create a grouped nominal IM3, IM5, and IM7 PA comparison.

        Processing details:
            Algorithm: Preserve PA model order and place three adjacent dBc
            bars per model using the common output-power and tone-spacing
            summary measurements.

        Args:
            summaryByModel: Model-grouped nominal nonlinear metrics.

        Returns:
            result: Matplotlib Figure with the odd-order comparison.
        """

        self.ValidateParameters()
        self.ValidatePaSummary(summaryByModel)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise RuntimeError(
                "matplotlib is required for PA nonlinearity plots"
            ) from error
        modelNames = list(summaryByModel)
        modelPositions = np.arange(len(modelNames), dtype=float)
        barWidth = 0.24
        figure, axes = plt.subplots(
            figsize=(
                float(self.parameters["figureWidthInches"]),
                float(self.parameters["figureHeightInches"]),
            )
        )
        for orderIndex, nonlinearOrder in enumerate((3, 5, 7)):
            metricName = f"nominalIm{nonlinearOrder}Dbc"
            axes.bar(
                modelPositions + (orderIndex - 1) * barWidth,
                [
                    float(summaryByModel[modelName][metricName])
                    for modelName in modelNames
                ],
                width=barWidth,
                label=f"IM{nonlinearOrder}",
            )
        axes.set_xticks(modelPositions)
        axes.set_xticklabels(modelNames)
        axes.set_ylabel(
            "Worst-side intermodulation (dBc, lower is better)"
        )
        axes.set_title(
            str(self.parameters["paNonlinearityPlotTitle"])
        )
        axes.grid(True, axis="y", linestyle=":", linewidth=0.7)
        axes.legend(loc="best")
        figure.tight_layout()
        return figure

    def SavePaNonlinearityComparison(
        self,
        summaryByModel: Mapping[str, Mapping[str, object]],
        outputDirectory: Path,
        fileStem: Optional[str] = None,
    ) -> Path:
        """Save the grouped nominal PA intermodulation comparison.

        Processing details:
            Algorithm: Resolve and validate the filename, create the output
            directory, render through the dedicated figure builder, save the
            PNG at configured DPI, and close plotting resources.

        Args:
            summaryByModel: Model-grouped nominal IM summaries.
            outputDirectory: Destination directory for the PNG.
            fileStem: Optional simple filename overriding the default.

        Returns:
            result: Path to the saved nonlinearity comparison image.
        """

        selectedFileStem = (
            str(self.parameters["paNonlinearityFileStem"])
            if fileStem is None
            else fileStem
        )
        if (
            not isinstance(selectedFileStem, str)
            or not selectedFileStem
            or any(
                character in selectedFileStem
                for character in '<>:"/\\|?*'
            )
        ):
            raise ValueError("fileStem must be a valid simple file name")
        outputPath = Path(outputDirectory)
        outputPath.mkdir(parents=True, exist_ok=True)
        figurePath = outputPath / f"{selectedFileStem}.png"
        figure = self.CreatePaNonlinearityComparisonFigure(
            summaryByModel
        )
        try:
            figure.savefig(
                figurePath,
                dpi=int(self.parameters["figureDpi"]),
                bbox_inches="tight",
            )
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)
        return figurePath

    def CreatePaPowerCharacteristicsFigure(
        self,
        powerSweepByModel: Mapping[
            str, Sequence[Mapping[str, object]]
        ],
    ) -> Any:
        """Create nonlinear and dynamic-memory curves versus PA output power.

        Processing details:
            Algorithm: Sort each model by measured output dBm, plot IM3,
            IM5/IM7, dynamic AM-AM hysteresis, and dynamic AM-PM hysteresis
            on four common-power panels, and retain separate line styles for
            the two higher odd orders.

        Args:
            powerSweepByModel: Model-grouped controlled-power point mappings.

        Returns:
            result: Matplotlib Figure containing four power-dependent panels.
        """

        self.ValidateParameters()
        self.ValidatePaSeries(
            powerSweepByModel,
            (
                "measuredOutputPowerDbm",
                "im3WorstDbc",
                "im5WorstDbc",
                "im7WorstDbc",
                "dynamicGainHysteresisDb",
                "dynamicPhaseHysteresisDegrees",
            ),
            "powerSweepByModel",
        )
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise RuntimeError(
                "matplotlib is required for PA power-characteristic plots"
            ) from error
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(
                float(self.parameters["figureWidthInches"]),
                float(self.parameters["figureHeightInches"]) * 1.25,
            ),
        )
        markerStyles = ("o", "s", "^", "D", "v")
        for modelIndex, (modelName, modelPoints) in enumerate(
            powerSweepByModel.items()
        ):
            sortedPoints = sorted(
                modelPoints,
                key=lambda point: float(
                    point["measuredOutputPowerDbm"]
                ),
            )
            measuredPowerDbm = [
                float(point["measuredOutputPowerDbm"])
                for point in sortedPoints
            ]
            markerStyle = markerStyles[
                modelIndex % len(markerStyles)
            ]
            commonPlotArguments = {
                "marker": markerStyle,
                "linewidth": float(self.parameters["lineWidth"]),
                "markersize": float(self.parameters["markerSize"]),
            }
            axes[0, 0].plot(
                measuredPowerDbm,
                [
                    float(point["im3WorstDbc"])
                    for point in sortedPoints
                ],
                label=modelName,
                **commonPlotArguments,
            )
            axes[0, 1].plot(
                measuredPowerDbm,
                [
                    float(point["im5WorstDbc"])
                    for point in sortedPoints
                ],
                label=f"{modelName} IM5",
                **commonPlotArguments,
            )
            axes[0, 1].plot(
                measuredPowerDbm,
                [
                    float(point["im7WorstDbc"])
                    for point in sortedPoints
                ],
                label=f"{modelName} IM7",
                linestyle="--",
                **commonPlotArguments,
            )
            axes[1, 0].plot(
                measuredPowerDbm,
                [
                    float(point["dynamicGainHysteresisDb"])
                    for point in sortedPoints
                ],
                label=modelName,
                **commonPlotArguments,
            )
            axes[1, 1].plot(
                measuredPowerDbm,
                [
                    float(
                        point["dynamicPhaseHysteresisDegrees"]
                    )
                    for point in sortedPoints
                ],
                label=modelName,
                **commonPlotArguments,
            )
        axes[0, 0].set_title("IM3 versus output power")
        axes[0, 0].set_ylabel("Worst-side IM3 (dBc)")
        axes[0, 1].set_title("IM5 and IM7 versus output power")
        axes[0, 1].set_ylabel("Worst-side IM level (dBc)")
        axes[1, 0].set_title("Dynamic AM-AM versus output power")
        axes[1, 0].set_ylabel("RMS hysteresis (dB)")
        axes[1, 1].set_title("Dynamic AM-PM versus output power")
        axes[1, 1].set_ylabel("RMS hysteresis (degrees)")
        for selectedAxes in axes.reshape(-1):
            selectedAxes.set_xlabel("Measured PA output power (dBm)")
            selectedAxes.grid(
                True, which="both", linestyle=":", linewidth=0.7
            )
            selectedAxes.legend(loc="best")
        figure.suptitle(str(self.parameters["paPowerPlotTitle"]))
        figure.tight_layout()
        return figure

    def SavePaPowerCharacteristics(
        self,
        powerSweepByModel: Mapping[
            str, Sequence[Mapping[str, object]]
        ],
        outputDirectory: Path,
        fileStem: Optional[str] = None,
    ) -> Path:
        """Save the four-panel PA output-power characteristic comparison.

        Processing details:
            Algorithm: Resolve a safe filename, create the output directory,
            render the validated power curves, save at configured DPI, and
            always close Matplotlib resources.

        Args:
            powerSweepByModel: Model-grouped controlled-power measurements.
            outputDirectory: Destination directory for the PNG.
            fileStem: Optional simple filename overriding the default.

        Returns:
            result: Path to the saved power-characteristic image.
        """

        selectedFileStem = (
            str(self.parameters["paPowerFileStem"])
            if fileStem is None
            else fileStem
        )
        if (
            not isinstance(selectedFileStem, str)
            or not selectedFileStem
            or any(
                character in selectedFileStem
                for character in '<>:"/\\|?*'
            )
        ):
            raise ValueError("fileStem must be a valid simple file name")
        outputPath = Path(outputDirectory)
        outputPath.mkdir(parents=True, exist_ok=True)
        figurePath = outputPath / f"{selectedFileStem}.png"
        figure = self.CreatePaPowerCharacteristicsFigure(
            powerSweepByModel
        )
        try:
            figure.savefig(
                figurePath,
                dpi=int(self.parameters["figureDpi"]),
                bbox_inches="tight",
            )
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)
        return figurePath

    def ValidateDpdGmpStages(
        self,
        stageResults: Sequence[Mapping[str, object]],
    ) -> None:
        """Validate DPD-GMP stage rows before comparison plotting.

        Processing details:
            Algorithm: Require at least two uniquely named stages, finite
            equal-power EVM/ACLR/IM metrics, and finite positive optional
            condition numbers while allowing baseline label diagnostics to
            remain None.

        Args:
            stageResults: Ordered flat benchmark stage mappings.

        Returns:
            result: None. Malformed or nonfinite values raise an exception.
        """

        stageRows = tuple(stageResults)
        if len(stageRows) < 2:
            raise ValueError(
                "stageResults must contain at least two stages"
            )
        stageNames = []
        for stageIndex, stageRow in enumerate(stageRows):
            if not isinstance(stageRow, Mapping):
                raise TypeError(
                    f"stageResults[{stageIndex}] must be a mapping"
                )
            stageName = stageRow.get("stageName")
            if not isinstance(stageName, str) or not stageName:
                raise ValueError(
                    "every DPD-GMP stage must have a nonempty stageName"
                )
            stageNames.append(stageName)
            for fieldName in (
                "evmDb",
                "aclrWorstDb",
                "im3WorstDbc",
            ):
                fieldValue = stageRow.get(fieldName)
                if (
                    not isinstance(fieldValue, (int, float))
                    or isinstance(fieldValue, bool)
                    or not np.isfinite(fieldValue)
                ):
                    raise ValueError(
                        f"{stageName}.{fieldName} must be finite"
                    )
            for fieldName in (
                "labelNmseDb",
                "peakWeightedLabelNmseDb",
                "regularizedConditionNumber",
            ):
                fieldValue = stageRow.get(fieldName)
                if fieldValue is None:
                    continue
                if (
                    not isinstance(fieldValue, (int, float))
                    or isinstance(fieldValue, bool)
                    or not np.isfinite(fieldValue)
                ):
                    raise ValueError(
                        f"{stageName}.{fieldName} must be finite or None"
                    )
                if (
                    fieldName == "regularizedConditionNumber"
                    and float(fieldValue) <= 0.0
                ):
                    raise ValueError(
                        "regularizedConditionNumber must be positive"
                    )
        if len(set(stageNames)) != len(stageNames):
            raise ValueError("DPD-GMP stage names must be unique")

    def CreateDpdGmpPerformanceFigure(
        self,
        stageResults: Sequence[Mapping[str, object]],
    ) -> Any:
        """Create one four-panel DPD-GMP improvement comparison figure.

        Processing details:
            Algorithm: Draw equal-output-power Wi-Fi EVM and two-tone IM3 for
            every stage, draw ordinary and peak-weighted label NMSE only for
            trained stages, and show the regularized solver condition number
            on a logarithmic scale.

        Args:
            stageResults: Ordered flat DPD-GMP benchmark stage mappings.

        Returns:
            result: Matplotlib Figure containing four labeled comparison axes.
        """

        self.ValidateParameters()
        stageRows = tuple(stageResults)
        self.ValidateDpdGmpStages(stageRows)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise RuntimeError(
                "matplotlib is required for DPD-GMP performance plots"
            ) from error
        stageNames = [str(stage["stageName"]) for stage in stageRows]
        stagePositions = np.arange(len(stageRows), dtype=float)
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(
                float(self.parameters["figureWidthInches"]) * 1.25,
                float(self.parameters["figureHeightInches"]) * 1.35,
            ),
        )
        axes[0, 0].bar(
            stagePositions,
            [float(stage["evmDb"]) for stage in stageRows],
        )
        axes[0, 0].set_title("Equal-output-power Wi-Fi EVM")
        axes[0, 0].set_ylabel("RMS EVM (dB, lower is better)")
        axes[0, 1].bar(
            stagePositions,
            [float(stage["im3WorstDbc"]) for stage in stageRows],
        )
        axes[0, 1].set_title("Equal-output-power two-tone IM3")
        axes[0, 1].set_ylabel("Worst-side IM3 (dBc, lower is better)")

        trainedPositions = np.asarray(
            [
                stageIndex
                for stageIndex, stage in enumerate(stageRows)
                if stage["labelNmseDb"] is not None
            ],
            dtype=float,
        )
        trainedRows = [
            stage
            for stage in stageRows
            if stage["labelNmseDb"] is not None
        ]
        barWidth = 0.38
        axes[1, 0].bar(
            trainedPositions - 0.5 * barWidth,
            [float(stage["labelNmseDb"]) for stage in trainedRows],
            width=barWidth,
            label="Uniform label NMSE",
        )
        axes[1, 0].bar(
            trainedPositions + 0.5 * barWidth,
            [
                float(stage["peakWeightedLabelNmseDb"])
                for stage in trainedRows
            ],
            width=barWidth,
            label="Peak-weighted label NMSE",
        )
        axes[1, 0].set_title("ILC-label modeling accuracy")
        axes[1, 0].set_ylabel("NMSE (dB, lower is better)")
        axes[1, 0].legend(loc="best")

        conditionPositions = np.asarray(
            [
                stageIndex
                for stageIndex, stage in enumerate(stageRows)
                if stage["regularizedConditionNumber"] is not None
            ],
            dtype=float,
        )
        conditionRows = [
            stage
            for stage in stageRows
            if stage["regularizedConditionNumber"] is not None
        ]
        axes[1, 1].bar(
            conditionPositions,
            [
                float(stage["regularizedConditionNumber"])
                for stage in conditionRows
            ],
        )
        axes[1, 1].set_yscale("log")
        axes[1, 1].set_title("Regularized normal-matrix conditioning")
        axes[1, 1].set_ylabel("Condition number (lower is better)")

        for selectedAxes in axes.reshape(-1):
            selectedAxes.set_xticks(stagePositions)
            selectedAxes.set_xticklabels(
                stageNames,
                rotation=28,
                ha="right",
            )
            selectedAxes.grid(
                True,
                axis="y",
                which="both",
                linestyle=":",
                linewidth=0.7,
            )
        figure.suptitle(str(self.parameters["dpdGmpPlotTitle"]))
        figure.tight_layout()
        return figure

    def SaveDpdGmpPerformance(
        self,
        stageResults: Sequence[Mapping[str, object]],
        outputDirectory: Path,
        fileStem: Optional[str] = None,
    ) -> Path:
        """Save the four-panel DPD-GMP improvement comparison PNG.

        Processing details:
            Algorithm: Resolve and validate a simple filename, create the
            output directory, render through CreateDpdGmpPerformanceFigure,
            save at configured DPI, and always close Matplotlib resources.

        Args:
            stageResults: Ordered flat DPD-GMP benchmark stage mappings.
            outputDirectory: Destination directory for the PNG.
            fileStem: Optional simple filename overriding dpdGmpFileStem.

        Returns:
            result: Path to the generated DPD-GMP performance figure.
        """

        selectedFileStem = (
            str(self.parameters["dpdGmpFileStem"])
            if fileStem is None
            else fileStem
        )
        if (
            not isinstance(selectedFileStem, str)
            or not selectedFileStem
            or any(
                character in selectedFileStem
                for character in '<>:"/\\|?*'
            )
        ):
            raise ValueError("fileStem must be a valid simple file name")
        outputPath = Path(outputDirectory)
        outputPath.mkdir(parents=True, exist_ok=True)
        figurePath = outputPath / f"{selectedFileStem}.png"
        figure = self.CreateDpdGmpPerformanceFigure(stageResults)
        try:
            figure.savefig(
                figurePath,
                dpi=int(self.parameters["figureDpi"]),
                bbox_inches="tight",
            )
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)
        return figurePath

    def ValidateConvergenceHistory(self, ilcHistory: Sequence[object]) -> None:
        """Validate independently analyzed convergence records.

        Processing details:
            Algorithm: Require at least one record, strictly increasing
            iteration indices, and finite raw, linear-compensated, and strict
            Wi-Fi EVM series produced by ``Analysis``.

        Args:
            ilcHistory: Ordered ``ILCPerformanceIteration`` records.

        Returns:
            result: None. Invalid histories raise a descriptive error.
        """

        historyRecords = tuple(ilcHistory)
        if not historyRecords:
            raise ValueError("ilcHistory cannot be empty")
        iterations = np.asarray(
            [record.iteration for record in historyRecords], dtype=int
        )
        if np.any(np.diff(iterations) <= 0):
            raise ValueError("ILC iterations must be strictly increasing")
        for fieldName in (
            "nmseDb",
            "linearCompensatedNmseDb",
            "evmDb",
            "snrDb",
            "aclrWorstDb",
        ):
            fieldValues = np.asarray(
                [getattr(record, fieldName) for record in historyRecords],
                dtype=float,
            )
            if not np.all(np.isfinite(fieldValues)):
                raise ValueError(f"{fieldName} values must be finite")

    def CreateConvergenceFigure(
        self, ilcHistory: Sequence[object]
    ) -> Any:
        """Create a native-NMSE and post-analysis EVM convergence figure.

        Processing details:
            Algorithm: Plot normalized metrics on one decibel axis so their
            iteration trends are directly comparable. The EVM series comes
            only from the independent post-ILC ``Analysis`` result.

        Args:
            ilcHistory: Ordered per-iteration ILC diagnostic records.

        Returns:
            result: Matplotlib Figure ready for display or PNG output.
        """

        self.ValidateParameters()
        historyRecords = tuple(ilcHistory)
        self.ValidateConvergenceHistory(historyRecords)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise RuntimeError(
                "matplotlib is required to create the convergence figure"
            ) from error

        iterations = np.asarray(
            [record.iteration for record in historyRecords], dtype=int
        )
        figure, axes = plt.subplots(
            figsize=(
                float(self.parameters["figureWidthInches"]),
                float(self.parameters["figureHeightInches"]),
            )
        )
        axes.plot(
            iterations,
            [record.nmseDb for record in historyRecords],
            marker="o",
            linewidth=float(self.parameters["lineWidth"]),
            markersize=float(self.parameters["markerSize"]),
            label="Raw time-domain NMSE",
        )
        axes.plot(
            iterations,
            [record.linearCompensatedNmseDb for record in historyRecords],
            marker="s",
            linewidth=float(self.parameters["lineWidth"]),
            markersize=float(self.parameters["markerSize"]),
            label="Complex-gain-compensated NMSE",
        )
        axes.plot(
            iterations,
            [record.evmDb for record in historyRecords],
            marker="^",
            linewidth=float(self.parameters["lineWidth"]),
            markersize=float(self.parameters["markerSize"]),
            label="Post-analysis Wi-Fi EVM",
        )
        axes.set_xlabel(str(self.parameters["convergenceXAxisLabel"]))
        axes.set_ylabel(str(self.parameters["convergenceYAxisLabel"]))
        axes.set_title(str(self.parameters["convergencePlotTitle"]))
        axes.set_xticks(iterations)
        axes.grid(True, which="both", linestyle=":", linewidth=0.7)
        axes.legend(loc="best")
        figure.tight_layout()
        return figure

    def SaveConvergenceCurve(
        self,
        ilcHistory: Sequence[object],
        outputDirectory: Path,
        fileStem: Optional[str] = None,
    ) -> Path:
        """Render and save all per-iteration MSE views in one PNG file.

        Processing details:
            Algorithm: Resolve the ChainMap-backed filename, create the output
            directory, render the validated convergence history, and always
            close Matplotlib resources after saving.

        Args:
            ilcHistory: Ordered post-analysis performance records.
            outputDirectory: Directory in which the PNG image is written.
            fileStem: Optional filename overriding ``convergenceFileStem``.

        Returns:
            result: Path to the generated convergence figure.
        """

        selectedFileStem = (
            str(self.parameters["convergenceFileStem"])
            if fileStem is None
            else fileStem
        )
        if not isinstance(selectedFileStem, str):
            raise TypeError("fileStem must be a string")
        if not selectedFileStem or any(
            character in selectedFileStem for character in '<>:"/\\|?*'
        ):
            raise ValueError("fileStem must be a valid simple file name")
        outputPath = Path(outputDirectory)
        outputPath.mkdir(parents=True, exist_ok=True)
        figurePath = outputPath / f"{selectedFileStem}.png"
        figure = self.CreateConvergenceFigure(ilcHistory)
        try:
            figure.savefig(
                figurePath,
                dpi=int(self.parameters["figureDpi"]),
                bbox_inches="tight",
            )
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)
        return figurePath
