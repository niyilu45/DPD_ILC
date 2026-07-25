"""Object-oriented plotting utilities for DPD-ILC result curves."""

from collections import ChainMap
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from ..lib.Analysis import PowerEvmCurve
from .ConfigUtils import (
    FilterRecognizedParameters,
    RecognizedParameterView,
)


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
                "figureWidthInches": 10.5,
                "figureHeightInches": 6.2,
                "figureDpi": 180,
                "lineWidth": 1.8,
                "markerSize": 5.0,
                "legendColumnThreshold": 6,
                "plotTitle": "Power-EVM comparison",
                "convergencePlotTitle": "ILC MSE convergence",
                "xAxisLabel": "PA output power per chain (dBm)",
                "yAxisLabel": "RMS EVM (dB, lower is better)",
                "convergenceXAxisLabel": "ILC iteration",
                "convergenceYAxisLabel": (
                    "Normalized error / EVM (dB, lower is better)"
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

        for parameterName in ("powerEvmFileStem", "convergenceFileStem"):
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
