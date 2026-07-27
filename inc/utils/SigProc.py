"""Shared synchronization, compensation, and RF-power utilities."""

import warnings
from collections import ChainMap
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Sequence, Tuple, cast

import numpy as np

from .ConfigUtils import (
    FilterRecognizedParameters,
    RecognizedParameterView,
)
from .FixedPoint import FixedPoint


class PowerCalibration:
    """Calibrate physical-voltage and normalized public waveforms in dBm.

    The project uses the explicit convention that the RMS magnitude of a
    complex baseband waveform is the RMS voltage delivered to the configured
    resistive port for the physical-voltage methods. Under this convention
    ``P = Vrms**2 / R``. Normalized public waveforms instead map active-region
    RMS equal to one onto ``maximumOutputPowerDbm``. Both paths exclude
    leading/trailing padding and long off intervals from the RMS denominator.
    """

    def __init__(
        self,
        loadResistanceOhm: Optional[float] = None,
        maximumOutputPowerDbm: Optional[float] = None,
        parameters: Optional[Mapping[str, object]] = None,
        width: Optional[int] = None,
        **parameterOverrides: object,
    ) -> None:
        """Initialize a live ChainMap-backed RF power calibration.

        Processing details:
            Algorithm: Define the standard 50-ohm default inside this
            constructor, layer caller values ahead of it, and validate the
            resolved resistance, active-region detector, full-scale power,
            and public width before any logarithmic conversion.

        Args:
            loadResistanceOhm: Optional resistive port value in ohms.
            maximumOutputPowerDbm: Optional rated PA output-power ceiling.
            parameters: Optional caller-owned mapping of calibration values.
            width: Optional public I/Q width used by normalized waveform
                calibration. None selects the internal 16-bit default.
            parameterOverrides: Highest-priority local calibration overrides.

        Returns:
            result: None. The converter is ready for dBm/RMS transformations.
        """

        self.defaultParameters: Mapping[str, object] = MappingProxyType(
            {
                "loadResistanceOhm": 50.0,
                "maximumOutputPowerDbm": 25.0,
                "outputPowerDbm": 20.0,
                "outputPowerDbmPerChain": None,
                "activePowerThresholdDb": -60.0,
                "activeGapToleranceSamples": 16,
                "width": 16,
            }
        )
        directOverrides = dict(parameterOverrides)
        if loadResistanceOhm is not None:
            directOverrides["loadResistanceOhm"] = loadResistanceOhm
        if maximumOutputPowerDbm is not None:
            directOverrides["maximumOutputPowerDbm"] = (
                maximumOutputPowerDbm
            )
        if width is not None:
            directOverrides["width"] = width
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping or None")
        externalParameters: Mapping[str, object] = (
            {}
            if parameters is None
            else RecognizedParameterView(
                parameters,
                self.defaultParameters,
                "PowerCalibration",
            )
        )
        recognizedOverrides = FilterRecognizedParameters(
            directOverrides,
            self.defaultParameters,
            "PowerCalibration",
        )
        self.parameters: ChainMap[str, object] = ChainMap(
            recognizedOverrides,
            externalParameters,
            self.defaultParameters,
        )
        self.Validate()

    @property
    def LoadResistanceOhm(self) -> float:
        """Return the resolved resistive port value in ohms.

        Processing details:
            Algorithm: Read the highest-priority ChainMap value after
            constructor and update validation.

        Returns:
            result: Positive finite resistance in ohms.
        """

        return float(cast(float, self.parameters["loadResistanceOhm"]))

    loadResistanceOhm = LoadResistanceOhm

    @property
    def MaximumOutputPowerDbm(self) -> float:
        """Return the rated per-PA output-power ceiling in dBm.

        Processing details:
            Algorithm: Read the validated ChainMap value used to convert
            requested output powers into normalized output-backoff drive.

        Returns:
            result: Finite maximum output power in dBm.
        """

        return float(
            cast(float, self.parameters["maximumOutputPowerDbm"])
        )

    maximumOutputPowerDbm = MaximumOutputPowerDbm

    @property
    def Width(self) -> int:
        """Return the public I/Q component width used for calibration.

        Processing details:
            Algorithm: Read the validated ChainMap value without changing the
            caller-owned parameter mapping.

        Returns:
            result: Zero for floating mode or a positive signed-code width.
        """

        return cast(int, self.parameters["width"])

    width = Width

    def GetParameters(self) -> Dict[str, object]:
        """Return a flattened calibration parameter snapshot.

        Processing details:
            Algorithm: Resolve all ChainMap layers without changing the live
            caller-owned mapping.

        Returns:
            result: Ordinary dictionary containing every resolved setting.
        """

        return dict(self.parameters)

    def UpdateParameters(self, **parameterOverrides: object) -> None:
        """Apply and validate high-priority calibration overrides.

        Processing details:
            Algorithm: Update the local ChainMap layer transactionally and
            restore the previous values if validation fails.

        Args:
            parameterOverrides: Local calibration values to replace.

        Returns:
            result: None. The active converter is updated in place.
        """

        recognizedOverrides = FilterRecognizedParameters(
            parameterOverrides,
            self.defaultParameters,
            "PowerCalibration.UpdateParameters",
        )
        previousOverrides = dict(self.parameters.maps[0])
        self.parameters.maps[0].update(recognizedOverrides)
        try:
            self.Validate()
        except (TypeError, ValueError):
            self.parameters.maps[0].clear()
            self.parameters.maps[0].update(previousOverrides)
            raise

    def Validate(self) -> None:
        """Validate power, active-region, and public-interface settings.

        Processing details:
            Algorithm: Check numeric type, finiteness, physical domains,
            active threshold and gap tolerance, then construct ``FixedPoint``
            to validate the requested public I/Q width.

        Returns:
            result: None. Invalid calibration raises an exception.
        """

        resistanceValue = self.parameters["loadResistanceOhm"]
        if (
            not isinstance(resistanceValue, (int, float))
            or isinstance(resistanceValue, bool)
            or not np.isfinite(resistanceValue)
            or resistanceValue <= 0.0
        ):
            raise ValueError(
                "loadResistanceOhm must be finite and positive"
            )
        maximumOutputPowerDbm = self.parameters[
            "maximumOutputPowerDbm"
        ]
        if (
            not isinstance(maximumOutputPowerDbm, (int, float))
            or isinstance(maximumOutputPowerDbm, bool)
            or not np.isfinite(maximumOutputPowerDbm)
        ):
            raise ValueError(
                "maximumOutputPowerDbm must be finite"
            )
        outputPowerDbm = self.parameters["outputPowerDbm"]
        if (
            not isinstance(outputPowerDbm, (int, float))
            or isinstance(outputPowerDbm, bool)
            or not np.isfinite(outputPowerDbm)
            or float(outputPowerDbm)
            > float(maximumOutputPowerDbm)
        ):
            raise ValueError(
                "outputPowerDbm must be finite and cannot exceed "
                "maximumOutputPowerDbm"
            )
        outputPowerDbmPerChain = self.parameters[
            "outputPowerDbmPerChain"
        ]
        if outputPowerDbmPerChain is not None:
            if (
                isinstance(outputPowerDbmPerChain, (str, bytes))
                or not isinstance(
                    outputPowerDbmPerChain,
                    (list, tuple, np.ndarray),
                )
            ):
                raise TypeError(
                    "outputPowerDbmPerChain must be a sequence or None"
                )
            powerArray = np.asarray(
                outputPowerDbmPerChain, dtype=object
            )
            if powerArray.ndim != 1 or powerArray.size == 0:
                raise ValueError(
                    "outputPowerDbmPerChain must be a nonempty "
                    "one-dimensional sequence"
                )
            for targetPowerDbm in powerArray:
                if (
                    not isinstance(targetPowerDbm, (int, float))
                    or isinstance(targetPowerDbm, bool)
                    or not np.isfinite(targetPowerDbm)
                    or float(targetPowerDbm)
                    > float(maximumOutputPowerDbm)
                ):
                    raise ValueError(
                        "every outputPowerDbmPerChain value must be "
                        "finite and cannot exceed maximumOutputPowerDbm"
                    )
        activePowerThresholdDb = self.parameters[
            "activePowerThresholdDb"
        ]
        if (
            not isinstance(activePowerThresholdDb, (int, float))
            or isinstance(activePowerThresholdDb, bool)
            or not np.isfinite(activePowerThresholdDb)
            or float(activePowerThresholdDb) >= 0.0
        ):
            raise ValueError(
                "activePowerThresholdDb must be finite and negative"
            )
        activeGapToleranceSamples = self.parameters[
            "activeGapToleranceSamples"
        ]
        if (
            not isinstance(activeGapToleranceSamples, int)
            or isinstance(activeGapToleranceSamples, bool)
            or activeGapToleranceSamples < 0
        ):
            raise ValueError(
                "activeGapToleranceSamples must be a nonnegative integer"
            )
        FixedPoint(self.width)

    def Calibrate(self, inputSignal: np.ndarray) -> np.ndarray:
        """Calibrate a public waveform using the configured power target.

        Processing details:
            Algorithm: Read ``outputPowerDbmPerChain`` from the live
            ChainMap when independent MIMO targets are configured; otherwise
            use the common ``outputPowerDbm`` value. Delegate effective-burst
            detection, arbitrary-RMS removal, floating/fixed conversion, and
            per-chain scaling to the existing waveform calibration engine.

        Args:
            inputSignal: Arbitrarily scaled public waveform vector or
                samples-by-chain matrix.

        Returns:
            result: A newly calibrated waveform with the same shape and
                public floating/fixed interface convention as the input.
        """

        targetPowers = self.parameters["outputPowerDbmPerChain"]
        if targetPowers is None:
            return self.CalibrateWaveformToOutputPower(
                inputSignal,
                float(cast(float, self.parameters["outputPowerDbm"])),
            )
        return self.CalibrateWaveformToOutputPowers(
            inputSignal,
            tuple(float(powerDbm) for powerDbm in targetPowers),
        )

    def DbmToRms(self, powerDbm: float) -> float:
        """Convert absolute port power in dBm to complex-envelope RMS volts.

        Processing details:
            Algorithm: Convert dBm to watts with the one-milliwatt reference,
            multiply by resistance, and take the positive RMS square root.

        Args:
            powerDbm: Absolute available power in dBm.

        Returns:
            result: Positive RMS voltage used to scale a unit-RMS waveform.
        """

        if (
            not isinstance(powerDbm, (int, float))
            or isinstance(powerDbm, bool)
            or not np.isfinite(powerDbm)
        ):
            raise ValueError("powerDbm must be finite")
        # Compute the voltage directly with a 20-log amplitude exponent. This
        # is algebraically identical to converting through watts, while NumPy
        # lets the explicit finite-range check handle overflow and underflow.
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            rmsVoltage = (
                np.sqrt(1.0e-3 * self.loadResistanceOhm)
                * np.power(10.0, float(powerDbm) / 20.0)
            )
        if not np.isfinite(rmsVoltage) or rmsVoltage <= 0.0:
            raise ValueError("powerDbm is outside the numeric range")
        return float(rmsVoltage)

    def RmsToDbm(self, signalRms: float) -> float:
        """Convert complex-envelope RMS volts to absolute port power in dBm.

        Processing details:
            Algorithm: Divide squared RMS voltage by resistance to obtain
            watts, normalize by one milliwatt, and take ten-base logarithms.

        Args:
            signalRms: Positive complex-envelope RMS voltage.

        Returns:
            result: Absolute resistive-port power in dBm.
        """

        if (
            not isinstance(signalRms, (int, float))
            or isinstance(signalRms, bool)
            or not np.isfinite(signalRms)
            or signalRms <= 0.0
        ):
            raise ValueError("signalRms must be finite and positive")
        # The logarithmic form avoids squaring a very large finite voltage.
        return float(
            20.0 * np.log10(float(signalRms))
            - 10.0 * np.log10(self.loadResistanceOhm * 1.0e-3)
        )

    def OutputPowerToDriveScale(
        self, outputPowerDbm: float
    ) -> float:
        """Convert requested PA output power to normalized backoff drive.

        Processing details:
            Algorithm: Treat ``maximumOutputPowerDbm`` as zero output
            backoff, subtract it from the requested per-PA output power, and
            convert the resulting dB backoff to a linear complex-envelope
            scale. This controls the normalized PA compression point without
            confusing output dBm with input-port voltage.

        Args:
            outputPowerDbm: Requested average output power for one PA.

        Returns:
            result: Positive normalized drive scale no greater than one.
        """

        if (
            not isinstance(outputPowerDbm, (int, float))
            or isinstance(outputPowerDbm, bool)
            or not np.isfinite(outputPowerDbm)
        ):
            raise ValueError("outputPowerDbm must be finite")
        numericPowerDbm = float(outputPowerDbm)
        if numericPowerDbm > self.maximumOutputPowerDbm:
            raise ValueError(
                "outputPowerDbm cannot exceed maximumOutputPowerDbm"
            )
        return float(
            np.power(
                10.0,
                (
                    numericPowerDbm
                    - self.maximumOutputPowerDbm
                )
                / 20.0,
            )
        )

    def NormalizedRmsToOutputPowerDbm(
        self, normalizedRms: float
    ) -> float:
        """Convert normalized full-scale RMS into output power in dBm.

        Processing details:
            Algorithm: Treat normalized RMS equal to one as
            ``maximumOutputPowerDbm`` and apply the amplitude-ratio
            twenty-log conversion.

        Args:
            normalizedRms: Positive normalized complex-envelope RMS.

        Returns:
            result: Absolute output power under the configured full-scale
                calibration.
        """

        if (
            not isinstance(normalizedRms, (int, float))
            or isinstance(normalizedRms, bool)
            or not np.isfinite(normalizedRms)
            or normalizedRms <= 0.0
        ):
            raise ValueError(
                "normalizedRms must be finite and positive"
            )
        return float(
            self.maximumOutputPowerDbm
            + 20.0 * np.log10(float(normalizedRms))
        )

    def FindActiveSampleMask(
        self, inputSignal: np.ndarray
    ) -> np.ndarray:
        """Detect burst-active samples while excluding padding and off time.

        Processing details:
            Algorithm: Compare instantaneous power with a configurable level
            below each chain's peak, exclude leading/trailing padding and
            long internal inactive runs, and fill only short inactive gaps so
            ordinary OFDM zero crossings remain part of the active burst.

        Args:
            inputSignal: Finite complex vector or samples-by-chain matrix in
                any consistent linear amplitude scale.

        Returns:
            result: Boolean mask with the same vector or matrix orientation.
        """

        complexSignal = np.asarray(inputSignal, dtype=np.complex128)
        if (
            complexSignal.ndim not in (1, 2)
            or complexSignal.size == 0
            or complexSignal.shape[0] == 0
            or not np.all(np.isfinite(complexSignal))
        ):
            raise ValueError(
                "inputSignal must be a finite nonempty vector or matrix"
            )
        inputWasVector = complexSignal.ndim == 1
        signalMatrix = (
            complexSignal.reshape(-1, 1)
            if inputWasVector
            else complexSignal
        )
        instantaneousPower = np.abs(signalMatrix) ** 2
        peakPowerPerChain = np.max(instantaneousPower, axis=0)
        if np.any(peakPowerPerChain <= np.finfo(float).tiny):
            raise ValueError(
                "cannot detect an active interval in a zero-power signal"
            )
        relativePowerThreshold = 10.0 ** (
            float(self.parameters["activePowerThresholdDb"]) / 10.0
        )
        activeMask = (
            instantaneousPower
            > peakPowerPerChain.reshape(1, -1)
            * relativePowerThreshold
        )
        gapTolerance = cast(
            int, self.parameters["activeGapToleranceSamples"]
        )
        if gapTolerance > 0:
            for chainIndex in range(signalMatrix.shape[1]):
                activeIndices = np.flatnonzero(
                    activeMask[:, chainIndex]
                )
                if activeIndices.size == 0:
                    raise ValueError(
                        "unable to identify active signal samples"
                    )
                for leftIndex, rightIndex in zip(
                    activeIndices[:-1], activeIndices[1:]
                ):
                    inactiveLength = int(
                        rightIndex - leftIndex - 1
                    )
                    if 0 < inactiveLength <= gapTolerance:
                        activeMask[
                            leftIndex:rightIndex + 1,
                            chainIndex,
                        ] = True
        if np.any(np.sum(activeMask, axis=0) == 0):
            raise ValueError(
                "unable to identify active signal samples"
            )
        return activeMask[:, 0] if inputWasVector else activeMask

    def CalculateActiveRmsPerChain(
        self, inputSignal: np.ndarray
    ) -> Tuple[float, ...]:
        """Measure RMS only over each chain's detected active samples.

        Processing details:
            Algorithm: Build a scale-invariant burst mask with
            ``FindActiveSampleMask`` and divide active sample energy by active
            sample count rather than total capture length.

        Args:
            inputSignal: Finite complex vector or samples-by-chain matrix.

        Returns:
            result: Chain-ordered active-region RMS values.
        """

        complexSignal = np.asarray(inputSignal, dtype=np.complex128)
        inputWasVector = complexSignal.ndim == 1
        signalMatrix = (
            complexSignal.reshape(-1, 1)
            if inputWasVector
            else complexSignal
        )
        activeMask = self.FindActiveSampleMask(complexSignal)
        maskMatrix = (
            activeMask.reshape(-1, 1)
            if activeMask.ndim == 1
            else activeMask
        )
        activeEnergy = np.sum(
            np.abs(signalMatrix) ** 2 * maskMatrix,
            axis=0,
        )
        activeSampleCount = np.sum(maskMatrix, axis=0)
        activeRms = np.sqrt(activeEnergy / activeSampleCount)
        return tuple(float(rmsValue) for rmsValue in activeRms)

    def CalibrateFixedColumn(
        self,
        normalizedColumn: np.ndarray,
        activeMask: np.ndarray,
        targetNormalizedRms: float,
    ) -> np.ndarray:
        """Quantize one normalized column while meeting its target RMS.

        Processing details:
            Algorithm: Search one nonnegative pre-quantization scale because
            signed-code rounding and saturation make achieved RMS piecewise
            constant. Retain the closest encoded candidate and warn when
            component clipping was required to attain the requested power.

        Args:
            normalizedColumn: Floating column whose active-region RMS is one.
            activeMask: Boolean active-sample mask for the same column.
            targetNormalizedRms: Desired post-quantization active RMS.

        Returns:
            result: Public fixed-point integer I/Q codes in complex128.
        """

        interfaceFormat = FixedPoint(self.width)
        if interfaceFormat.IsFloatingPoint():
            return (
                targetNormalizedRms * normalizedColumn
            ).astype(np.complex128, copy=False)
        columnArray = np.asarray(
            normalizedColumn, dtype=np.complex128
        ).reshape(-1)
        maskArray = np.asarray(activeMask, dtype=bool).reshape(-1)
        if (
            columnArray.size == 0
            or maskArray.shape != columnArray.shape
            or not np.any(maskArray)
        ):
            raise ValueError(
                "normalizedColumn and activeMask must be aligned and nonempty"
            )

        scaleLower = 0.0
        scaleUpper = float(targetNormalizedRms)
        bestCodes = interfaceFormat.EncodeComplex(
            scaleUpper * columnArray
        )
        bestDecoded = interfaceFormat.DecodeComplex(bestCodes)
        bestRms = float(
            np.sqrt(np.mean(np.abs(bestDecoded[maskArray]) ** 2))
        )
        bestError = abs(bestRms - targetNormalizedRms)
        upperRms = bestRms

        # Expand the search until quantized RMS brackets the target. If every
        # active I/Q component is saturated, further scale increases no longer
        # change RMS and the target is physically unreachable at this width.
        previousUpperRms = -1.0
        expansionCount = 0
        while (
            upperRms < targetNormalizedRms
            and expansionCount < 60
            and abs(upperRms - previousUpperRms)
            > np.finfo(float).eps
        ):
            previousUpperRms = upperRms
            scaleUpper *= 2.0
            candidateCodes = interfaceFormat.EncodeComplex(
                scaleUpper * columnArray
            )
            candidateDecoded = interfaceFormat.DecodeComplex(
                candidateCodes
            )
            candidateRms = float(
                np.sqrt(
                    np.mean(
                        np.abs(candidateDecoded[maskArray]) ** 2
                    )
                )
            )
            upperRms = candidateRms
            candidateError = abs(
                candidateRms - targetNormalizedRms
            )
            if candidateError < bestError:
                bestCodes = candidateCodes
                bestRms = candidateRms
                bestError = candidateError
            expansionCount += 1
        if upperRms < targetNormalizedRms:
            raise ValueError(
                "target output power is unreachable at the configured width"
            )

        for _ in range(64):
            scaleMiddle = 0.5 * (scaleLower + scaleUpper)
            candidateCodes = interfaceFormat.EncodeComplex(
                scaleMiddle * columnArray
            )
            candidateDecoded = interfaceFormat.DecodeComplex(
                candidateCodes
            )
            candidateRms = float(
                np.sqrt(
                    np.mean(
                        np.abs(candidateDecoded[maskArray]) ** 2
                    )
                )
            )
            candidateError = abs(
                candidateRms - targetNormalizedRms
            )
            if candidateError < bestError:
                bestCodes = candidateCodes
                bestRms = candidateRms
                bestError = candidateError
            if candidateRms < targetNormalizedRms:
                scaleLower = scaleMiddle
            else:
                scaleUpper = scaleMiddle

        achievedPowerDbm = self.NormalizedRmsToOutputPowerDbm(
            bestRms
        )
        targetPowerDbm = self.NormalizedRmsToOutputPowerDbm(
            targetNormalizedRms
        )
        if abs(achievedPowerDbm - targetPowerDbm) > 0.01:
            raise ValueError(
                "fixed-point quantization cannot meet target power within "
                "0.01 dB"
            )
        formatInfo = interfaceFormat.GetFormatInfo()
        if np.any(
            bestCodes.real
            >= cast(float, formatInfo["maximumCode"])
        ) or np.any(
            bestCodes.real
            <= cast(float, formatInfo["minimumCode"])
        ) or np.any(
            bestCodes.imag
            >= cast(float, formatInfo["maximumCode"])
        ) or np.any(
            bestCodes.imag
            <= cast(float, formatInfo["minimumCode"])
        ):
            warnings.warn(
                (
                    "PowerCalibration met the requested fixed-point power "
                    "using component clipping; EVM may change"
                ),
                UserWarning,
                stacklevel=3,
            )
        return bestCodes

    def CalibrateWaveformToOutputPower(
        self,
        inputSignal: np.ndarray,
        outputPowerDbm: float,
    ) -> np.ndarray:
        """Regenerate every waveform chain at one requested output power.

        Processing details:
            Algorithm: Detect effective burst samples, remove arbitrary input
            RMS normalization, apply the target normalized RMS implied by the
            rated full-scale output power, and return floating samples or
            fixed integer codes according to ``width``.

        Args:
            inputSignal: Arbitrarily scaled public waveform vector or matrix.
            outputPowerDbm: Requested active-region output power per chain.

        Returns:
            result: Newly calibrated waveform with unchanged shape.
        """

        complexSignal = np.asarray(inputSignal, dtype=np.complex128)
        if complexSignal.ndim not in (1, 2):
            raise ValueError(
                "inputSignal must be a vector or samples-by-chain matrix"
            )
        chainCount = 1 if complexSignal.ndim == 1 else complexSignal.shape[1]
        return self.CalibrateWaveformToOutputPowers(
            complexSignal,
            tuple(float(outputPowerDbm) for _ in range(chainCount)),
        )

    def CalibrateWaveformToOutputPowers(
        self,
        inputSignal: np.ndarray,
        outputPowerDbmPerChain: Sequence[float],
    ) -> np.ndarray:
        """Regenerate waveform chains at independent requested powers.

        Processing details:
            Algorithm: Decode the public interface, detect each chain's
            effective burst mask, normalize active RMS independently, scale
            to each dBm-derived normalized target, and encode back through the
            same floating or signed-code interface.

        Args:
            inputSignal: Arbitrarily scaled public vector or matrix.
            outputPowerDbmPerChain: One target dBm value per waveform chain.

        Returns:
            result: Power-calibrated public waveform preserving orientation.
        """

        interfaceFormat = FixedPoint(self.width)
        floatingSignal = interfaceFormat.DecodeComplex(inputSignal)
        if (
            floatingSignal.ndim not in (1, 2)
            or floatingSignal.size == 0
            or floatingSignal.shape[0] == 0
            or not np.all(np.isfinite(floatingSignal))
        ):
            raise ValueError(
                "inputSignal must be a finite nonempty vector or matrix"
            )
        inputWasVector = floatingSignal.ndim == 1
        signalMatrix = (
            floatingSignal.reshape(-1, 1)
            if inputWasVector
            else floatingSignal
        )
        targetPowers = tuple(outputPowerDbmPerChain)
        if len(targetPowers) != signalMatrix.shape[1]:
            raise ValueError(
                "outputPowerDbmPerChain must contain one value per chain"
            )
        activeMask = self.FindActiveSampleMask(signalMatrix)
        maskMatrix = (
            activeMask.reshape(-1, 1)
            if activeMask.ndim == 1
            else activeMask
        )
        calibratedColumns = []
        for chainIndex, outputPowerDbm in enumerate(targetPowers):
            targetNormalizedRms = self.OutputPowerToDriveScale(
                outputPowerDbm
            )
            if targetNormalizedRms <= np.finfo(float).tiny:
                raise ValueError(
                    "outputPowerDbm is outside the numeric range"
                )
            chainMask = maskMatrix[:, chainIndex]
            currentRms = float(
                np.sqrt(
                    np.mean(
                        np.abs(signalMatrix[chainMask, chainIndex]) ** 2
                    )
                )
            )
            if currentRms <= np.finfo(float).tiny:
                raise ValueError(
                    "cannot calibrate a zero-power active waveform"
                )
            normalizedColumn = (
                signalMatrix[:, chainIndex] / currentRms
            )
            if interfaceFormat.IsFloatingPoint():
                calibratedColumns.append(
                    targetNormalizedRms * normalizedColumn
                )
            else:
                calibratedColumns.append(
                    self.CalibrateFixedColumn(
                        normalizedColumn,
                        chainMask,
                        targetNormalizedRms,
                    )
                )
        calibratedMatrix = np.column_stack(calibratedColumns)
        if interfaceFormat.IsFloatingPoint():
            calibratedMatrix = interfaceFormat.EncodeComplex(
                calibratedMatrix
            )
        if inputWasVector:
            return calibratedMatrix[:, 0]
        return calibratedMatrix

    def ScaleSignalToOutputPower(
        self,
        signal: np.ndarray,
        outputPowerDbm: float,
    ) -> np.ndarray:
        """Scale every PA output chain to one absolute average power.

        Processing details:
            Algorithm: Validate a complex vector or samples-by-chain matrix,
            calculate each chain active-region RMS independently, convert the
            requested per-chain dBm value to RMS voltage, and apply one
            constant gain per chain. Constant post-PA gain preserves EVM and
            ACLR ratios.

        Args:
            signal: One PA output vector or a samples-by-PA matrix.
            outputPowerDbm: Requested average output power per PA chain.

        Returns:
            result: Complex signal with every chain at the requested dBm.
        """

        complexSignal = np.asarray(signal, dtype=np.complex128)
        if (
            complexSignal.ndim not in (1, 2)
            or complexSignal.size == 0
            or complexSignal.shape[0] == 0
            or not np.all(np.isfinite(complexSignal))
        ):
            raise ValueError(
                "signal must be a finite nonempty vector or matrix"
            )
        signalMatrix = (
            complexSignal.reshape(-1, 1)
            if complexSignal.ndim == 1
            else complexSignal
        )
        scaledMatrix = self.ScaleSignalToOutputPowers(
            signalMatrix,
            tuple(
                float(outputPowerDbm)
                for _ in range(signalMatrix.shape[1])
            ),
        )
        if complexSignal.ndim == 1:
            return scaledMatrix[:, 0]
        return scaledMatrix

    def ScaleSignalToOutputPowers(
        self,
        signal: np.ndarray,
        outputPowerDbmPerChain: Sequence[float],
    ) -> np.ndarray:
        """Scale PA output columns to independent absolute powers.

        Processing details:
            Algorithm: Validate one target per matrix column, measure each
            current active-region RMS, convert every dBm target through the
            common port resistance, and apply independent constant gains.

        Args:
            signal: Samples-by-PA complex matrix.
            outputPowerDbmPerChain: Requested dBm target for every column.

        Returns:
            result: Calibrated matrix with unchanged column ordering.
        """

        complexSignal = np.asarray(signal, dtype=np.complex128)
        if (
            complexSignal.ndim not in (1, 2)
            or complexSignal.size == 0
            or complexSignal.shape[0] == 0
            or not np.all(np.isfinite(complexSignal))
        ):
            raise ValueError(
                "signal must be a finite nonempty vector or matrix"
            )
        inputWasVector = complexSignal.ndim == 1
        signalMatrix = (
            complexSignal.reshape(-1, 1)
            if inputWasVector
            else complexSignal
        )
        targetPowers = tuple(outputPowerDbmPerChain)
        if len(targetPowers) != signalMatrix.shape[1]:
            raise ValueError(
                "outputPowerDbmPerChain must contain one value per chain"
            )
        targetRmsList = []
        for targetPowerDbm in targetPowers:
            self.OutputPowerToDriveScale(targetPowerDbm)
            targetRmsList.append(
                self.DbmToRms(float(targetPowerDbm))
            )
        targetRmsValues = np.asarray(targetRmsList, dtype=float)
        currentRms = np.asarray(
            self.CalculateActiveRmsPerChain(signalMatrix),
            dtype=float,
        )
        if np.any(currentRms <= np.finfo(float).tiny):
            raise ValueError(
                "cannot calibrate zero-power PA output"
            )
        scaledMatrix = (
            signalMatrix
            * (targetRmsValues / currentRms).reshape(1, -1)
        )
        if inputWasVector:
            return scaledMatrix[:, 0]
        return scaledMatrix


@dataclass(frozen=True)
class SignalProcessingResult:
    """Store one signal-processing pass and all estimated impairments."""

    processedSignal: np.ndarray
    integerDelaySamples: int
    fractionalDelaySamples: float
    carrierFrequencyOffsetHz: float
    samplingFrequencyOffsetPpm: float
    complexGain: complex

    def ToDict(self) -> Dict[str, float]:
        """Return synchronization estimates in a serialization-ready form.

        Processing details:
            Algorithm: Split the complex gain into real and imaginary parts
            while preserving every timing and frequency estimate numerically.

        Returns:
            result: Dictionary containing scalar impairment estimates. The
                processed sample array is intentionally excluded.
        """

        return {
            "integerDelaySamples": float(self.integerDelaySamples),
            "fractionalDelaySamples": float(self.fractionalDelaySamples),
            "carrierFrequencyOffsetHz": float(
                self.carrierFrequencyOffsetHz
            ),
            "samplingFrequencyOffsetPpm": float(
                self.samplingFrequencyOffsetPpm
            ),
            "complexGainReal": float(np.real(self.complexGain)),
            "complexGainImag": float(np.imag(self.complexGain)),
            "complexGainMagnitude": float(np.abs(self.complexGain)),
            "complexGainPhaseDegrees": float(
                np.degrees(np.angle(self.complexGain))
            ),
        }


@dataclass(frozen=True)
class SignalOverlapResult:
    """Store the common sample interval found between two waveforms."""

    receivedStartSample: int
    referenceStartSample: int
    overlapLength: int
    confidence: float

    def ToDict(self) -> Dict[str, float]:
        """Return overlap coordinates in a serialization-ready dictionary.

        Processing details:
            Algorithm: Convert integer sample coordinates and normalized
            correlation confidence without changing their physical meaning.

        Returns:
            result: Dictionary containing both starts, length, and confidence.
        """

        return {
            "receivedStartSample": float(self.receivedStartSample),
            "referenceStartSample": float(self.referenceStartSample),
            "overlapLength": float(self.overlapLength),
            "confidence": float(self.confidence),
        }


class SigProc:
    """Estimate and compensate deterministic baseband signal impairments.

    The processor uses a known complex reference waveform. A data-aided
    approach is appropriate for this DPD-ILC project because every measured
    PA output is generated from, or captured in response to, that reference.
    Timing and frequency synchronization therefore remain independent from
    the later SNR, EVM, and ACLR definitions.
    """

    def __init__(
        self,
        referenceSignal: np.ndarray,
        sampleRateHz: float,
        parameters: Optional[Mapping[str, object]] = None,
        **parameterOverrides: object,
    ) -> None:
        """Initialize the processor with constructor-internal defaults.

        Processing details:
            Algorithm: Validate the reference, define immutable default
            synchronization settings locally, and layer caller overrides in
            front of those defaults with ``ChainMap`` precedence.

        Args:
            referenceSignal: Known finite one-dimensional complex waveform.
            sampleRateHz: Nominal complex sample rate in samples per second.
            parameters: Optional caller-owned mapping containing only values
                that differ from the internal defaults.
            parameterOverrides: Highest-priority per-instance overrides.

        Returns:
            result: None. Validated reference and configuration state are
                retained for subsequent ``Process`` calls.
        """

        self.defaultParameters: Mapping[str, object] = MappingProxyType(
            {
                "enableIntegerDelayCompensation": True,
                "enableFractionalDelayCompensation": True,
                "enableCarrierFrequencyOffsetCompensation": True,
                "enableSamplingFrequencyOffsetCompensation": True,
                "enableComplexGainCompensation": True,
                "maxIntegerDelaySamples": None,
                "maxCarrierFrequencyOffsetHz": None,
                "maxSamplingFrequencyOffsetPpm": 200.0,
                "timingWindowCount": 9,
                "timingWindowLength": 2048,
                "interpolationHalfLength": 12,
            }
        )
        self.referenceSignal = self.ValidateSignal(
            referenceSignal, "referenceSignal"
        )
        if (
            not isinstance(sampleRateHz, (int, float))
            or isinstance(sampleRateHz, bool)
            or not np.isfinite(sampleRateHz)
            or sampleRateHz <= 0.0
        ):
            raise ValueError("sampleRateHz must be finite and positive")
        self.sampleRateHz = float(sampleRateHz)
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping or None")
        externalParameters: Mapping[str, object] = (
            {}
            if parameters is None
            else RecognizedParameterView(
                parameters,
                self.defaultParameters,
                "SigProc",
            )
        )
        recognizedOverrides = FilterRecognizedParameters(
            parameterOverrides,
            self.defaultParameters,
            "SigProc",
        )
        self.parameters: ChainMap[str, object] = ChainMap(
            recognizedOverrides,
            externalParameters,
            self.defaultParameters,
        )
        self.ValidateParameters()
        self.lastResult: Optional[SignalProcessingResult] = None

    @staticmethod
    def ValidateSignal(inputSignal: np.ndarray, signalName: str) -> np.ndarray:
        """Convert and validate one finite complex baseband signal.

        Processing details:
            Algorithm: Flatten the input into a deterministic one-dimensional
            complex128 array and reject empty or non-finite data.

        Args:
            inputSignal: Array-like complex samples to validate.
            signalName: Human-readable name used in validation messages.

        Returns:
            result: One-dimensional finite complex128 sample array.
        """

        complexSignal = np.asarray(
            inputSignal, dtype=np.complex128
        ).reshape(-1)
        if complexSignal.size == 0:
            raise ValueError(f"{signalName} cannot be empty")
        if not np.all(np.isfinite(complexSignal)):
            raise ValueError(f"{signalName} contains NaN or infinite values")
        return complexSignal

    @staticmethod
    def EstimateSignalOverlap(
        measuredSignal: np.ndarray,
        referenceSignal: np.ndarray,
        maximumMeasuredOffsetSamples: int,
        maximumProbeLength: int,
        minimumConfidence: float,
    ) -> SignalOverlapResult:
        """Find the strongest valid common interval of two waveforms.

        Processing details:
            Algorithm: Remove negligible zero padding from the reference,
            enumerate signed lags with useful overlap, calculate normalized
            correlation on every physical chain, and prefer the longest,
            earliest interval when correlation scores tie. A negative lag
            means that the measured record starts inside a cropped reference.

        Args:
            measuredSignal: Measured vector or samples-by-chains matrix.
            referenceSignal: Known transmitted vector or matrix.
            maximumMeasuredOffsetSamples: Largest searched leading offset on
                the measured side.
            maximumProbeLength: Maximum samples used for each correlation.
            minimumConfidence: Minimum accepted normalized correlation.

        Returns:
            result: Common measured/reference starts, overlap length, and
                normalized multi-chain confidence.
        """

        complexMeasured = np.asarray(
            measuredSignal, dtype=np.complex128
        )
        complexReference = np.asarray(
            referenceSignal, dtype=np.complex128
        )
        for signalName, complexSignal in (
            ("measuredSignal", complexMeasured),
            ("referenceSignal", complexReference),
        ):
            if complexSignal.ndim not in (1, 2):
                raise ValueError(
                    f"{signalName} must be a vector or matrix"
                )
            if complexSignal.shape[0] == 0:
                raise ValueError(f"{signalName} cannot be empty")
            if not np.all(np.isfinite(complexSignal)):
                raise ValueError(
                    f"{signalName} contains NaN or infinite values"
                )
        if (
            not isinstance(maximumMeasuredOffsetSamples, int)
            or isinstance(maximumMeasuredOffsetSamples, bool)
            or maximumMeasuredOffsetSamples < 0
        ):
            raise ValueError(
                "maximumMeasuredOffsetSamples must be a nonnegative integer"
            )
        if (
            not isinstance(maximumProbeLength, int)
            or isinstance(maximumProbeLength, bool)
            or maximumProbeLength < 16
        ):
            raise ValueError(
                "maximumProbeLength must be an integer of at least 16"
            )
        if (
            not isinstance(minimumConfidence, (int, float))
            or isinstance(minimumConfidence, bool)
            or not np.isfinite(minimumConfidence)
            or not 0.0 <= minimumConfidence <= 1.0
        ):
            raise ValueError(
                "minimumConfidence must be finite and between zero and one"
            )

        measuredMatrix = (
            complexMeasured.reshape(-1, 1)
            if complexMeasured.ndim == 1
            else complexMeasured
        )
        referenceMatrix = (
            complexReference.reshape(-1, 1)
            if complexReference.ndim == 1
            else complexReference
        )
        if measuredMatrix.shape[1] != referenceMatrix.shape[1]:
            raise ValueError(
                "measuredSignal and referenceSignal must have the same "
                "number of physical chains"
            )

        referencePower = np.sum(
            np.abs(referenceMatrix) ** 2,
            axis=1,
        )
        peakReferencePower = float(np.max(referencePower))
        if peakReferencePower <= np.finfo(float).tiny:
            raise ValueError(
                "referenceSignal must contain nonzero finite samples"
            )
        activeReferenceIndices = np.flatnonzero(
            referencePower
            > peakReferencePower * np.finfo(float).eps
        )
        activeReferenceStart = int(activeReferenceIndices[0])
        activeReferenceStop = int(activeReferenceIndices[-1]) + 1
        activeReference = referenceMatrix[
            activeReferenceStart:activeReferenceStop
        ]

        measuredLength = int(measuredMatrix.shape[0])
        referenceLength = int(activeReference.shape[0])
        shorterLength = min(measuredLength, referenceLength)
        minimumOverlap = min(
            shorterLength,
            max(16, min(64, shorterLength // 4)),
        )
        minimumLag = -(referenceLength - minimumOverlap)
        maximumLag = min(
            maximumMeasuredOffsetSamples,
            measuredLength - minimumOverlap,
        )
        numericFloor = np.finfo(float).tiny
        bestScore = float("-inf")
        bestMeasuredStart = 0
        bestReferenceStart = activeReferenceStart
        bestOverlapLength = 0
        bestTieBreak = (float("-inf"), -1, 0)
        for candidateLag in range(minimumLag, maximumLag + 1):
            activeReferenceOffset = max(0, -candidateLag)
            measuredStart = max(0, candidateLag)
            overlapLength = min(
                referenceLength - activeReferenceOffset,
                measuredLength - measuredStart,
            )
            if overlapLength < minimumOverlap:
                continue
            probeLength = min(overlapLength, maximumProbeLength)
            chainPowers = []
            for chainIndex in range(referenceMatrix.shape[1]):
                referenceProbe = activeReference[
                    activeReferenceOffset:
                    activeReferenceOffset + probeLength,
                    chainIndex,
                ]
                measuredProbe = measuredMatrix[
                    measuredStart:measuredStart + probeLength,
                    chainIndex,
                ]
                referenceEnergy = float(
                    np.vdot(referenceProbe, referenceProbe).real
                )
                measuredEnergy = float(
                    np.vdot(measuredProbe, measuredProbe).real
                )
                correlationPower = float(
                    np.abs(
                        np.vdot(referenceProbe, measuredProbe)
                    )
                    ** 2
                )
                chainPowers.append(
                    correlationPower
                    / max(
                        referenceEnergy * measuredEnergy,
                        numericFloor,
                    )
                )
            candidateScore = float(np.mean(chainPowers))
            candidateTieBreak = (
                candidateScore,
                overlapLength,
                -measuredStart,
            )
            if candidateTieBreak > bestTieBreak:
                bestTieBreak = candidateTieBreak
                bestScore = candidateScore
                bestMeasuredStart = measuredStart
                bestReferenceStart = (
                    activeReferenceStart + activeReferenceOffset
                )
                bestOverlapLength = overlapLength

        if not np.isfinite(bestScore):
            raise ValueError(
                "measuredSignal and referenceSignal do not have a "
                "nonempty searchable overlap"
            )
        bestConfidence = float(np.sqrt(max(bestScore, 0.0)))
        if bestConfidence < float(minimumConfidence):
            raise ValueError(
                "waveform correlation is below minimumConfidence"
            )
        return SignalOverlapResult(
            receivedStartSample=bestMeasuredStart,
            referenceStartSample=bestReferenceStart,
            overlapLength=bestOverlapLength,
            confidence=bestConfidence,
        )

    def GetParameters(self) -> Dict[str, object]:
        """Return a flattened snapshot of effective processing parameters.

        Processing details:
            Algorithm: Resolve all ``ChainMap`` layers and copy the result so
            the returned dictionary cannot mutate processor state.

        Returns:
            result: Dictionary containing every effective setting.
        """

        return dict(self.parameters)

    def UpdateParameters(self, **parameterOverrides: object) -> None:
        """Apply validated highest-priority processing overrides.

        Processing details:
            Algorithm: Update the local ``ChainMap`` layer transactionally and
            restore its prior contents when validation fails.

        Args:
            parameterOverrides: Supported settings to update.

        Returns:
            result: None. Valid values affect subsequent processing calls.
        """

        recognizedOverrides = FilterRecognizedParameters(
            parameterOverrides,
            self.defaultParameters,
            "SigProc.UpdateParameters",
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
        """Validate all resolved synchronization parameters.

        Processing details:
            Algorithm: Validate switches, physical search limits, estimator
            window sizes, and interpolation support after unknown keys have
            been warned about and filtered at the configuration boundary.

        Returns:
            result: None. Invalid settings raise a descriptive exception.
        """

        switchNames = (
            "enableIntegerDelayCompensation",
            "enableFractionalDelayCompensation",
            "enableCarrierFrequencyOffsetCompensation",
            "enableSamplingFrequencyOffsetCompensation",
            "enableComplexGainCompensation",
        )
        for switchName in switchNames:
            if not isinstance(self.parameters[switchName], bool):
                raise TypeError(f"{switchName} must be boolean")

        maxIntegerDelaySamples = self.parameters["maxIntegerDelaySamples"]
        if maxIntegerDelaySamples is not None and (
            not isinstance(maxIntegerDelaySamples, int)
            or isinstance(maxIntegerDelaySamples, bool)
            or maxIntegerDelaySamples < 0
        ):
            raise ValueError(
                "maxIntegerDelaySamples must be a nonnegative integer or None"
            )

        maxCarrierFrequencyOffsetHz = self.parameters[
            "maxCarrierFrequencyOffsetHz"
        ]
        if maxCarrierFrequencyOffsetHz is not None and (
            not isinstance(maxCarrierFrequencyOffsetHz, (int, float))
            or isinstance(maxCarrierFrequencyOffsetHz, bool)
            or not np.isfinite(maxCarrierFrequencyOffsetHz)
            or maxCarrierFrequencyOffsetHz <= 0.0
            or maxCarrierFrequencyOffsetHz >= self.sampleRateHz / 2.0
        ):
            raise ValueError(
                "maxCarrierFrequencyOffsetHz must be positive, below Nyquist, or None"
            )

        maxSamplingFrequencyOffsetPpm = self.parameters[
            "maxSamplingFrequencyOffsetPpm"
        ]
        if (
            not isinstance(maxSamplingFrequencyOffsetPpm, (int, float))
            or isinstance(maxSamplingFrequencyOffsetPpm, bool)
            or not np.isfinite(maxSamplingFrequencyOffsetPpm)
            or maxSamplingFrequencyOffsetPpm < 0.0
        ):
            raise ValueError(
                "maxSamplingFrequencyOffsetPpm must be finite and nonnegative"
            )

        integerMinimums = {
            "timingWindowCount": 3,
            "timingWindowLength": 32,
            "interpolationHalfLength": 2,
        }
        for parameterName, minimumValue in integerMinimums.items():
            parameterValue = self.parameters[parameterName]
            if (
                not isinstance(parameterValue, int)
                or isinstance(parameterValue, bool)
                or parameterValue < minimumValue
            ):
                raise ValueError(
                    f"{parameterName} must be an integer of at least {minimumValue}"
                )

    def ResolveMaximumIntegerDelay(self) -> int:
        """Resolve the configured or automatic integer-delay search radius.

        Processing details:
            Algorithm: Use the explicit caller limit when provided; otherwise
            search up to one quarter of the reference with a 4096-sample cap.

        Returns:
            result: Nonnegative integer lag radius in samples.
        """

        configuredMaximum = self.parameters["maxIntegerDelaySamples"]
        if configuredMaximum is not None:
            return min(int(configuredMaximum), self.referenceSignal.size - 1)
        automaticMaximum = max(32, self.referenceSignal.size // 4)
        return min(4096, automaticMaximum, self.referenceSignal.size - 1)

    def EstimateIntegerDelay(self, measuredSignal: np.ndarray) -> int:
        """Estimate the signed integer delay of measurement versus reference.

        A positive result means that the measured waveform occurs later and
        must be sampled at ``n + delay`` to align it with reference sample
        ``n``.

        Processing details:
            Algorithm: Compute linear cross-correlation with an FFT, restrict
            the lag search, normalize every candidate by overlap energy, and
            select the maximum normalized magnitude.

        Args:
            measuredSignal: Finite measured complex waveform.

        Returns:
            result: Signed integer delay in nominal samples.
        """

        complexMeasured = self.ValidateSignal(
            measuredSignal, "measuredSignal"
        )
        referenceLength = self.referenceSignal.size
        measuredLength = complexMeasured.size
        fullLength = referenceLength + measuredLength - 1
        fftLength = 1 << int(np.ceil(np.log2(max(fullLength, 2))))
        correlation = np.fft.ifft(
            np.fft.fft(complexMeasured, fftLength)
            * np.fft.fft(np.conj(self.referenceSignal[::-1]), fftLength)
        )[:fullLength]
        lags = np.arange(fullLength, dtype=int) - (referenceLength - 1)
        maximumDelay = self.ResolveMaximumIntegerDelay()
        minimumOverlap = max(16, min(referenceLength, measuredLength) // 4)
        referenceEnergyPrefix = np.r_[
            0.0, np.cumsum(np.abs(self.referenceSignal) ** 2)
        ]
        measuredEnergyPrefix = np.r_[
            0.0, np.cumsum(np.abs(complexMeasured) ** 2)
        ]
        bestScore = -np.inf
        bestLag = 0

        for correlationIndex, lagValue in enumerate(lags):
            if abs(int(lagValue)) > maximumDelay:
                continue
            referenceStart = max(0, -int(lagValue))
            referenceStop = min(
                referenceLength, measuredLength - int(lagValue)
            )
            overlapLength = referenceStop - referenceStart
            if overlapLength < minimumOverlap:
                continue
            measuredStart = referenceStart + int(lagValue)
            measuredStop = measuredStart + overlapLength
            referenceEnergy = (
                referenceEnergyPrefix[referenceStop]
                - referenceEnergyPrefix[referenceStart]
            )
            measuredEnergy = (
                measuredEnergyPrefix[measuredStop]
                - measuredEnergyPrefix[measuredStart]
            )
            normalization = np.sqrt(
                max(referenceEnergy * measuredEnergy, np.finfo(float).tiny)
            )
            candidateScore = abs(correlation[correlationIndex]) / normalization
            if candidateScore > bestScore:
                bestScore = float(candidateScore)
                bestLag = int(lagValue)

        if not np.isfinite(bestScore):
            raise RuntimeError("unable to estimate integer delay")
        return bestLag

    def ExtractIntegerAligned(
        self, measuredSignal: np.ndarray, integerDelaySamples: int
    ) -> np.ndarray:
        """Extract a reference-length signal using one signed integer delay.

        Processing details:
            Algorithm: Map reference index ``n`` to measured index
            ``n + integerDelaySamples`` and zero-fill unavailable boundaries.

        Args:
            measuredSignal: Finite measured complex waveform.
            integerDelaySamples: Signed delay estimate in samples.

        Returns:
            result: Reference-length integer-aligned complex array.
        """

        complexMeasured = self.ValidateSignal(
            measuredSignal, "measuredSignal"
        )
        outputSignal = np.zeros_like(self.referenceSignal)
        referenceStart = max(0, -integerDelaySamples)
        referenceStop = min(
            self.referenceSignal.size,
            complexMeasured.size - integerDelaySamples,
        )
        if referenceStop <= referenceStart:
            raise ValueError("integer delay leaves no overlapping samples")
        measuredStart = referenceStart + integerDelaySamples
        measuredStop = measuredStart + (referenceStop - referenceStart)
        outputSignal[referenceStart:referenceStop] = complexMeasured[
            measuredStart:measuredStop
        ]
        return outputSignal

    def EstimateCarrierFrequencyOffset(
        self,
        integerAlignedSignal: np.ndarray,
    ) -> float:
        """Estimate data-aided carrier-frequency offset from block gains.

        Processing details:
            Algorithm: Estimate one least-squares complex gain in each of
            several time windows, unwrap the gain phases, fit their weighted
            linear slope versus sample index, convert radians per sample to
            hertz, and apply both a resolution deadband and the search bound.
            Block gains suppress nonlinear sample-to-sample PA phase changes
            that would otherwise resemble a false carrier offset.

        Args:
            integerAlignedSignal: Reference-length measurement after coarse
                integer-delay alignment.

        Returns:
            result: Estimated carrier-frequency offset in hertz.
        """

        complexAligned = self.ValidateSignal(
            integerAlignedSignal, "integerAlignedSignal"
        )
        if complexAligned.size != self.referenceSignal.size:
            raise ValueError(
                "integerAlignedSignal must match the reference length"
            )
        signalLength = self.referenceSignal.size
        requestedWindowLength = int(self.parameters["timingWindowLength"])
        windowLength = min(
            requestedWindowLength,
            max(64, signalLength // 12),
        )
        if windowLength >= signalLength:
            windowLength = max(16, signalLength // 3)
        halfWindow = windowLength // 2
        centerCount = min(
            max(int(self.parameters["timingWindowCount"]), 5),
            max(5, signalLength // max(windowLength, 1)),
        )
        firstCenter = halfWindow
        lastCenter = signalLength - halfWindow - 1
        if lastCenter <= firstCenter:
            return 0.0
        centerIndices = np.linspace(
            firstCenter, lastCenter, centerCount
        ).round().astype(int)
        validCenters = []
        blockPhases = []
        blockWeights = []
        for centerIndex in centerIndices:
            startIndex = int(centerIndex) - halfWindow
            stopIndex = startIndex + windowLength
            referenceWindow = self.referenceSignal[startIndex:stopIndex]
            measuredWindow = complexAligned[startIndex:stopIndex]
            referenceEnergy = float(
                np.vdot(referenceWindow, referenceWindow).real
            )
            measuredEnergy = float(
                np.vdot(measuredWindow, measuredWindow).real
            )
            if (
                referenceEnergy <= np.finfo(float).tiny
                or measuredEnergy <= np.finfo(float).tiny
            ):
                continue
            gainNumerator = np.vdot(referenceWindow, measuredWindow)
            normalizedCorrelation = abs(gainNumerator) / np.sqrt(
                referenceEnergy * measuredEnergy
            )
            validCenters.append(float(centerIndex))
            blockPhases.append(float(np.angle(gainNumerator)))
            blockWeights.append(max(float(normalizedCorrelation), 1.0e-6))

        if len(validCenters) < 3:
            return 0.0
        centerArray = np.asarray(validCenters, dtype=float)
        phaseArray = np.unwrap(np.asarray(blockPhases, dtype=float))
        weightArray = np.asarray(blockWeights, dtype=float) ** 2
        weightedCenter = np.average(centerArray, weights=weightArray)
        weightedPhase = np.average(phaseArray, weights=weightArray)
        centeredCoordinates = centerArray - weightedCenter
        slopeDenominator = np.sum(weightArray * centeredCoordinates**2)
        if slopeDenominator <= np.finfo(float).tiny:
            return 0.0
        radiansPerSample = np.sum(
            weightArray
            * centeredCoordinates
            * (phaseArray - weightedPhase)
        ) / slopeDenominator
        frequencyOffsetHz = (
            radiansPerSample * self.sampleRateHz / (2.0 * np.pi)
        )
        frequencyResolutionHz = self.sampleRateHz / max(signalLength, 1)
        if abs(frequencyOffsetHz) < 0.1 * frequencyResolutionHz:
            frequencyOffsetHz = 0.0
        configuredMaximum = self.parameters["maxCarrierFrequencyOffsetHz"]
        maximumOffsetHz = (
            self.sampleRateHz / 4.0
            if configuredMaximum is None
            else float(configuredMaximum)
        )
        return float(
            np.clip(frequencyOffsetHz, -maximumOffsetHz, maximumOffsetHz)
        )

    def CompensateCarrierFrequencyOffset(
        self, measuredSignal: np.ndarray, frequencyOffsetHz: float
    ) -> np.ndarray:
        """Remove a carrier-frequency offset from measured samples.

        Processing details:
            Algorithm: Multiply measured sample ``m`` by
            ``exp(-j*2*pi*frequencyOffsetHz*m/sampleRateHz)``.

        Args:
            measuredSignal: Finite measured complex waveform.
            frequencyOffsetHz: Offset estimate in hertz.

        Returns:
            result: Frequency-corrected complex waveform of unchanged length.
        """

        complexMeasured = self.ValidateSignal(
            measuredSignal, "measuredSignal"
        )
        sampleIndices = np.arange(complexMeasured.size, dtype=float)
        correction = np.exp(
            -1j
            * 2.0
            * np.pi
            * float(frequencyOffsetHz)
            * sampleIndices
            / self.sampleRateHz
        )
        return complexMeasured * correction

    @staticmethod
    def RefineCorrelationPeak(
        lowerScore: float, centerScore: float, upperScore: float
    ) -> float:
        """Refine one discrete correlation maximum with a parabola.

        Processing details:
            Algorithm: Fit a three-point quadratic around the integer maximum
            and bound the vertex to half a sample for numerical robustness.

        Args:
            lowerScore: Correlation magnitude at lag ``k-1``.
            centerScore: Correlation magnitude at lag ``k``.
            upperScore: Correlation magnitude at lag ``k+1``.

        Returns:
            result: Fractional correction in the interval ``[-0.5, 0.5]``.
        """

        denominator = lowerScore - 2.0 * centerScore + upperScore
        if abs(denominator) <= np.finfo(float).eps:
            return 0.0
        peakOffset = 0.5 * (lowerScore - upperScore) / denominator
        return float(np.clip(peakOffset, -0.5, 0.5))

    def EstimateTimingOffsets(
        self,
        frequencyCorrectedSignal: np.ndarray,
        integerDelaySamples: int,
    ) -> Tuple[int, float, float]:
        """Estimate residual fractional delay and sampling-frequency offset.

        Processing details:
            Algorithm: Correlate several reference windows against local
            measured windows, refine each timing peak to sub-sample accuracy,
            and fit ``delay(n) = fractionalDelay + slope*n``. The slope is
            reported in parts per million and the intercept is normalized to
            a signed half-sample interval by adjusting the integer delay.

        Args:
            frequencyCorrectedSignal: Measured waveform after CFO removal.
            integerDelaySamples: Coarse signed integer-delay estimate.

        Returns:
            result: Tuple containing adjusted integer delay, fractional delay
                in samples, and sampling-frequency offset in ppm.
        """

        complexMeasured = self.ValidateSignal(
            frequencyCorrectedSignal, "frequencyCorrectedSignal"
        )
        referenceLength = self.referenceSignal.size
        requestedWindowLength = int(self.parameters["timingWindowLength"])
        windowLength = min(requestedWindowLength, max(32, referenceLength // 4))
        if windowLength >= referenceLength:
            windowLength = max(16, referenceLength // 2)
        windowCount = min(
            int(self.parameters["timingWindowCount"]),
            max(3, referenceLength // max(windowLength, 1)),
        )
        maximumSamplingOffsetPpm = float(
            self.parameters["maxSamplingFrequencyOffsetPpm"]
        )
        maximumDriftSamples = (
            maximumSamplingOffsetPpm * referenceLength / 1.0e6
        )
        localSearchRadius = max(2, int(np.ceil(maximumDriftSamples)) + 2)
        halfWindow = windowLength // 2
        firstCenter = halfWindow + localSearchRadius
        lastCenter = referenceLength - halfWindow - localSearchRadius - 1
        if lastCenter <= firstCenter:
            return integerDelaySamples, 0.0, 0.0
        centerIndices = np.linspace(
            firstCenter,
            lastCenter,
            windowCount,
        ).round().astype(int)
        timingCenters = []
        timingDelays = []
        timingWeights = []

        for centerIndex in centerIndices:
            referenceStart = int(centerIndex) - halfWindow
            referenceStop = referenceStart + windowLength
            referenceWindow = self.referenceSignal[
                referenceStart:referenceStop
            ]
            referenceEnergy = max(
                float(np.vdot(referenceWindow, referenceWindow).real),
                np.finfo(float).tiny,
            )
            lagValues = np.arange(
                -localSearchRadius,
                localSearchRadius + 1,
                dtype=int,
            )
            lagScores = np.full(lagValues.size, -np.inf, dtype=float)
            for lagIndex, localLag in enumerate(lagValues):
                measuredStart = (
                    referenceStart + integerDelaySamples + int(localLag)
                )
                measuredStop = measuredStart + windowLength
                if measuredStart < 0 or measuredStop > complexMeasured.size:
                    continue
                measuredWindow = complexMeasured[measuredStart:measuredStop]
                measuredEnergy = max(
                    float(np.vdot(measuredWindow, measuredWindow).real),
                    np.finfo(float).tiny,
                )
                lagScores[lagIndex] = abs(
                    np.vdot(referenceWindow, measuredWindow)
                ) / np.sqrt(referenceEnergy * measuredEnergy)

            peakIndex = int(np.argmax(lagScores))
            if not np.isfinite(lagScores[peakIndex]):
                continue
            fractionalPeak = 0.0
            if 0 < peakIndex < lagScores.size - 1:
                fractionalPeak = self.RefineCorrelationPeak(
                    float(lagScores[peakIndex - 1]),
                    float(lagScores[peakIndex]),
                    float(lagScores[peakIndex + 1]),
                )
            timingCenters.append(float(centerIndex))
            timingDelays.append(
                float(lagValues[peakIndex]) + fractionalPeak
            )
            timingWeights.append(max(float(lagScores[peakIndex]), 1.0e-6))

        if not timingCenters:
            return integerDelaySamples, 0.0, 0.0
        centerArray = np.asarray(timingCenters, dtype=float)
        delayArray = np.asarray(timingDelays, dtype=float)
        weightArray = np.asarray(timingWeights, dtype=float) ** 2
        enableSamplingOffset = bool(
            self.parameters["enableSamplingFrequencyOffsetCompensation"]
        )
        if enableSamplingOffset and centerArray.size >= 3:
            weightedCenter = np.average(centerArray, weights=weightArray)
            weightedDelay = np.average(delayArray, weights=weightArray)
            centeredCoordinates = centerArray - weightedCenter
            slopeDenominator = np.sum(
                weightArray * centeredCoordinates**2
            )
            if slopeDenominator > np.finfo(float).tiny:
                timingSlope = np.sum(
                    weightArray
                    * centeredCoordinates
                    * (delayArray - weightedDelay)
                ) / slopeDenominator
            else:
                timingSlope = 0.0
            maximumSlope = maximumSamplingOffsetPpm / 1.0e6
            timingSlope = float(
                np.clip(timingSlope, -maximumSlope, maximumSlope)
            )
            timingIntercept = weightedDelay - timingSlope * weightedCenter
        else:
            timingSlope = 0.0
            timingIntercept = float(np.median(delayArray))

        enableFractionalDelay = bool(
            self.parameters["enableFractionalDelayCompensation"]
        )
        if enableFractionalDelay:
            integerAdjustment = int(np.floor(timingIntercept + 0.5))
            fractionalDelaySamples = timingIntercept - integerAdjustment
        else:
            integerAdjustment = int(np.round(timingIntercept))
            fractionalDelaySamples = 0.0
        adjustedIntegerDelay = integerDelaySamples + integerAdjustment
        samplingFrequencyOffsetPpm = timingSlope * 1.0e6
        return (
            adjustedIntegerDelay,
            float(fractionalDelaySamples),
            float(samplingFrequencyOffsetPpm),
        )

    def InterpolateSignal(
        self,
        inputSignal: np.ndarray,
        samplePositions: np.ndarray,
    ) -> np.ndarray:
        """Evaluate a complex signal at arbitrary fractional sample positions.

        Processing details:
            Algorithm: Return exact indexed values for an all-integer grid;
            otherwise apply a normalized finite-support Lanczos sinc kernel in
            bounded chunks and zero-fill positions outside the input record.

        Args:
            inputSignal: Finite one-dimensional complex samples.
            samplePositions: Floating-point source indices for each output.

        Returns:
            result: Complex samples evaluated at the requested positions.
        """

        complexInput = self.ValidateSignal(inputSignal, "inputSignal")
        positionArray = np.asarray(samplePositions, dtype=float).reshape(-1)
        if not np.all(np.isfinite(positionArray)):
            raise ValueError("samplePositions contains NaN or infinite values")
        roundedPositions = np.rint(positionArray).astype(np.int64)
        if np.all(np.abs(positionArray - roundedPositions) < 1.0e-12):
            outputSignal = np.zeros(positionArray.size, dtype=np.complex128)
            validMask = (
                (roundedPositions >= 0)
                & (roundedPositions < complexInput.size)
            )
            outputSignal[validMask] = complexInput[
                roundedPositions[validMask]
            ]
            return outputSignal

        halfLength = int(self.parameters["interpolationHalfLength"])
        tapOffsets = np.arange(-halfLength + 1, halfLength + 1)
        outputSignal = np.zeros(positionArray.size, dtype=np.complex128)
        chunkLength = 32768
        for startIndex in range(0, positionArray.size, chunkLength):
            stopIndex = min(startIndex + chunkLength, positionArray.size)
            chunkPositions = positionArray[startIndex:stopIndex]
            centerIndices = np.floor(chunkPositions).astype(np.int64)
            sourceIndices = centerIndices[:, None] + tapOffsets[None, :]
            distances = chunkPositions[:, None] - sourceIndices
            interpolationWeights = (
                np.sinc(distances)
                * np.sinc(distances / float(halfLength))
            )
            validMask = (
                (sourceIndices >= 0)
                & (sourceIndices < complexInput.size)
                & (np.abs(distances) < halfLength)
            )
            interpolationWeights *= validMask
            weightSums = np.sum(interpolationWeights, axis=1)
            safeWeightSums = np.where(
                np.abs(weightSums) > np.finfo(float).eps,
                weightSums,
                1.0,
            )
            clippedIndices = np.clip(
                sourceIndices, 0, complexInput.size - 1
            )
            outputSignal[startIndex:stopIndex] = np.sum(
                complexInput[clippedIndices] * interpolationWeights,
                axis=1,
            ) / safeWeightSums
        return outputSignal

    @staticmethod
    def EstimateComplexGain(
        referenceSignal: np.ndarray, measuredSignal: np.ndarray
    ) -> complex:
        """Estimate the least-squares gain mapping reference to measurement.

        Processing details:
            Algorithm: Evaluate ``reference^H * measured`` divided by
            ``reference^H * reference`` with a positive numerical floor.

        Args:
            referenceSignal: Known finite complex samples.
            measuredSignal: Aligned finite complex samples of equal length.

        Returns:
            result: Least-squares complex gain applied by the measured path.
        """

        complexReference = SigProc.ValidateSignal(
            referenceSignal, "referenceSignal"
        )
        complexMeasured = SigProc.ValidateSignal(
            measuredSignal, "measuredSignal"
        )
        if complexReference.size != complexMeasured.size:
            raise ValueError(
                "referenceSignal and measuredSignal must have equal length"
            )
        denominator = max(
            float(np.vdot(complexReference, complexReference).real),
            np.finfo(float).tiny,
        )
        return complex(
            np.vdot(complexReference, complexMeasured) / denominator
        )

    @staticmethod
    def ResolveEstimationSlice(
        estimationSlice: Optional[slice], signalLength: int
    ) -> slice:
        """Normalize an optional gain-estimation slice to valid boundaries.

        Processing details:
            Algorithm: Expand ``None`` to the complete signal, apply standard
            slice boundary normalization, and reject empty or strided regions.

        Args:
            estimationSlice: Optional contiguous region used for gain fitting.
            signalLength: Positive available reference length.

        Returns:
            result: Valid contiguous unit-step slice.
        """

        if estimationSlice is None:
            return slice(0, signalLength, 1)
        if not isinstance(estimationSlice, slice):
            raise TypeError("estimationSlice must be a slice or None")
        startIndex, stopIndex, stepSize = estimationSlice.indices(signalLength)
        if stepSize != 1:
            raise ValueError("estimationSlice must use a unit step")
        if stopIndex <= startIndex:
            raise ValueError("estimationSlice cannot be empty")
        return slice(startIndex, stopIndex, 1)

    def Process(
        self,
        measuredSignal: np.ndarray,
        estimationSlice: Optional[slice] = None,
    ) -> SignalProcessingResult:
        """Estimate and compensate all enabled signal impairments.

        Processing details:
            Algorithm: Estimate coarse integer timing, remove CFO, estimate
            residual fractional timing and sample-rate drift, interpolate the
            measured record onto the reference grid, estimate complex gain on
            the requested region, and divide out that gain. Every impairment
            switch can disable its corresponding estimate and correction.

        Args:
            measuredSignal: Captured or simulated complex waveform. Its length
                may differ from the reference because alignment performs the
                final extraction.
            estimationSlice: Optional reference-grid region used to estimate
                complex gain, normally the Wi-Fi data field.

        Returns:
            result: ``SignalProcessingResult`` containing the compensated
                reference-length signal and all scalar estimates.
        """

        self.ValidateParameters()
        complexMeasured = self.ValidateSignal(
            measuredSignal, "measuredSignal"
        )
        enableIntegerDelay = bool(
            self.parameters["enableIntegerDelayCompensation"]
        )
        integerDelaySamples = (
            self.EstimateIntegerDelay(complexMeasured)
            if enableIntegerDelay
            else 0
        )
        integerAlignedSignal = self.ExtractIntegerAligned(
            complexMeasured, integerDelaySamples
        )

        enableCarrierFrequencyOffset = bool(
            self.parameters["enableCarrierFrequencyOffsetCompensation"]
        )
        carrierFrequencyOffsetHz = (
            self.EstimateCarrierFrequencyOffset(integerAlignedSignal)
            if enableCarrierFrequencyOffset
            else 0.0
        )
        frequencyCorrectedSignal = self.CompensateCarrierFrequencyOffset(
            complexMeasured, carrierFrequencyOffsetHz
        )

        enableFractionalDelay = bool(
            self.parameters["enableFractionalDelayCompensation"]
        )
        enableSamplingOffset = bool(
            self.parameters["enableSamplingFrequencyOffsetCompensation"]
        )
        coarseAlignedSignal = self.ExtractIntegerAligned(
            frequencyCorrectedSignal, integerDelaySamples
        )
        coarseGain = self.EstimateComplexGain(
            self.referenceSignal, coarseAlignedSignal
        )
        coarseError = coarseAlignedSignal - coarseGain * self.referenceSignal
        coarseErrorRatio = np.sum(np.abs(coarseError) ** 2) / max(
            np.sum(np.abs(coarseGain * self.referenceSignal) ** 2),
            np.finfo(float).tiny,
        )
        coarseAlignmentIsExact = coarseErrorRatio < 1.0e-24
        if (
            (enableFractionalDelay or enableSamplingOffset)
            and not coarseAlignmentIsExact
        ):
            (
                integerDelaySamples,
                fractionalDelaySamples,
                samplingFrequencyOffsetPpm,
            ) = self.EstimateTimingOffsets(
                frequencyCorrectedSignal,
                integerDelaySamples,
            )
            # Correlation interpolation has a finite numerical floor even for
            # an exactly aligned record. Deadbands preserve bit-identical
            # ideal paths without masking physically meaningful impairments.
            if abs(fractionalDelaySamples) < 1.0e-3:
                fractionalDelaySamples = 0.0
            if abs(samplingFrequencyOffsetPpm) < 5.0e-2:
                samplingFrequencyOffsetPpm = 0.0
        else:
            fractionalDelaySamples = 0.0
            samplingFrequencyOffsetPpm = 0.0

        referenceIndices = np.arange(
            self.referenceSignal.size, dtype=float
        )
        sourcePositions = (
            float(integerDelaySamples)
            + float(fractionalDelaySamples)
            + referenceIndices
            * (1.0 + float(samplingFrequencyOffsetPpm) / 1.0e6)
        )
        alignedSignal = self.InterpolateSignal(
            frequencyCorrectedSignal, sourcePositions
        )

        gainSlice = self.ResolveEstimationSlice(
            estimationSlice, self.referenceSignal.size
        )
        enableComplexGain = bool(
            self.parameters["enableComplexGainCompensation"]
        )
        if enableComplexGain:
            complexGain = self.EstimateComplexGain(
                self.referenceSignal[gainSlice],
                alignedSignal[gainSlice],
            )
            if abs(complexGain) <= np.finfo(float).tiny:
                raise ValueError("estimated complex gain is numerically zero")
            processedSignal = alignedSignal / complexGain
        else:
            complexGain = 1.0 + 0.0j
            processedSignal = alignedSignal

        self.lastResult = SignalProcessingResult(
            processedSignal=processedSignal,
            integerDelaySamples=int(integerDelaySamples),
            fractionalDelaySamples=float(fractionalDelaySamples),
            carrierFrequencyOffsetHz=float(carrierFrequencyOffsetHz),
            samplingFrequencyOffsetPpm=float(samplingFrequencyOffsetPpm),
            complexGain=complex(complexGain),
        )
        return self.lastResult
