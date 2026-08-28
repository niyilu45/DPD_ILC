"""Represent signed fixed-point I/Q codes in complex NumPy arrays."""

from typing import Dict

import numpy as np


class FixedPoint:
    """Convert between physical floating samples and public fixed-point codes.

    A width of zero selects an unquantized floating-point interface. A
    positive width selects one signed integer code per I and Q component.
    ``fullScaleAmplitude`` defines the positive physical component amplitude
    represented by the nominal code scale in fixed mode. Its default of one
    preserves the historical normalized convention exactly.
    Public arrays always use ``complex128`` as a common storage container, but
    their real and imaginary components are integer-valued codes in fixed
    mode. Modules call ``DecodeComplex`` before floating-point processing and
    ``EncodeComplex`` before returning data through a public interface.
    """

    def __init__(
        self,
        width: int = 16,
        fullScaleAmplitude: float = 1.0,
    ) -> None:
        """Validate and store one external-interface numerical convention.

        Processing details:
            Algorithm: Accept zero as floating-point bypass, accept positive
            integer widths as signed public I/Q code formats, require a finite
            positive physical component full scale in both modes, and reject
            booleans, negative widths, and noninteger widths. Floating mode
            retains the scale as metadata but does not apply it to samples.

        Args:
            width: Total bits per I or Q component, including one sign bit.
            fullScaleAmplitude: Positive physical I/Q component amplitude
                represented by the nominal positive full-scale code. The
                default one selects the historical normalized mapping.

        Returns:
            result: None. The immutable numerical convention is retained.
        """

        if (
            not isinstance(width, int)
            or isinstance(width, bool)
            or width < 0
        ):
            raise ValueError("width must be a nonnegative integer")
        if width > 53:
            raise ValueError(
                "width cannot exceed 53 with a float64 external data type"
            )
        if (
            not isinstance(
                fullScaleAmplitude,
                (int, float, np.integer, np.floating),
            )
            or isinstance(fullScaleAmplitude, (bool, np.bool_))
            or not np.isfinite(fullScaleAmplitude)
            or float(fullScaleAmplitude) <= 0.0
        ):
            raise ValueError(
                "fullScaleAmplitude must be a finite positive real number"
            )
        self.width = int(width)
        self.fullScaleAmplitude = float(fullScaleAmplitude)

    def IsFloatingPoint(self) -> bool:
        """Return whether quantization is bypassed.

        Processing details:
            Algorithm: Treat exactly zero bits as the documented floating
            interface mode and every positive width as fixed-point emulation.

        Returns:
            result: True only when ``width`` equals zero.
        """

        return self.width == 0

    def GetFormatInfo(self) -> Dict[str, object]:
        """Return the resolved interface-format definition.

        Processing details:
            Algorithm: Report the integer code range, physical normalized
            range relative to the configured full-scale amplitude, sign
            allocation, and one-code physical step without changing samples.
            Floating mode reports its retained full-scale metadata while its
            unbounded physical range and zero quantization step remain exact.

        Returns:
            result: Ordinary dictionary describing the active interface mode.
        """

        if self.IsFloatingPoint():
            return {
                "mode": "floating",
                "width": 0,
                "fullScaleAmplitude": self.fullScaleAmplitude,
                "signBits": 0,
                "fractionalBits": 0,
                "quantizationStep": 0.0,
                "minimumValue": float("-inf"),
                "maximumValue": float("inf"),
                "minimumCode": None,
                "maximumCode": None,
                "physicalMinimumValue": float("-inf"),
                "physicalMaximumValue": float("inf"),
            }
        integerScale = float(2 ** (self.width - 1))
        quantizationStep = self.fullScaleAmplitude / integerScale
        minimumCode = -integerScale
        maximumCode = integerScale - 1.0
        return {
            "mode": "fixed",
            "width": self.width,
            "fullScaleAmplitude": self.fullScaleAmplitude,
            "signBits": 1,
            "fractionalBits": self.width - 1,
            "quantizationStep": quantizationStep,
            "minimumValue": minimumCode,
            "maximumValue": maximumCode,
            "minimumCode": minimumCode,
            "maximumCode": maximumCode,
            "physicalMinimumValue": -self.fullScaleAmplitude,
            "physicalMaximumValue": (
                self.fullScaleAmplitude - quantizationStep
            ),
        }

    def QuantizeCodes(self, inputSignal: np.ndarray) -> np.ndarray:
        """Round and saturate public I/Q codes without changing their scale.

        Processing details:
            Algorithm: Convert to complex128, round real and imaginary code
            components independently with NumPy's ties-to-even rule, and
            saturate them to the selected signed integer range. Width zero
            returns an equal-valued complex128 copy.

        Args:
            inputSignal: Finite public code vector or matrix of any shape.

        Returns:
            result: Complex128 array whose fixed-mode components are integers.
        """

        complexInput = np.asarray(inputSignal, dtype=np.complex128)
        if not np.all(np.isfinite(complexInput)):
            raise ValueError("inputSignal contains NaN or infinite values")
        if self.IsFloatingPoint():
            return complexInput.copy()
        integerScale = float(2 ** (self.width - 1))
        minimumCode = -integerScale
        maximumCode = integerScale - 1.0
        realCodes = np.clip(
            np.rint(complexInput.real),
            minimumCode,
            maximumCode,
        )
        imaginaryCodes = np.clip(
            np.rint(complexInput.imag),
            minimumCode,
            maximumCode,
        )
        return (realCodes + 1j * imaginaryCodes).astype(
            np.complex128, copy=False
        )

    def EncodeComplex(self, inputSignal: np.ndarray) -> np.ndarray:
        """Encode physical I/Q values as public integer codes.

        Processing details:
            Algorithm: Divide each physical component by the configured
            full-scale amplitude, multiply by ``2**(width-1)``, round to the
            nearest integer, saturate to the signed code range, and retain
            complex128 as the public container. Width zero bypasses both the
            full-scale mapping and quantization and returns an equal copy.

        Args:
            inputSignal: Finite physical samples of any shape.

        Returns:
            result: Public complex128 samples containing fixed integer codes.
        """

        complexInput = np.asarray(inputSignal, dtype=np.complex128)
        if not np.all(np.isfinite(complexInput)):
            raise ValueError("inputSignal contains NaN or infinite values")
        if self.IsFloatingPoint():
            return complexInput.copy()
        integerScale = float(2 ** (self.width - 1))
        return self.QuantizeCodes(
            complexInput / self.fullScaleAmplitude * integerScale
        )

    def DecodeComplex(self, inputSignal: np.ndarray) -> np.ndarray:
        """Decode public integer I/Q codes to physical samples.

        Processing details:
            Algorithm: First round and saturate the external code values, then
            divide both components by ``2**(width-1)`` and multiply by the
            configured full-scale amplitude. Width zero bypasses the retained
            scale and returns an equal-valued complex128 copy.

        Args:
            inputSignal: Finite public complex samples containing I/Q codes.

        Returns:
            result: Complex128 physical values of the same shape.
        """

        quantizedCodes = self.QuantizeCodes(inputSignal)
        if self.IsFloatingPoint():
            return quantizedCodes
        integerScale = float(2 ** (self.width - 1))
        return (
            quantizedCodes / integerScale * self.fullScaleAmplitude
        ).astype(
            np.complex128, copy=False
        )

    def QuantizeComplex(self, inputSignal: np.ndarray) -> np.ndarray:
        """Encode physical samples using the historical method name.

        Processing details:
            Algorithm: Delegate to ``EncodeComplex``. The alias keeps older
            callers source-compatible while changing fixed-mode public values
            to the required raw integer-code convention.

        Args:
            inputSignal: Finite physical samples of any shape.

        Returns:
            result: Public complex128 samples containing fixed integer codes.
        """

        return self.EncodeComplex(inputSignal)
