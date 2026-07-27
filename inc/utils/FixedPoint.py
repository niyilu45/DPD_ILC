"""Represent signed fixed-point I/Q codes in complex NumPy arrays."""

from typing import Dict

import numpy as np


class FixedPoint:
    """Convert between physical floating samples and public fixed-point codes.

    A width of zero selects an unquantized floating-point interface. A
    positive width selects one signed integer code per I and Q component.
    Public arrays always use ``complex128`` as a common storage container, but
    their real and imaginary components are integer-valued codes in fixed
    mode. Modules call ``DecodeComplex`` before floating-point processing and
    ``EncodeComplex`` before returning data through a public interface.
    """

    def __init__(self, width: int = 16) -> None:
        """Validate and store one external-interface word width.

        Processing details:
            Algorithm: Accept zero as floating-point bypass, accept positive
            integer widths as signed public I/Q code formats, and
            reject booleans or negative and noninteger values.

        Args:
            width: Total bits per I or Q component, including one sign bit.

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
        self.width = int(width)

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
            range, sign allocation, and one-code physical step without
            changing any samples.

        Returns:
            result: Ordinary dictionary describing the active interface mode.
        """

        if self.IsFloatingPoint():
            return {
                "mode": "floating",
                "width": 0,
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
        quantizationStep = 1.0 / integerScale
        minimumCode = -integerScale
        maximumCode = integerScale - 1.0
        return {
            "mode": "fixed",
            "width": self.width,
            "signBits": 1,
            "fractionalBits": self.width - 1,
            "quantizationStep": quantizationStep,
            "minimumValue": minimumCode,
            "maximumValue": maximumCode,
            "minimumCode": minimumCode,
            "maximumCode": maximumCode,
            "physicalMinimumValue": -1.0,
            "physicalMaximumValue": 1.0 - quantizationStep,
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
        """Encode normalized physical I/Q values as public integer codes.

        Processing details:
            Algorithm: Multiply each physical component by
            ``2**(width-1)``, round to the nearest integer, saturate to the
            signed code range, and retain complex128 as the public container.
            Width zero bypasses scaling and returns a complex128 copy.

        Args:
            inputSignal: Finite normalized physical samples of any shape.

        Returns:
            result: Public complex128 samples containing fixed integer codes.
        """

        complexInput = np.asarray(inputSignal, dtype=np.complex128)
        if not np.all(np.isfinite(complexInput)):
            raise ValueError("inputSignal contains NaN or infinite values")
        if self.IsFloatingPoint():
            return complexInput.copy()
        integerScale = float(2 ** (self.width - 1))
        return self.QuantizeCodes(complexInput * integerScale)

    def DecodeComplex(self, inputSignal: np.ndarray) -> np.ndarray:
        """Decode public integer I/Q codes to normalized physical samples.

        Processing details:
            Algorithm: First round and saturate the external code values, then
            divide both components by ``2**(width-1)``. This makes all module
            internals operate on approximately unit-scale complex envelopes.
            Width zero returns an equal-valued complex128 copy.

        Args:
            inputSignal: Finite public complex samples containing I/Q codes.

        Returns:
            result: Normalized complex128 physical values of the same shape.
        """

        quantizedCodes = self.QuantizeCodes(inputSignal)
        if self.IsFloatingPoint():
            return quantizedCodes
        integerScale = float(2 ** (self.width - 1))
        return (quantizedCodes / integerScale).astype(
            np.complex128, copy=False
        )

    def QuantizeComplex(self, inputSignal: np.ndarray) -> np.ndarray:
        """Encode normalized samples using the historical method name.

        Processing details:
            Algorithm: Delegate to ``EncodeComplex``. The alias keeps older
            callers source-compatible while changing fixed-mode public values
            to the required raw integer-code convention.

        Args:
            inputSignal: Finite normalized physical samples of any shape.

        Returns:
            result: Public complex128 samples containing fixed integer codes.
        """

        return self.EncodeComplex(inputSignal)
