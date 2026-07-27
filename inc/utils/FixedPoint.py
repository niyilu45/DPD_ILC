"""Emulate signed fixed-point signal interfaces with floating-point arrays."""

from typing import Dict

import numpy as np


class FixedPoint:
    """Quantize external I/Q samples while preserving NumPy data types.

    A width of zero selects an unquantized floating-point interface. A
    positive width selects a signed normalized Q1.(width-1) format for each
    real and imaginary component. Quantized values are dequantized back to
    ``float64`` or ``complex128`` so all modules keep one public data type and
    continue to execute their internal algorithms in floating point.
    """

    def __init__(self, width: int = 16) -> None:
        """Validate and store one external-interface word width.

        Processing details:
            Algorithm: Accept zero as floating-point bypass, accept positive
            integer widths as signed normalized fixed-point formats, and
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
            Algorithm: Report one sign bit, the remaining fractional bits,
            quantization step, and representable component limits without
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
            }
        integerScale = float(2 ** (self.width - 1))
        quantizationStep = 1.0 / integerScale
        return {
            "mode": "fixed",
            "width": self.width,
            "signBits": 1,
            "fractionalBits": self.width - 1,
            "quantizationStep": quantizationStep,
            "minimumValue": -1.0,
            "maximumValue": 1.0 - quantizationStep,
        }

    def QuantizeComplex(self, inputSignal: np.ndarray) -> np.ndarray:
        """Quantize I and Q independently and return ``complex128`` samples.

        Processing details:
            Algorithm: Convert to complex128, round each component to nearest
            with NumPy's ties-to-even rule, saturate signed integer codes, and
            dequantize back to the original physical normalized scale. Width
            zero returns an equal-valued complex128 copy.

        Args:
            inputSignal: Finite complex vector or matrix of any shape.

        Returns:
            result: Complex128 array with the same shape in both modes.
        """

        complexInput = np.asarray(
            inputSignal, dtype=np.complex128
        )
        if not np.all(np.isfinite(complexInput)):
            raise ValueError(
                "inputSignal contains NaN or infinite values"
            )
        if self.IsFloatingPoint():
            return complexInput.copy()
        integerScale = float(2 ** (self.width - 1))
        minimumCode = -integerScale
        maximumCode = integerScale - 1.0
        realCodes = np.clip(
            np.rint(complexInput.real * integerScale),
            minimumCode,
            maximumCode,
        )
        imaginaryCodes = np.clip(
            np.rint(complexInput.imag * integerScale),
            minimumCode,
            maximumCode,
        )
        return (
            realCodes / integerScale
            + 1j * imaginaryCodes / integerScale
        ).astype(np.complex128, copy=False)
