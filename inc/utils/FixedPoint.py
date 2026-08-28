"""Represent signed fixed-point I/Q codes in complex NumPy arrays."""

from typing import Dict, Optional, Tuple

import numpy as np


class FixedPointArray(np.ndarray):
    """Carry public I/Q codes together with their numerical reference plane."""

    def __array_finalize__(self, source: Optional[np.ndarray]) -> None:
        """Propagate fixed-point metadata to NumPy-derived views.

        Processing details:
            Algorithm: Copy the private width and full-scale fields from the
            source view when present. A raw ndarray view without these fields
            remains unannotated and is ignored by ``GetFixedPointFormat``.

        Args:
            source: Source array used by NumPy to create the new view.

        Returns:
            result: None. Metadata is attached to the new array in place.
        """

        if source is None:
            return
        self._fixedPointWidth = getattr(
            source, "_fixedPointWidth", None
        )
        self._fullScaleAmplitude = getattr(
            source, "_fullScaleAmplitude", None
        )

    @property
    def FixedPointWidth(self) -> int:
        """Return the public I/Q component width carried by this array.

        Returns:
            result: Zero for floating bypass or the positive code width.
        """

        return int(self._fixedPointWidth)

    fixedPointWidth = FixedPointWidth

    @property
    def FullScaleAmplitude(self) -> float:
        """Return the physical component magnitude represented by code rails.

        Returns:
            result: Positive full-scale amplitude associated with the codes.
        """

        return float(self._fullScaleAmplitude)

    fullScaleAmplitude = FullScaleAmplitude


def CreateFixedPointArray(
    inputSignal: np.ndarray,
    width: int,
    fullScaleAmplitude: float,
) -> FixedPointArray:
    """Create a complex code array with immutable format metadata.

    Processing details:
        Algorithm: Validate the supplied fixed-point convention, view a
        complex128 copy as ``FixedPointArray``, and attach the component width
        and physical code-rail amplitude. NumPy slices and copies retain these
        fields through ``FixedPointArray.__array_finalize__``.

    Args:
        inputSignal: Finite public code or floating-bypass samples.
        width: Nonnegative public I/Q component width.
        fullScaleAmplitude: Positive physical component code-rail value.

    Returns:
        result: Metadata-bearing complex128 NumPy array.
    """

    formatDefinition = FixedPoint(width, fullScaleAmplitude)
    complexInput = np.asarray(inputSignal, dtype=np.complex128)
    if not np.all(np.isfinite(complexInput)):
        raise ValueError("inputSignal contains NaN or infinite values")
    result = complexInput.copy().view(FixedPointArray)
    result._fixedPointWidth = formatDefinition.width
    result._fullScaleAmplitude = formatDefinition.fullScaleAmplitude
    return result


def GetFixedPointFormat(
    inputSignal: object,
) -> Optional[Tuple[int, float]]:
    """Read validated fixed-point metadata from a public signal array.

    Processing details:
        Algorithm: Recognize only ``FixedPointArray`` instances whose private
        metadata survived NumPy propagation, revalidate the convention, and
        return an immutable pair. Plain ndarrays deliberately return ``None``
        because their physical code scale cannot be inferred from values.

    Args:
        inputSignal: Candidate public signal object.

    Returns:
        result: ``(width, fullScaleAmplitude)`` or ``None`` when unannotated.
    """

    if not isinstance(inputSignal, FixedPointArray):
        return None
    rawWidth = getattr(inputSignal, "_fixedPointWidth", None)
    rawFullScaleAmplitude = getattr(
        inputSignal, "_fullScaleAmplitude", None
    )
    if rawWidth is None or rawFullScaleAmplitude is None:
        return None
    formatDefinition = FixedPoint(rawWidth, rawFullScaleAmplitude)
    return (
        formatDefinition.width,
        formatDefinition.fullScaleAmplitude,
    )


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
            return CreateFixedPointArray(
                complexInput, self.width, self.fullScaleAmplitude
            )
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
        return CreateFixedPointArray(
            (realCodes + 1j * imaginaryCodes).astype(
                np.complex128, copy=False
            ),
            self.width,
            self.fullScaleAmplitude,
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
            return CreateFixedPointArray(
                complexInput, self.width, self.fullScaleAmplitude
            )
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
            return np.asarray(
                quantizedCodes, dtype=np.complex128
            ).copy()
        integerScale = float(2 ** (self.width - 1))
        decodedSignal = (
            np.asarray(quantizedCodes, dtype=np.complex128)
            / integerScale
            * self.fullScaleAmplitude
        )
        return np.asarray(decodedSignal, dtype=np.complex128)

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
