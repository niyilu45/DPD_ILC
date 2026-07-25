"""Utility configuration filtering for ChainMap-backed project classes.

Unknown configuration keys are nonfatal. They are excluded from parameter
resolution and reported through Python warnings so a long simulation can keep
running while still exposing caller spelling mistakes or stale options.
"""

import warnings
from typing import Dict, Iterator, Mapping, Set, Tuple


def FindUnknownParameterNames(
    parameters: Mapping[object, object],
    knownParameterNames: Mapping[str, object],
) -> Tuple[str, ...]:
    """Return sorted display names for keys not supported by one owner.

    Processing details:
        Algorithm: Compare every caller key with the keys in the owner's
        immutable default mapping, convert unsupported keys to readable text,
        remove duplicate display names, and sort the result deterministically.

    Args:
        parameters: Caller mapping that may contain supported or stale keys.
        knownParameterNames: Default mapping whose keys define the public
            configuration vocabulary.

    Returns:
        result: Sorted tuple of unknown key names.
    """

    return tuple(
        sorted(
            {
                str(parameterName)
                for parameterName in parameters
                if parameterName not in knownParameterNames
            }
        )
    )


def WarnUnknownParameters(
    ownerName: str,
    unknownParameterNames: Tuple[str, ...],
    stacklevel: int = 3,
) -> None:
    """Warn that unsupported configuration keys will be ignored.

    Processing details:
        Algorithm: Emit one ``UserWarning`` containing all new unsupported
        names. An empty tuple is a no-op.

    Args:
        ownerName: Class or function that owns the supported parameter set.
        unknownParameterNames: Unsupported names to report together.
        stacklevel: Warning-frame depth used to point toward caller code.

    Returns:
        result: None. Warning delivery uses Python's standard warning system.
    """

    if not unknownParameterNames:
        return
    warnings.warn(
        (
            f"{ownerName} ignored unknown configuration parameter(s): "
            + ", ".join(unknownParameterNames)
        ),
        UserWarning,
        stacklevel=stacklevel,
    )


def FilterRecognizedParameters(
    parameters: Mapping[object, object],
    knownParameterNames: Mapping[str, object],
    ownerName: str,
) -> Dict[str, object]:
    """Copy recognized keys and warn once for unsupported keys.

    Processing details:
        Algorithm: Report unknown keys, then build a normal dictionary from
        only keys present in the owner's default mapping.

    Args:
        parameters: Input mapping supplied to a constructor or update method.
        knownParameterNames: Default mapping defining recognized keys.
        ownerName: Name included in warning messages.

    Returns:
        result: Dictionary containing only recognized configuration entries.
    """

    unknownParameterNames = FindUnknownParameterNames(
        parameters,
        knownParameterNames,
    )
    WarnUnknownParameters(ownerName, unknownParameterNames, stacklevel=4)
    return {
        str(parameterName): parameterValue
        for parameterName, parameterValue in parameters.items()
        if parameterName in knownParameterNames
    }


class RecognizedParameterView(Mapping[str, object]):
    """Expose a live external mapping while hiding unsupported keys."""

    def __init__(
        self,
        parameters: Mapping[object, object],
        knownParameterNames: Mapping[str, object],
        ownerName: str,
    ) -> None:
        """Create a live filtered view and report current unknown keys.

        Processing details:
            Algorithm: Keep the caller mapping by reference, store the
            recognized key vocabulary, and remember warned names so later
            accesses warn only for newly introduced unsupported keys.

        Args:
            parameters: Live caller-owned mapping.
            knownParameterNames: Default mapping defining recognized keys.
            ownerName: Name included in warning messages.

        Returns:
            result: None. Mapping protocol methods expose recognized live data.
        """

        self.sourceParameters = parameters
        self.knownParameterNames = knownParameterNames
        self.ownerName = ownerName
        self.warnedParameterNames: Set[str] = set()
        self.WarnForNewUnknownParameters()

    def WarnForNewUnknownParameters(self) -> None:
        """Warn once for each unsupported key observed in the live mapping.

        Processing details:
            Algorithm: Compare current unknown names with the remembered set,
            emit one warning for newly observed names, and update that set.

        Returns:
            result: None. Supported keys and the source mapping are unchanged.
        """

        currentUnknownNames = set(
            FindUnknownParameterNames(
                self.sourceParameters,
                self.knownParameterNames,
            )
        )
        newUnknownNames = tuple(
            sorted(currentUnknownNames.difference(
                self.warnedParameterNames
            ))
        )
        WarnUnknownParameters(
            self.ownerName,
            newUnknownNames,
            stacklevel=5,
        )
        self.warnedParameterNames.update(currentUnknownNames)

    def __getitem__(self, parameterName: str) -> object:
        """Return one recognized value from the live source mapping.

        Processing details:
            Algorithm: Check for newly added unsupported keys, reject access
            outside the recognized vocabulary with ``KeyError``, and otherwise
            read the current value directly from the caller-owned mapping.

        Args:
            parameterName: Recognized configuration key requested by ChainMap.

        Returns:
            result: Current caller-owned value for the recognized key.
        """

        self.WarnForNewUnknownParameters()
        if parameterName not in self.knownParameterNames:
            raise KeyError(parameterName)
        return self.sourceParameters[parameterName]

    def __iter__(self) -> Iterator[str]:
        """Iterate recognized keys currently present in the live source.

        Processing details:
            Algorithm: Warn for newly observed unsupported keys and lazily
            yield only source keys included in the recognized vocabulary.

        Returns:
            result: Iterator over recognized string configuration keys.
        """

        self.WarnForNewUnknownParameters()
        return (
            str(parameterName)
            for parameterName in self.sourceParameters
            if parameterName in self.knownParameterNames
        )

    def __len__(self) -> int:
        """Return the number of recognized keys in the live source.

        Processing details:
            Algorithm: Warn for newly observed unsupported keys and count only
            source keys included in the recognized vocabulary.

        Returns:
            result: Number of recognized external configuration entries.
        """

        self.WarnForNewUnknownParameters()
        return sum(
            1
            for parameterName in self.sourceParameters
            if parameterName in self.knownParameterNames
        )
