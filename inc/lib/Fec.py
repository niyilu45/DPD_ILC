"""Provide forward-error-correction primitives for project PHY metadata.

The module currently implements the deterministic short-block LDPC code used
by the project Wi-Fi descriptor. It deliberately contains only coding-domain
operations: parity-check construction, systematic encoding, and soft-input
iterative decoding. OFDM placement, pilots, interleaving, and semantic field
parsing remain in ``ParseWifi``.
"""

from functools import lru_cache
from itertools import combinations
from typing import Tuple

import numpy as np


@lru_cache(maxsize=1)
def BuildDescriptorLdpcMatrices() -> Tuple[np.ndarray, np.ndarray]:
    """Build the deterministic systematic short-block LDPC matrices.

    Processing details:
        Algorithm: Greedily choose three-check column neighborhoods while
        balancing check degrees and avoiding repeated row pairs, then append a
        lower-bidiagonal accumulator parity submatrix. A codeword is ordered
        as 55 message bits followed by 35 parity bits, giving a rate-55/90
        irregular repeat-accumulate LDPC code. The function-local deterministic
        construction is cached without module-level matrix variables and is
        bit-identical on Python 3.9 and Python 3.12.

    Returns:
        result: Pair containing the 35-by-90 parity-check matrix and its
            35-by-55 message submatrix as unsigned binary arrays.
    """

    messageBitCount = 55
    parityBitCount = 35
    messageMatrix = np.zeros(
        (parityBitCount, messageBitCount),
        dtype=np.uint8,
    )
    rowDegrees = np.zeros(parityBitCount, dtype=np.int64)
    usedRowPairs = set()
    candidateTriples = tuple(combinations(range(parityBitCount), 3))
    for messageIndex in range(messageBitCount):
        def TripleScore(
            candidateRows: Tuple[int, int, int],
        ) -> Tuple[int, int, int, int]:
            """Score one deterministic LDPC column-neighborhood candidate.

            Processing details:
                Algorithm: Penalize reused check-row pairs first, then peak
                and total row degree, and finally use a column-dependent
                cyclic tie break to distribute otherwise equal candidates.

            Args:
                candidateRows: Three distinct parity-check row indices.

            Returns:
                result: Lexicographically ordered integer score tuple.
            """

            candidatePairs = tuple(
                combinations(candidateRows, 2)
            )
            repeatedPairCount = sum(
                pair in usedRowPairs for pair in candidatePairs
            )
            candidateDegrees = rowDegrees[
                np.asarray(candidateRows, dtype=np.int64)
            ]
            cyclicTieBreak = sum(
                (
                    rowIndex
                    - 11 * messageIndex
                )
                % parityBitCount
                for rowIndex in candidateRows
            )
            return (
                repeatedPairCount,
                int(np.max(candidateDegrees)),
                int(np.sum(candidateDegrees)),
                int(cyclicTieBreak),
            )

        selectedRows = min(candidateTriples, key=TripleScore)
        usedRowPairs.update(combinations(selectedRows, 2))
        rowDegrees[
            np.asarray(selectedRows, dtype=np.int64)
        ] += 1
        for rowIndex in selectedRows:
            messageMatrix[rowIndex, messageIndex] = 1
    parityMatrix = np.eye(parityBitCount, dtype=np.uint8)
    parityMatrix[
        np.arange(1, parityBitCount),
        np.arange(parityBitCount - 1),
    ] = 1
    parityCheckMatrix = np.c_[messageMatrix, parityMatrix]
    parityCheckMatrix.setflags(write=False)
    messageMatrix.setflags(write=False)
    return parityCheckMatrix, messageMatrix


def EncodeDescriptorLdpc(messageBits: np.ndarray) -> np.ndarray:
    """Encode one 55-bit descriptor payload into a 90-bit LDPC codeword.

    Processing details:
        Algorithm: Keep the payload systematic, calculate the 35 check-source
        bits ``A m`` over GF(2), and solve the lower-bidiagonal accumulator
        recursively. For ``H=[A B]``, the emitted word ``[m, p]`` satisfies
        ``B p = A m`` and therefore ``H c = 0`` over GF(2).

    Args:
        messageBits: Binary descriptor payload containing exactly 55 bits.

    Returns:
        result: Systematic 90-bit LDPC codeword.
    """

    messageArray = np.asarray(
        messageBits,
        dtype=np.uint8,
    ).reshape(-1)
    if messageArray.size != 55 or np.any(messageArray > 1):
        raise ValueError("messageBits must contain exactly 55 binary values")
    parityCheckMatrix, messageMatrix = BuildDescriptorLdpcMatrices()
    paritySource = np.mod(
        messageMatrix.astype(np.int64)
        @ messageArray.astype(np.int64),
        2,
    ).astype(np.uint8)
    parityBits = np.zeros(35, dtype=np.uint8)
    parityBits[0] = paritySource[0]
    for parityIndex in range(1, parityBits.size):
        parityBits[parityIndex] = (
            paritySource[parityIndex]
            ^ parityBits[parityIndex - 1]
        )
    codeword = np.r_[messageArray, parityBits].astype(np.uint8)
    if np.any(
        np.mod(
            parityCheckMatrix.astype(np.int64)
            @ codeword.astype(np.int64),
            2,
        )
    ):
        raise RuntimeError("internal descriptor LDPC encoding failure")
    return codeword


def DecodeDescriptorLdpc(
    softCodeword: np.ndarray,
    maximumIterations: int = 60,
) -> np.ndarray:
    """Decode one soft 90-value LDPC observation with normalized min-sum.

    Processing details:
        Algorithm: Treat positive soft values as evidence for bit zero,
        exchange variable-to-check and check-to-variable log-likelihood
        messages on the sparse Tanner graph, apply 0.5-normalized min-sum
        check updates to stabilize low-degree accumulator nodes, and stop when
        every parity check is satisfied. The pure NumPy implementation avoids
        compiled-extension version restrictions and is intentionally
        compatible with Python 3.9 through Python 3.12.

    Args:
        softCodeword: Finite real soft observations in codeword order;
            positive values favor zero and negative values favor one.
        maximumIterations: Positive belief-propagation iteration limit.

    Returns:
        result: Corrected 55-bit systematic descriptor payload.
    """

    softValues = np.asarray(softCodeword, dtype=float).reshape(-1)
    if softValues.size != 90 or not np.all(np.isfinite(softValues)):
        raise ValueError(
            "softCodeword must contain exactly 90 finite real values"
        )
    if (
        not isinstance(maximumIterations, int)
        or isinstance(maximumIterations, bool)
        or maximumIterations < 1
    ):
        raise ValueError("maximumIterations must be a positive integer")
    parityCheckMatrix, _ = BuildDescriptorLdpcMatrices()
    edgeMask = parityCheckMatrix.astype(bool)
    nonzeroMagnitudes = np.abs(softValues[np.abs(softValues) > 0.0])
    softScale = (
        float(np.median(nonzeroMagnitudes))
        if nonzeroMagnitudes.size
        else 1.0
    )
    channelLlr = np.clip(
        2.0 * softValues / max(softScale, np.finfo(float).eps),
        -30.0,
        30.0,
    )
    variableToCheck = np.zeros(
        parityCheckMatrix.shape,
        dtype=float,
    )
    checkToVariable = np.zeros_like(variableToCheck)
    edgeRows, edgeColumns = np.nonzero(edgeMask)
    variableToCheck[edgeRows, edgeColumns] = channelLlr[edgeColumns]
    decodedBits = (channelLlr < 0.0).astype(np.uint8)
    for _ in range(maximumIterations):
        for checkIndex in range(parityCheckMatrix.shape[0]):
            variableIndices = np.flatnonzero(edgeMask[checkIndex])
            incomingMessages = variableToCheck[
                checkIndex,
                variableIndices,
            ]
            incomingSigns = np.where(
                incomingMessages < 0.0,
                -1.0,
                1.0,
            )
            incomingMagnitudes = np.abs(incomingMessages)
            totalSign = float(np.prod(incomingSigns))
            if incomingMagnitudes.size <= 1:
                extrinsicMinimums = np.zeros(
                    incomingMagnitudes.shape, dtype=float
                )
            else:
                # Every outgoing check message needs the minimum magnitude of
                # all other edges. One smallest/second-smallest reduction gives
                # every result at once and replaces one np.delete allocation
                # per Tanner-graph edge. Repeated minima retain the smallest
                # value because removing one occurrence leaves another.
                twoSmallest = np.partition(
                    incomingMagnitudes, 1
                )[:2]
                minimumMagnitude = float(twoSmallest[0])
                secondMinimumMagnitude = float(twoSmallest[1])
                extrinsicMinimums = np.full(
                    incomingMagnitudes.shape,
                    minimumMagnitude,
                    dtype=float,
                )
                minimumMask = incomingMagnitudes == minimumMagnitude
                if np.count_nonzero(minimumMask) == 1:
                    extrinsicMinimums[minimumMask] = (
                        secondMinimumMagnitude
                    )
            checkToVariable[checkIndex, variableIndices] = (
                0.5
                * totalSign
                * incomingSigns
                * extrinsicMinimums
            )
        posteriorLlr = channelLlr + np.sum(
            checkToVariable,
            axis=0,
        )
        decodedBits = (posteriorLlr < 0.0).astype(np.uint8)
        syndrome = np.mod(
            parityCheckMatrix.astype(np.int64)
            @ decodedBits.astype(np.int64),
            2,
        )
        if not np.any(syndrome):
            return decodedBits[:55].copy()
        variableToCheck[edgeRows, edgeColumns] = (
            posteriorLlr[edgeColumns]
            - checkToVariable[edgeRows, edgeColumns]
        )
    raise ValueError(
        "descriptor LDPC decoder did not converge to a valid codeword"
    )
