"""Independent per-PA ILC and GMP deployment for MIMO transmit matrices."""

from dataclasses import dataclass, replace
from typing import Tuple

import numpy as np

from .DpdIlc import (
    FitGmpPredistorter,
    GMPPredistorter,
    ILCConfig,
    ILCResult,
    RunFrequencyDomainIlc,
)
from .PaModel import MimoPaModel


class MimoPaChain:
    """Expose one chain of ``MimoPaModel`` as an ordinary ILC plant."""

    def __init__(self, mimoPaModel: MimoPaModel, chainIndex: int) -> None:
        """Store the parent PA bank and selected zero-based chain index.

        Processing details:
            Algorithm: Validate the index against the live MIMO PA count and
            retain references without copying nonlinear model state.

        Args:
            mimoPaModel: Parent object owning all independent PA models.
            chainIndex: Zero-based chain processed by this view.

        Returns:
            result: None. A lightweight chain adapter is initialized.
        """

        if chainIndex < 0 or chainIndex >= mimoPaModel.numTransmitChains:
            raise IndexError("chainIndex is outside the configured chain range")
        self.mimoPaModel = mimoPaModel
        self.chainIndex = chainIndex

    def Process(self, inputSignal: np.ndarray) -> np.ndarray:
        """Process one vector through the selected physical PA path.

        Processing details:
            Algorithm: Delegate nonlinear and power-calibration operations to
            ``MimoPaModel.ProcessChain`` so ILC observes the deployed plant.

        Args:
            inputSignal: One-dimensional complex samples for this chain.

        Returns:
            result: Processed one-dimensional PA output samples.
        """

        return self.mimoPaModel.ProcessChain(inputSignal, self.chainIndex)


@dataclass(frozen=True)
class MimoIlcResult:
    """Store matrix-valued MIMO ILC results and each chain history."""

    learnedInput: np.ndarray
    outputSignal: np.ndarray
    chainResults: Tuple[ILCResult, ...]


class MimoGmpPredistorter:
    """Apply one independently fitted GMP predistorter to each PA input."""

    def __init__(
        self, chainPredistorters: Tuple[GMPPredistorter, ...]
    ) -> None:
        """Store one fitted GMP model per transmit chain.

        Processing details:
            Algorithm: Require at least one model and preserve tuple ordering
            as the physical RF-chain ordering used throughout the project.

        Args:
            chainPredistorters: Chain-ordered fitted GMP models.

        Returns:
            result: None. Predistorters are ready for matrix processing.
        """

        if not chainPredistorters:
            raise ValueError("chainPredistorters cannot be empty")
        self.chainPredistorters = tuple(chainPredistorters)

    def Process(self, inputSignal: np.ndarray) -> np.ndarray:
        """Predistort every column of a samples-by-chains matrix.

        Processing details:
            Algorithm: Validate exact chain count, evaluate each GMP on its
            corresponding column, and restore the common matrix orientation.

        Args:
            inputSignal: Complex matrix shaped samples by transmit chains.

        Returns:
            result: Predistorted matrix with the same shape as the input.
        """

        complexInput = np.asarray(inputSignal, dtype=np.complex128)
        if (
            complexInput.ndim != 2
            or complexInput.shape[1] != len(self.chainPredistorters)
            or complexInput.shape[0] == 0
        ):
            raise ValueError(
                "inputSignal must have one column per chain predistorter"
            )
        if not np.all(np.isfinite(complexInput)):
            raise ValueError("inputSignal must contain finite samples")
        return np.column_stack(
            [
                predistorter.Process(complexInput[:, chainIndex])
                for chainIndex, predistorter in enumerate(
                    self.chainPredistorters
                )
            ]
        )


def RunMimoFrequencyDomainIlc(
    referenceSignal: np.ndarray,
    mimoPaModel: MimoPaModel,
    sampleRateHz: float,
    channelBandwidthHz: float,
    config: ILCConfig = ILCConfig(),
) -> MimoIlcResult:
    """Run waveform ILC independently on every conducted transmit path.

    Processing details:
        Algorithm: Treat each PA as a separate repeatable SISO plant, offset
        feedback random seeds by chain, run identical regularized ILC, and
        stack learned inputs and outputs into the original MIMO orientation.

    Args:
        referenceSignal: Complex samples-by-transmit-chains target matrix.
        mimoPaModel: Independent nonlinear PA bank and power settings.
        sampleRateHz: Complex sampling rate in samples per second.
        channelBandwidthHz: Wanted Wi-Fi channel bandwidth in hertz.
        config: Shared ILC convergence and feedback configuration.

    Returns:
        result: Matrix ILC result plus all chain-specific histories.
    """

    complexReference = np.asarray(referenceSignal, dtype=np.complex128)
    if (
        complexReference.ndim != 2
        or complexReference.shape[1] != mimoPaModel.numTransmitChains
        or complexReference.shape[0] == 0
    ):
        raise ValueError(
            "referenceSignal must have one column per MIMO PA chain"
        )
    chainResults = []
    for chainIndex in range(mimoPaModel.numTransmitChains):
        chainConfig = replace(
            config,
            randomSeed=config.randomSeed + chainIndex,
            # A per-chain waveform cannot be evaluated by the full MIMO
            # post-spatial-demapping EVM callback. The common complex-gain
            # compensated MSE remains available for every physical PA.
            evmMseEvaluator=None,
        )
        chainResults.append(
            RunFrequencyDomainIlc(
                complexReference[:, chainIndex],
                MimoPaChain(mimoPaModel, chainIndex),
                sampleRateHz,
                channelBandwidthHz,
                chainConfig,
            )
        )
    resultTuple = tuple(chainResults)
    return MimoIlcResult(
        learnedInput=np.column_stack(
            [chainResult.learnedInput for chainResult in resultTuple]
        ),
        outputSignal=np.column_stack(
            [chainResult.outputSignal for chainResult in resultTuple]
        ),
        chainResults=resultTuple,
    )


def FitMimoGmpPredistorter(
    referenceSignal: np.ndarray,
    learnedInput: np.ndarray,
    nonlinearOrders: Tuple[int, ...] = (1, 3, 5, 7),
    memoryDepth: int = 3,
    crossMemoryDepth: int = 2,
    ridgeFactor: float = 1e-6,
) -> MimoGmpPredistorter:
    """Fit one deployable GMP to the ILC labels of every transmit chain.

    Processing details:
        Algorithm: Validate equal samples-by-chains matrices, independently
        regress each learned PA input on its original chain waveform, and
        assemble the fitted models in physical-chain order.

    Args:
        referenceSignal: Original MIMO waveform used as regression input.
        learnedInput: MIMO PA drive matrix learned by per-chain ILC.
        nonlinearOrders: Odd GMP main and cross nonlinear orders.
        memoryDepth: Main-branch sample memory depth.
        crossMemoryDepth: Lagging/leading envelope memory depth.
        ridgeFactor: Positive coefficient regularization strength.

    Returns:
        result: Matrix-capable collection of chain GMP predistorters.
    """

    complexReference = np.asarray(referenceSignal, dtype=np.complex128)
    complexLearned = np.asarray(learnedInput, dtype=np.complex128)
    if (
        complexReference.ndim != 2
        or complexReference.shape != complexLearned.shape
        or complexReference.shape[0] == 0
    ):
        raise ValueError(
            "referenceSignal and learnedInput must be equal nonempty matrices"
        )
    chainPredistorters = tuple(
        FitGmpPredistorter(
            complexReference[:, chainIndex],
            complexLearned[:, chainIndex],
            nonlinearOrders=nonlinearOrders,
            memoryDepth=memoryDepth,
            crossMemoryDepth=crossMemoryDepth,
            ridgeFactor=ridgeFactor,
        )
        for chainIndex in range(complexReference.shape[1])
    )
    return MimoGmpPredistorter(chainPredistorters)
