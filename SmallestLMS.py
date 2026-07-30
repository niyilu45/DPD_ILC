"""Minimal sample-by-sample GMP-NLMS reference for software or HDL ports."""

import numpy as np

from inc.lib.DpdLms import DpdLms


def RunSmallestLms() -> None:
    """Train two complex GMP coefficients one sample at a time.

    Processing details:
        Algorithm: Generate a deterministic normalized complex stream, create
        a known linear-plus-cubic target, prepare one frozen feature scale,
        call UpdateSample exactly once per chronological sample, commit the
        shadow coefficients after the frame, and print fit improvement.

    Returns:
        result: None. Coefficients and before/after NMSE are printed.
    """

    randomGenerator = np.random.default_rng(7)
    referenceSignal = (
        randomGenerator.standard_normal(8192)
        + 1j * randomGenerator.standard_normal(8192)
    )
    referenceSignal *= 0.25 / np.sqrt(
        np.mean(np.abs(referenceSignal) ** 2)
    )

    # This known target lets a port compare recovered coefficients directly.
    targetSignal = (
        1.03 * referenceSignal
        + 0.18
        * referenceSignal
        * np.abs(referenceSignal) ** 2
    )

    dpdLms = DpdLms(
        parameters={
            "nonlinearOrders": (1, 3),
            "memoryDepth": 1,
            "crossMemoryDepth": 0,
            "adaptationMode": "nlms",
            "learningRate": 0.10,
            "featureScaleMode": "frame",
            "coefficientCommitMode": "frame",
            "maximumOutputMagnitude": None,
            "width": 0,
        }
    )

    beforeNmseDb = dpdLms.CalculateNmse(
        referenceSignal,
        targetSignal,
    )

    # Frame normalization is the only full-frame preparation operation.
    dpdLms.BeginFrame(referenceSignal)

    # This loop is the portable per-sample adaptive kernel.
    for referenceSample, targetSample in zip(
        referenceSignal,
        targetSignal,
    ):
        dpdLms.UpdateSample(
            complex(referenceSample),
            complex(targetSample),
        )

    # The shadow vector changed every sample; deployment changes only here.
    dpdLms.CommitCoefficients()

    afterNmseDb = dpdLms.CalculateNmse(
        referenceSignal,
        targetSignal,
    )
    featureSpecs = dpdLms.GetFeatureSpecs()
    coefficients = dpdLms.GetCoefficients()

    print(f"Before NMSE: {beforeNmseDb:.3f} dB")
    print(f"After NMSE:  {afterNmseDb:.3f} dB")
    for featureSpec, coefficient in zip(
        featureSpecs,
        coefficients,
    ):
        print(f"{featureSpec}: {coefficient}")


if __name__ == "__main__":
    RunSmallestLms()
