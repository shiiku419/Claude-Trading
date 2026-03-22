"""Composite signal aggregator.

Combines multiple :class:`~signals.base.SignalGenerator` instances using
configurable per-generator weights to produce a single high-conviction signal
or no signal at all.

Pure function implementation — no I/O, no global state mutations.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from signals.base import Signal, SignalDirection, SignalGenerator


class CompositeSignal:
    """Aggregates multiple signals with configurable weights.

    Firing conditions (ALL must be true for a signal to be emitted):

    1. At least one individual signal is non-NEUTRAL.
    2. A single direction commands the majority of non-NEUTRAL signals
       (strict majority: more than any other direction).
    3. The weighted strength of signals agreeing with the majority direction
       meets or exceeds ``threshold``.

    Weighted strength is calculated as:

        weighted_strength = sum(weight_i * strength_i for agreeing signals)
                           / sum(weight_i for agreeing signals)

    That is, it is a weighted *average* of strength among the agreeing
    signals — not a sum — so that adding more generators does not inflate
    the score.

    Attributes:
        generators: List of ``(generator, weight)`` pairs.  Weights need
            not sum to 1; they are used only for relative weighting.
        threshold: Minimum weighted average strength required to fire
            (default 0.6).
    """

    _NAME = "composite"

    def __init__(
        self,
        generators: list[tuple[SignalGenerator, float]],
        threshold: float = 0.6,
    ) -> None:
        """Initialise the CompositeSignal aggregator.

        Args:
            generators: Non-empty list of ``(generator, weight)`` pairs.
                Weights must be positive.
            threshold: Weighted-average strength threshold in ``(0, 1]``
                (default 0.6).

        Raises:
            ValueError: If ``generators`` is empty, any weight is not
                positive, or ``threshold`` is outside ``(0, 1]``.
        """
        if not generators:
            raise ValueError("generators must not be empty.")
        for gen, w in generators:
            if w <= 0.0:
                raise ValueError(
                    f"All weights must be positive; got {w} for {gen!r}."
                )
        if not (0.0 < threshold <= 1.0):
            raise ValueError(f"threshold must be in (0, 1]; got {threshold}.")
        self.generators = generators
        self.threshold = threshold

    def evaluate(self, pair: str, candles: np.ndarray) -> Signal | None:
        """Evaluate all generators and return a composite signal if it fires.

        Args:
            pair: Trading pair symbol, e.g. ``"BTC/USDT"``.
            candles: Numpy structured array with dtype ``CANDLE_DTYPE``,
                ordered oldest-first.

        Returns:
            A composite :class:`~signals.base.Signal` if all firing conditions
            are met; ``None`` otherwise.
        """
        # Collect individual signals.
        individual_signals: list[tuple[Signal, float]] = []
        for gen, weight in self.generators:
            sig = gen.generate(pair, candles)
            individual_signals.append((sig, weight))

        # Gather non-NEUTRAL signals.
        non_neutral = [
            (sig, w) for sig, w in individual_signals if sig.direction != SignalDirection.NEUTRAL
        ]
        if not non_neutral:
            return None

        # Count raw votes per direction.
        vote_counts: defaultdict[SignalDirection, int] = defaultdict(int)
        for sig, _ in non_neutral:
            vote_counts[sig.direction] += 1

        majority_direction = max(vote_counts, key=lambda d: vote_counts[d])

        # Ensure strict majority (more votes than any other direction).
        majority_votes = vote_counts[majority_direction]
        other_max = max(
            (count for d, count in vote_counts.items() if d != majority_direction),
            default=0,
        )
        if majority_votes <= other_max:
            # Tie — no majority direction.
            return None

        # Compute weighted average strength for majority-agreeing signals.
        agreeing = [
            (sig, w) for sig, w in non_neutral if sig.direction == majority_direction
        ]
        total_weight = sum(w for _, w in agreeing)
        weighted_strength = sum(sig.strength * w for sig, w in agreeing) / total_weight

        if weighted_strength < self.threshold:
            return None

        # Build metadata from all individual signals.
        timestamp = int(candles[-1]["timestamp"]) if len(candles) > 0 else 0
        metadata: dict[str, float] = {}
        for sig, w in individual_signals:
            prefix = sig.indicator_name
            metadata[f"{prefix}.direction"] = {
                SignalDirection.LONG: 1.0,
                SignalDirection.SHORT: -1.0,
                SignalDirection.NEUTRAL: 0.0,
            }[sig.direction]
            metadata[f"{prefix}.strength"] = sig.strength
            metadata[f"{prefix}.weight"] = w

        metadata["agreeing_count"] = float(len(agreeing))
        metadata["total_non_neutral"] = float(len(non_neutral))
        metadata["weighted_strength"] = weighted_strength

        return Signal(
            pair=pair,
            direction=majority_direction,
            strength=float(np.clip(weighted_strength, 0.0, 1.0)),
            indicator_name=self._NAME,
            timestamp=timestamp,
            metadata=metadata,
        )
