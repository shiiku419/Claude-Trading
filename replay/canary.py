"""Paper vs replay signal comparison (canary check).

:class:`CanaryComparison` compares the signal lists produced by the paper
executor against those produced by the replayer.  Significant divergences
indicate one of:

* Look-ahead bias in the paper executor (e.g. a signal using future candle
  data).
* Data feed gaps that caused the paper executor to miss market moves.
* Time-dependent bugs in signal logic that produce different outputs when
  evaluated at different wall-clock times.

The comparison is intentionally simple for the MVP: it checks signal counts,
directional agreement, and per-pair strength deltas.  A ``tolerance`` parameter
controls how large a normalised deviation must be before it is flagged.
"""

from __future__ import annotations

import structlog

from signals.base import Signal

log: structlog.BoundLogger = structlog.get_logger(__name__)


class CanaryComparison:
    """Compares paper trading results with replay results.

    Flags significant divergences that might indicate:

    * Look-ahead bias in the paper executor.
    * Data feed gaps that affected paper trading.
    * Time-dependent bugs.

    The comparison works on two flat lists of :class:`~signals.base.Signal`
    objects.  Timestamps are used to match signals between the two runs when
    both lists have the same length.

    Example::

        canary = CanaryComparison()
        result = canary.compare(paper_signals, replay_signals)
        if result["has_divergence"]:
            log.warning("canary.divergence_detected", **result)
    """

    def __init__(self) -> None:
        pass

    def compare(
        self,
        paper_signals: list[Signal],
        replay_signals: list[Signal],
        tolerance: float = 0.01,
    ) -> dict[str, object]:
        """Compare signal lists and return divergence metrics.

        The method computes the following metrics and flags a divergence when
        any of them exceed ``tolerance``:

        * ``count_delta``: absolute difference in number of signals.
        * ``count_delta_pct``: ``count_delta / max(len(paper), len(replay))``.
        * ``direction_mismatch_rate``: fraction of positionally-matched signals
          that disagree on :class:`~signals.base.SignalDirection`.
        * ``avg_strength_delta``: mean absolute difference in ``strength``
          between positionally-matched signals.

        A ``has_divergence`` flag is set to ``True`` when any normalised metric
        exceeds *tolerance*.

        Args:
            paper_signals: Signals produced by the paper executor during live
                trading.
            replay_signals: Signals produced by the replayer from the same
                historical candles.
            tolerance: Threshold above which a metric is considered a
                divergence (default ``0.01``, i.e. 1 %).

        Returns:
            Dict with keys:

            * ``"paper_signal_count"`` – ``int``
            * ``"replay_signal_count"`` – ``int``
            * ``"count_delta"`` – ``int``
            * ``"count_delta_pct"`` – ``float``
            * ``"matched_pairs"`` – ``int`` (min of both lengths)
            * ``"direction_mismatches"`` – ``int``
            * ``"direction_mismatch_rate"`` – ``float``
            * ``"avg_strength_delta"`` – ``float``
            * ``"max_strength_delta"`` – ``float``
            * ``"has_divergence"`` – ``bool``
            * ``"divergence_reasons"`` – ``list[str]``
        """
        paper_count = len(paper_signals)
        replay_count = len(replay_signals)
        count_delta = abs(paper_count - replay_count)
        max_count = max(paper_count, replay_count, 1)
        count_delta_pct = count_delta / max_count

        matched_pairs = min(paper_count, replay_count)

        direction_mismatches = 0
        strength_deltas: list[float] = []

        for paper_sig, replay_sig in zip(paper_signals, replay_signals):
            if paper_sig.direction != replay_sig.direction:
                direction_mismatches += 1
            strength_deltas.append(abs(paper_sig.strength - replay_sig.strength))

        direction_mismatch_rate = (
            direction_mismatches / matched_pairs if matched_pairs > 0 else 0.0
        )
        avg_strength_delta = (
            sum(strength_deltas) / len(strength_deltas) if strength_deltas else 0.0
        )
        max_strength_delta = max(strength_deltas) if strength_deltas else 0.0

        divergence_reasons: list[str] = []

        if count_delta_pct > tolerance:
            divergence_reasons.append(
                f"signal count divergence: paper={paper_count} replay={replay_count} "
                f"({count_delta_pct:.1%} > {tolerance:.1%} tolerance)"
            )

        if direction_mismatch_rate > tolerance:
            divergence_reasons.append(
                f"direction mismatch rate {direction_mismatch_rate:.1%} "
                f"> {tolerance:.1%} tolerance "
                f"({direction_mismatches}/{matched_pairs} matched pairs)"
            )

        if avg_strength_delta > tolerance:
            divergence_reasons.append(
                f"avg strength delta {avg_strength_delta:.4f} "
                f"> {tolerance:.4f} tolerance"
            )

        has_divergence = bool(divergence_reasons)

        result: dict[str, object] = {
            "paper_signal_count": paper_count,
            "replay_signal_count": replay_count,
            "count_delta": count_delta,
            "count_delta_pct": count_delta_pct,
            "matched_pairs": matched_pairs,
            "direction_mismatches": direction_mismatches,
            "direction_mismatch_rate": direction_mismatch_rate,
            "avg_strength_delta": avg_strength_delta,
            "max_strength_delta": max_strength_delta,
            "has_divergence": has_divergence,
            "divergence_reasons": divergence_reasons,
        }

        if has_divergence:
            log.warning(
                "canary.divergence_detected",
                count_delta_pct=count_delta_pct,
                direction_mismatch_rate=direction_mismatch_rate,
                avg_strength_delta=avg_strength_delta,
                reasons=divergence_reasons,
            )
        else:
            log.info(
                "canary.no_divergence",
                paper_signal_count=paper_count,
                replay_signal_count=replay_count,
                matched_pairs=matched_pairs,
            )

        return result
