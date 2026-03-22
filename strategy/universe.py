"""Pair universe management.

For the MVP the allowed universe is a static list supplied at construction
time from :class:`~config.settings.TradingConfig`.  Future iterations will
let Claude dynamically suggest additional pairs that must still be whitelisted
here before the signal engine will evaluate them.
"""

from __future__ import annotations

import structlog

log: structlog.BoundLogger = structlog.get_logger(__name__)


class PairUniverse:
    """Manages the set of tradeable pairs.

    For MVP: static list from config.  Future: dynamic based on Claude
    suggestions, subject to whitelist approval.

    Args:
        allowed_pairs: Explicit list of CCXT-formatted pair symbols that
            the bot is permitted to trade, e.g.
            ``["BTC/USDT", "ETH/USDT"]``.
    """

    def __init__(self, allowed_pairs: list[str]) -> None:
        self._allowed: frozenset[str] = frozenset(allowed_pairs)
        log.debug(
            "pair_universe.initialised",
            count=len(self._allowed),
            pairs=sorted(self._allowed),
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_allowed(self, pair: str) -> bool:
        """Return ``True`` if *pair* is in the allowed universe.

        Args:
            pair: CCXT-formatted trading pair symbol, e.g. ``"BTC/USDT"``.

        Returns:
            ``True`` when *pair* is in the allowed set, ``False`` otherwise.
        """
        return pair in self._allowed

    def filter_pairs(self, pairs: list[str]) -> list[str]:
        """Return only those pairs that belong to the allowed universe.

        The original ordering of *pairs* is preserved.

        Args:
            pairs: Candidate pair symbols to filter.

        Returns:
            A new list containing only elements that pass
            :meth:`is_allowed`.
        """
        allowed = [p for p in pairs if self.is_allowed(p)]
        rejected = [p for p in pairs if not self.is_allowed(p)]
        if rejected:
            log.debug(
                "pair_universe.pairs_rejected",
                rejected=rejected,
            )
        return allowed

    @property
    def all_pairs(self) -> list[str]:
        """Sorted list of every pair in the universe.

        Returns:
            A new sorted list of all allowed pair symbols.
        """
        return sorted(self._allowed)

    def __len__(self) -> int:
        return len(self._allowed)

    def __contains__(self, pair: object) -> bool:
        return pair in self._allowed
