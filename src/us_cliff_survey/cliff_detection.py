"""Cliff detection on (earnings, net_income) curves.

A cliff at point i exists when net_income[i] - net_income[i+1] > earnings[i+1] - earnings[i].
That is: a small earnings increment costs more than its size in net income.

We separately track:
- The size of the maximum cliff in absolute dollars (max drop in net income)
- The location (earnings level) where it occurs
- The implied marginal "tax rate" at the cliff (drop / earnings step)
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Cliff:
    """A single detected cliff."""

    earnings_at_cliff: float
    earnings_step: float
    net_income_drop: float
    implied_marginal_rate: float

    @property
    def is_cliff(self) -> bool:
        """True if the drop exceeds the earnings step (>100% marginal rate)."""
        return self.net_income_drop > self.earnings_step


def find_cliffs(
    earnings: np.ndarray,
    net_income: np.ndarray,
    min_drop: float = 0.0,
) -> list[Cliff]:
    """Find all points where net_income decreases as earnings increase.

    Parameters
    ----------
    earnings
        Sorted ascending earnings levels.
    net_income
        Net income at each earnings level (same length).
    min_drop
        Minimum dollar drop to register as a cliff. Useful to filter noise.

    Returns
    -------
    List of Cliff records, sorted by net_income_drop descending.
    """
    earnings = np.asarray(earnings, dtype=float)
    net_income = np.asarray(net_income, dtype=float)
    if earnings.shape != net_income.shape:
        raise ValueError("earnings and net_income must have the same shape")
    if earnings.ndim != 1:
        raise ValueError("earnings must be 1-D")
    if not np.all(np.diff(earnings) >= 0):
        raise ValueError("earnings must be sorted ascending")

    earnings_step = np.diff(earnings)
    net_income_step = np.diff(net_income)

    drop_mask = -net_income_step > min_drop
    cliffs = []
    for i in np.where(drop_mask)[0]:
        step = float(earnings_step[i])
        drop = float(-net_income_step[i])
        cliffs.append(
            Cliff(
                earnings_at_cliff=float(earnings[i + 1]),
                earnings_step=step,
                net_income_drop=drop,
                implied_marginal_rate=drop / step if step > 0 else np.inf,
            )
        )
    cliffs.sort(key=lambda c: c.net_income_drop, reverse=True)
    return cliffs


def max_cliff(
    earnings: np.ndarray,
    net_income: np.ndarray,
) -> Cliff | None:
    """Return the largest cliff, or None if net_income is monotonic non-decreasing."""
    cliffs = find_cliffs(earnings, net_income, min_drop=0.0)
    return cliffs[0] if cliffs else None


def refine_cliff(
    sweep_fn,
    lo: float,
    hi: float,
    tol: float = 1.0,
    max_iter: int = 30,
) -> Cliff:
    """Bisect to localize a cliff between earnings lo and hi.

    sweep_fn(earnings) -> net_income at that earnings level.
    Returns the cliff with earnings_step ~ tol.
    """
    if lo >= hi:
        raise ValueError("lo must be < hi")
    ni_lo = sweep_fn(lo)
    ni_hi = sweep_fn(hi)
    if ni_lo <= ni_hi:
        raise ValueError(f"no cliff between {lo} and {hi}: ni went up")

    for _ in range(max_iter):
        if hi - lo <= tol:
            break
        mid = (lo + hi) / 2.0
        ni_mid = sweep_fn(mid)
        # The drop is between [lo, mid] if ni_mid < ni_lo, else [mid, hi].
        # We pick whichever side still contains the drop.
        if ni_mid < ni_lo:
            hi = mid
            ni_hi = ni_mid
        else:
            lo = mid
            ni_lo = ni_mid

    step = hi - lo
    drop = ni_lo - ni_hi
    return Cliff(
        earnings_at_cliff=hi,
        earnings_step=step,
        net_income_drop=drop,
        implied_marginal_rate=drop / step if step > 0 else np.inf,
    )
