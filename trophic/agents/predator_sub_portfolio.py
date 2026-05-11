"""PredatorSubPortfolio: per-apex-predator capital slice + thesis book.

Composes with `ApexPortfolio.sub_portfolios` to support the DEBATE path
(phase3-A-05c). Each apex predator in a debate owns:
  - its own slice of starting capital,
  - its own cash + positions,
  - its own tax_owed_accrued (per-predator tax isolation is the key
    invariant — a momentum predator that realizes gains pays its own tax
    from its own slice; the other predators are unaffected),
  - its own rolling daily realized P&L history,
  - its own `ThesisBook` (typed-checked against predator_id by ThesisBook
    itself on add()).

The aggregate equity / cash / invested_pct / tax_owed_accrued on the
parent `ApexPortfolio` is summed across sub-portfolios when in
debate mode. Validators continue to operate on `PortfolioState` +
positions (phase2-D contract); `ApexPortfolio.state_for_predator()`
returns those scoped to a single predator's view so validators can run
per-predator unchanged.

Methods on this class are scoped to ONE sub-portfolio's books — they do
not touch other predators' state.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from trophic.agents.apex_portfolio import _Position
from trophic.beliefs.investment_thesis import ThesisBook

RECENT_PNL_WINDOW = 5  # days; mirrors apex_portfolio.RECENT_PNL_WINDOW


def _empty_thesis_book() -> ThesisBook:
    # ThesisBook.predator_id is required; placeholder is rewritten in
    # __post_init__ once the dataclass knows its own predator_id.
    return ThesisBook(predator_id="__pending__")


@dataclass
class PredatorSubPortfolio:
    """Capital + positions + thesis book owned by one apex predator.

    Construction example:

        sub = PredatorSubPortfolio(
            predator_id="momentum",
            philosophy="momentum",
            starting_cash=25_000.0,
        )
        # cash auto-fills to starting_cash; thesis_book.predator_id auto-fills.
    """
    predator_id: str
    philosophy: str
    starting_cash: float
    cash: float = 0.0
    positions: dict[str, _Position] = field(default_factory=dict)
    tax_owed_accrued: float = 0.0
    realized_pnl_history: list[tuple[str, float]] = field(default_factory=list)
    thesis_book: ThesisBook = field(default_factory=_empty_thesis_book)

    def __post_init__(self) -> None:
        # Auto-wire thesis_book.predator_id from the dataclass owner.
        if self.thesis_book.predator_id == "__pending__" or not self.thesis_book.predator_id:
            # ThesisBook is a Pydantic model with default config; assignment works.
            self.thesis_book.predator_id = self.predator_id
        # Default cash to starting_cash on first construction.
        if self.cash == 0.0 and self.starting_cash > 0:
            self.cash = self.starting_cash

    # ── Value snapshots ────────────────────────────────────────────────

    def equity(self, prices: dict[str, float]) -> float:
        """Cash + mark-to-market positions − tax_owed_accrued.

        Mirrors `ApexPortfolio.equity` semantics: tax_owed is already
        netted out of equity so the apex sees its post-tax position.
        """
        held = sum(
            p.shares * prices.get(t, 0.0)
            for t, p in self.positions.items()
        )
        return self.cash + held - self.tax_owed_accrued

    def invested_pct(self, prices: dict[str, float]) -> float:
        eq = self.equity(prices)
        if eq <= 0:
            return 0.0
        held = sum(
            p.shares * prices.get(t, 0.0)
            for t, p in self.positions.items()
        )
        return 100.0 * held / eq

    def recent_realized_pnl_5d(self, today: str) -> float:
        """Sum of the most recent RECENT_PNL_WINDOW days of realized P&L.

        `today` is accepted for API parity with future calendar-aware
        callers; the current implementation simply takes the tail of the
        history (the runner appends one entry per trading day).
        """
        if not self.realized_pnl_history:
            return 0.0
        return sum(v for _, v in self.realized_pnl_history[-RECENT_PNL_WINDOW:])

    # ── Trade methods (scoped to this sub-portfolio only) ─────────────

    def buy(
        self,
        ticker: str,
        dollars: float,
        slip_bps: float,
        today: str,
        primary_horizon: str = "",
        thesis_id: str = "",
        *,
        price: float,
    ) -> None:
        """Acquire `dollars` worth of `ticker` at `price` (per share).

        Slippage is paid on the gross cost. Updates this sub-portfolio's
        cash and positions only. A re-buy on an already-open position
        keeps the original `primary_horizon`, `opened_on`, and
        `bought_at_date` (the v4 gameability fix from phase2-D).

        `price` is passed as a keyword arg so the signature stays
        symmetric with `ApexPortfolio.execute`, which receives prices
        from the runner's daily price map.
        """
        if dollars <= 0 or price <= 0:
            return
        bps = slip_bps / 1e4
        cost_with_slip = dollars * (1 + bps)
        if cost_with_slip > self.cash:
            cost_with_slip = self.cash
            dollars = cost_with_slip / (1 + bps) if (1 + bps) > 0 else 0.0
        if dollars <= 0:
            return
        shares = dollars / price
        self.cash -= cost_with_slip
        pos = self.positions.setdefault(
            ticker, _Position(opened_on=today, ticker=ticker),
        )
        if pos.shares == 0:
            pos.opened_on = today
            pos.bought_at_date = today
            pos.ticker = ticker
            if primary_horizon:
                pos.primary_horizon = primary_horizon
            if thesis_id:
                pos.thesis_id = thesis_id
        else:
            # Existing position: do not reset horizon / opened_on.
            if not pos.primary_horizon and primary_horizon:
                pos.primary_horizon = primary_horizon
            if not pos.thesis_id and thesis_id:
                pos.thesis_id = thesis_id
        pos.shares += shares
        pos.cost_basis += cost_with_slip

    def sell(
        self,
        ticker: str,
        fraction_of_position: float,
        slip_bps: float,
        tax_fraction: float,
        today: str,
        *,
        price: float,
    ) -> tuple[float, float]:
        """Sell `fraction_of_position` (0–1) of `ticker` at `price`.

        Returns `(proceeds_net, realized_pnl)`. Mutates THIS
        sub-portfolio only:
          - cash += proceeds_net
          - tax_owed_accrued += realized_pnl * tax_fraction if gain > 0
          - realized_pnl_history.append((today, realized_pnl))

        Other predators' books are untouched.
        """
        pos = self.positions.get(ticker)
        if not pos or pos.shares <= 0 or price <= 0:
            return 0.0, 0.0
        frac = max(0.0, min(1.0, fraction_of_position))
        if frac <= 0:
            return 0.0, 0.0
        bps = slip_bps / 1e4
        shares_to_sell = pos.shares * frac
        proceeds_gross = shares_to_sell * price
        proceeds_net = proceeds_gross * (1 - bps)
        cost_for_those = (shares_to_sell / pos.shares) * pos.cost_basis
        realized_pnl = proceeds_net - cost_for_those
        self.cash += proceeds_net
        if realized_pnl > 0 and tax_fraction > 0:
            self.tax_owed_accrued += realized_pnl * tax_fraction
        pos.shares -= shares_to_sell
        pos.cost_basis -= cost_for_those
        if pos.shares < 1e-9:
            del self.positions[ticker]
        self.realized_pnl_history.append((today, realized_pnl))
        return proceeds_net, realized_pnl

    def rotate(
        self,
        from_ticker: str,
        to_ticker: str,
        fraction_of_from: float,
        slip_bps: float,
        tax_fraction: float,
        today: str,
        primary_horizon: str = "",
        thesis_id: str = "",
        *,
        from_price: float,
        to_price: float,
    ) -> tuple[float, float]:
        """SELL `fraction_of_from` of `from_ticker`, BUY `to_ticker` with proceeds.

        Returns `(proceeds_net, realized_pnl)` from the SELL leg. The
        BUY uses `proceeds_net` as its dollar intent. Atomic for THIS
        sub-portfolio: both legs operate on the same cash bucket here.
        """
        proceeds_net, realized_pnl = self.sell(
            from_ticker, fraction_of_from, slip_bps, tax_fraction, today,
            price=from_price,
        )
        if proceeds_net > 0:
            self.buy(
                to_ticker, proceeds_net, slip_bps, today,
                primary_horizon=primary_horizon, thesis_id=thesis_id,
                price=to_price,
            )
        return proceeds_net, realized_pnl
