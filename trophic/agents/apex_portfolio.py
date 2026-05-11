"""Per-pipeline portfolio state + guardrail validator for the apex-PM mode.

Used by run_firehose_loop.py when --apex-pm is set. The apex emits
structured orders (ticker, side, size_pct, reasoning); this module:
  1. Builds the portfolio_state + open_positions snapshot the apex sees
     before deciding.
  2. Validates + clamps the apex's orders against per-name caps, total
     exposure, and available cash (no margin).
  3. Executes the validated orders against an indexed price series
     (each ticker normalized to $100 on the first day, compounded by
     `actual_return` thereafter — same convention as scripts/portfolio_sim.py).

Each pipeline (ecology, bare) gets its OWN ApexPortfolio instance — they
are blind to each other.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:
    from trophic.agents.predator_sub_portfolio import PredatorSubPortfolio

# Same defaults as scripts/portfolio_sim.py so the live runner and the
# retroactive sim agree on guardrails.
PER_NAME_CAP_PCT = 20.0
MAX_INVESTED_PCT = 100.0
DEFAULT_SLIPPAGE_BPS = 5.0
DEFAULT_TAX_RATE = 0.37
DEFAULT_CASH_YIELD_ANNUAL = 0.04   # 4% annualized; ~3-month T-bill equivalent
TRADING_DAYS_PER_YEAR = 252
RECENT_PNL_WINDOW = 5  # days
MIN_TRIM_PCT = 5.0  # Reject SELLs/trims smaller than this — they're noise


@dataclass
class _Position:
    shares: float = 0.0
    cost_basis: float = 0.0
    opened_on: str = ""
    # v4 tax-aware: the committed primary_horizon at buy time. The
    # MinHoldByHorizon validator reads THIS (not the order's horizon) on
    # SELL, so re-buys cannot reset the clock. Default "" means "no
    # commitment recorded" — the tax-aware chain treats it as
    # pass-through (legacy compatibility).
    primary_horizon: str = ""
    # Mirror of opened_on under the v4 contract name; populated alongside.
    bought_at_date: str = ""
    # Ticker is mirrored from the dict key so duck-typed validators can
    # match by .ticker attribute without needing the parent dict.
    ticker: str = ""
    # v4 debate: optional reference back to the InvestmentThesis that
    # motivated this position. Empty string = legacy / non-debate path.
    # Setting this on first BUY follows the same gameability rule as
    # primary_horizon — a re-buy never resets it.
    thesis_id: str = ""

    def days_held(self, today: str, calendar: list[str]) -> int:
        if not self.opened_on or self.opened_on not in calendar:
            return 0
        try:
            return calendar.index(today) - calendar.index(self.opened_on)
        except ValueError:
            return 0


@dataclass
class ApexPortfolio:
    label: str
    starting_cash: float = 100_000.0
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS
    tax_rate: float = DEFAULT_TAX_RATE
    cash_yield_annual: float = DEFAULT_CASH_YIELD_ANNUAL
    cash: float = 0.0
    positions: dict[str, _Position] = field(default_factory=dict)
    realized_gains_cum: float = 0.0
    tax_owed: float = 0.0
    cash_yield_cum: float = 0.0
    cash_yield_today: float = 0.0
    n_buys: int = 0
    n_sells: int = 0
    notional_traded: float = 0.0
    # rolling per-day realized P&L, used to expose recent_realized_pnl_5d
    daily_realized: list[tuple[str, float]] = field(default_factory=list)
    # internal price index: ticker -> last price (starts at 100)
    _last_price: dict[str, float] = field(default_factory=dict)
    # ordered list of dates we've seen, for days_held math
    _dates_seen: list[str] = field(default_factory=list)
    # v4 DEBATE: optional per-predator capital sub-portfolios. None for
    # v3.3/v4 non-debate paths (legacy behaviour intact). When populated,
    # aggregate equity / cash / invested_pct / tax_owed_accrued sum
    # across sub-portfolios; validators run per-predator via
    # `state_for_predator()`. Type is `dict[str, PredatorSubPortfolio]`
    # but we keep the annotation loose to avoid a circular import at
    # module load time.
    sub_portfolios: Optional[dict[str, "PredatorSubPortfolio"]] = None

    def __post_init__(self) -> None:
        if self.cash == 0.0:
            self.cash = self.starting_cash

    # ── Debate mode (v4) ──────────────────────────────────────────────

    def is_debate_mode(self) -> bool:
        """True iff `sub_portfolios` is populated with at least one predator."""
        return self.sub_portfolios is not None and len(self.sub_portfolios) > 0

    # ── Price index (mirrors scripts/portfolio_sim.py) ────────────────

    def advance_prices(self, date: str, ground_truth: list[dict]) -> dict[str, float]:
        """Compound each ticker by actual_return AND accrue daily cash yield
        on the cash bucket. Returns prices[ticker] for `date`. Cash yield is
        accrued *before* the apex sees state so the apex sees today's
        already-credited cash.
        """
        new_day = date not in self._dates_seen
        if new_day:
            self._dates_seen.append(date)
        for g in ground_truth:
            t = g["ticker"]
            ret = float(g.get("actual_return") or 0.0)
            if t not in self._last_price:
                self._last_price[t] = 100.0
            else:
                self._last_price[t] *= (1.0 + ret)

        # Daily cash yield (compounded). Only on first sighting of this date
        # (defensive — advance_prices may be called once per day per portfolio).
        if new_day and self.cash_yield_annual > 0 and self.cash > 0:
            daily_rate = (1.0 + self.cash_yield_annual) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
            yield_today = self.cash * daily_rate
            self.cash += yield_today
            self.cash_yield_cum += yield_today
            self.cash_yield_today = yield_today
        else:
            self.cash_yield_today = 0.0

        return dict(self._last_price)

    # ── State the apex sees ───────────────────────────────────────────

    def equity(self, prices: dict[str, float]) -> float:
        if self.is_debate_mode():
            return sum(s.equity(prices) for s in self.sub_portfolios.values())
        held = sum(
            p.shares * prices.get(t, self._last_price.get(t, 0.0))
            for t, p in self.positions.items()
        )
        return self.cash + held - self.tax_owed

    def invested_pct(self, prices: dict[str, float]) -> float:
        eq = self.equity(prices)
        if eq <= 0:
            return 0.0
        if self.is_debate_mode():
            held = 0.0
            for sub in self.sub_portfolios.values():
                held += sum(
                    p.shares * prices.get(t, 0.0)
                    for t, p in sub.positions.items()
                )
            return 100.0 * held / eq
        held = sum(
            p.shares * prices.get(t, self._last_price.get(t, 0.0))
            for t, p in self.positions.items()
        )
        return 100.0 * held / eq

    # ── Debate-mode aggregate accessors ────────────────────────────────

    @property
    def total_cash(self) -> float:
        """Total cash across sub-portfolios in debate mode; legacy cash otherwise."""
        if self.is_debate_mode():
            return sum(s.cash for s in self.sub_portfolios.values())
        return self.cash

    @property
    def total_tax_owed_accrued(self) -> float:
        """Aggregate tax_owed across sub-portfolios in debate mode.

        Per-predator tax isolation invariant: each predator's tax is
        accrued on its own slice; the aggregate is simply the sum.
        """
        if self.is_debate_mode():
            return sum(s.tax_owed_accrued for s in self.sub_portfolios.values())
        return self.tax_owed

    def recent_realized_pnl(self) -> float:
        if not self.daily_realized:
            return 0.0
        return sum(v for _, v in self.daily_realized[-RECENT_PNL_WINDOW:])

    def state_for_predator(
        self,
        predator_id: str,
        today: str,
        prices: dict[str, float],
    ):
        """Return (PortfolioState, list[PositionSnapshot]) scoped to ONE predator's
        sub-portfolio. Used by the debate mechanism (phase3-A-05d) to give each
        predator its own view of its own book so validators can run per-predator
        without modification.

        Raises ValueError if not in debate mode or predator_id is unknown.
        """
        from trophic.beliefs.apex_signatures import PortfolioState, PositionSnapshot

        if not self.is_debate_mode():
            raise ValueError("state_for_predator requires debate mode")
        if predator_id not in self.sub_portfolios:
            raise ValueError(
                f"unknown predator_id={predator_id!r}; "
                f"known: {sorted(self.sub_portfolios.keys())}"
            )
        sub = self.sub_portfolios[predator_id]
        eq = sub.equity(prices)
        inv_pct = max(0.0, min(100.0, sub.invested_pct(prices)))
        portfolio_state = PortfolioState(
            equity=round(eq, 2),
            cash=round(sub.cash, 2),
            invested_pct=round(inv_pct, 2),
            tax_owed_accrued=round(sub.tax_owed_accrued, 2),
            recent_realized_pnl_5d=round(sub.recent_realized_pnl_5d(today), 2),
            cash_yield_annual_pct=round(100.0 * self.cash_yield_annual, 2),
            cash_yield_today=0.0,  # cash yield is accrued at the parent level
            cash_yield_ytd=0.0,
            profit_definition=(
                f"net_profit = realized_pnl - tax_owed - slippage + cash_yield. "
                f"Currently {round(sub.tax_owed_accrued, 2):.2f} owed in tax accrual "
                f"(predator={predator_id!r}, already netted from this slice's "
                f"equity). Tax rate {round(100.0 * self.tax_rate, 1):.1f}% on "
                f"short-term gains; slippage {self.slippage_bps:.0f}bps per leg."
            ),
        )
        open_positions = []
        for t, p in sub.positions.items():
            px = prices.get(t, self._last_price.get(t, 0.0))
            mv = p.shares * px
            unrealized = mv - p.cost_basis
            weight_raw = 100.0 * mv / eq if eq > 0 else 0.0
            open_positions.append(PositionSnapshot(
                ticker=t,
                weight_pct=round(max(0.0, min(100.0, weight_raw)), 2),
                unrealized_pnl_pct=round(
                    100.0 * unrealized / p.cost_basis if p.cost_basis > 0 else 0.0,
                    2,
                ),
                days_held=p.days_held(today, self._dates_seen),
            ))
        return portfolio_state, open_positions

    def state_for_apex(self, today: str, prices: dict[str, float]):
        """Return (PortfolioState, list[PositionSnapshot]) Pydantic instances
        that DSPy signatures accept directly without the 'Type mismatch'
        coercion warning.
        """
        from trophic.beliefs.apex_signatures import PortfolioState, PositionSnapshot
        eq = self.equity(prices)
        # Clamp invested_pct to [0, 100] — float rounding can push it
        # microscopically past 100 (e.g. 100.01) which would violate the
        # PortfolioState Pydantic constraint. The validator downstream
        # still enforces no-margin via the actual numbers.
        inv_pct = max(0.0, min(100.0, self.invested_pct(prices)))
        portfolio_state = PortfolioState(
            equity=round(eq, 2),
            cash=round(self.cash, 2),
            invested_pct=round(inv_pct, 2),
            tax_owed_accrued=round(self.tax_owed, 2),
            recent_realized_pnl_5d=round(self.recent_realized_pnl(), 2),
            cash_yield_annual_pct=round(100.0 * self.cash_yield_annual, 2),
            cash_yield_today=round(self.cash_yield_today, 2),
            cash_yield_ytd=round(self.cash_yield_cum, 2),
            profit_definition=(
                f"net_profit = realized_pnl - tax_owed - slippage + cash_yield. "
                f"Currently {round(self.tax_owed, 2):.2f} owed in tax accrual "
                f"(already netted from equity). Tax rate "
                f"{round(100.0 * self.tax_rate, 1):.1f}% on short-term gains; "
                f"slippage {self.slippage_bps:.0f}bps per leg; cash yield "
                f"{round(100.0 * self.cash_yield_annual, 2):.2f}%/yr."
            ),
        )
        open_positions = []
        for t, p in self.positions.items():
            px = prices.get(t, self._last_price.get(t, 0.0))
            mv = p.shares * px
            unrealized = mv - p.cost_basis
            weight_raw = 100.0 * mv / eq if eq > 0 else 0.0
            open_positions.append(PositionSnapshot(
                ticker=t,
                weight_pct=round(max(0.0, min(100.0, weight_raw)), 2),
                unrealized_pnl_pct=round(
                    100.0 * unrealized / p.cost_basis if p.cost_basis > 0 else 0.0,
                    2,
                ),
                days_held=p.days_held(today, self._dates_seen),
            ))
        return portfolio_state, open_positions

    # ── Validate + clamp apex orders ──────────────────────────────────

    def validate_orders(
        self,
        raw_orders: Iterable[dict],
        universe: set[str],
        prices: dict[str, float],
    ) -> tuple[list[dict], list[str]]:
        """Return (validated_orders, rejection_reasons).

        validated_orders have keys: ticker, side, dollars_intent, size_pct,
        reasoning. dollars_intent is the clamped notional. The runner then
        feeds these to execute().
        """
        eq = self.equity(prices)
        if eq <= 0:
            return [], ["equity ≤ 0"]

        rejected: list[str] = []
        sells: list[dict] = []
        buys: list[dict] = []

        # Pre-pass: expand ROTATE orders into a (SELL, BUY) pair. Marks
        # both halves with _rotate_id so we can fail them atomically.
        # ROTATE size_pct is expressed as PERCENT OF EQUITY (consistent
        # with BUY's convention). The SELL leg gets converted to its
        # equivalent "percent of position" so the existing validator math
        # still works.
        flat_orders: list[dict] = []
        for o in raw_orders:
            side = (o.get("side") or "").upper().strip()
            if side == "ROTATE":
                t_from = (o.get("from") or "").upper().strip()
                t_to = (o.get("to") or "").upper().strip()
                size_pct_eq = float(o.get("size_pct") or 0)  # % of equity
                reasoning = (o.get("reasoning") or "")[:200]
                if not t_from or not t_to or size_pct_eq <= 0:
                    rejected.append(
                        f"ROTATE bad shape (from={t_from!r}, to={t_to!r}, "
                        f"size_pct={size_pct_eq}): {o}"
                    )
                    continue
                # Convert SELL leg's "% of equity" → "% of position".
                pos = self.positions.get(t_from)
                if not pos or pos.shares <= 0:
                    rejected.append(
                        f"ROTATE {t_from}→{t_to}: source {t_from} not held; "
                        f"cannot rotate from an unheld position. Use a plain "
                        f"BUY for {t_to} instead."
                    )
                    continue
                from_px = prices.get(t_from)
                if not from_px:
                    rejected.append(f"{t_from}: no price available (ROTATE)")
                    continue
                from_position_value = pos.shares * from_px
                position_pct_of_eq = 100.0 * from_position_value / eq if eq > 0 else 0.0
                # Cap requested % of equity at what's actually held in t_from
                effective_pct_eq = min(size_pct_eq, position_pct_of_eq)
                # SELL fraction of POSITION = effective_pct_eq / position_pct_of_eq * 100
                sell_pct_of_pos = (
                    100.0 * effective_pct_eq / position_pct_of_eq
                    if position_pct_of_eq > 0 else 0.0
                )
                rotate_id = f"rot_{len(flat_orders)}"
                flat_orders.append({
                    "ticker": t_from, "side": "SELL",
                    "size_pct": sell_pct_of_pos,  # validator-expected: % of position
                    "reasoning": f"[ROTATE {t_from}→{t_to}] {reasoning}",
                    "_rotate_id": rotate_id,
                    "_rotate_eq_pct": effective_pct_eq,  # for budget earmarking
                })
                flat_orders.append({
                    "ticker": t_to, "side": "BUY",
                    "size_pct": effective_pct_eq,  # validator-expected: % of equity
                    "reasoning": f"[ROTATE {t_from}→{t_to}] {reasoning}",
                    "_rotate_id": rotate_id,
                })
            else:
                flat_orders.append(o)

        for o in flat_orders:
            try:
                t = (o.get("ticker") or "").upper().strip()
                side = (o.get("side") or "").upper().strip()
                size_pct = float(o.get("size_pct") or 0)
                reasoning = (o.get("reasoning") or "")[:200]
                rotate_id = o.get("_rotate_id")
                primary_horizon = o.get("primary_horizon") or ""
                expected_alpha_bps = float(o.get("expected_alpha_bps") or 0.0)
            except Exception as e:
                rejected.append(f"unparsable order ({e}): {o}")
                continue

            if t not in universe:
                rejected.append(f"{t}: not in universe")
                continue
            if side not in ("BUY", "SELL", "HOLD"):
                rejected.append(f"{t}: bad side={side!r}")
                continue
            if side == "HOLD":
                continue
            if size_pct <= 0:
                rejected.append(f"{t}: size_pct={size_pct} (non-positive)")
                continue
            if t not in prices:
                rejected.append(f"{t}: no price available")
                continue

            if side == "SELL":
                pos = self.positions.get(t)
                if not pos or pos.shares <= 0:
                    rejected.append(f"{t}: SELL on unheld name")
                    continue
                pct = min(100.0, size_pct)
                # Reject noise trims (only fully closing positions or ≥MIN_TRIM_PCT trims)
                if pct < MIN_TRIM_PCT and pct < 100.0:
                    rejected.append(
                        f"{t}: SELL size_pct={pct:.1f} below MIN_TRIM_PCT "
                        f"({MIN_TRIM_PCT:.0f}). Either trim ≥{MIN_TRIM_PCT:.0f}% "
                        f"or drop the order — small trims are tax/slippage drag."
                    )
                    continue
                shares_to_sell = pos.shares * pct / 100.0
                dollars = shares_to_sell * prices[t]
                sells.append({
                    "ticker": t,
                    "side": "SELL",
                    "size_pct": round(pct, 2),
                    "dollars_intent": round(dollars, 2),
                    "reasoning": reasoning,
                    "primary_horizon": primary_horizon,
                    "expected_alpha_bps": expected_alpha_bps,
                    "_rotate_id": rotate_id,
                })
            else:  # BUY
                pct = min(PER_NAME_CAP_PCT, size_pct)
                buys.append({
                    "ticker": t,
                    "side": "BUY",
                    "size_pct": round(pct, 2),
                    "dollars_intent": 0.0,  # set after sells settle
                    "reasoning": reasoning,
                    "primary_horizon": primary_horizon,
                    "expected_alpha_bps": expected_alpha_bps,
                    "_raw_pct": pct,
                    "_rotate_id": rotate_id,
                })

        # If any rotate-pair SELL was rejected, drop the matching BUY too —
        # the apex's intent was atomic, partial execution would leave the
        # portfolio over-exposed.
        rejected_rotate_ids = set()
        for r in rejected:
            # rotate sells already failed validation will be in `rejected`;
            # they don't have IDs in there, but unheld-SELLs in a rotate
            # are detectable: if their _rotate_id was set, we skipped them
            # before adding to sells. Detect by checking if _rotate_id
            # appears in sells.
            pass
        sell_rotate_ids = {s.get("_rotate_id") for s in sells if s.get("_rotate_id")}
        # Any BUY whose rotate_id is set but missing from sell_rotate_ids
        # means the matching SELL was rejected upstream — drop the BUY.
        kept_buys = []
        for b in buys:
            rid = b.get("_rotate_id")
            if rid and rid not in sell_rotate_ids:
                rejected.append(
                    f"{b['ticker']}: ROTATE BUY dropped — matching SELL leg "
                    f"was rejected; rotate is atomic."
                )
                continue
            kept_buys.append(b)
        buys = kept_buys

        # Apply sells first (frees up cash + capacity for buys)
        validated = list(sells)

        # After sells, recompute capacity. Use *current* invested_pct (sells
        # haven't actually executed yet, but we know how much cash they'll
        # free up). We approximate: total budget for new BUYs =
        # max(0, MAX_INVESTED_PCT - (current_invested - sell_pct)).
        sell_freed_pct = sum(
            (self.positions[s["ticker"]].shares * prices[s["ticker"]] / eq) * 100.0
            * s["size_pct"] / 100.0
            for s in sells if s["ticker"] in self.positions
        )
        # Earmark each ROTATE-paired BUY's matching SELL capacity so the
        # BUY won't be scaled down. The pre-pass set _rotate_eq_pct (% of
        # equity) on the SELL leg, so we can read it directly.
        rotate_freed_by_id: dict[str, float] = {}
        for s in sells:
            rid = s.get("_rotate_id")
            eq_pct = s.get("_rotate_eq_pct")
            if rid and eq_pct is not None:
                rotate_freed_by_id[rid] = rotate_freed_by_id.get(rid, 0.0) + eq_pct

        current_inv_pct = self.invested_pct(prices)
        post_sell_inv_pct = max(0.0, current_inv_pct - sell_freed_pct)
        buy_budget_pct = max(0.0, MAX_INVESTED_PCT - post_sell_inv_pct)

        # Pre-compute "weakest holdings" candidate list for directive
        # rejection messages: positions ordered by ascending unrealized P&L%
        # (worst first). Used in messages that suggest what to SELL.
        weakest = []
        for t, p in self.positions.items():
            px = prices.get(t, self._last_price.get(t, 0.0))
            mv = p.shares * px
            weight = 100.0 * mv / eq if eq > 0 else 0.0
            unreal_pct = (
                100.0 * (mv - p.cost_basis) / p.cost_basis
                if p.cost_basis > 0 else 0.0
            )
            if (
                # Don't suggest names already being sold this turn
                not any(s["ticker"] == t for s in sells)
                and weight > 0.5
            ):
                weakest.append((t, weight, unreal_pct))
        weakest.sort(key=lambda x: x[2])  # ascending unrealized %
        weakest_str = ", ".join(
            f"{t} (w={w:.0f}%, unreal={u:+.1f}%)"
            for t, w, u in weakest[:3]
        ) or "(no held positions to rotate)"

        # Greedy-allocate buys against budget. ROTATE-paired BUYs are
        # earmarked their matching SELL's freed capacity FIRST, then any
        # remainder competes for the general budget.
        rotate_buy_pcts = {}  # rotate_id -> reserved pct already taken
        non_rotate_total = 0.0
        for b in buys:
            rid = b.get("_rotate_id")
            if rid and rid in rotate_freed_by_id:
                # This BUY gets up to its matching SELL's freed capacity.
                reserved = min(b["_raw_pct"], rotate_freed_by_id[rid])
                rotate_buy_pcts[id(b)] = reserved
                # If the apex asked for more than the SELL freed, the
                # remainder is non-rotate and competes for general budget.
                non_rotate_total += max(0.0, b["_raw_pct"] - reserved)
            else:
                non_rotate_total += b["_raw_pct"]

        general_budget = max(0.0, buy_budget_pct - sum(rotate_buy_pcts.values()))
        scale = 1.0
        if non_rotate_total > general_budget and non_rotate_total > 0:
            scale = general_budget / non_rotate_total

        for b in buys:
            reserved = rotate_buy_pcts.get(id(b), 0.0)
            extra_raw = max(0.0, b["_raw_pct"] - reserved)
            b["_raw_pct"] = reserved + extra_raw * scale

        # Re-derive total_requested for the diagnostic message
        total_requested = non_rotate_total + sum(rotate_buy_pcts.values())
        if non_rotate_total > general_budget and non_rotate_total > 0:
            if buy_budget_pct < 1.0:
                # No real room — directive: tell apex what to sell
                rejected.append(
                    f"BUY total {total_requested:.1f}% blocked: portfolio is "
                    f"{post_sell_inv_pct:.0f}% invested with only {buy_budget_pct:.1f}% "
                    f"budget. To free capacity, add a SELL on one of: {weakest_str}. "
                    f"Stock-to-stock trades go through cash; you must SELL first."
                )
            else:
                rejected.append(
                    f"BUY total {total_requested:.1f}% > budget "
                    f"{buy_budget_pct:.1f}% — scaled to fit. To get full size, "
                    f"add a SELL on one of: {weakest_str}."
                )

        # Per-name cap relative to existing weight
        for b in buys:
            t = b["ticker"]
            pos = self.positions.get(t)
            current_weight = (pos.shares * prices[t] / eq * 100.0) if pos else 0.0
            room = max(0.0, PER_NAME_CAP_PCT - current_weight)
            allowed_pct = min(b["_raw_pct"], room)
            if allowed_pct <= 0 and current_weight > 0:
                rejected.append(
                    f"{t}: BUY blocked — already at per-name cap "
                    f"({current_weight:.0f}% of equity, max {PER_NAME_CAP_PCT:.0f}%). "
                    f"Either drop this BUY or SELL part of {t} first."
                )
                continue
            if allowed_pct <= 0:
                # Zero size after total-budget scaling, name not held —
                # already covered by the budget rejection above; skip.
                continue
            dollars = eq * allowed_pct / 100.0
            b["size_pct"] = round(allowed_pct, 2)
            b["dollars_intent"] = round(dollars, 2)
            b.pop("_raw_pct", None)
            validated.append(b)

        # Strip internal markers before returning
        for v in validated:
            v.pop("_rotate_id", None)
            v.pop("_rotate_eq_pct", None)
        return validated, rejected

    # ── v4 tax-aware filter (runs AFTER validate_orders) ──────────────

    def filter_tax_aware(
        self,
        validated_orders: list[dict],
        today: str,
        views: list = (),
        slice_fraction: float = 1.0,
        universe: "Optional[Iterable[str]]" = None,
    ) -> tuple[list[dict], list[str]]:
        """Second-layer filter: enforce v4 tax-aware discipline.

        Inputs are the *already-validated* dicts returned by
        `validate_orders` (so structural/cash/cap checks have run). This
        layer runs `run_tax_aware_chain` on each order and partitions
        accepted vs rejected. Rejected orders are not executed; the
        returned reasons feed into the apex retry loop.

        - `views`: iterable of TickerView pydantic instances (or
          duck-typed objects) — needed for forecast consistency, edge
          floor, and regime-invalidation overrides.
        - `slice_fraction`: forwarded to the horizon-sizing validator
          for debate-mode predator slices (default 1.0 = whole portfolio).
        - `universe`: optional iterable of in-universe tickers. When set,
          `validate_in_universe` runs FIRST in the chain so off-universe
          hallucinations (cross-predator extension leaks) fail-loud.
          When None, the universe check is skipped (backward-compat).
        """
        from trophic.beliefs.apex_signatures import Order
        from trophic.beliefs.validators import run_tax_aware_chain

        # Materialize positions with the .ticker attribute so the
        # duck-typed validator can match.
        positions_view = []
        for ticker_key, p in self.positions.items():
            if not p.ticker:
                p.ticker = ticker_key
            positions_view.append(p)

        accepted: list[dict] = []
        rejections: list[str] = []
        for od in validated_orders:
            # Skip orders that don't carry a primary_horizon — they came
            # from a non-PM path (legacy watchlist mode) and the
            # tax-aware chain isn't applicable.
            if not od.get("primary_horizon"):
                accepted.append(od)
                continue
            try:
                order_obj = Order(
                    side=od["side"],
                    ticker=od.get("ticker"),
                    size_pct=float(od.get("size_pct") or 0),
                    reasoning=od.get("reasoning") or "tax-aware-chain",
                    primary_horizon=od["primary_horizon"],
                    expected_alpha_bps=float(od.get("expected_alpha_bps") or 0.0),
                )
            except Exception as e:  # pragma: no cover - defensive
                rejections.append(
                    f"{od.get('ticker', '?')}: tax-aware-chain "
                    f"could not rebuild Order ({e})"
                )
                continue
            result = run_tax_aware_chain(
                order_obj,
                positions_view,
                today,
                self._dates_seen,
                views,
                slice_fraction=slice_fraction,
                universe=universe,
            )
            if result.accepted:
                accepted.append(od)
            else:
                rejections.append(result.reason)
        return accepted, rejections

    # ── Execute (mirrors scripts/portfolio_sim.execute) ───────────────

    def execute(
        self,
        date: str,
        validated_orders: list[dict],
        prices: dict[str, float],
    ) -> None:
        bps = self.slippage_bps / 1e4
        day_realized = 0.0
        for o in validated_orders:
            t = o["ticker"]
            side = o["side"]
            dollars = o["dollars_intent"]
            px = prices.get(t)
            if not px:
                continue
            if side == "BUY":
                cost_with_slip = dollars * (1 + bps)
                if cost_with_slip > self.cash:
                    cost_with_slip = self.cash
                    dollars = cost_with_slip / (1 + bps)
                shares = dollars / px
                self.cash -= cost_with_slip
                pos = self.positions.setdefault(
                    t, _Position(opened_on=date, ticker=t),
                )
                # v4: record the committed primary_horizon at buy time so
                # MinHoldByHorizon can enforce on SELL. A re-buy on an
                # already-open position does NOT reset the horizon — the
                # original commitment stands (otherwise the apex could
                # game min_hold by adding 1 share to refresh the clock).
                order_horizon = o.get("primary_horizon") or ""
                if pos.shares == 0:
                    pos.opened_on = date
                    pos.bought_at_date = date
                    pos.ticker = t
                    if order_horizon:
                        pos.primary_horizon = order_horizon
                else:
                    # Existing position: keep stored horizon unchanged.
                    # Only fill in if it was missing (e.g. pre-v4 legacy).
                    if not pos.primary_horizon and order_horizon:
                        pos.primary_horizon = order_horizon
                pos.shares += shares
                pos.cost_basis += cost_with_slip
                self.n_buys += 1
                self.notional_traded += dollars
            elif side == "SELL":
                pos = self.positions.get(t)
                if not pos or pos.shares <= 0:
                    continue
                shares_to_sell = min(pos.shares, dollars / px)
                proceeds_gross = shares_to_sell * px
                proceeds_net = proceeds_gross * (1 - bps)
                cost_for_those = (shares_to_sell / pos.shares) * pos.cost_basis
                gain = proceeds_net - cost_for_those
                self.realized_gains_cum += gain
                day_realized += gain
                if gain > 0 and self.tax_rate > 0:
                    self.tax_owed += gain * self.tax_rate
                self.cash += proceeds_net
                pos.shares -= shares_to_sell
                pos.cost_basis -= cost_for_those
                if pos.shares < 1e-9:
                    del self.positions[t]
                self.n_sells += 1
                self.notional_traded += proceeds_gross

        self.daily_realized.append((date, day_realized))

    # ── Snapshot for persistence ──────────────────────────────────────

    def snapshot(self, date: str, prices: dict[str, float]) -> dict:
        eq = self.equity(prices)
        positions = []
        for t, p in self.positions.items():
            px = prices.get(t, self._last_price.get(t, 0.0))
            mv = p.shares * px
            positions.append({
                "ticker": t,
                "shares": round(p.shares, 6),
                "cost_basis": round(p.cost_basis, 2),
                "mv": round(mv, 2),
                "unrealized_pnl": round(mv - p.cost_basis, 2),
                "weight_pct": round(100.0 * mv / eq, 2) if eq > 0 else 0.0,
                "days_held": p.days_held(date, self._dates_seen),
            })
        return {
            "label": self.label,
            "date": date,
            "equity": round(eq, 2),
            "cash": round(self.cash, 2),
            "invested_pct": round(self.invested_pct(prices), 2),
            "tax_owed_accrued": round(self.tax_owed, 2),
            "realized_gains_cum": round(self.realized_gains_cum, 2),
            "cash_yield_today": round(self.cash_yield_today, 2),
            "cash_yield_cum": round(self.cash_yield_cum, 2),
            "n_buys_total": self.n_buys,
            "n_sells_total": self.n_sells,
            "open_positions": positions,
        }


# ── Debate-mode factory ───────────────────────────────────────────────

DEFAULT_DEBATE_PREDATORS: tuple[tuple[str, str], ...] = (
    ("momentum",     "momentum"),
    ("value",        "value"),
    ("mean_revert",  "mean_revert"),
    ("event_driven", "event_driven"),
)


def make_debate_portfolio(
    total_starting_cash: float = 100_000.0,
    predators: tuple[tuple[str, str], ...] = DEFAULT_DEBATE_PREDATORS,
    label: str = "DEBATE",
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    tax_rate: float = DEFAULT_TAX_RATE,
    cash_yield_annual: float = DEFAULT_CASH_YIELD_ANNUAL,
) -> "ApexPortfolio":
    """Create an ApexPortfolio in DEBATE mode with N predators each starting
    with `total_starting_cash / N`.

    Each `predators` entry is `(predator_id, philosophy)`. Defaults to the
    4-predator setup (momentum / value / mean_revert / event_driven) used
    by phase 3's DEBATE path.

    The returned ApexPortfolio has:
      - `sub_portfolios` populated (one PredatorSubPortfolio per predator),
      - `is_debate_mode()` → True,
      - legacy `cash` / `tax_owed` left at 0 (the per-predator slices
        carry capital + tax accrual instead).
    """
    from trophic.agents.predator_sub_portfolio import PredatorSubPortfolio

    if not predators:
        raise ValueError("predators tuple must not be empty")
    n = len(predators)
    per_slice = total_starting_cash / n

    subs: dict[str, PredatorSubPortfolio] = {}
    for predator_id, philosophy in predators:
        if predator_id in subs:
            raise ValueError(f"duplicate predator_id in factory: {predator_id!r}")
        subs[predator_id] = PredatorSubPortfolio(
            predator_id=predator_id,
            philosophy=philosophy,
            starting_cash=per_slice,
        )

    return ApexPortfolio(
        label=label,
        starting_cash=0.0,  # capital lives in sub-portfolios
        cash=0.0,
        slippage_bps=slippage_bps,
        tax_rate=tax_rate,
        cash_yield_annual=cash_yield_annual,
        sub_portfolios=subs,
    )
