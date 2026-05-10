"""Perfect-foresight oracle (math only, no LLM) — absolute upper bound.

For each trading day, given `actual_return` per ticker, the oracle:
  1. Ranks the universe by tomorrow_return (= today's actual_return,
     since pass artifacts use the close-to-close convention).
  2. SELLs every held position whose tomorrow_return < cash_daily_yield.
  3. BUYs the top-K positive-return tickers, equal-weighted up to the
     per-name cap (20%), filling cash until 100% invested or until the
     next candidate's return ≤ cash yield.
  4. Settles trades at today's close (same convention as portfolio_sim).

Two flavors:
  --frictionless : zero slippage, zero tax — pure math ceiling.
  (default)      : 5 bps slippage + 37% tax — what an oracle bound by
                   real frictions would actually pocket.

Output: same schema as scripts/portfolio_sim.py for drop-in comparison.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from trophic.agents.apex_portfolio import (
    ApexPortfolio,
    PER_NAME_CAP_PCT,
    MAX_INVESTED_PCT,
    DEFAULT_SLIPPAGE_BPS,
    DEFAULT_TAX_RATE,
    DEFAULT_CASH_YIELD_ANNUAL,
    TRADING_DAYS_PER_YEAR,
)

PASS_DIR = ROOT / "data" / "firehose_eval" / "passes"


def oracle_orders(
    portfolio: ApexPortfolio,
    tomorrow_returns: list[dict],
    prices: dict[str, float],
) -> list[dict]:
    """Greedy: SELL losing held names, BUY top-positive names up to cap."""
    daily_yield = (1.0 + portfolio.cash_yield_annual) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0

    by_ticker: dict[str, float] = {
        r["ticker"]: float(r.get("actual_return") or 0.0)
        for r in tomorrow_returns
    }

    orders: list[dict] = []
    eq = portfolio.equity(prices)

    # 1. SELL anything held with tomorrow_return < daily_yield (don't even
    # tie up capital in something that would underperform cash).
    for t, pos in list(portfolio.positions.items()):
        ret = by_ticker.get(t, 0.0)
        if ret < daily_yield:
            orders.append({
                "ticker": t,
                "side": "SELL",
                "size_pct": 100.0,
                "reasoning": f"oracle: tomorrow_return={ret:.4f} < daily_yield={daily_yield:.4f}",
            })

    # 2. Rank candidates by tomorrow_return desc, BUY up to per-name cap
    # until we hit 100% invested or candidates fall below the yield.
    ranked = sorted(by_ticker.items(), key=lambda kv: -kv[1])

    # Track planned exposure (after sells settle and before buys execute).
    held_pct: dict[str, float] = {}
    for t, p in portfolio.positions.items():
        if any(o["ticker"] == t and o["side"] == "SELL" for o in orders):
            continue  # we just told it to sell entirely
        mv = p.shares * prices.get(t, 0.0)
        held_pct[t] = 100.0 * mv / eq if eq > 0 else 0.0

    planned_invested = sum(held_pct.values())
    budget = max(0.0, MAX_INVESTED_PCT - planned_invested)

    for ticker, ret in ranked:
        if ret <= daily_yield:
            break  # nothing above yield is left
        if budget <= 0.5:
            break
        current = held_pct.get(ticker, 0.0)
        room = max(0.0, PER_NAME_CAP_PCT - current)
        if room <= 0.5:
            continue
        size = min(room, budget)
        if size <= 0.5:
            continue
        orders.append({
            "ticker": ticker,
            "side": "BUY",
            "size_pct": round(size, 2),
            "reasoning": f"oracle: tomorrow_return={ret:.4f}, sized to {size:.0f}% of equity",
        })
        budget -= size
        held_pct[ticker] = current + size

    return orders


def simulate_oracle(
    pass_paths: list[Path],
    starting_cash: float,
    slippage_bps: float,
    tax_rate: float,
    cash_yield: float,
    universe: set[str],
) -> dict:
    pf = ApexPortfolio(
        label="ORACLE_MATH",
        starting_cash=starting_cash,
        slippage_bps=slippage_bps,
        tax_rate=tax_rate,
        cash_yield_annual=cash_yield,
    )
    daily_states: list[dict] = []
    equity_curve: list[tuple[str, float]] = []

    for path in pass_paths:
        rec = json.load(gzip.open(path, "rt"))
        date = rec["date"]
        tomorrow = rec.get("ground_truth", []) or []
        prices = pf.advance_prices(date, tomorrow)
        orders = oracle_orders(pf, tomorrow, prices)
        validated, rejected = pf.validate_orders(orders, universe, prices)
        pf.execute(date, validated, prices)
        snap = pf.snapshot(date, prices)
        daily_states.append(snap)
        eq = pf.equity(prices)
        equity_curve.append((date, eq))

    final_eq = equity_curve[-1][1]
    return {
        "label": "ORACLE_MATH",
        "starting": starting_cash,
        "final": final_eq,
        "total_return": (final_eq - starting_cash) / starting_cash,
        "equity_curve": equity_curve,
        "daily_states": daily_states,
        "n_buys": pf.n_buys,
        "n_sells": pf.n_sells,
        "realized_gains": pf.realized_gains_cum,
        "tax_owed": pf.tax_owed,
        "cash_yield_cum": pf.cash_yield_cum,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--starting-cash", type=float, default=100_000.0)
    ap.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    ap.add_argument("--tax-rate", type=float, default=DEFAULT_TAX_RATE)
    ap.add_argument("--cash-yield", type=float, default=DEFAULT_CASH_YIELD_ANNUAL)
    ap.add_argument("--frictionless", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if args.frictionless:
        slip, tax = 0.0, 0.0
        print("Mode: ORACLE FRICTIONLESS (zero slippage, zero tax — absolute ceiling)")
    else:
        slip, tax = args.slippage_bps, args.tax_rate
        print(f"Mode: ORACLE REALISTIC (slippage={slip}bps, tax={tax:.0%}, "
              f"cash_yield={args.cash_yield:.1%}/yr)")

    pass_paths = sorted(PASS_DIR.glob("*.json.gz"))
    print(f"Loaded {len(pass_paths)} pass artifacts: "
          f"{pass_paths[0].stem.removesuffix('.json')} → "
          f"{pass_paths[-1].stem.removesuffix('.json')}")

    universe = set()
    for p in pass_paths[:1]:
        rec = json.load(gzip.open(p, "rt"))
        for g in rec.get("ground_truth", []):
            universe.add(g["ticker"])
    print(f"Universe: {len(universe)} tickers")

    result = simulate_oracle(
        pass_paths, args.starting_cash, slip, tax, args.cash_yield, universe,
    )

    print()
    print("=" * 70)
    print(f"{'STRATEGY':<18} {'FINAL EQUITY':>14} {'P&L':>12} {'RETURN':>10} "
          f"{'#BUY':>6} {'#SELL':>6}")
    print("-" * 70)
    pnl = result["final"] - result["starting"]
    print(f"{result['label']:<18} ${result['final']:>13,.0f} "
          f"${pnl:>+11,.0f} {result['total_return']:>+9.2%} "
          f"{result['n_buys']:>6} {result['n_sells']:>6}")
    print("=" * 70)
    print(f"realized_gains: ${result['realized_gains']:+,.0f}  "
          f"tax_owed: ${result['tax_owed']:,.0f}  "
          f"cash_yield_cum: ${result['cash_yield_cum']:,.0f}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=1))
        print(f"\nWritten to {args.json_out}")


if __name__ == "__main__":
    main()
