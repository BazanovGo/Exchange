"""The backtest engine.

A deliberately small, explicit, bar-by-bar simulator. It replaces the
audited project's vectorbt-based engine for one reason that outweighs every
convenience vectorbt offered: **the timing of execution has to be a
structural property, not a parameter**. Here, a signal computed on bar ``t``
is physically unable to trade on bar ``t`` — the loop reads the target for
bar ``t`` and submits it against bar ``t + delay``. There is no argument
combination that can turn that off except the explicitly flagged
``same_close`` assumption, which marks its own results invalid.

Contract with the strategy layer
--------------------------------
A strategy produces a **target position** series in units of contracts (or
shares), indexed by bar and *unshifted*: the value at bar ``t`` is what the
strategy wants to hold having seen bar ``t`` close. The engine applies the
delay. Strategies therefore never need to remember to ``.shift()``, which is
the mistake that shift-based conventions invite.

Accounting
----------
Mark-to-market on bar closes, in currency:

    ``equity_t = cash_t + position_t * close_t * multiplier``

Cash moves only on fills, by ``-quantity * fill_price * multiplier - fees``.
Realised P&L is tracked per closed trade using average-cost accounting;
position reversals (long to short in one order) are split into a closing
trade and an opening trade, because reporting them as one trade misstates
both the win rate and the holding period.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from exchange.core.errors import ConfigError, DataError
from exchange.core.hashing import hash_config
from exchange.core.logging import get_logger
from exchange.core.timeframe import empirical_bars_per_year
from exchange.data.schema import assert_canonical
from exchange.execution.assumptions import BASE, ExecutionAssumptions
from exchange.execution.orders import Bar, Order, OrderType, Side, match, takes_liquidity

logger = get_logger(__name__)


@dataclass(frozen=True)
class InstrumentSpec:
    """What the engine needs to know to price a position in money."""

    symbol: str = "ASSET"
    multiplier: float = 1.0     # currency per 1.0 of price per contract
    tick_size: float = 0.0
    allow_short: bool = True
    allow_fractional: bool = True   # False for futures: whole contracts only

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "multiplier": self.multiplier,
            "tick_size": self.tick_size,
            "allow_short": self.allow_short,
            "allow_fractional": self.allow_fractional,
        }


@dataclass(frozen=True)
class BacktestConfig:
    """Everything that is not the data, the signal or the cost model."""

    initial_cash: float = 100_000.0
    instrument: InstrumentSpec = field(default_factory=InstrumentSpec)
    execution: ExecutionAssumptions = BASE
    volatility_lookback: int = 20   # bars used for the cost models' volatility proxy
    annualisation: float | None = None  # None -> inferred from the bar index

    def to_dict(self) -> dict:
        return {
            "initial_cash": self.initial_cash,
            "instrument": self.instrument.to_dict(),
            "execution": self.execution.to_dict(),
            "volatility_lookback": self.volatility_lookback,
            "annualisation": self.annualisation,
        }

    @property
    def fingerprint(self) -> str:
        return hash_config(self.to_dict())


@dataclass
class Trade:
    """One round trip, from opening a position to closing it."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int              # +1 long, -1 short
    quantity: float
    entry_price: float
    exit_price: float
    gross_pnl: float
    fees: float
    net_pnl: float
    bars_held: int
    entry_tag: str = ""
    exit_tag: str = ""

    @property
    def return_pct(self) -> float:
        """Return on the position's own notional at entry."""
        notional = abs(self.quantity * self.entry_price)
        return self.net_pnl / notional if notional > 0 else np.nan


@dataclass
class BacktestResult:
    """Everything one simulation produced."""

    equity: pd.Series
    cash: pd.Series
    position: pd.Series
    marks: pd.Series             # mark price used each bar
    orders: pd.DataFrame
    trades: pd.DataFrame
    config: BacktestConfig
    symbol: str
    diagnostics: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().fillna(0.0).rename("return")

    @property
    def bars_per_year(self) -> float:
        if self.config.annualisation is not None:
            return float(self.config.annualisation)
        return empirical_bars_per_year(self.equity.index)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def total_costs(self) -> float:
        if len(self.orders) == 0:
            return 0.0
        return float(self.orders["fees"].sum() + self.orders["slippage_cost"].sum())

    @property
    def valid(self) -> bool:
        """False when the run knowingly used an unreachable execution price."""
        return not self.config.execution.admits_lookahead

    def metrics(self):
        from exchange.backtest.metrics import performance_metrics

        return performance_metrics(self)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        total = self.equity.iloc[-1] / self.equity.iloc[0] - 1 if len(self.equity) else np.nan
        return (
            f"BacktestResult({self.symbol}, {len(self.equity)} bars, {self.n_trades} trades, "
            f"total_return={total:.2%}, costs={self.total_costs:,.0f})"
        )


def run_backtest(
    bars: pd.DataFrame,
    target_position: pd.Series,
    *,
    config: BacktestConfig | None = None,
    symbol: str | None = None,
    entry_tags: pd.Series | None = None,
) -> BacktestResult:
    """Simulate holding ``target_position`` on ``bars``.

    Parameters
    ----------
    bars:
        Canonical OHLCV (see :mod:`exchange.data.schema`).
    target_position:
        Desired holding in contracts, indexed like ``bars``, **unshifted**.
        The value at bar ``t`` is what the strategy wants after seeing bar
        ``t``. The engine executes it at bar ``t + execution_delay``.
    entry_tags:
        Optional per-bar label recorded on orders, so a report can attribute
        trades to the rule that generated them.
    """
    config = config or BacktestConfig()
    assert_canonical(bars, name=symbol or "bars")
    inst = config.instrument
    ex = config.execution
    symbol = symbol or inst.symbol

    target = _align_target(target_position, bars.index)
    if not inst.allow_short and (target.to_numpy() < 0).any():
        raise ConfigError(f"{symbol}: instrument disallows shorts but the target goes negative")
    if not inst.allow_fractional:
        target = np.trunc(target)

    n = len(bars)
    close = bars["close"].to_numpy(float)
    open_ = bars["open"].to_numpy(float) if "open" in bars else close
    high = bars["high"].to_numpy(float) if "high" in bars else close
    low = bars["low"].to_numpy(float) if "low" in bars else close
    volume = bars["volume"].to_numpy(float) if "volume" in bars else np.full(n, np.nan)
    tags = _align_tags(entry_tags, bars.index)

    volatility = _bar_volatility(bars, config.volatility_lookback)
    targets = target.to_numpy(float)
    delay = ex.execution_delay

    cash = np.full(n, np.nan)
    position = np.zeros(n)
    equity = np.full(n, np.nan)

    order_rows: list[dict] = []
    trades: list[Trade] = []

    # Average-cost position state.
    cash_now = float(config.initial_cash)
    pos_now = 0.0
    avg_price = 0.0
    entry_time: pd.Timestamp | None = None
    entry_bar = 0
    entry_tag = ""
    open_fees = 0.0

    pending: list[tuple[int, Order]] = []   # (submitted_at_bar, order)
    rejected = 0

    for i in range(n):
        # 1. Execute orders whose execution bar is `i`.
        still_pending: list[tuple[int, Order]] = []
        for submitted_at, order in pending:
            if submitted_at + delay > i:
                still_pending.append((submitted_at, order))
                continue
            bar = Bar(open_[i], high[i], low[i], close[i], volume[i])
            cap = (
                ex.costs.impact.fillable(order.quantity, volume[i])
                if ex.allow_partial_fills
                else order.quantity
            )
            fill = match(order, bar, touch_fills=ex.touch_fills, max_fillable=cap)

            if fill.did_fill:
                offset = 0.0
                if takes_liquidity(order.order_type):
                    offset = ex.costs.price_offset(
                        fill.price, fill.filled,
                        tick_size=inst.tick_size,
                        volatility=volatility[i],
                        bar_volume=volume[i],
                    )
                exec_price = fill.price + order.side.sign * offset
                fees = ex.costs.commission.charge(fill.filled, exec_price, inst.multiplier)
                signed = order.side.sign * fill.filled

                cash_now, pos_now, avg_price, realised, closed = _apply_fill(
                    cash_now, pos_now, avg_price, signed, exec_price, inst.multiplier
                )
                cash_now -= fees

                if closed is not None:
                    closed_qty, closed_dir, closed_entry = closed
                    trades.append(
                        Trade(
                            entry_time=entry_time if entry_time is not None else bars.index[i],
                            exit_time=bars.index[i],
                            direction=closed_dir,
                            quantity=closed_qty,
                            entry_price=closed_entry,
                            exit_price=exec_price,
                            gross_pnl=realised,
                            fees=fees + open_fees,
                            net_pnl=realised - fees - open_fees,
                            bars_held=i - entry_bar,
                            entry_tag=entry_tag,
                            exit_tag=order.tag,
                        )
                    )
                    open_fees = 0.0

                if pos_now == 0:
                    entry_time, entry_tag, open_fees = None, "", 0.0
                elif entry_time is None or closed is not None:
                    # A fresh position, or the remainder of a reversal.
                    entry_time, entry_bar, entry_tag = bars.index[i], i, order.tag
                    open_fees = fees if closed is None else 0.0
                else:
                    open_fees += fees

                order_rows.append({
                    "timestamp": bars.index[i],
                    "submitted_at": bars.index[submitted_at],
                    "side": order.side.value,
                    "type": order.order_type.value,
                    "requested": order.quantity,
                    "filled": fill.filled,
                    "raw_price": fill.price,
                    "exec_price": exec_price,
                    "slippage_cost": offset * fill.filled * inst.multiplier,
                    "fees": fees,
                    "tag": order.tag,
                })
                remainder = order.quantity - fill.filled
                if remainder > 1e-12 and ex.allow_partial_fills and i - submitted_at + 1 < ex.max_fill_bars:
                    still_pending.append((submitted_at, _resize(order, remainder)))
                elif remainder > 1e-12:
                    rejected += 1
            else:
                if i - submitted_at + 1 < ex.max_fill_bars and order.time_in_force.value == "gtc":
                    still_pending.append((submitted_at, order))
                else:
                    order_rows.append({
                        "timestamp": bars.index[i],
                        "submitted_at": bars.index[submitted_at],
                        "side": order.side.value,
                        "type": order.order_type.value,
                        "requested": order.quantity,
                        "filled": 0.0,
                        "raw_price": np.nan,
                        "exec_price": np.nan,
                        "slippage_cost": 0.0,
                        "fees": 0.0,
                        "tag": f"{order.tag}|unfilled:{fill.reason}",
                    })
                    rejected += 1
        pending = still_pending

        # 2. Mark to market on this bar's close.
        cash[i] = cash_now
        position[i] = pos_now
        equity[i] = cash_now + pos_now * close[i] * inst.multiplier

        # 3. Decide, using only bars up to and including `i`. The order is
        #    submitted now and can only execute from bar i + delay onwards.
        wanted = targets[i]
        if np.isfinite(wanted):
            outstanding = sum(o.signed_quantity for _, o in pending)
            delta = wanted - pos_now - outstanding
            if abs(delta) > 1e-12 and i + delay < n:
                pending.append((
                    i,
                    Order(
                        side=Side.of(delta),
                        quantity=abs(delta),
                        order_type=OrderType.MARKET,
                        tag=tags[i],
                    ),
                ))

    index = bars.index
    result = BacktestResult(
        equity=pd.Series(equity, index=index, name="equity"),
        cash=pd.Series(cash, index=index, name="cash"),
        position=pd.Series(position, index=index, name="position"),
        marks=pd.Series(close, index=index, name="mark"),
        orders=pd.DataFrame(order_rows) if order_rows else _empty_orders(),
        trades=_trades_frame(trades),
        config=config,
        symbol=symbol,
        diagnostics={
            "execution_delay_bars": delay,
            "unfilled_or_capped_orders": rejected,
            "target_changes": int((np.diff(targets, prepend=0.0) != 0).sum()),
            "admits_lookahead": ex.admits_lookahead,
        },
    )
    if ex.admits_lookahead:
        logger.warning(
            "%s: run used fill_timing='same_close' — results include a known lookahead bias "
            "and must not be used for a verdict.", symbol,
        )
    return result


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------


def _align_target(target: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    if not isinstance(target, pd.Series):
        raise DataError("target_position must be a Series")
    if not target.index.equals(index):
        if not target.index.isin(index).all():
            raise DataError("target_position has timestamps that are not in the bar index")
        target = target.reindex(index)
    return target.astype(float).fillna(0.0)


def _align_tags(tags: pd.Series | None, index: pd.DatetimeIndex) -> np.ndarray:
    if tags is None:
        return np.array([""] * len(index), dtype=object)
    return tags.reindex(index).fillna("").astype(str).to_numpy()


def _bar_volatility(bars: pd.DataFrame, lookback: int) -> np.ndarray:
    """Trailing realised volatility used to scale spread/slippage/impact.

    Strictly backward-looking (``shift(1)`` after the rolling window), so the
    cost charged on bar ``i`` never depends on bar ``i``'s own range.
    """
    ret = bars["close"].pct_change()
    vol = ret.rolling(lookback, min_periods=max(2, lookback // 4)).std().shift(1)
    fallback = float(ret.abs().median()) if np.isfinite(ret.abs().median()) else 0.0
    return vol.fillna(fallback).fillna(0.0).to_numpy(float)


def _resize(order: Order, quantity: float) -> Order:
    return Order(
        side=order.side,
        quantity=quantity,
        order_type=order.order_type,
        limit_price=order.limit_price,
        stop_price=order.stop_price,
        time_in_force=order.time_in_force,
        tag=order.tag,
        meta=dict(order.meta),
    )


def _apply_fill(
    cash: float,
    position: float,
    avg_price: float,
    signed_qty: float,
    price: float,
    multiplier: float,
) -> tuple[float, float, float, float, tuple[float, int, float] | None]:
    """Average-cost accounting for one fill.

    Returns ``(cash, position, avg_price, realised_pnl, closed)`` where
    ``closed`` is ``(quantity, direction, entry_price)`` when this fill shut a
    position (fully or in part), else ``None``.
    """
    cash -= signed_qty * price * multiplier
    new_position = position + signed_qty

    same_direction = position == 0 or np.sign(signed_qty) == np.sign(position)
    if same_direction:
        total = abs(position) + abs(signed_qty)
        avg_price = (avg_price * abs(position) + price * abs(signed_qty)) / total if total else 0.0
        return cash, new_position, avg_price, 0.0, None

    # Reducing, closing, or reversing.
    closing_qty = min(abs(signed_qty), abs(position))
    direction = int(np.sign(position))
    realised = direction * (price - avg_price) * closing_qty * multiplier
    closed = (closing_qty, direction, avg_price)

    if abs(signed_qty) > abs(position):
        # Reversal: the remainder opens a new position at this fill's price.
        avg_price = price
    elif new_position == 0:
        avg_price = 0.0
    return cash, new_position, avg_price, realised, closed


def _empty_orders() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "timestamp", "submitted_at", "side", "type", "requested", "filled",
            "raw_price", "exec_price", "slippage_cost", "fees", "tag",
        ]
    )


def _trades_frame(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=[
                "entry_time", "exit_time", "direction", "quantity", "entry_price",
                "exit_price", "gross_pnl", "fees", "net_pnl", "bars_held",
                "return_pct", "entry_tag", "exit_tag",
            ]
        )
    return pd.DataFrame(
        [
            {
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "direction": t.direction,
                "quantity": t.quantity,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "gross_pnl": t.gross_pnl,
                "fees": t.fees,
                "net_pnl": t.net_pnl,
                "bars_held": t.bars_held,
                "return_pct": t.return_pct,
                "entry_tag": t.entry_tag,
                "exit_tag": t.exit_tag,
            }
            for t in trades
        ]
    )
