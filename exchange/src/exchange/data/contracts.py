"""Futures contract identity and metadata.

A futures "symbol" is not one instrument: ``NG`` is a *root*, and what
trades is a ladder of dated contracts (``NGV26``, ``NGX26``, ``NGZ26``,
``NGF27``, …). Every quantity that turns a price move into money — the
multiplier, the tick value, the currency — lives on the contract, not on the
root, so it has to be carried explicitly. The audited project had no
concept of a contract at all: a user researching NG would load Yahoo's
``NG=F``, whose roll rule and adjustment are undocumented.

Symbol grammar
--------------
``<root><month code><year>`` where the month code is the standard CME letter
and the year is 1, 2 or 4 digits::

    NGV26   -> root NG, October 2026
    CLZ5    -> root CL, December 2025 (1-digit year, resolved near `today`)
    ESH2026 -> root ES, March 2026
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from datetime import date

import pandas as pd

from exchange.core.errors import ConfigError

# CME month codes. The ordering is the calendar month, 1-based.
MONTH_CODES = "FGHJKMNQUVXZ"
CODE_TO_MONTH = {c: i + 1 for i, c in enumerate(MONTH_CODES)}
MONTH_TO_CODE = {i + 1: c for i, c in enumerate(MONTH_CODES)}

_SYMBOL_RE = re.compile(r"^(?P<root>[A-Z0-9]{1,4}?)(?P<code>[FGHJKMNQUVXZ])(?P<year>\d{1,4})$")


@dataclass(frozen=True, order=True)
class ContractMonth:
    """A delivery month: ``(year, month)`` with a canonical symbol form."""

    year: int
    month: int

    def __post_init__(self) -> None:
        if not 1 <= self.month <= 12:
            raise ConfigError(f"month must be 1..12, got {self.month}")
        if not 1900 <= self.year <= 2200:
            raise ConfigError(f"implausible contract year {self.year}")

    @property
    def code(self) -> str:
        return MONTH_TO_CODE[self.month]

    def symbol(self, root: str, *, year_digits: int = 2) -> str:
        year = self.year % 100 if year_digits == 2 else self.year
        return f"{root}{self.code}{year:0{year_digits}d}"

    def next(self, step: int = 1) -> ContractMonth:
        total = (self.year * 12 + self.month - 1) + step
        return ContractMonth(total // 12, total % 12 + 1)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.year}-{self.month:02d}"


def parse_contract_symbol(symbol: str, *, pivot_year: int | None = None) -> tuple[str, ContractMonth]:
    """Split ``NGV26`` into ``("NG", ContractMonth(2026, 10))``.

    A 1- or 2-digit year is resolved to the window
    ``[pivot_year - 20, pivot_year + 79]`` (default pivot: the current
    year), which is the convention exchanges use and avoids resolving a 2026
    contract to 1926.
    """
    text = str(symbol).strip().upper().replace("-", "").replace("_", "")
    match = _SYMBOL_RE.match(text)
    if match is None:
        raise ConfigError(
            f"cannot parse futures symbol {symbol!r}; expected <root><month code><year>, e.g. NGV26"
        )
    root = match.group("root")
    month = CODE_TO_MONTH[match.group("code")]
    raw_year = match.group("year")
    year = _resolve_year(int(raw_year), len(raw_year), pivot_year)
    return root, ContractMonth(year, month)


def _resolve_year(value: int, digits: int, pivot_year: int | None) -> int:
    if digits >= 4:
        return value
    pivot = pivot_year if pivot_year is not None else date.today().year
    if digits == 2:
        century = (pivot - 20) // 100 * 100
        candidate = century + value
        while candidate < pivot - 20:
            candidate += 100
        return candidate
    # Single digit: pick the nearest year ending in `value` within [-2, +7].
    base = pivot - 2
    return base + ((value - base) % 10)


def contract_symbol(root: str, year: int, month: int, *, year_digits: int = 2) -> str:
    """Build the canonical symbol for a root and delivery month."""
    return ContractMonth(year, month).symbol(root.upper(), year_digits=year_digits)


@dataclass(frozen=True)
class ContractSpec:
    """Everything needed to turn a price move on one contract into money.

    Dates are calendar dates (no time-of-day): the roll logic compares them
    against the *date* part of a bar timestamp, which is unambiguous once
    ``exchange_tz`` is known.
    """

    symbol: str
    root: str
    month: ContractMonth
    exchange: str = ""
    currency: str = "USD"
    exchange_tz: str = "UTC"
    multiplier: float = 1.0          # contract size, e.g. 10_000 MMBtu for NG
    tick_size: float = 0.01
    tick_value: float | None = None  # money per tick; defaults to tick_size * multiplier
    last_trade_date: date | None = None
    first_notice_date: date | None = None
    expiration_date: date | None = None
    initial_margin: float | None = None
    maintenance_margin: float | None = None
    trading_session: str = ""        # free-form label, e.g. "CME Globex 18:00-17:00 ET"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.multiplier <= 0:
            raise ConfigError(f"{self.symbol}: multiplier must be positive")
        if self.tick_size <= 0:
            raise ConfigError(f"{self.symbol}: tick_size must be positive")

    @property
    def money_per_tick(self) -> float:
        """Currency value of one minimum price increment."""
        return self.tick_value if self.tick_value is not None else self.tick_size * self.multiplier

    @property
    def point_value(self) -> float:
        """Currency value of a one-unit price move (== multiplier)."""
        return self.money_per_tick / self.tick_size

    @property
    def roll_reference_date(self) -> date | None:
        """The date the position must be out by.

        First notice date when it exists (physically delivered contracts must
        be exited before it), otherwise last trade date, otherwise expiration.
        """
        for candidate in (self.first_notice_date, self.last_trade_date, self.expiration_date):
            if candidate is not None:
                return candidate
        return None

    def round_to_tick(self, price: float, *, mode: str = "nearest") -> float:
        """Snap a price to the contract's tick grid."""
        ticks = price / self.tick_size
        if mode == "nearest":
            ticks = round(ticks)
        elif mode == "up":
            ticks = math.ceil(ticks - 1e-9)
        elif mode == "down":
            ticks = math.floor(ticks + 1e-9)
        else:
            raise ConfigError(f"unknown rounding mode {mode!r}")
        return float(ticks * self.tick_size)

    def with_dates(self, **kwargs: date | None) -> ContractSpec:
        return replace(self, **kwargs)

    def to_dict(self) -> dict:
        out = {
            "symbol": self.symbol,
            "root": self.root,
            "year": self.month.year,
            "month": self.month.month,
            "exchange": self.exchange,
            "currency": self.currency,
            "exchange_tz": self.exchange_tz,
            "multiplier": self.multiplier,
            "tick_size": self.tick_size,
            "tick_value": self.money_per_tick,
            "trading_session": self.trading_session,
            "initial_margin": self.initial_margin,
            "maintenance_margin": self.maintenance_margin,
        }
        for name in ("last_trade_date", "first_notice_date", "expiration_date"):
            value = getattr(self, name)
            out[name] = None if value is None else value.isoformat()
        if self.meta:
            out["meta"] = dict(self.meta)
        return out


@dataclass(frozen=True)
class RootSpec:
    """Defaults shared by every contract of one root.

    ``months`` is the set of listed delivery months (all twelve for NG and
    CL; ``HMUZ`` for the equity index futures), used when generating a
    contract ladder.
    """

    root: str
    name: str = ""
    exchange: str = ""
    currency: str = "USD"
    exchange_tz: str = "UTC"
    multiplier: float = 1.0
    tick_size: float = 0.01
    tick_value: float | None = None
    months: str = MONTH_CODES
    trading_session: str = ""
    sector: str = ""
    meta: dict = field(default_factory=dict)

    def contract(self, year: int, month: int, **overrides) -> ContractSpec:
        """Instantiate one dated contract from these defaults."""
        cm = ContractMonth(year, month)
        base = {
            "symbol": cm.symbol(self.root),
            "root": self.root,
            "month": cm,
            "exchange": self.exchange,
            "currency": self.currency,
            "exchange_tz": self.exchange_tz,
            "multiplier": self.multiplier,
            "tick_size": self.tick_size,
            "tick_value": self.tick_value,
            "trading_session": self.trading_session,
            "meta": dict(self.meta),
        }
        base.update(overrides)
        return ContractSpec(**base)

    def ladder(self, start: ContractMonth, n: int) -> list[ContractSpec]:
        """The next ``n`` listed contracts from ``start`` inclusive."""
        out: list[ContractSpec] = []
        cursor = start
        allowed = {CODE_TO_MONTH[c] for c in self.months}
        guard = 0
        while len(out) < n and guard < 12 * n + 24:
            if cursor.month in allowed:
                out.append(self.contract(cursor.year, cursor.month))
            cursor = cursor.next()
            guard += 1
        return out


# --------------------------------------------------------------------------
# A small built-in catalogue.
#
# These are *reference defaults for research*, not a substitute for the
# exchange's own contract specifications. Margins move constantly and are
# deliberately left unset. Anything that decides real money should come from
# a provider.
# --------------------------------------------------------------------------

BUILTIN_ROOTS: dict[str, RootSpec] = {
    "NG": RootSpec(
        root="NG", name="Henry Hub Natural Gas", exchange="NYMEX", currency="USD",
        exchange_tz="America/New_York", multiplier=10_000.0, tick_size=0.001,
        tick_value=10.0, months=MONTH_CODES, sector="energy",
        trading_session="CME Globex Sun-Fri 18:00-17:00 ET",
    ),
    "CL": RootSpec(
        root="CL", name="WTI Crude Oil", exchange="NYMEX", currency="USD",
        exchange_tz="America/New_York", multiplier=1_000.0, tick_size=0.01,
        tick_value=10.0, months=MONTH_CODES, sector="energy",
        trading_session="CME Globex Sun-Fri 18:00-17:00 ET",
    ),
    "BZ": RootSpec(
        root="BZ", name="Brent Crude Oil (NYMEX listed)", exchange="NYMEX",
        currency="USD", exchange_tz="America/New_York", multiplier=1_000.0,
        tick_size=0.01, tick_value=10.0, sector="energy",
    ),
    "RB": RootSpec(
        root="RB", name="RBOB Gasoline", exchange="NYMEX", currency="USD",
        exchange_tz="America/New_York", multiplier=42_000.0, tick_size=0.0001,
        tick_value=4.2, sector="energy",
    ),
    "HO": RootSpec(
        root="HO", name="NY Harbor ULSD (heating oil)", exchange="NYMEX", currency="USD",
        exchange_tz="America/New_York", multiplier=42_000.0, tick_size=0.0001,
        tick_value=4.2, sector="energy",
    ),
    "ES": RootSpec(
        root="ES", name="E-mini S&P 500", exchange="CME", currency="USD",
        exchange_tz="America/Chicago", multiplier=50.0, tick_size=0.25,
        tick_value=12.5, months="HMUZ", sector="equity_index",
    ),
    "GC": RootSpec(
        root="GC", name="Gold", exchange="COMEX", currency="USD",
        exchange_tz="America/New_York", multiplier=100.0, tick_size=0.1,
        tick_value=10.0, months="GJMQVZ", sector="metals",
    ),
    "ZC": RootSpec(
        root="ZC", name="Corn", exchange="CBOT", currency="USD",
        exchange_tz="America/Chicago", multiplier=5_000.0, tick_size=0.25,
        tick_value=12.5, months="HKNUZ", sector="agriculture",
    ),
}


class ContractRegistry:
    """Lookup of contract specifications, with a synthesising fallback.

    Real research needs a provider-backed registry (expirations and notice
    dates are exchange data). This class keeps that interface but can also
    *derive* a plausible spec from the built-in root defaults so the pipeline
    is testable without a data vendor. Derived specs are marked
    ``meta["derived"] = True`` and their dates carry
    ``meta["dates_estimated"] = True`` — the report states when a result
    depends on estimated roll dates.
    """

    def __init__(self, roots: dict[str, RootSpec] | None = None) -> None:
        self._roots = dict(roots or BUILTIN_ROOTS)
        self._contracts: dict[str, ContractSpec] = {}

    def register_root(self, spec: RootSpec) -> None:
        self._roots[spec.root.upper()] = spec

    def register(self, spec: ContractSpec) -> None:
        self._contracts[spec.symbol.upper()] = spec

    def register_many(self, specs) -> None:
        for spec in specs:
            self.register(spec)

    @property
    def roots(self) -> list[str]:
        return sorted(self._roots)

    def root_spec(self, root: str) -> RootSpec:
        key = root.upper()
        if key not in self._roots:
            raise ConfigError(f"unknown futures root {root!r}; known: {self.roots}")
        return self._roots[key]

    def get(self, symbol: str, *, derive: bool = True) -> ContractSpec:
        """Return the spec for a dated contract symbol."""
        key = str(symbol).strip().upper()
        if key in self._contracts:
            return self._contracts[key]
        if not derive:
            raise ConfigError(f"no registered contract spec for {symbol!r}")
        root, month = parse_contract_symbol(key)
        return self.root_spec(root).contract(
            month.year, month.month, meta={"derived": True, "dates_estimated": True}
        )

    def known(self) -> list[str]:
        return sorted(self._contracts)


def estimate_last_trade_date(spec: ContractSpec, *, business_days_before_month: int = 3) -> date:
    """A crude last-trade-date estimate for contracts with no vendor data.

    Energy futures (NG, CL) stop trading a few business days before the first
    of the delivery month. This helper exists so a pipeline can run
    end-to-end without a vendor feed; it is **not** accurate enough for
    production and every series built on it is flagged accordingly.
    """
    first_of_month = pd.Timestamp(year=spec.month.year, month=spec.month.month, day=1)
    estimate = first_of_month - pd.tseries.offsets.BDay(business_days_before_month)
    return estimate.date()
