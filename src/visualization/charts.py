"""Chart builders and GitHub-safe rendering.

GitHub renders notebooks statically: interactive plotly output (including
every vectorbt ``.plot()``) shows up as a blank cell. The helpers here
force static PNG output so committed notebooks display all figures.

Call :func:`setup_github_rendering` once at the top of a notebook, then
display any plotly figure with :func:`show`.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEFAULT_WIDTH = 1000
DEFAULT_HEIGHT = 500

# Kaleido >= 1.0 renders through a locally installed Chrome/Chromium.
# Headless environments (CI, cloud sandboxes) often ship Chromium outside
# PATH — check the usual suspects and expose the binary via BROWSER_PATH,
# which choreographer (kaleido's backend) honors as an override.
_CHROMIUM_FALLBACK_PATHS = (
    "/opt/pw-browsers/chromium",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/usr/bin/google-chrome",
)


def _ensure_kaleido_browser() -> None:
    if os.environ.get("BROWSER_PATH"):
        return
    for name in ("google-chrome", "chrome", "chromium", "chromium-browser"):
        if shutil.which(name):
            return
    for candidate in _CHROMIUM_FALLBACK_PATHS:
        if Path(candidate).exists():
            os.environ["BROWSER_PATH"] = candidate
            return


def setup_github_rendering(width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT) -> None:
    """Make PNG the default plotly renderer (requires kaleido).

    After this call ``fig.show()`` — and therefore vectorbt's own plots —
    embed static images into the notebook, which GitHub can display.
    """
    import plotly.io as pio

    _ensure_kaleido_browser()
    pio.renderers.default = "png"
    pio.renderers["png"].width = width
    pio.renderers["png"].height = height


def _json_safe(fig: go.Figure) -> go.Figure:
    """Round-trip a figure through plotly's JSON encoder.

    vectorbt figures may carry ``pd.Timestamp`` objects that kaleido's
    strict orjson serializer rejects; plotly's own encoder converts them
    to ISO strings.
    """
    import json

    import plotly.io as pio

    return go.Figure(json.loads(pio.to_json(fig, validate=False)))


def show(fig: go.Figure, *, width: int | None = None, height: int | None = None) -> None:
    """Display a plotly figure as a static PNG (GitHub-safe)."""
    safe = _json_safe(fig)
    safe.show(renderer="png", width=width or fig.layout.width or DEFAULT_WIDTH,
              height=height or fig.layout.height or DEFAULT_HEIGHT)


def price_chart(
    ohlcv: pd.DataFrame,
    *,
    title: str = "Price",
    overlays: dict[str, pd.Series] | None = None,
    show_volume: bool = True,
) -> go.Figure:
    """Candlestick chart with optional indicator overlays and volume pane."""
    rows = 2 if show_volume else 1
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25] if show_volume else [1.0],
        vertical_spacing=0.03,
    )
    fig.add_trace(
        go.Candlestick(
            x=ohlcv.index, open=ohlcv["Open"], high=ohlcv["High"],
            low=ohlcv["Low"], close=ohlcv["Close"], name="OHLC",
        ),
        row=1, col=1,
    )
    for name, series in (overlays or {}).items():
        fig.add_trace(go.Scatter(x=series.index, y=series, name=name, line=dict(width=1.5)), row=1, col=1)
    if show_volume and "Volume" in ohlcv.columns:
        fig.add_trace(
            go.Bar(x=ohlcv.index, y=ohlcv["Volume"], name="Volume", marker_color="rgba(100,110,130,0.5)"),
            row=rows, col=1,
        )
    fig.update_layout(
        title=title, xaxis_rangeslider_visible=False,
        width=DEFAULT_WIDTH, height=600, legend=dict(orientation="h", y=1.02),
        margin=dict(l=60, r=30, t=60, b=40),
    )
    return fig


def equity_drawdown_chart(
    equity: pd.Series,
    drawdown: pd.Series,
    *,
    title: str = "Equity & Drawdown",
) -> go.Figure:
    """Equity curve with the drawdown profile underneath."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3], vertical_spacing=0.04,
        subplot_titles=("Equity", "Drawdown"),
    )
    fig.add_trace(go.Scatter(x=equity.index, y=equity, name="Equity", line=dict(width=1.8)), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=drawdown.index, y=drawdown * 100, name="Drawdown %",
            fill="tozeroy", line=dict(width=1, color="crimson"),
        ),
        row=2, col=1,
    )
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    fig.update_layout(
        title=title, width=DEFAULT_WIDTH, height=600, showlegend=False,
        margin=dict(l=60, r=30, t=60, b=40),
    )
    return fig
