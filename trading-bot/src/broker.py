"""Abstract broker interface and Alpaca paper implementation."""
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_PAPER_BASE_URL = "https://paper-api.alpaca.markets"
_LIVE_BASE_URL = "https://api.alpaca.markets"

_RISK_CLI_FLAG_ENV = "_ALPACA_RISK_FLAG_CONFIRMED"


@dataclass
class AccountInfo:
    equity: float
    cash: float
    buying_power: float
    currency: str


@dataclass
class Position:
    symbol: str
    qty: float
    market_value: float
    avg_entry_price: float
    unrealized_pl: float


@dataclass
class Order:
    id: str
    symbol: str
    qty: float
    side: str
    order_type: str
    status: str
    filled_avg_price: float | None = None


class Broker(ABC):
    """Abstract broker interface. All implementations must honour this contract."""

    @abstractmethod
    def get_account(self) -> AccountInfo:
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> Position | None:
        ...

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
    ) -> Order:
        ...

    @abstractmethod
    def cancel_all(self) -> None:
        ...

    @abstractmethod
    def is_paper(self) -> bool:
        ...


class AlpacaPaperBroker(Broker):
    """Alpaca broker that defaults to paper trading.

    Live trading requires BOTH:
      - LIVE_TRADING=true  in the environment
      - risk_flag=True     passed explicitly at construction
    If either is absent, paper endpoint is used unconditionally.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        risk_flag: bool = False,
    ) -> None:
        live_env = os.getenv("LIVE_TRADING", "false").lower() == "true"
        self._paper = not (live_env and risk_flag)

        if self._paper:
            base_url = _PAPER_BASE_URL
            reason = "paper endpoint (default)"
            if live_env and not risk_flag:
                reason = "paper endpoint (LIVE_TRADING=true but --i-understand-the-risk flag missing)"
            elif not live_env:
                reason = "paper endpoint (LIVE_TRADING != true)"
        else:
            base_url = _LIVE_BASE_URL
            reason = "LIVE endpoint (LIVE_TRADING=true + risk flag confirmed)"

        logger.warning("AlpacaBroker initialised using %s", reason)

        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
        except ImportError as exc:
            raise ImportError("alpaca-py is required: pip install alpaca-py") from exc

        self._trading = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=self._paper,
        )
        self._data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key,
        )

    # ------------------------------------------------------------------
    # Mandatory double-check at every method call (not just __init__)
    # ------------------------------------------------------------------
    def _assert_paper_if_required(self) -> None:
        live_env = os.getenv("LIVE_TRADING", "false").lower() == "true"
        if not live_env and not self._paper:
            raise RuntimeError(
                "Safety violation: broker was constructed for live trading but "
                "LIVE_TRADING env var is now false. Refusing to submit orders."
            )

    def is_paper(self) -> bool:
        return self._paper

    def get_account(self) -> AccountInfo:
        self._assert_paper_if_required()
        acct = self._trading.get_account()
        return AccountInfo(
            equity=float(acct.equity),
            cash=float(acct.cash),
            buying_power=float(acct.buying_power),
            currency=acct.currency,
        )

    def get_position(self, symbol: str) -> Position | None:
        self._assert_paper_if_required()
        try:
            pos = self._trading.get_open_position(symbol)
            return Position(
                symbol=pos.symbol,
                qty=float(pos.qty),
                market_value=float(pos.market_value),
                avg_entry_price=float(pos.avg_entry_price),
                unrealized_pl=float(pos.unrealized_pl),
            )
        except Exception:
            return None

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
    ) -> Order:
        self._assert_paper_if_required()
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        tif_enum = TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC

        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side_enum,
            time_in_force=tif_enum,
        )
        order = self._trading.submit_order(req)
        logger.info(
            "Order submitted symbol=%s qty=%s side=%s paper=%s id=%s",
            symbol, qty, side, self._paper, order.id,
        )
        return Order(
            id=str(order.id),
            symbol=order.symbol,
            qty=float(order.qty),
            side=str(order.side),
            order_type=str(order.order_type),
            status=str(order.status),
            filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else None,
        )

    def cancel_all(self) -> None:
        self._assert_paper_if_required()
        self._trading.cancel_orders()
        logger.info("All open orders cancelled (paper=%s)", self._paper)


def create_broker(cfg: dict[str, Any], risk_flag: bool = False) -> AlpacaPaperBroker:
    """Factory: build broker from loaded config dict."""
    return AlpacaPaperBroker(
        api_key=cfg["_env"]["alpaca_api_key"],
        secret_key=cfg["_env"]["alpaca_secret_key"],
        risk_flag=risk_flag,
    )
