"""
Basic data structure used for general trading function in the trading platform.
"""

from dataclasses import dataclass, field
from datetime import datetime
from logging import INFO
from typing import Optional

from .constant import Direction, Exchange, Interval, Offset, Status, Product, OptionType, OrderType

ACTIVE_STATUSES = set([Status.SUBMITTING, Status.NOTTRADED, Status.PARTTRADED])


@dataclass
class BaseData:
    """
    Any data object needs a gateway_name as source
    and should inherit base data.
    """

    gateway_name = None

    extra = field(default=None, init=False)


@dataclass
class TickData(BaseData):
    """
    Tick data contains information about:
        * last trade in market
        * orderbook snapshot
        * intraday market statistics.
    """

    symbol = None
    exchange = None
    datetime = None

    name = ""
    volume = 0
    turnover = 0
    open_interest = 0
    last_price = 0
    last_volume = 0
    limit_up = 0
    limit_down = 0

    open_price = 0
    high_price = 0
    low_price = 0
    pre_close = 0

    bid_price_1 = 0
    bid_price_2 = 0
    bid_price_3 = 0
    bid_price_4 = 0
    bid_price_5 = 0

    ask_price_1 = 0
    ask_price_2 = 0
    ask_price_3 = 0
    ask_price_4 = 0
    ask_price_5 = 0

    bid_volume_1 = 0
    bid_volume_2 = 0
    bid_volume_3 = 0
    bid_volume_4 = 0
    bid_volume_5 = 0

    ask_volume_1 = 0
    ask_volume_2 = 0
    ask_volume_3 = 0
    ask_volume_4 = 0
    ask_volume_5 = 0

    localtime = None

    def __post_init__(self):
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"


@dataclass
class BarData(BaseData):
    """
    Candlestick bar data of a certain trading period.
    """

    symbol = None
    exchange = None
    datetime = None

    interval = None
    volume = 0
    turnover = 0
    open_interest = 0
    open_price = 0
    high_price = 0
    low_price = 0
    close_price = 0

    def __post_init__(self):
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"


@dataclass
class OrderData(BaseData):
    """
    Order data contains information for tracking lastest status
    of a specific order.
    """

    symbol = None
    exchange = None
    orderid = None

    type = OrderType.LIMIT
    direction = None
    offset = Offset.NONE
    price = 0
    volume = 0
    traded = 0
    status = Status.SUBMITTING
    datetime = None
    reference = ""

    def __post_init__(self):
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"
        self.vt_orderid: str = f"{self.gateway_name}.{self.orderid}"

    def is_active(self):
        """
        Check if the order is active.
        """
        return self.status in ACTIVE_STATUSES

    def create_cancel_request(self):
        """
        Create cancel request object from order.
        """
        req: CancelRequest = CancelRequest(
            orderid=self.orderid, symbol=self.symbol, exchange=self.exchange
        )
        return req


@dataclass
class TradeData(BaseData):
    """
    Trade data contains information of a fill of an order. One order
    can have several trade fills.
    """

    symbol = None
    exchange = None
    orderid = None
    tradeid = None
    direction = None

    offset = Offset.NONE
    price = 0
    volume = 0
    datetime = None

    def __post_init__(self):
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"
        self.vt_orderid: str = f"{self.gateway_name}.{self.orderid}"
        self.vt_tradeid: str = f"{self.gateway_name}.{self.tradeid}"


@dataclass
class PositionData(BaseData):
    """
    Position data is used for tracking each individual position holding.
    """

    symbol = None
    exchange = None
    direction = None

    volume = 0
    frozen = 0
    price = 0
    pnl = 0
    yd_volume = 0

    def __post_init__(self):
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"
        self.vt_positionid: str = f"{self.gateway_name}.{self.vt_symbol}.{self.direction.value}"


@dataclass
class AccountData(BaseData):
    """
    Account data contains information about balance, frozen and
    available.
    """

    accountid = None

    balance = 0
    frozen = 0

    def __post_init__(self):
        """"""
        self.available: float = self.balance - self.frozen
        self.vt_accountid: str = f"{self.gateway_name}.{self.accountid}"


@dataclass
class LogData(BaseData):
    """
    Log data is used for recording log messages on GUI or in log files.
    """

    msg = None
    level = INFO

    def __post_init__(self):
        """"""
        self.time: datetime = datetime.now()


@dataclass
class ContractData(BaseData):
    """
    Contract data contains basic information about each contract traded.
    """

    symbol = None
    exchange = None
    name = None
    product = None
    size = None
    pricetick = None

    min_volume = 1           # minimum order volume
    max_volume = None        # maximum order volume
    stop_supported = False    # whether server supports stop order
    net_position = False      # whether gateway uses net position volume
    history_data = False      # whether gateway provides bar history data

    option_strike = 0
    option_underlying = ""     # vt_symbol of underlying contract
    option_type = None
    option_listed = None
    option_expiry = None
    option_portfolio = ""
    option_index = ""          # for identifying options with same strike price

    def __post_init__(self):
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"


@dataclass
class QuoteData(BaseData):
    """
    Quote data contains information for tracking lastest status
    of a specific quote.
    """

    symbol = None
    exchange = None
    quoteid = None

    bid_price = 0.0
    bid_volume = 0
    ask_price = 0.0
    ask_volume = 0
    bid_offset = Offset.NONE
    ask_offset = Offset.NONE
    status = Status.SUBMITTING
    datetime = None
    reference = ""

    def __post_init__(self):
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"
        self.vt_quoteid: str = f"{self.gateway_name}.{self.quoteid}"

    def is_active(self):
        """
        Check if the quote is active.
        """
        return self.status in ACTIVE_STATUSES

    def create_cancel_request(self):
        """
        Create cancel request object from quote.
        """
        req: CancelRequest = CancelRequest(
            orderid=self.quoteid, symbol=self.symbol, exchange=self.exchange
        )
        return req


@dataclass
class SubscribeRequest:
    """
    Request sending to specific gateway for subscribing tick data update.
    """

    symbol = None
    exchange = None

    def __post_init__(self):
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"


@dataclass
class OrderRequest:
    """
    Request sending to specific gateway for creating a new order.
    """

    symbol = None
    exchange = None
    direction = None
    type = None
    volume = None
    price = 0
    offset = Offset.NONE
    reference = ""

    def __post_init__(self):
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"

    def create_order_data(self, orderid, gateway_name):
        """
        Create order data from request.
        """
        order: OrderData = OrderData(
            symbol=self.symbol,
            exchange=self.exchange,
            orderid=orderid,
            type=self.type,
            direction=self.direction,
            offset=self.offset,
            price=self.price,
            volume=self.volume,
            reference=self.reference,
            gateway_name=gateway_name,
        )
        return order


@dataclass
class CancelRequest:
    """
    Request sending to specific gateway for canceling an existing order.
    """

    orderid = None
    symbol = None
    exchange = None

    def __post_init__(self):
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"


@dataclass
class HistoryRequest:
    """
    Request sending to specific gateway for querying history data.
    """

    symbol = None
    exchange = None
    start = None
    end = None
    interval = None

    def __post_init__(self):
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"


@dataclass
class QuoteRequest:
    """
    Request sending to specific gateway for creating a new quote.
    """

    symbol = None
    exchange = None
    bid_price = None
    bid_volume = None
    ask_price = None
    ask_volume = None
    bid_offset = Offset.NONE
    ask_offset = Offset.NONE
    reference = ""

    def __post_init__(self):
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"

    def create_quote_data(self, quoteid, gateway_name):
        """
        Create quote data from request.
        """
        quote: QuoteData = QuoteData(
            symbol=self.symbol,
            exchange=self.exchange,
            quoteid=quoteid,
            bid_price=self.bid_price,
            bid_volume=self.bid_volume,
            ask_price=self.ask_price,
            ask_volume=self.ask_volume,
            bid_offset=self.bid_offset,
            ask_offset=self.ask_offset,
            reference=self.reference,
            gateway_name=gateway_name,
        )
        return quote
