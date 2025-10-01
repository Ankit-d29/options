"""
Zerodha Kite Connect integration for the options trading system.

This module provides a complete implementation of the BaseBroker interface
using the Zerodha Kite Connect API.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable
import os
from dotenv import load_dotenv

from kiteconnect import KiteConnect, KiteTicker
from kiteconnect.exceptions import KiteException

from .base_broker import (
    BaseBroker, BrokerOrder, BrokerPosition, BrokerQuote,
    OrderType, OrderSide, OrderStatus, ProductType, Variety
)

# Load environment variables
load_dotenv()

# Setup logging
kite_logger = logging.getLogger("kite_connect")


@dataclass
class KiteConnectConfig:
    """Configuration for Kite Connect broker."""
    api_key: str
    api_secret: str
    access_token: Optional[str] = None
    user_id: Optional[str] = None
    password: Optional[str] = None
    twofa: Optional[str] = None
    pin: Optional[str] = None
    request_token: Optional[str] = None
    redirect_url: Optional[str] = None
    debug: bool = False


class KiteConnectBroker(BaseBroker):
    """
    Zerodha Kite Connect broker implementation.
    
    Provides full integration with Zerodha's trading platform including:
    - Order placement and management
    - Real-time market data
    - Position tracking
    - Account information
    """

    def __init__(self, config: KiteConnectConfig):
        """
        Initialize Kite Connect broker.
        
        Args:
            config: Kite Connect configuration
        """
        self.config = config
        self.kite = KiteConnect(api_key=config.api_key)
        self.ticker = None
        self.is_connected = False
        self.connection_status = "disconnected"
        self._subscribers: List[Callable] = []
        self._instrument_tokens: Dict[str, str] = {}
        
        # Set access token if provided
        if config.access_token:
            self.kite.set_access_token(config.access_token)
            self.is_connected = True
            self.connection_status = "connected"
            kite_logger.info("Kite Connect initialized with access token")
        else:
            kite_logger.info("Kite Connect initialized without access token")

    async def connect(self) -> bool:
        """
        Establish connection to Kite Connect.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.config.access_token:
                # Test connection with existing token
                profile = self.kite.profile()
                if profile:
                    self.is_connected = True
                    self.connection_status = "connected"
                    kite_logger.info(f"Connected to Kite Connect as {profile.get('user_name', 'Unknown')}")
                    return True
            else:
                kite_logger.warning("No access token provided. Manual login required.")
                return False
                
        except KiteException as e:
            kite_logger.error(f"Kite Connect connection failed: {e}")
            self.is_connected = False
            self.connection_status = "disconnected"
            return False
        except Exception as e:
            kite_logger.error(f"Unexpected error during connection: {e}")
            self.is_connected = False
            self.connection_status = "disconnected"
            return False

    async def disconnect(self) -> bool:
        """
        Disconnect from Kite Connect.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        try:
            if self.ticker:
                self.ticker.stop()
                self.ticker = None
            
            self.is_connected = False
            self.connection_status = "disconnected"
            kite_logger.info("Disconnected from Kite Connect")
            return True
            
        except Exception as e:
            kite_logger.error(f"Error during disconnection: {e}")
            return False

    async def get_profile(self) -> Dict[str, Any]:
        """
        Get user profile information.
        
        Returns:
            Dictionary containing user profile data
        """
        try:
            profile = self.kite.profile()
            return profile
        except KiteException as e:
            kite_logger.error(f"Failed to get profile: {e}")
            raise

    async def get_margins(self) -> Dict[str, Any]:
        """
        Get account margins information.
        
        Returns:
            Dictionary containing margin data
        """
        try:
            margins = self.kite.margins()
            return margins
        except KiteException as e:
            kite_logger.error(f"Failed to get margins: {e}")
            raise

    async def place_order(self, order: BrokerOrder) -> BrokerOrder:
        """
        Place an order through Kite Connect.
        
        Args:
            order: Order to place
            
        Returns:
            Updated order with broker-assigned order_id
        """
        try:
            # Convert order to Kite format
            kite_order = {
                "variety": order.variety.value,
                "exchange": "NSE",  # Default to NSE, can be made configurable
                "tradingsymbol": order.symbol,
                "transaction_type": order.side.value,
                "order_type": order.order_type.value,
                "quantity": order.quantity,
                "product": order.product.value,
            }
            
            if order.price:
                kite_order["price"] = float(order.price)
            if order.stop_price:
                kite_order["trigger_price"] = float(order.stop_price)
            if order.tag:
                kite_order["tag"] = order.tag
                
            # Place order
            order_id = self.kite.place_order(**kite_order)
            
            # Update order with broker response
            order.order_id = str(order_id)
            order.status = OrderStatus.OPEN
            order.order_timestamp = datetime.now()
            
            kite_logger.info(f"Order placed successfully: {order_id}")
            return order
            
        except KiteException as e:
            kite_logger.error(f"Failed to place order: {e}")
            order.status = OrderStatus.REJECTED
            order.status_message = str(e)
            return order
        except Exception as e:
            kite_logger.error(f"Unexpected error placing order: {e}")
            order.status = OrderStatus.REJECTED
            order.status_message = str(e)
            return order

    async def modify_order(self, order_id: str, **kwargs) -> BrokerOrder:
        """
        Modify an existing order.
        
        Args:
            order_id: ID of the order to modify
            **kwargs: Parameters to modify
            
        Returns:
            Updated order object
        """
        try:
            # Get current order details
            orders = self.kite.orders()
            current_order = None
            for o in orders:
                if str(o['order_id']) == str(order_id):
                    current_order = o
                    break
            
            if not current_order:
                raise ValueError(f"Order {order_id} not found")
            
            # Prepare modification parameters
            modify_params = {"order_id": order_id}
            
            if "price" in kwargs:
                modify_params["price"] = float(kwargs["price"])
            if "quantity" in kwargs:
                modify_params["quantity"] = kwargs["quantity"]
            if "order_type" in kwargs:
                modify_params["order_type"] = kwargs["order_type"].value
            if "validity" in kwargs:
                modify_params["validity"] = kwargs["validity"]
                
            # Modify order
            self.kite.modify_order(**modify_params)
            
            # Return updated order
            updated_order = self._kite_order_to_broker_order(current_order)
            kite_logger.info(f"Order {order_id} modified successfully")
            return updated_order
            
        except KiteException as e:
            kite_logger.error(f"Failed to modify order {order_id}: {e}")
            raise
        except Exception as e:
            kite_logger.error(f"Unexpected error modifying order {order_id}: {e}")
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            True if cancellation successful, False otherwise
        """
        try:
            self.kite.cancel_order(order_id=order_id)
            kite_logger.info(f"Order {order_id} cancelled successfully")
            return True
            
        except KiteException as e:
            kite_logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
        except Exception as e:
            kite_logger.error(f"Unexpected error cancelling order {order_id}: {e}")
            return False

    async def get_orders(self, order_ids: Optional[List[str]] = None) -> List[BrokerOrder]:
        """
        Get order information.
        
        Args:
            order_ids: Optional list of specific order IDs to retrieve
            
        Returns:
            List of order objects
        """
        try:
            orders = self.kite.orders()
            
            if order_ids:
                # Filter specific orders
                filtered_orders = [o for o in orders if str(o['order_id']) in order_ids]
            else:
                filtered_orders = orders
            
            broker_orders = [self._kite_order_to_broker_order(o) for o in filtered_orders]
            return broker_orders
            
        except KiteException as e:
            kite_logger.error(f"Failed to get orders: {e}")
            raise

    async def get_positions(self) -> List[BrokerPosition]:
        """
        Get current positions.
        
        Returns:
            List of position objects
        """
        try:
            positions = self.kite.positions()
            
            broker_positions = []
            for p in positions:
                if p['quantity'] != 0:  # Only include non-zero positions
                    broker_position = self._kite_position_to_broker_position(p)
                    broker_positions.append(broker_position)
            
            return broker_positions
            
        except KiteException as e:
            kite_logger.error(f"Failed to get positions: {e}")
            raise

    async def get_quote(self, symbols: List[str]) -> Dict[str, BrokerQuote]:
        """
        Get market quotes for symbols.
        
        Args:
            symbols: List of symbol names
            
        Returns:
            Dictionary mapping symbol to quote object
        """
        try:
            # Convert symbols to Kite format (NSE:SYMBOL)
            kite_symbols = [f"NSE:{symbol}" for symbol in symbols]
            
            quotes = self.kite.quote(kite_symbols)
            
            broker_quotes = {}
            for symbol in symbols:
                kite_symbol = f"NSE:{symbol}"
                if kite_symbol in quotes:
                    quote_data = quotes[kite_symbol]
                    broker_quote = self._kite_quote_to_broker_quote(symbol, quote_data)
                    broker_quotes[symbol] = broker_quote
            
            return broker_quotes
            
        except KiteException as e:
            kite_logger.error(f"Failed to get quotes: {e}")
            raise

    async def get_historical_data(self, symbol: str, from_date: datetime, 
                                to_date: datetime, interval: str) -> List[Dict[str, Any]]:
        """
        Get historical data for a symbol.
        
        Args:
            symbol: Symbol name
            from_date: Start date
            to_date: End date
            interval: Data interval (minute, day, etc.)
            
        Returns:
            List of historical data points
        """
        try:
            # Convert to Kite format
            kite_symbol = f"NSE:{symbol}"
            kite_interval = self._convert_interval(interval)
            
            historical_data = self.kite.historical_data(
                instrument_token=self._get_instrument_token(symbol),
                from_date=from_date,
                to_date=to_date,
                interval=kite_interval
            )
            
            return historical_data
            
        except KiteException as e:
            kite_logger.error(f"Failed to get historical data for {symbol}: {e}")
            raise

    async def get_instruments(self, exchange: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of tradable instruments.
        
        Args:
            exchange: Optional exchange filter
            
        Returns:
            List of instrument dictionaries
        """
        try:
            instruments = self.kite.instruments(exchange or "NSE")
            return instruments
            
        except KiteException as e:
            kite_logger.error(f"Failed to get instruments: {e}")
            raise

    async def subscribe_market_data(self, symbols: List[str], 
                                  callback: Callable) -> bool:
        """
        Subscribe to real-time market data.
        
        Args:
            symbols: List of symbols to subscribe to
            callback: Function to call when data is received
            
        Returns:
            True if subscription successful, False otherwise
        """
        try:
            # Get instrument tokens
            instrument_tokens = []
            for symbol in symbols:
                token = self._get_instrument_token(symbol)
                if token:
                    instrument_tokens.append(token)
            
            if not instrument_tokens:
                kite_logger.error("No valid instrument tokens found")
                return False
            
            # Initialize ticker
            if not self.ticker:
                self.ticker = KiteTicker(self.config.api_key, self.config.access_token)
                
                # Set up callbacks
                self.ticker.on_ticks = callback
                self.ticker.on_connect = self._on_connect
                self.ticker.on_close = self._on_close
                self.ticker.on_error = self._on_error
            
            # Subscribe to instruments
            self.ticker.subscribe(instrument_tokens)
            self.ticker.set_mode(self.ticker.MODE_QUOTE, instrument_tokens)
            
            # Start ticker in background
            asyncio.create_task(self._start_ticker())
            
            kite_logger.info(f"Subscribed to market data for {len(instrument_tokens)} instruments")
            return True
            
        except Exception as e:
            kite_logger.error(f"Failed to subscribe to market data: {e}")
            return False

    async def unsubscribe_market_data(self, symbols: List[str]) -> bool:
        """
        Unsubscribe from real-time market data.
        
        Args:
            symbols: List of symbols to unsubscribe from
            
        Returns:
            True if unsubscription successful, False otherwise
        """
        try:
            if not self.ticker:
                return True
            
            # Get instrument tokens
            instrument_tokens = []
            for symbol in symbols:
                token = self._get_instrument_token(symbol)
                if token:
                    instrument_tokens.append(token)
            
            if instrument_tokens:
                self.ticker.unsubscribe(instrument_tokens)
                kite_logger.info(f"Unsubscribed from market data for {len(instrument_tokens)} instruments")
            
            return True
            
        except Exception as e:
            kite_logger.error(f"Failed to unsubscribe from market data: {e}")
            return False

    async def _start_ticker(self):
        """Start the ticker in background."""
        try:
            self.ticker.connect()
        except Exception as e:
            kite_logger.error(f"Failed to start ticker: {e}")

    def _on_connect(self, ws, response):
        """Handle ticker connection."""
        kite_logger.info("Market data ticker connected")

    def _on_close(self, ws, code, reason):
        """Handle ticker disconnection."""
        kite_logger.info(f"Market data ticker disconnected: {code} - {reason}")

    def _on_error(self, ws, code, reason):
        """Handle ticker error."""
        kite_logger.error(f"Market data ticker error: {code} - {reason}")

    def _get_instrument_token(self, symbol: str) -> Optional[str]:
        """
        Get instrument token for a symbol.
        
        Args:
            symbol: Symbol name
            
        Returns:
            Instrument token or None if not found
        """
        try:
            # Check cache first
            if symbol in self._instrument_tokens:
                return self._instrument_tokens[symbol]
            
            # Get from instruments
            instruments = self.kite.instruments("NSE")
            for instrument in instruments:
                if instrument['tradingsymbol'] == symbol:
                    token = str(instrument['instrument_token'])
                    self._instrument_tokens[symbol] = token
                    return token
            
            kite_logger.warning(f"Instrument token not found for {symbol}")
            return None
            
        except Exception as e:
            kite_logger.error(f"Error getting instrument token for {symbol}: {e}")
            return None

    def _convert_interval(self, interval: str) -> str:
        """
        Convert interval to Kite format.
        
        Args:
            interval: Interval string
            
        Returns:
            Kite-compatible interval string
        """
        interval_map = {
            "minute": "minute",
            "1minute": "minute",
            "3minute": "3minute",
            "5minute": "5minute",
            "15minute": "15minute",
            "30minute": "30minute",
            "60minute": "60minute",
            "day": "day",
            "1day": "day"
        }
        
        return interval_map.get(interval, "minute")

    def _kite_order_to_broker_order(self, kite_order: Dict[str, Any]) -> BrokerOrder:
        """
        Convert Kite order to BrokerOrder.
        
        Args:
            kite_order: Kite order dictionary
            
        Returns:
            BrokerOrder object
        """
        return BrokerOrder(
            order_id=str(kite_order['order_id']),
            symbol=kite_order['tradingsymbol'],
            side=OrderSide(kite_order['transaction_type']),
            order_type=OrderType(kite_order['order_type']),
            quantity=kite_order['quantity'],
            price=Decimal(str(kite_order.get('price', 0))) if kite_order.get('price') else None,
            stop_price=Decimal(str(kite_order.get('trigger_price', 0))) if kite_order.get('trigger_price') else None,
            product=ProductType(kite_order['product']),
            variety=Variety(kite_order['variety']),
            status=OrderStatus(kite_order['status']),
            filled_quantity=kite_order['filled_quantity'],
            filled_price=Decimal(str(kite_order['average_price'])) if kite_order.get('average_price') else None,
            average_price=Decimal(str(kite_order['average_price'])) if kite_order.get('average_price') else None,
            pending_quantity=kite_order['pending_quantity'],
            cancelled_quantity=kite_order['cancelled_quantity'],
            order_timestamp=datetime.strptime(kite_order['order_timestamp'], '%Y-%m-%d %H:%M:%S') if kite_order.get('order_timestamp') else None,
            exchange_timestamp=datetime.strptime(kite_order['exchange_timestamp'], '%Y-%m-%d %H:%M:%S') if kite_order.get('exchange_timestamp') else None,
            status_message=kite_order.get('status_message'),
            tag=kite_order.get('tag')
        )

    def _kite_position_to_broker_position(self, kite_position: Dict[str, Any]) -> BrokerPosition:
        """
        Convert Kite position to BrokerPosition.
        
        Args:
            kite_position: Kite position dictionary
            
        Returns:
            BrokerPosition object
        """
        return BrokerPosition(
            symbol=kite_position['tradingsymbol'],
            quantity=kite_position['quantity'],
            average_price=Decimal(str(kite_position['average_price'])),
            last_price=Decimal(str(kite_position['last_price'])),
            day_change=Decimal(str(kite_position['day_change'])),
            day_change_percent=Decimal(str(kite_position['day_change_percent'])),
            unrealized_pnl=Decimal(str(kite_position['unrealised'])),
            realized_pnl=Decimal(str(kite_position['realised'])),
            product=ProductType(kite_position['product']),
            instrument_token=str(kite_position['instrument_token']),
            exchange=kite_position['exchange']
        )

    def _kite_quote_to_broker_quote(self, symbol: str, kite_quote: Dict[str, Any]) -> BrokerQuote:
        """
        Convert Kite quote to BrokerQuote.
        
        Args:
            symbol: Symbol name
            kite_quote: Kite quote dictionary
            
        Returns:
            BrokerQuote object
        """
        return BrokerQuote(
            symbol=symbol,
            last_price=Decimal(str(kite_quote['last_price'])),
            last_quantity=kite_quote['last_quantity'],
            average_price=Decimal(str(kite_quote['average_price'])),
            volume=kite_quote['volume'],
            buy_quantity=kite_quote['buy_quantity'],
            sell_quantity=kite_quote['sell_quantity'],
            ohlc={
                'open': Decimal(str(kite_quote['ohlc']['open'])),
                'high': Decimal(str(kite_quote['ohlc']['high'])),
                'low': Decimal(str(kite_quote['ohlc']['low'])),
                'close': Decimal(str(kite_quote['ohlc']['close']))
            },
            net_change=Decimal(str(kite_quote['net_change'])),
            oi=kite_quote.get('oi'),
            oi_day_high=kite_quote.get('oi_day_high'),
            oi_day_low=kite_quote.get('oi_day_low'),
            timestamp=datetime.strptime(kite_quote['timestamp'], '%Y-%m-%d %H:%M:%S') if kite_quote.get('timestamp') else None,
            depth=kite_quote.get('depth')
        )
